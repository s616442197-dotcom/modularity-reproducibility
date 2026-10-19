from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

from dataset import BinaryPairDataset
from loss import WeightedMSELoss
from model import ModularMLP
from task import BinaryArithmeticTask, ExperimentData


def iter_batches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    if int(batch_size) >= len(x):
        yield x, y
        return
    order = np.arange(len(x))
    rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        yield x[idx], y[idx]


def rule_accuracy(pred: np.ndarray, target: np.ndarray, data: ExperimentData, threshold: float = 0.5) -> dict[int, float]:
    binary = (pred > threshold).astype(np.float32)
    acc = {}
    for pos, rule_id in enumerate(data.rule_indices):
        start = pos * data.pattern_size
        end = start + data.pattern_size
        acc[rule_id] = float(np.mean(binary[:, start:end] == target[:, start:end]))
    return acc


def mean_accuracy(acc: dict[int, float]) -> float:
    return float(np.mean(list(acc.values()))) if acc else float('nan')


def named_accuracy(acc: dict[int, float], data: ExperimentData) -> dict[str, float]:
    return {data.rule_names[pos]: float(acc[rule_id]) for pos, rule_id in enumerate(data.rule_indices)}


def rule_acc_matrix(history_values: list[dict], rule_indices: list[int]) -> np.ndarray:
    matrix = np.zeros((len(history_values), len(rule_indices)), dtype=float)
    for epoch_idx, acc in enumerate(history_values):
        for rule_pos, rule_id in enumerate(rule_indices):
            matrix[epoch_idx, rule_pos] = float(acc.get(rule_id, acc.get(str(rule_id), np.nan)))
    return matrix


def format_rule_accuracy(acc: dict[int, float], data: ExperimentData) -> str:
    parts = []
    for pos, rule_id in enumerate(data.rule_indices):
        name = data.rule_names[pos]
        parts.append('{}={:.3f}'.format(name, acc[rule_id]))
    return ', '.join(parts)


def torch_rule_accuracy(pred, target, data: ExperimentData, threshold: float = 0.5) -> dict[int, float]:
    binary = (pred > threshold).to(target.dtype)
    acc = {}
    for pos, rule_id in enumerate(data.rule_indices):
        start = pos * data.pattern_size
        end = start + data.pattern_size
        acc[rule_id] = float((binary[:, start:end] == target[:, start:end]).float().mean().item())
    return acc


def should_save_snapshot(epoch: int, snapshot_every: int | None) -> bool:
    return snapshot_every is not None and int(snapshot_every) > 0 and epoch % int(snapshot_every) == 0


def should_record_epoch(epoch: int, epochs: int, log_every: int | None, snapshot_every: int | None, eval_every: int | None = None) -> bool:
    if epoch == 1 or epoch == epochs:
        return True
    if should_save_snapshot(epoch, snapshot_every):
        return True
    return eval_every is not None and int(eval_every) > 0 and epoch % int(eval_every) == 0


def train_one_run_torch(model, data: ExperimentData, loss, seed: int, epochs: int = 100, batch_size: int = 32, lr: float = 0.05, log_every: int = 20, optimizer_name: str = 'sgd', weight_decay: float = 0.0, snapshot_every: int | None = 1000, snapshot_callback=None, eval_every: int | None = None):
    try:
        import torch
    except Exception as exc:
        raise ImportError('PyTorch is required for Torch models') from exc

    device = model.device
    torch.manual_seed(seed + 200)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed + 200)

    x_train = torch.as_tensor(data.x_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(data.y_train, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(data.x_test, dtype=torch.float32, device=device)
    y_test = torch.as_tensor(data.y_test, dtype=torch.float32, device=device)

    if hasattr(loss, 'output_weights'):
        output_weights = loss.output_weights(data)
    else:
        output_weights = np.ones((1, data.y_train.shape[1]), dtype=np.float32)
    output_weights_t = torch.as_tensor(output_weights, dtype=torch.float32, device=device)

    trainable_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named_params]
    l1_params = [p for name, p in trainable_named_params if p.dim() >= 2 and 'bias' not in name.lower()]
    if not trainable_params:
        raise ValueError('Torch model has no trainable parameters')
    optimizer_key = str(optimizer_name).lower().strip()
    if optimizer_key == 'adam':
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_key == 'adamw':
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optimizer_key == 'sgd':
        optimizer = torch.optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_name must be 'sgd', 'adam', or 'adamw', got {!r}".format(optimizer_name))
    history = {'epoch': [], 'loss': [], 'train_mean_acc': [], 'test_mean_acc': [], 'train_rule_acc': [], 'test_rule_acc': [], 'train_task_acc': [], 'test_task_acc': []}
    n_train = x_train.shape[0]
    use_full_batch = int(batch_size) >= int(n_train)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        if use_full_batch:
            batch_iter = ((x_train, y_train),)
        else:
            order = torch.randperm(n_train, device=device)
            batch_iter = ((x_train.index_select(0, order[start : start + batch_size]), y_train.index_select(0, order[start : start + batch_size])) for start in range(0, n_train, batch_size))
        for xb, yb in batch_iter:
            optimizer.zero_grad(set_to_none=True)
            pred = model.forward_tensor(xb)
            diff = pred - yb
            batch_loss = torch.mean(output_weights_t * diff * diff)
            objective = batch_loss
            if getattr(loss, 'l1', 0.0) and l1_params:
                gamma = float(getattr(loss, 'gamma', 1.0))
                eps = float(getattr(loss, 'regularization_epsilon', 1e-8))
                l_penalty = sum((p.abs() + eps).pow(gamma).mean() for p in l1_params)
                objective = objective + float(loss.l1) * l_penalty
            objective.backward()
            optimizer.step()
            losses.append(float(batch_loss.detach().item()))

        if should_record_epoch(epoch, epochs, log_every, snapshot_every, eval_every):
            model.eval()
            with torch.no_grad():
                train_acc = torch_rule_accuracy(model.forward_tensor(x_train), y_train, data)
                test_acc = torch_rule_accuracy(model.forward_tensor(x_test), y_test, data)
            history['epoch'].append(int(epoch))
            history['loss'].append(float(np.mean(losses)))
            history['train_mean_acc'].append(mean_accuracy(train_acc))
            history['test_mean_acc'].append(mean_accuracy(test_acc))
            history['train_rule_acc'].append(train_acc)
            history['test_rule_acc'].append(test_acc)
            history['train_task_acc'].append(named_accuracy(train_acc, data))
            history['test_task_acc'].append(named_accuracy(test_acc, data))
            if snapshot_callback is not None and should_save_snapshot(epoch, snapshot_every):
                snapshot_callback(epoch, model, history)
            if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                print('  epoch {:4d}/{} | loss={:.5f} | test_mean={:.3f} | test_by_task=[{}] | device={}'.format(epoch, epochs, history['loss'][-1], history['test_mean_acc'][-1], format_rule_accuracy(test_acc, data), device))
    return model, history


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def save_json(path: Path, data: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def make_output_dir(output_root: str | Path, experiment_name: str) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    out = root / experiment_name
    if not out.exists():
        out.mkdir()
        return out
    i = 1
    while True:
        candidate = root / '{}_{}'.format(experiment_name, i)
        if not candidate.exists():
            candidate.mkdir()
            return candidate
        i += 1


def train_one_run(model_template, data: ExperimentData, loss, seed: int, epochs: int = 100, batch_size: int = 32, lr: float = 0.05, log_every: int = 20, optimizer_name: str = 'sgd', weight_decay: float = 0.0, snapshot_every: int | None = 1000, snapshot_callback=None, eval_every: int | None = None):
    model = model_template.build(input_dim=data.x_train.shape[1], output_dim=data.y_train.shape[1], seed=seed + 100)
    if getattr(model, 'is_torch_model', False):
        return train_one_run_torch(model, data, loss, seed=seed, epochs=epochs, batch_size=batch_size, lr=lr, log_every=log_every, optimizer_name=optimizer_name, weight_decay=weight_decay, snapshot_every=snapshot_every, snapshot_callback=snapshot_callback, eval_every=eval_every)

    rng = np.random.default_rng(seed + 200)
    history = {'epoch': [], 'loss': [], 'train_mean_acc': [], 'test_mean_acc': [], 'train_rule_acc': [], 'test_rule_acc': [], 'train_task_acc': [], 'test_task_acc': []}
    for epoch in range(1, epochs + 1):
        losses = []
        for xb, yb in iter_batches(data.x_train, data.y_train, batch_size, rng):
            pred, cache = model.forward(xb)
            batch_loss, grad = loss.loss_and_grad(pred, yb, data)
            loss_l1 = getattr(loss, 'l1', 0.0)
            loss_gamma = getattr(loss, 'gamma', 1.0)
            regularization_epsilon = getattr(loss, 'regularization_epsilon', 1e-8)
            if hasattr(model, 'train_step_from_cache'):
                model.train_step_from_cache(
                    xb,
                    grad,
                    cache,
                    lr=lr,
                    l1=loss_l1,
                    gamma=loss_gamma,
                    regularization_epsilon=regularization_epsilon,
                )
            else:
                model.train_step(
                    xb,
                    grad,
                    lr=lr,
                    l1=loss_l1,
                    gamma=loss_gamma,
                    regularization_epsilon=regularization_epsilon,
                )
            losses.append(batch_loss)
        if should_record_epoch(epoch, epochs, log_every, snapshot_every, eval_every):
            train_acc = rule_accuracy(model.predict(data.x_train), data.y_train, data)
            test_acc = rule_accuracy(model.predict(data.x_test), data.y_test, data)
            history['epoch'].append(int(epoch))
            history['loss'].append(float(np.mean(losses)))
            history['train_mean_acc'].append(mean_accuracy(train_acc))
            history['test_mean_acc'].append(mean_accuracy(test_acc))
            history['train_rule_acc'].append(train_acc)
            history['test_rule_acc'].append(test_acc)
            history['train_task_acc'].append(named_accuracy(train_acc, data))
            history['test_task_acc'].append(named_accuracy(test_acc, data))
            if snapshot_callback is not None and should_save_snapshot(epoch, snapshot_every):
                snapshot_callback(epoch, model, history)
            if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                print('  epoch {:4d}/{} | loss={:.5f} | test_mean={:.3f} | test_by_task=[{}]'.format(epoch, epochs, history['loss'][-1], history['test_mean_acc'][-1], format_rule_accuracy(test_acc, data)))
    return model, history


def save_run_data(run_dir: Path, model, data: ExperimentData, history: dict) -> None:
    state = model.state_dict()
    np.savez(
        run_dir / 'run_data.npz',
        **state,
        x_test=data.x_test,
        y_test=data.y_test,
        pattern_size=np.asarray(data.pattern_size),
        rule_indices=np.asarray(data.rule_indices),
        rule_names=np.asarray(data.rule_names),
    )
    savemat(
        run_dir / 'run_data.mat',
        {
            **state,
            'images': data.x_test,
            'labels': data.y_test,
            'loss_history': np.asarray(history['loss']),
            'history_epoch': np.asarray(history.get('epoch', np.arange(1, len(history.get('loss', [])) + 1))),
            'train_mean_acc': np.asarray(history['train_mean_acc']),
            'test_mean_acc': np.asarray(history['test_mean_acc']),
            'train_rule_acc': rule_acc_matrix(history.get('train_rule_acc', []), data.rule_indices),
            'test_rule_acc': rule_acc_matrix(history.get('test_rule_acc', []), data.rule_indices),
            'pattern_size': data.pattern_size,
            'rule_indices': np.asarray(data.rule_indices),
        },
    )


def save_run(run_dir: Path, model, data: ExperimentData, history: dict, config: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / 'config.json', config)
    save_json(run_dir / 'history.json', history)
    save_run_data(run_dir, model, data, history)


def snapshot_run_dir(experiment_dir: Path, epoch: int, repeat_idx: int) -> Path:
    return experiment_dir / 'snapshots' / 'epoch_{:06d}'.format(epoch) / 'run_{:03d}'.format(repeat_idx)


def run_repeats(
    dataset,
    task,
    model,
    loss,
    experiment_name: str,
    output_root: str | Path = 'results/simulation',
    repeat: int = 5,
    seed: int = 0,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.05,
    log_every: int = 20,
    optimizer_name: str = 'sgd',
    weight_decay: float = 0.0,
    snapshot_every: int | None = 1000,
    eval_every: int | None = None,
) -> Path:
    experiment_dir = make_output_dir(output_root, experiment_name)
    summaries = []
    for repeat_idx in range(repeat):
        run_seed = seed + repeat_idx
        print('\nRun {}/{}: seed={}'.format(repeat_idx + 1, repeat, run_seed))
        bundle = dataset.build(seed=run_seed)
        data = task.prepare(bundle)
        run_dir = experiment_dir / 'run_{:03d}'.format(repeat_idx)
        config = {
            'dataset': dataset.__class__.__name__,
            'task': task.__class__.__name__,
            'model': model.__class__.__name__,
            'loss': loss.__class__.__name__,
            'loss_config': loss.to_config() if hasattr(loss, 'to_config') else {},
            'seed': run_seed,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr,
            'optimizer_name': optimizer_name,
            'weight_decay': weight_decay,
            'snapshot_every': snapshot_every,
            'eval_every': eval_every,
            'rule_indices': data.rule_indices,
            'rule_names': data.rule_names,
        }

        def save_epoch_snapshot(epoch: int, current_model, current_history: dict) -> None:
            snapshot_config = dict(config)
            snapshot_config['snapshot_epoch'] = int(epoch)
            snapshot_config['snapshot_source_run'] = 'run_{:03d}'.format(repeat_idx)
            save_run(snapshot_run_dir(experiment_dir, epoch, repeat_idx), current_model, data, current_history, snapshot_config)

        trained_model, history = train_one_run(
            model,
            data,
            loss,
            seed=run_seed,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            log_every=log_every,
            optimizer_name=optimizer_name,
            weight_decay=weight_decay,
            snapshot_every=snapshot_every,
            snapshot_callback=save_epoch_snapshot,
            eval_every=eval_every,
        )
        save_run(run_dir, trained_model, data, history, config)
        summaries.append({
            'run_dir': str(run_dir),
            'seed': run_seed,
            'final_loss': history['loss'][-1],
            'final_train_mean_acc': history['train_mean_acc'][-1],
            'final_test_mean_acc': history['test_mean_acc'][-1],
            'final_train_task_acc': history['train_task_acc'][-1],
            'final_test_task_acc': history['test_task_acc'][-1],
            'snapshot_every': snapshot_every,
            'eval_every': eval_every,
            'snapshots_root': str(experiment_dir / 'snapshots'),
            'loss_config': loss.to_config() if hasattr(loss, 'to_config') else {},
        })
    save_json(experiment_dir / 'summary.json', summaries)
    return experiment_dir


if __name__ == '__main__':
    dataset = BinaryPairDataset(num_samples=1024, pattern_size=5, train_ratio=0.5)
    task = BinaryArithmeticTask(rule_indices=[0, 1, 2])
    model = ModularMLP(hidden_dim=64, num_modules=3)
    loss = WeightedMSELoss(weighted_rule=0, weighted_loss=1.0, l1=0.0)

    output_dir = run_repeats(
        dataset=dataset,
        task=task,
        model=model,
        loss=loss,
        experiment_name='main_mlp_binary',
        repeat=5,
        epochs=100,
        batch_size=32,
        lr=0.05,
    )
    print('\nSaved simulation results to: {}'.format(output_dir))
