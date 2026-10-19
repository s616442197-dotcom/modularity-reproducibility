from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import analysis as ana
from dataset import BinaryPairDataset
from loss import WeightedMSELoss
from main import named_accuracy, save_run, train_one_run
from model import TorchModularMLP
from task import BinaryArithmeticTask


DEFAULT_OUTPUT_ROOT = ROOT / "example/composite/results"
RULE_INDICES = [0, 1, 2, 3, 4]
EXPORT_EXTENSIONS = ("png", "pdf", "svg")


def configure_style() -> None:
    ana.configure_publication_style()
    matplotlib.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "svg.fonttype": "none",
        }
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False), encoding="utf-8")


def unique_dir(root: Path, name: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / name
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate
    idx = 1
    while True:
        candidate = root / "{}_{:02d}".format(name, idx)
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        idx += 1


def state_parameter_count(state: dict[str, np.ndarray]) -> int:
    total = 0
    for value in state.values():
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.number):
            total += int(array.size)
    return total


def model_parameter_count(model: Any) -> int:
    return state_parameter_count(model.state_dict())


def train_cost_proxy(param_count: int, train_samples: int, epochs: int) -> float:
    return float(param_count) * float(train_samples) * float(epochs)


def sync_device(device: Any) -> None:
    try:
        import torch

        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        return


def warmup_torch(args: argparse.Namespace) -> None:
    try:
        import torch

        model = TorchModularMLP(
            input_dim=args.pattern_size * 2,
            output_dim=args.pattern_size * len(RULE_INDICES),
            hidden_dim=max(4, min(args.hidden_dim, 32)),
            num_modules=args.num_modules,
            device=args.device,
        )
        x = torch.zeros((2, args.pattern_size * 2), dtype=torch.float32, device=model.device)
        y = model.forward_tensor(x).sum()
        y.backward()
        sync_device(model.device)
    except Exception:
        return


def rule_label(rule_id: int) -> str:
    return "rule {}".format(int(rule_id) + 1)


def extraction_proxy(combo_result: dict[str, Any], test_samples: int, perturbation_batch_size: int) -> dict[str, float]:
    layers = combo_result.get("layers", [])
    layer_widths = [np.asarray(layer.get("important_mask", [])).shape[1] for layer in layers]
    total_neurons = int(np.sum(layer_widths)) if layer_widths else 0
    batched_forward_calls = int(np.sum([int(np.ceil(width / max(float(perturbation_batch_size), 1.0))) for width in layer_widths]))
    return {
        "extraction_total_hidden_neurons": float(total_neurons),
        "extraction_batched_forward_calls": float(batched_forward_calls),
        "extraction_sample_forward_proxy": float(batched_forward_calls * int(test_samples)),
    }


def minimal_neuron_extraction(
    run_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float]:
    loaded = ana.load_run_npz(run_dir)
    start = time.perf_counter()
    combo_subnet = ana.combo_subnetwork_analysis(
        loaded["state"],
        loaded["x_test"],
        loaded["y_test"],
        int(loaded["pattern_size"]),
        [int(rule) for rule in loaded["rule_indices"]],
        mask_method=args.mask_method,
        importance_min_score=args.importance_min_score,
        importance_mask_mode="score_ratio",
        importance_score_ratio_threshold=args.importance_neuron_threshold,
        importance_output_eps=args.importance_output_eps,
        perturbation_batch_size=args.perturbation_batch_size,
        analysis_device=args.analysis_device,
    )
    seconds = time.perf_counter() - start
    result = {
        "combo_subnetwork": combo_subnet,
        "extraction_seconds": seconds,
        **extraction_proxy(combo_subnet, test_samples=len(loaded["x_test"]), perturbation_batch_size=args.perturbation_batch_size),
    }
    save_json(output_dir / "analysis.json", result)
    return result, seconds


def diagonal_token_retained(combo_result: dict[str, Any], rule_count: int) -> list[float]:
    retained = np.asarray(combo_result.get("network_retained_accuracy", []), dtype=float)
    if retained.ndim != 2 or retained.shape[0] < rule_count or retained.shape[1] < rule_count:
        return [float("nan")] * rule_count
    return [float(retained[idx, idx]) for idx in range(rule_count)]


def train_one_saved_run(
    *,
    run_dir: Path,
    dataset: BinaryPairDataset,
    task: BinaryArithmeticTask,
    model_template: TorchModularMLP,
    loss: WeightedMSELoss,
    seed: int,
    repeat_idx: int,
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    bundle = dataset.build(seed=seed)
    data = task.prepare(bundle)
    sync_device(model_template.device)
    start = time.perf_counter()
    trained_model, history = train_one_run(
        model_template,
        data,
        loss,
        seed=seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        optimizer_name=args.optimizer_name,
        weight_decay=args.weight_decay,
        snapshot_every=None,
    )
    sync_device(trained_model.device)
    train_seconds = time.perf_counter() - start
    config = {
        "experiment": label,
        "dataset": dataset.__class__.__name__,
        "task": task.__class__.__name__,
        "model": model_template.__class__.__name__,
        "loss": loss.__class__.__name__,
        "loss_config": loss.to_config(),
        "seed": seed,
        "repeat_idx": repeat_idx,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer_name": args.optimizer_name,
        "weight_decay": args.weight_decay,
        "rule_indices": data.rule_indices,
        "rule_names": data.rule_names,
    }
    save_run(run_dir, trained_model, data, history, config)
    param_count = model_parameter_count(trained_model)
    final_test = history["test_rule_acc"][-1]
    return {
        "run_dir": str(run_dir),
        "label": label,
        "seed": seed,
        "repeat_idx": repeat_idx,
        "rule_indices": list(data.rule_indices),
        "rule_names": list(data.rule_names),
        "train_seconds": float(train_seconds),
        "param_count": int(param_count),
        "train_samples": int(data.x_train.shape[0]),
        "test_samples": int(data.x_test.shape[0]),
        "output_dim": int(data.y_train.shape[1]),
        "train_proxy": train_cost_proxy(param_count, int(data.x_train.shape[0]), args.epochs),
        "final_test_mean_acc": float(history["test_mean_acc"][-1]),
        "final_test_task_acc": named_accuracy(final_test, data),
    }


def run_experiment(args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    run_root = unique_dir(args.output_root, args.run_name)
    train_root = run_root / "training"
    analysis_root = run_root / "analysis"
    dataset = BinaryPairDataset(num_samples=args.num_samples, pattern_size=args.pattern_size, train_ratio=args.train_ratio)

    multitask_rows = []
    extraction_rows = []
    separate_rows = []
    for repeat_idx in range(args.repeat):
        seed = args.seed + repeat_idx
        print("\nRepeat {}/{} seed={}".format(repeat_idx + 1, args.repeat, seed), flush=True)

        multitask_task = BinaryArithmeticTask(rule_indices=RULE_INDICES)
        multitask_model = TorchModularMLP(hidden_dim=args.hidden_dim, num_modules=args.num_modules, device=args.device)
        multitask_loss = WeightedMSELoss(l1=args.l1, gamma=args.gamma, regularization_epsilon=args.regularization_epsilon)
        multitask_run_dir = train_root / "multitask" / "run_{:03d}".format(repeat_idx)
        print("  training multitask model", flush=True)
        multitask_row = train_one_saved_run(
            run_dir=multitask_run_dir,
            dataset=dataset,
            task=multitask_task,
            model_template=multitask_model,
            loss=multitask_loss,
            seed=seed,
            repeat_idx=repeat_idx,
            args=args,
            label="multitask",
        )
        multitask_rows.append(multitask_row)

        if not args.skip_extraction:
            print("  extracting functional-token subnetworks", flush=True)
            extraction_result, extraction_seconds = minimal_neuron_extraction(
                multitask_run_dir,
                analysis_root / "multitask" / "run_{:03d}".format(repeat_idx),
                args,
            )
            retained_diag = diagonal_token_retained(extraction_result["combo_subnetwork"], len(RULE_INDICES))
            extraction_rows.append(
                {
                    "run_dir": str(multitask_run_dir),
                    "repeat_idx": repeat_idx,
                    "seed": seed,
                    "extraction_seconds": float(extraction_seconds),
                    "token_retained_relative_accuracy_mean": float(np.nanmean(retained_diag)),
                    "token_retained_relative_accuracy_by_rule": {
                        rule_label(rule_id): float(value) for rule_id, value in zip(RULE_INDICES, retained_diag)
                    },
                    "extraction_total_hidden_neurons": extraction_result["extraction_total_hidden_neurons"],
                    "extraction_batched_forward_calls": extraction_result["extraction_batched_forward_calls"],
                    "extraction_sample_forward_proxy": extraction_result["extraction_sample_forward_proxy"],
                }
            )

        for rule_id in RULE_INDICES:
            print("  training separate {}".format(rule_label(rule_id)), flush=True)
            separate_task = BinaryArithmeticTask(rule_indices=[rule_id])
            separate_model = TorchModularMLP(hidden_dim=args.hidden_dim, num_modules=args.num_modules, device=args.device)
            separate_loss = WeightedMSELoss(l1=args.single_l1, gamma=args.gamma, regularization_epsilon=args.regularization_epsilon)
            separate_run_dir = train_root / "separate_{}".format(rule_label(rule_id).replace(" ", "_")) / "run_{:03d}".format(repeat_idx)
            row = train_one_saved_run(
                run_dir=separate_run_dir,
                dataset=dataset,
                task=separate_task,
                model_template=separate_model,
                loss=separate_loss,
                seed=seed,
                repeat_idx=repeat_idx,
                args=args,
                label="separate_{}".format(rule_label(rule_id)),
            )
            row["target_rule"] = rule_label(rule_id)
            row["target_rule_id"] = int(rule_id)
            separate_rows.append(row)

    return run_root, multitask_rows, extraction_rows, separate_rows


def aggregate_costs(
    multitask_rows: list[dict[str, Any]],
    extraction_rows: list[dict[str, Any]],
    separate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    repeats = sorted({int(row["repeat_idx"]) for row in multitask_rows})
    per_repeat = []
    for repeat_idx in repeats:
        multi = [row for row in multitask_rows if int(row["repeat_idx"]) == repeat_idx][0]
        extraction = [row for row in extraction_rows if int(row["repeat_idx"]) == repeat_idx]
        sep = [row for row in separate_rows if int(row["repeat_idx"]) == repeat_idx]
        extraction_seconds = float(extraction[0]["extraction_seconds"]) if extraction else 0.0
        extraction_proxy_value = float(extraction[0]["extraction_sample_forward_proxy"]) if extraction else 0.0
        row = {
            "repeat_idx": repeat_idx,
            "multitask_train_seconds": float(multi["train_seconds"]),
            "multitask_plus_extraction_seconds": float(multi["train_seconds"]) + extraction_seconds,
            "separate_train_seconds": float(np.sum([item["train_seconds"] for item in sep])),
            "multitask_train_proxy": float(multi["train_proxy"]),
            "multitask_plus_extraction_proxy": float(multi["train_proxy"]) + extraction_proxy_value,
            "separate_train_proxy": float(np.sum([item["train_proxy"] for item in sep])),
            "multitask_param_count": int(multi["param_count"]),
            "separate_param_count_sum": int(np.sum([item["param_count"] for item in sep])),
            "multitask_mean_acc": float(multi["final_test_mean_acc"]),
            "separate_mean_acc": float(np.mean([item["final_test_mean_acc"] for item in sep])),
            "token_retained_relative_accuracy_mean": float(extraction[0]["token_retained_relative_accuracy_mean"]) if extraction else float("nan"),
        }
        row["separate_vs_multitask_train_proxy_ratio"] = row["separate_train_proxy"] / max(row["multitask_train_proxy"], 1.0)
        row["separate_vs_multitask_plus_extraction_proxy_ratio"] = row["separate_train_proxy"] / max(row["multitask_plus_extraction_proxy"], 1.0)
        row["separate_vs_multitask_wall_ratio"] = row["separate_train_seconds"] / max(row["multitask_train_seconds"], 1e-12)
        row["separate_vs_multitask_plus_extraction_wall_ratio"] = row["separate_train_seconds"] / max(row["multitask_plus_extraction_seconds"], 1e-12)
        per_repeat.append(row)

    summary = {
        "repeat_count": len(per_repeat),
        "per_repeat": per_repeat,
        "mean": {},
    }
    numeric_keys = [key for key in per_repeat[0].keys() if key != "repeat_idx"] if per_repeat else []
    for key in numeric_keys:
        values = np.asarray([row[key] for row in per_repeat], dtype=float)
        summary["mean"][key] = float(np.nanmean(values))
        summary["sem_{}".format(key)] = float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_figure(fig: plt.Figure, stem: Path) -> list[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in EXPORT_EXTENSIONS:
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=ana.PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.03)
        written.append(str(path))
    return written


def bar_with_points(ax: plt.Axes, labels: list[str], values_by_group: list[np.ndarray], colors: list[str], ylabel: str, title: str) -> None:
    means = [float(np.nanmean(values)) for values in values_by_group]
    sems = [
        float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        for values in values_by_group
    ]
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.78, capsize=2.5, width=0.62)
    rng = np.random.default_rng(20260807)
    for idx, values in enumerate(values_by_group):
        jitter = rng.normal(0.0, 0.035, size=values.size)
        ax.scatter(np.full(values.size, x[idx]) + jitter, values, s=13, color="#222222", alpha=0.62, linewidths=0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ana._style_publication_axes(ax, grid=True)


def plot_outputs(run_root: Path, summary: dict[str, Any]) -> list[str]:
    rows = summary["per_repeat"]
    labels = ["multitask", "multitask+\nextraction", "5 separate"]
    colors = ["#0072B2", "#009E73", "#D55E00"]
    written = []
    proxy_values = [
        np.asarray([row["multitask_train_proxy"] for row in rows], dtype=float),
        np.asarray([row["multitask_plus_extraction_proxy"] for row in rows], dtype=float),
        np.asarray([row["separate_train_proxy"] for row in rows], dtype=float),
    ]
    fig, ax = plt.subplots(figsize=(ana.NATURE_SINGLE_COLUMN_IN, 2.7))
    bar_with_points(ax, labels, proxy_values, colors, "training cost proxy", "a")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    written.extend(export_figure(fig, run_root / "figures" / "cost_proxy_comparison"))
    plt.close(fig)

    wall_values = [
        np.asarray([row["multitask_train_seconds"] for row in rows], dtype=float),
        np.asarray([row["multitask_plus_extraction_seconds"] for row in rows], dtype=float),
        np.asarray([row["separate_train_seconds"] for row in rows], dtype=float),
    ]
    fig, ax = plt.subplots(figsize=(ana.NATURE_SINGLE_COLUMN_IN, 2.7))
    bar_with_points(ax, labels, wall_values, colors, "wall time (s)", "b")
    written.extend(export_figure(fig, run_root / "figures" / "wall_time_comparison"))
    plt.close(fig)

    accuracy_labels = ["multitask", "5 separate", "token retained"]
    accuracy_values = [
        np.asarray([row["multitask_mean_acc"] for row in rows], dtype=float),
        np.asarray([row["separate_mean_acc"] for row in rows], dtype=float),
        np.asarray([row["token_retained_relative_accuracy_mean"] for row in rows], dtype=float),
    ]
    fig, ax = plt.subplots(figsize=(ana.NATURE_SINGLE_COLUMN_IN, 2.7))
    bar_with_points(ax, accuracy_labels, accuracy_values, ["#0072B2", "#D55E00", "#009E73"], "accuracy / relative accuracy", "c")
    written.extend(export_figure(fig, run_root / "figures" / "accuracy_comparison"))
    plt.close(fig)
    return written


def write_article_note(run_root: Path, summary: dict[str, Any], figure_paths: list[str]) -> None:
    mean = summary["mean"]
    lines = [
        "# Multitask Cost Comparison",
        "",
        "This experiment complements `article/modularity_new.tex` subsection `Compositional Reuse of Functional Tokens`.",
        "",
        "It compares one shared five-rule multitask MLP followed by neuron-token extraction against training five independent single-rule MLPs with the same hidden architecture.",
        "",
        "Cost proxy is `trainable parameter count x train samples x epochs`. The extraction proxy counts test-sample forward evaluations used by batched neuron ablation.",
        "",
        "Mean ratios over repeats:",
        "- separate / multitask train proxy: {:.3f}".format(mean.get("separate_vs_multitask_train_proxy_ratio", float("nan"))),
        "- separate / multitask+extraction proxy: {:.3f}".format(mean.get("separate_vs_multitask_plus_extraction_proxy_ratio", float("nan"))),
        "- separate / multitask wall time: {:.3f}".format(mean.get("separate_vs_multitask_wall_ratio", float("nan"))),
        "- separate / multitask+extraction wall time: {:.3f}".format(mean.get("separate_vs_multitask_plus_extraction_wall_ratio", float("nan"))),
        "",
        "Generated figures:",
    ]
    lines.extend(["- {}".format(path) for path in figure_paths])
    (run_root / "article_note.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multitask+subnet extraction cost with five separately trained task models.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="multitask_vs_separate_cost")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=830)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--pattern-size", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--optimizer-name", default="SGD")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--num-modules", type=int, default=3)
    parser.add_argument("--l1", type=float, default=1.0)
    parser.add_argument("--single-l1", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--regularization-epsilon", type=float, default=1e-8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--mask-method", choices=["ablation", "taylor"], default="ablation")
    parser.add_argument("--importance-neuron-threshold", type=float, default=1e-5)
    parser.add_argument("--importance-min-score", type=float, default=1e-9)
    parser.add_argument("--importance-output-eps", type=float, default=1e-8)
    parser.add_argument("--perturbation-batch-size", type=int, default=64)
    parser.add_argument("--analysis-device", default="auto")
    args = parser.parse_args()

    configure_style()
    args.output_root = args.output_root.resolve()
    warmup_torch(args)
    run_root, multitask_rows, extraction_rows, separate_rows = run_experiment(args)
    summary = aggregate_costs(multitask_rows, extraction_rows, separate_rows)
    save_json(run_root / "summary.json", summary)
    save_json(run_root / "multitask_runs.json", multitask_rows)
    save_json(run_root / "extraction_runs.json", extraction_rows)
    save_json(run_root / "separate_runs.json", separate_rows)
    write_csv(run_root / "cost_per_repeat.csv", summary["per_repeat"])
    figure_paths = plot_outputs(run_root, summary)
    write_article_note(run_root, summary, figure_paths)
    print("\nSaved composite cost comparison to {}".format(run_root))
    print("separate / multitask train proxy = {:.3f}".format(summary["mean"]["separate_vs_multitask_train_proxy_ratio"]))
    print(
        "separate / multitask+extraction proxy = {:.3f}".format(
            summary["mean"]["separate_vs_multitask_plus_extraction_proxy_ratio"]
        )
    )


if __name__ == "__main__":
    main()
