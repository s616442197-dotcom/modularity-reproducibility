from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.spatial.distance import pdist, squareform

try:
    import torch
    import torch.nn.functional as torch_f
except Exception:
    torch = None
    torch_f = None

LEGACY_HESSIAN_EPSILON = 1e-2


PUBLICATION_DPI = 300
DEFAULT_TOKEN_EXTRACTION_METHOD = 'score_quantile'
DEFAULT_TOKEN_EXTRACTION_FRACTION = 0.75
DEFAULT_RULE_SCORE_AVERAGE_FRACTION = 1.0 / 3.0
TOKEN_EXTRACTION_METHOD_CHOICES = ('pt', 'score_mean_fraction', 'score_median_fraction', 'score_quantile')
NATURE_SINGLE_COLUMN_IN = 3.5
NATURE_DOUBLE_COLUMN_IN = 7.2
NATURE_SUPPLEMENT_MAX_IN = 10.5
NATURE_COLORS = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # green
    '#CC79A7',  # reddish purple
    '#56B4E9',  # sky blue
    '#E69F00',  # orange
    '#F0E442',  # yellow
    '#000000',  # black
]


def configure_publication_style() -> None:
    matplotlib.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 8,
        'axes.labelsize': 8,
        'axes.linewidth': 0.8,
        'axes.edgecolor': '#222222',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'legend.fontsize': 7,
        'legend.title_fontsize': 7,
        'legend.frameon': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': PUBLICATION_DPI,
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
    })


def _nature_color(index: int) -> str:
    return NATURE_COLORS[index % len(NATURE_COLORS)]


def _style_publication_axes(ax, grid: bool = False) -> None:
    ax.set_facecolor('white')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color('#222222')
    ax.tick_params(axis='both', direction='out', length=3.0, width=0.8, color='#222222', labelcolor='#222222', pad=2)
    ax.grid(False)
    if grid:
        ax.yaxis.grid(True, color='#D9D9D9', linewidth=0.45, alpha=0.7)
    ax.set_axisbelow(True)


configure_publication_style()


def load_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(convert(data), f, indent=2, ensure_ascii=False)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-12)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def sorted_mlp_layer_ids(state: dict[str, np.ndarray]) -> list[int]:
    ids = []
    for key in state:
        if key.startswith('modules_list_') and key.endswith('_linear1_weight'):
            ids.append(int(key.split('_')[2]))
    return sorted(ids)


def sorted_transformer_layer_ids(state: dict[str, np.ndarray]) -> list[int]:
    ids = set()
    for key in state:
        if key.startswith('transformer_layer_'):
            ids.add(int(key.split('_')[2]))
    return sorted(ids)


def mlp_forward(state: dict[str, np.ndarray], x: np.ndarray, silence: tuple[int, list[int]] | None = None):
    current = x
    activations = []
    for idx in sorted_mlp_layer_ids(state):
        w = state['modules_list_{}_linear1_weight'.format(idx)]
        b = state['modules_list_{}_linear1_bias'.format(idx)]
        z = current @ w.T + b
        if silence is not None and silence[0] == idx:
            z = z.copy()
            z[:, silence[1]] = 0.0
        if silence is not None and len(silence) == 3 and silence[0] == 'keep_only' and silence[1] == idx:
            z = z.copy()
            mask = np.ones(z.shape[1], dtype=bool)
            mask[silence[2]] = False
            z[:, mask] = 0.0
        activations.append(z)
        current = relu(z)
    out = current @ state['output_block_linear1_weight'].T + state['output_block_linear1_bias']
    activations.append(out)
    return out, activations


def transformer_forward(state: dict[str, np.ndarray], x: np.ndarray, silence: tuple[int, list[int]] | None = None):
    input_proj = state['transformer_input_proj']
    token_bias = state['transformer_token_bias']
    half_dim = x.shape[1] // 2
    model_dim = input_proj.shape[1]
    left = x[:, :half_dim] @ input_proj
    right = x[:, half_dim:] @ input_proj
    tokens = np.stack([left, right], axis=1) + token_bias[None, :, :]
    activations = []
    for idx in sorted_transformer_layer_ids(state):
        q = tokens @ state['transformer_layer_{}_wq'.format(idx)]
        k = tokens @ state['transformer_layer_{}_wk'.format(idx)]
        v = tokens @ state['transformer_layer_{}_wv'.format(idx)]
        attention = softmax(q @ np.swapaxes(k, 1, 2) / np.sqrt(model_dim), axis=-1)
        tokens = layer_norm(tokens + (attention @ v) @ state['transformer_layer_{}_wo'.format(idx)])
        ff = relu(tokens @ state['transformer_layer_{}_ff1'.format(idx)] + state['transformer_layer_{}_b1'.format(idx)])
        tokens = layer_norm(tokens + ff @ state['transformer_layer_{}_ff2'.format(idx)] + state['transformer_layer_{}_b2'.format(idx)])
        if silence is not None and silence[0] == idx:
            tokens = tokens.copy()
            tokens[:, :, silence[1]] = 0.0
        if silence is not None and len(silence) == 3 and silence[0] == 'keep_only' and silence[1] == idx:
            tokens = tokens.copy()
            mask = np.ones(tokens.shape[2], dtype=bool)
            mask[silence[2]] = False
            tokens[:, :, mask] = 0.0
        activations.append(tokens.mean(axis=1))
    features = tokens.mean(axis=1)
    out = features @ state['output_block_linear1_weight'].T + state['output_block_linear1_bias']
    activations.append(out)
    return out, activations


def forward_from_state(state: dict[str, np.ndarray], x: np.ndarray, silence: tuple[int, list[int]] | None = None):
    if 'transformer_input_proj' in state:
        return transformer_forward(state, x, silence=silence)
    return mlp_forward(state, x, silence=silence)


def torch_analysis_device(device: str = 'auto'):
    if torch is None:
        return None
    if device is None:
        return None
    value = str(device).lower().strip()
    if value in {'none', 'off', 'numpy'}:
        return None
    if value == 'auto':
        return torch.device('cuda') if torch.cuda.is_available() else None
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        return None
    return dev


def torch_importance_device(device: str = 'auto'):
    if torch is None:
        return None
    if device is None:
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    value = str(device).lower().strip()
    if value in {'none', 'off', 'numpy'}:
        return None
    if value == 'auto':
        return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dev = torch.device(device)
    if dev.type == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return dev


def torch_state_from_numpy(state: dict[str, np.ndarray], device) -> dict[str, Any]:
    return {key: torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=device) for key, value in state.items()}


def torch_state_from_numpy_for_importance(state: dict[str, np.ndarray], device, weight_keys: set[str]) -> dict[str, Any]:
    tstate = {}
    for key, value in state.items():
        tensor = torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=device).clone().detach()
        if key in weight_keys:
            tensor.requires_grad_(True)
        tstate[key] = tensor
    return tstate


def torch_ablation_context(state: dict[str, np.ndarray], x: np.ndarray, baseline_pred: np.ndarray, device: str = 'auto') -> tuple[dict[str, Any], Any, Any] | None:
    dev = torch_analysis_device(device)
    if dev is None:
        return None
    try:
        tstate = torch_state_from_numpy(state, dev)
        x_tensor = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
        baseline = torch.as_tensor(np.ascontiguousarray(baseline_pred), dtype=torch.float32, device=dev)
        return tstate, x_tensor, baseline
    except Exception:
        return None


def torch_linear_out_in(current, weight, bias):
    if weight.dim() == 3:
        if current.dim() == 2:
            out = torch.einsum('ni,koi->kno', current, weight)
        else:
            out = torch.einsum('kni,koi->kno', current, weight)
        return out + bias.view(1, 1, -1)
    out = torch.matmul(current, weight.t())
    if current.dim() == 3:
        return out + bias.view(1, 1, -1)
    return out + bias


def torch_linear_in_out(current, weight):
    if weight.dim() == 3:
        if current.dim() == 2:
            return torch.einsum('ni,kio->kno', current, weight)
        if current.dim() == 3:
            return torch.einsum('nti,kio->knto', current, weight)
        return torch.einsum('knti,kio->knto', current, weight)
    return torch.matmul(current, weight)


def torch_output_linear(features, weight, bias):
    if weight.dim() == 3:
        if features.dim() == 2:
            out = torch.einsum('nd,kod->kno', features, weight)
        else:
            out = torch.einsum('knd,kod->kno', features, weight)
        return out + bias.view(1, 1, -1)
    out = torch.matmul(features, weight.t())
    if features.dim() == 3:
        return out + bias.view(1, 1, -1)
    return out + bias


def torch_mlp_forward_state(tstate: dict[str, Any], x_tensor):
    current = x_tensor
    for idx in sorted_mlp_layer_ids(tstate):
        weight = tstate['modules_list_{}_linear1_weight'.format(idx)]
        bias = tstate['modules_list_{}_linear1_bias'.format(idx)]
        current = torch.relu(torch_linear_out_in(current, weight, bias))
    return torch_output_linear(current, tstate['output_block_linear1_weight'], tstate['output_block_linear1_bias'])


def torch_transformer_forward_state(tstate: dict[str, Any], x_tensor):
    input_proj = tstate['transformer_input_proj']
    token_bias = tstate['transformer_token_bias']
    half_dim = x_tensor.shape[1] // 2
    model_dim = int(input_proj.shape[-1])
    if input_proj.dim() == 3:
        left = torch.einsum('ni,kio->kno', x_tensor[:, :half_dim], input_proj)
        right = torch.einsum('ni,kio->kno', x_tensor[:, half_dim:], input_proj)
        tokens = torch.stack([left, right], dim=2) + token_bias.view(1, 1, 2, model_dim)
    else:
        left = x_tensor[:, :half_dim] @ input_proj
        right = x_tensor[:, half_dim:] @ input_proj
        tokens = torch.stack([left, right], dim=1) + token_bias.view(1, 2, model_dim)

    scale = float(np.sqrt(model_dim))
    for idx in sorted_transformer_layer_ids(tstate):
        q = torch_linear_in_out(tokens, tstate['transformer_layer_{}_wq'.format(idx)])
        k = torch_linear_in_out(tokens, tstate['transformer_layer_{}_wk'.format(idx)])
        v = torch_linear_in_out(tokens, tstate['transformer_layer_{}_wv'.format(idx)])
        attention = torch.softmax(q @ k.transpose(-1, -2) / scale, dim=-1)
        tokens = torch_f.layer_norm(tokens + torch_linear_in_out(attention @ v, tstate['transformer_layer_{}_wo'.format(idx)]), (model_dim,))
        ff = torch.relu(torch_linear_in_out(tokens, tstate['transformer_layer_{}_ff1'.format(idx)]) + tstate['transformer_layer_{}_b1'.format(idx)])
        tokens = torch_f.layer_norm(tokens + torch_linear_in_out(ff, tstate['transformer_layer_{}_ff2'.format(idx)]) + tstate['transformer_layer_{}_b2'.format(idx)], (model_dim,))
    features = tokens.mean(dim=-2)
    return torch_output_linear(features, tstate['output_block_linear1_weight'], tstate['output_block_linear1_bias'])


def torch_forward_state(tstate: dict[str, Any], x_tensor):
    if 'transformer_input_proj' in tstate:
        return torch_transformer_forward_state(tstate, x_tensor)
    return torch_mlp_forward_state(tstate, x_tensor)


def torch_mlp_forward_state_tracked(tstate: dict[str, Any], x_tensor):
    current = x_tensor
    activations = []
    for idx in sorted_mlp_layer_ids(tstate):
        weight = tstate['modules_list_{}_linear1_weight'.format(idx)]
        bias = tstate['modules_list_{}_linear1_bias'.format(idx)]
        z = torch_linear_out_in(current, weight, bias)
        activations.append(z)
        current = torch.relu(z)
    out = torch_output_linear(current, tstate['output_block_linear1_weight'], tstate['output_block_linear1_bias'])
    return out, activations


def torch_transformer_forward_state_tracked(tstate: dict[str, Any], x_tensor):
    input_proj = tstate['transformer_input_proj']
    token_bias = tstate['transformer_token_bias']
    half_dim = x_tensor.shape[1] // 2
    model_dim = int(input_proj.shape[-1])
    left = x_tensor[:, :half_dim] @ input_proj
    right = x_tensor[:, half_dim:] @ input_proj
    tokens = torch.stack([left, right], dim=1) + token_bias.view(1, 2, model_dim)
    activations = []
    scale = float(np.sqrt(model_dim))
    for idx in sorted_transformer_layer_ids(tstate):
        q = torch_linear_in_out(tokens, tstate['transformer_layer_{}_wq'.format(idx)])
        k = torch_linear_in_out(tokens, tstate['transformer_layer_{}_wk'.format(idx)])
        v = torch_linear_in_out(tokens, tstate['transformer_layer_{}_wv'.format(idx)])
        attention = torch.softmax(q @ k.transpose(-1, -2) / scale, dim=-1)
        tokens = torch_f.layer_norm(tokens + torch_linear_in_out(attention @ v, tstate['transformer_layer_{}_wo'.format(idx)]), (model_dim,))
        ff = torch.relu(torch_linear_in_out(tokens, tstate['transformer_layer_{}_ff1'.format(idx)]) + tstate['transformer_layer_{}_b1'.format(idx)])
        tokens = torch_f.layer_norm(tokens + torch_linear_in_out(ff, tstate['transformer_layer_{}_ff2'.format(idx)]) + tstate['transformer_layer_{}_b2'.format(idx)], (model_dim,))
        activations.append(tokens)
    features = tokens.mean(dim=-2)
    out = torch_output_linear(features, tstate['output_block_linear1_weight'], tstate['output_block_linear1_bias'])
    return out, activations


def torch_forward_state_tracked(tstate: dict[str, Any], x_tensor):
    if 'transformer_input_proj' in tstate:
        return torch_transformer_forward_state_tracked(tstate, x_tensor)
    return torch_mlp_forward_state_tracked(tstate, x_tensor)


def taylor_activation_contribution(activation, grad):
    local = activation.detach() * grad.detach()
    if local.dim() == 2:
        return local.square().sum(dim=0)
    if local.dim() == 3:
        return local.sum(dim=1).square().sum(dim=0)
    return local.reshape(local.shape[0], -1).square().sum(dim=0)


def taylor_importance_scores(
    state: dict[str, np.ndarray],
    x: np.ndarray,
    pattern_size: int,
    rule_indices: list[int],
    batch_size: int = 64,
    device: str = 'auto',
    probes: int = 4,
    seed: int = 0,
) -> dict[str, Any]:
    dev = torch_importance_device(device)
    if dev is None:
        raise RuntimeError("mask_method='taylor' requires PyTorch; use analysis_device='auto', 'cpu', or 'cuda'.")
    if len(x) == 0:
        weight_keys = [spec['key'] for spec in weight_block_specs(state)]
        return {
            'neuron_layers': [np.zeros((len(rule_indices), width), dtype=float) for width in analysis_layer_sizes(state)],
            'weights': {key: np.zeros((len(rule_indices),) + state[key].shape, dtype=float) for key in weight_keys},
            'probes': probes,
            'batch_size': batch_size,
        }

    probes = max(1, int(probes))
    weight_keys = [spec['key'] for spec in weight_block_specs(state)]
    weight_key_set = set(weight_keys)
    tstate = torch_state_from_numpy_for_importance(state, dev, weight_key_set)
    weight_tensors = [tstate[key] for key in weight_keys]
    neuron_scores = [np.zeros((len(rule_indices), width), dtype=float) for width in analysis_layer_sizes(state)]
    weight_scores = {key: np.zeros((len(rule_indices),) + state[key].shape, dtype=float) for key in weight_keys}
    x_tensor = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
    generator = torch.Generator(device=dev)
    generator.manual_seed(int(seed))
    sample_count = 0

    for start in range(0, len(x), batch_size):
        xb = x_tensor[start : start + batch_size]
        sample_count += int(xb.shape[0])
        for _probe in range(probes):
            output, activations = torch_forward_state_tracked(tstate, xb)
            targets = weight_tensors + activations
            for rule_pos, _rule_id in enumerate(rule_indices):
                out_start = rule_pos * pattern_size
                out_end = out_start + pattern_size
                segment = output[:, out_start:out_end]
                signs = torch.randint(0, 2, segment.shape, generator=generator, device=dev, dtype=torch.int8).to(segment.dtype)
                signs = signs.mul(2.0).sub(1.0)
                retain_graph = rule_pos < len(rule_indices) - 1
                grads = torch.autograd.grad((segment * signs).sum(), targets, allow_unused=True, retain_graph=retain_graph)
                weight_grads = grads[:len(weight_tensors)]
                activation_grads = grads[len(weight_tensors):]
                for key, param, grad in zip(weight_keys, weight_tensors, weight_grads):
                    if grad is not None:
                        weight_scores[key][rule_pos] += (param.detach() * grad.detach()).square().cpu().numpy()
                for layer_idx, (activation, grad) in enumerate(zip(activations, activation_grads)):
                    if grad is not None:
                        neuron_scores[layer_idx][rule_pos] += taylor_activation_contribution(activation, grad).cpu().numpy()

    denom = float(max(sample_count * probes, 1))
    neuron_scores = [scores / denom for scores in neuron_scores]
    weight_scores = {key: scores / denom for key, scores in weight_scores.items()}
    return {'neuron_layers': neuron_scores, 'weights': weight_scores, 'probes': probes, 'batch_size': batch_size}


def torch_legacy_hessian_neuron_scores(
    state: dict[str, np.ndarray],
    x: np.ndarray,
    baseline_pred: np.ndarray,
    pattern_size: int,
    rule_indices: list[int],
    layer_idx: int,
    width: int,
    batch_size: int = 64,
    epsilon: float = LEGACY_HESSIAN_EPSILON,
    device: str = 'auto',
    context: tuple[dict[str, Any], Any, Any] | None = None,
) -> np.ndarray | None:
    context = context or torch_ablation_context(state, x, baseline_pred, device=device)
    if context is None:
        return None
    try:
        tstate, x_tensor, _baseline = context
        if 'transformer_input_proj' in tstate:
            return None
        layer_ids = sorted_mlp_layer_ids(tstate)
        if layer_idx not in layer_ids:
            return None

        values = np.zeros((len(rule_indices), width), dtype=float)
        chunk_size = max(1, min(int(batch_size), 16))
        eps = float(epsilon)
        with torch.no_grad():
            current = x_tensor
            target_z = None
            layer_pos = 0
            for pos, idx in enumerate(layer_ids):
                weight = tstate['modules_list_{}_linear1_weight'.format(idx)]
                bias = tstate['modules_list_{}_linear1_bias'.format(idx)]
                z = torch_linear_out_in(current, weight, bias)
                if idx == layer_idx:
                    target_z = z
                    layer_pos = pos
                    break
                current = torch.relu(z)
            if target_z is None:
                return None

            later_ids = layer_ids[layer_pos + 1 :]
            for start in range(0, width, chunk_size):
                neurons = np.arange(start, min(start + chunk_size, width), dtype=int)
                idx_tensor = torch.as_tensor(neurons, dtype=torch.long, device=x_tensor.device)
                batch = int(idx_tensor.numel())
                batch_range = torch.arange(batch, device=x_tensor.device)

                plus = target_z.unsqueeze(0).expand(batch, -1, -1).clone()
                minus = target_z.unsqueeze(0).expand(batch, -1, -1).clone()
                plus[batch_range, :, idx_tensor] += eps
                minus[batch_range, :, idx_tensor] -= eps

                current_plus = torch.relu(plus)
                current_minus = torch.relu(minus)
                for idx in later_ids:
                    weight = tstate['modules_list_{}_linear1_weight'.format(idx)]
                    bias = tstate['modules_list_{}_linear1_bias'.format(idx)]
                    current_plus = torch.relu(torch_linear_out_in(current_plus, weight, bias))
                    current_minus = torch.relu(torch_linear_out_in(current_minus, weight, bias))

                out_plus = torch_output_linear(current_plus, tstate['output_block_linear1_weight'], tstate['output_block_linear1_bias'])
                out_minus = torch_output_linear(current_minus, tstate['output_block_linear1_weight'], tstate['output_block_linear1_bias'])
                deriv = (out_plus - out_minus) / (2.0 * eps)
                for rule_pos, _rule_id in enumerate(rule_indices):
                    out_start = rule_pos * pattern_size
                    out_end = out_start + pattern_size
                    segment = deriv[:, :, out_start:out_end]
                    values[rule_pos, start : start + batch] = segment.square().mean(dim=(1, 2)).detach().cpu().numpy()
        return values
    except Exception:
        return None


def mlp_forward_from_layer_activation(state: dict[str, np.ndarray], current: np.ndarray, layer_idx: int) -> np.ndarray:
    past_layer = False
    for idx in sorted_mlp_layer_ids(state):
        if idx == layer_idx:
            past_layer = True
            continue
        if not past_layer:
            continue
        w = state['modules_list_{}_linear1_weight'.format(idx)]
        b = state['modules_list_{}_linear1_bias'.format(idx)]
        current = relu(current @ w.T + b)
    return current @ state['output_block_linear1_weight'].T + state['output_block_linear1_bias']


def legacy_hessian_neuron_scores(
    state: dict[str, np.ndarray],
    x: np.ndarray,
    baseline_pred: np.ndarray,
    pattern_size: int,
    rule_indices: list[int],
    layer_idx: int,
    width: int,
    batch_size: int = 64,
    epsilon: float = LEGACY_HESSIAN_EPSILON,
    device: str = 'auto',
    context: tuple[dict[str, Any], Any, Any] | None = None,
) -> np.ndarray:
    torch_scores = torch_legacy_hessian_neuron_scores(
        state,
        x,
        baseline_pred,
        pattern_size,
        rule_indices,
        layer_idx,
        width,
        batch_size=batch_size,
        epsilon=epsilon,
        device=device,
        context=context,
    )
    if torch_scores is not None:
        return torch_scores
    if 'transformer_input_proj' in state:
        raise ValueError("legacy_hessian currently supports MLP neuron scoring only")

    _pred, activations = forward_from_state(state, x)
    layer_ids = sorted_mlp_layer_ids(state)
    if layer_idx not in layer_ids:
        raise ValueError('layer_idx {} is not present in the MLP state'.format(layer_idx))
    z = np.asarray(activations[layer_ids.index(layer_idx)], dtype=float)
    values = np.zeros((len(rule_indices), width), dtype=float)
    eps = float(epsilon)
    chunk_size = max(1, min(int(batch_size), 16))
    for start in range(0, width, chunk_size):
        neurons = np.arange(start, min(start + chunk_size, width), dtype=int)
        batch = len(neurons)
        batch_range = np.arange(batch)
        z_plus = np.broadcast_to(z[None, :, :], (batch, z.shape[0], z.shape[1])).copy()
        z_minus = z_plus.copy()
        z_plus[batch_range, :, neurons] += eps
        z_minus[batch_range, :, neurons] -= eps
        out_plus = mlp_forward_from_layer_activation(state, relu(z_plus), layer_idx)
        out_minus = mlp_forward_from_layer_activation(state, relu(z_minus), layer_idx)
        deriv = (out_plus - out_minus) / (2.0 * eps)
        for rule_pos, _rule_id in enumerate(rule_indices):
            out_start = rule_pos * pattern_size
            out_end = out_start + pattern_size
            values[rule_pos, start : start + batch] = np.mean(deriv[:, :, out_start:out_end] ** 2, axis=(1, 2))
    return values


def torch_forward_with_neuron_silence(tstate: dict[str, Any], x_tensor, layer_idx: int, neuron_indices):
    if len(neuron_indices) == 0:
        return torch_forward_state(tstate, x_tensor)
    idx_tensor = torch.as_tensor(neuron_indices, dtype=torch.long, device=x_tensor.device)
    batch = idx_tensor.numel()
    batch_range = torch.arange(batch, device=x_tensor.device)

    if 'transformer_input_proj' in tstate:
        input_proj = tstate['transformer_input_proj']
        token_bias = tstate['transformer_token_bias']
        half_dim = x_tensor.shape[1] // 2
        model_dim = int(input_proj.shape[-1])
        left = x_tensor[:, :half_dim] @ input_proj
        right = x_tensor[:, half_dim:] @ input_proj
        tokens = torch.stack([left, right], dim=1) + token_bias.view(1, 2, model_dim)
        scale = float(np.sqrt(model_dim))
        for idx in sorted_transformer_layer_ids(tstate):
            q = torch_linear_in_out(tokens, tstate['transformer_layer_{}_wq'.format(idx)])
            k = torch_linear_in_out(tokens, tstate['transformer_layer_{}_wk'.format(idx)])
            v = torch_linear_in_out(tokens, tstate['transformer_layer_{}_wv'.format(idx)])
            attention = torch.softmax(q @ k.transpose(-1, -2) / scale, dim=-1)
            tokens = torch_f.layer_norm(tokens + torch_linear_in_out(attention @ v, tstate['transformer_layer_{}_wo'.format(idx)]), (model_dim,))
            ff = torch.relu(torch_linear_in_out(tokens, tstate['transformer_layer_{}_ff1'.format(idx)]) + tstate['transformer_layer_{}_b1'.format(idx)])
            tokens = torch_f.layer_norm(tokens + torch_linear_in_out(ff, tstate['transformer_layer_{}_ff2'.format(idx)]) + tstate['transformer_layer_{}_b2'.format(idx)], (model_dim,))
            if idx == layer_idx:
                tokens = tokens.unsqueeze(0).expand(batch, -1, -1, -1).clone()
                tokens[batch_range, :, :, idx_tensor] = 0.0
        features = tokens.mean(dim=-2)
        return torch_output_linear(features, tstate['output_block_linear1_weight'], tstate['output_block_linear1_bias'])

    current = x_tensor
    for idx in sorted_mlp_layer_ids(tstate):
        weight = tstate['modules_list_{}_linear1_weight'.format(idx)]
        bias = tstate['modules_list_{}_linear1_bias'.format(idx)]
        z = torch_linear_out_in(current, weight, bias)
        if idx == layer_idx:
            z = z.unsqueeze(0).expand(batch, -1, -1).clone()
            z[batch_range, :, idx_tensor] = 0.0
        current = torch.relu(z)
    return torch_output_linear(current, tstate['output_block_linear1_weight'], tstate['output_block_linear1_bias'])


def torch_rule_output_change_to_baseline(pred, baseline, pattern_size: int, rule_indices: list[int]) -> np.ndarray:
    if pred.dim() == 2:
        pred = pred.unsqueeze(0)
    values = []
    for rule_pos, _rule_id in enumerate(rule_indices):
        start = rule_pos * pattern_size
        end = start + pattern_size
        diff = pred[:, :, start:end] - baseline.view(1, baseline.shape[0], baseline.shape[1])[:, :, start:end]
        values.append(diff.square().sum(dim=2).mean(dim=1).detach().cpu().numpy())
    return np.stack(values, axis=0)


def torch_rule_similarity_to_baseline(pred, baseline, pattern_size: int, rule_indices: list[int]) -> np.ndarray:
    return 1.0 - torch_rule_output_change_to_baseline(pred, baseline, pattern_size, rule_indices) / max(pattern_size, 1)


def torch_neuron_silence_similarity(
    state: dict[str, np.ndarray],
    x: np.ndarray,
    baseline_pred: np.ndarray,
    pattern_size: int,
    rule_indices: list[int],
    layer_idx: int,
    width: int,
    batch_size: int = 64,
    device: str = 'auto',
    context: tuple[dict[str, Any], Any, Any] | None = None,
) -> np.ndarray | None:
    context = context or torch_ablation_context(state, x, baseline_pred, device=device)
    if context is None:
        return None
    try:
        values = np.zeros((len(rule_indices), width), dtype=float)
        with torch.no_grad():
            tstate, x_tensor, baseline = context
            for start in range(0, width, batch_size):
                neurons = np.arange(start, min(start + batch_size, width), dtype=int)
                pred = torch_forward_with_neuron_silence(tstate, x_tensor, layer_idx, neurons)
                values[:, start : start + len(neurons)] = torch_rule_similarity_to_baseline(pred, baseline, pattern_size, rule_indices)
        return values
    except Exception:
        return None


def torch_weight_zero_similarity(
    state: dict[str, np.ndarray],
    x: np.ndarray,
    baseline_pred: np.ndarray,
    pattern_size: int,
    rule_indices: list[int],
    key: str,
    candidates: np.ndarray,
    batch_size: int = 64,
    device: str = 'auto',
    context: tuple[dict[str, Any], Any, Any] | None = None,
) -> np.ndarray | None:
    if len(candidates) == 0:
        return np.zeros((len(rule_indices), 0), dtype=float)
    context = context or torch_ablation_context(state, x, baseline_pred, device=device)
    if context is None:
        return None
    try:
        values = np.zeros((len(rule_indices), len(candidates)), dtype=float)
        with torch.no_grad():
            tstate, x_tensor, baseline = context
            base_weight = tstate[key]
            for start in range(0, len(candidates), batch_size):
                chunk = np.asarray(candidates[start : start + batch_size], dtype=np.int64)
                idx_tensor = torch.as_tensor(chunk, dtype=torch.long, device=x_tensor.device)
                batched_weight = base_weight.unsqueeze(0).repeat(len(chunk), *([1] * base_weight.dim())).clone()
                flat = batched_weight.reshape(len(chunk), -1)
                flat[torch.arange(len(chunk), device=x_tensor.device), idx_tensor] = 0.0
                local_state = dict(tstate)
                local_state[key] = batched_weight
                pred = torch_forward_state(local_state, x_tensor)
                values[:, start : start + len(chunk)] = torch_rule_similarity_to_baseline(pred, baseline, pattern_size, rule_indices)
        return values
    except Exception:
        return None


def rule_accuracy(pred: np.ndarray, target: np.ndarray, pattern_size: int, rule_indices: list[int]) -> dict[int, float]:
    binary = (pred > 0.5).astype(np.float32)
    acc = {}
    for pos, rule_id in enumerate(rule_indices):
        start = pos * pattern_size
        end = start + pattern_size
        acc[rule_id] = float(np.mean(binary[:, start:end] == target[:, start:end]))
    return acc


def centered_kernel(k: np.ndarray) -> np.ndarray:
    n = k.shape[0]
    h = np.eye(n) - np.ones((n, n)) / n
    return h @ k @ h


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    kx = centered_kernel(x @ x.T)
    ky = centered_kernel(y @ y.T)
    return float(np.sum(kx * ky) / (np.sqrt(np.sum(kx * kx) * np.sum(ky * ky)) + 1e-12))


def rbf_cka(x: np.ndarray, y: np.ndarray) -> float:
    def kernel(a: np.ndarray) -> np.ndarray:
        dist = squareform(pdist(a, metric='euclidean'))
        upper = dist[np.triu_indices_from(dist, k=1)]
        sigma = float(np.median(upper[upper > 0])) if np.any(upper > 0) else 1.0
        return np.exp(-(dist ** 2) / (2.0 * sigma * sigma + 1e-12))
    kx = centered_kernel(kernel(x))
    ky = centered_kernel(kernel(y))
    return float(np.sum(kx * ky) / (np.sqrt(np.sum(kx * kx) * np.sum(ky * ky)) + 1e-12))


def cpa_analysis(activations: list[np.ndarray]) -> dict[str, np.ndarray]:
    n = len(activations)
    linear = np.zeros((n, n), dtype=float)
    rbf = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            linear_value = linear_cka(activations[i], activations[j])
            rbf_value = rbf_cka(activations[i], activations[j])
            linear[i, j] = linear[j, i] = linear_value
            rbf[i, j] = rbf[j, i] = rbf_value
    return {'linear_cka': linear, 'rbf_cka': rbf}


def pca_entropy(activations: list[np.ndarray]) -> np.ndarray:
    values = []
    for act in activations:
        x = act - act.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(x, compute_uv=False)
        latent = singular ** 2
        if latent.sum() <= 0:
            values.append(0.0)
        else:
            p = latent / latent.sum()
            values.append(float(-np.sum(p * np.log(p + 1e-12))))
    return np.asarray(values)


def neuron_correlation_distribution(activations: list[np.ndarray], sample_size: int = 5000, seed: int = 0) -> dict[str, Any]:
    layers = []
    signed_bins = np.linspace(-1.0, 1.0, 81)
    abs_bins = np.linspace(0.0, 1.0, 51)
    rng = np.random.default_rng(seed)
    for layer_idx, act in enumerate(activations[:-1]):
        values = np.asarray(act, dtype=float)
        if values.ndim != 2:
            values = values.reshape(values.shape[0], -1)
        neuron_count = int(values.shape[1]) if values.ndim == 2 else 0
        if values.shape[0] < 2 or neuron_count < 2:
            corr_values = np.zeros(0, dtype=float)
        else:
            centered = values - values.mean(axis=0, keepdims=True)
            std = centered.std(axis=0, ddof=1)
            valid = std > 1e-12
            centered = centered[:, valid]
            if centered.shape[1] < 2:
                corr_values = np.zeros(0, dtype=float)
            else:
                z = centered / (std[valid][None, :] + 1e-12)
                corr = (z.T @ z) / max(z.shape[0] - 1, 1)
                corr = np.clip(corr, -1.0, 1.0)
                corr_values = corr[np.triu_indices_from(corr, k=1)]
                corr_values = corr_values[np.isfinite(corr_values)]
        abs_values = np.abs(corr_values)
        take = min(int(sample_size), corr_values.size) if sample_size is not None else corr_values.size
        if take > 0 and corr_values.size > take:
            sampled_idx = rng.choice(corr_values.size, size=take, replace=False)
            sampled = corr_values[sampled_idx]
        else:
            sampled = corr_values.copy()
        if corr_values.size:
            signed_quantiles = np.quantile(corr_values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
            abs_quantiles = np.quantile(abs_values, [0.50, 0.90, 0.95, 0.99])
            signed_hist, _ = np.histogram(corr_values, bins=signed_bins)
            abs_hist, _ = np.histogram(abs_values, bins=abs_bins)
            stats = {
                'layer': layer_idx,
                'neuron_count': neuron_count,
                'pair_count': int(corr_values.size),
                'mean_corr': float(np.mean(corr_values)),
                'std_corr': float(np.std(corr_values)),
                'q01_corr': float(signed_quantiles[0]),
                'q05_corr': float(signed_quantiles[1]),
                'q25_corr': float(signed_quantiles[2]),
                'q50_corr': float(signed_quantiles[3]),
                'q75_corr': float(signed_quantiles[4]),
                'q95_corr': float(signed_quantiles[5]),
                'q99_corr': float(signed_quantiles[6]),
                'mean_abs_corr': float(np.mean(abs_values)),
                'median_abs_corr': float(abs_quantiles[0]),
                'p90_abs_corr': float(abs_quantiles[1]),
                'p95_abs_corr': float(abs_quantiles[2]),
                'p99_abs_corr': float(abs_quantiles[3]),
                'max_abs_corr': float(np.max(abs_values)),
                'frac_abs_ge_0p1': float(np.mean(abs_values >= 0.1)),
                'frac_abs_ge_0p3': float(np.mean(abs_values >= 0.3)),
                'frac_abs_ge_0p5': float(np.mean(abs_values >= 0.5)),
                'signed_hist': signed_hist,
                'signed_hist_bins': signed_bins,
                'abs_hist': abs_hist,
                'abs_hist_bins': abs_bins,
                'sampled_correlations': sampled.astype(float),
                'sampled_abs_correlations': np.abs(sampled).astype(float),
            }
        else:
            stats = {
                'layer': layer_idx,
                'neuron_count': neuron_count,
                'pair_count': 0,
                'mean_corr': float('nan'),
                'std_corr': float('nan'),
                'q01_corr': float('nan'),
                'q05_corr': float('nan'),
                'q25_corr': float('nan'),
                'q50_corr': float('nan'),
                'q75_corr': float('nan'),
                'q95_corr': float('nan'),
                'q99_corr': float('nan'),
                'mean_abs_corr': float('nan'),
                'median_abs_corr': float('nan'),
                'p90_abs_corr': float('nan'),
                'p95_abs_corr': float('nan'),
                'p99_abs_corr': float('nan'),
                'max_abs_corr': float('nan'),
                'frac_abs_ge_0p1': float('nan'),
                'frac_abs_ge_0p3': float('nan'),
                'frac_abs_ge_0p5': float('nan'),
                'signed_hist': np.zeros(len(signed_bins) - 1, dtype=int),
                'signed_hist_bins': signed_bins,
                'abs_hist': np.zeros(len(abs_bins) - 1, dtype=int),
                'abs_hist_bins': abs_bins,
                'sampled_correlations': np.zeros(0, dtype=float),
                'sampled_abs_correlations': np.zeros(0, dtype=float),
            }
        layers.append(stats)
    return {'sample_size': sample_size, 'layers': layers}


def neuron_correlation_stat_matrix(neuron_corr: dict, key: str) -> np.ndarray:
    return np.asarray([layer.get(key, np.nan) for layer in neuron_corr.get('layers', [])], dtype=float)


def neuron_correlation_hist_stack(neuron_corr: dict, key: str) -> np.ndarray:
    layers = neuron_corr.get('layers', [])
    if not layers:
        return np.zeros((0, 0), dtype=float)
    return np.stack([np.asarray(layer[key], dtype=float) for layer in layers], axis=0)

def analysis_layer_sizes(state: dict[str, np.ndarray]) -> list[int]:
    if 'transformer_input_proj' in state:
        layer_count = len(sorted_transformer_layer_ids(state))
        return [state['transformer_input_proj'].shape[1] for _ in range(layer_count)]
    return [state['modules_list_{}_linear1_weight'.format(i)].shape[0] for i in sorted_mlp_layer_ids(state)]


def importance_weighted_specialization_summary(
    score_matrix: np.ndarray,
    membership_mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float | int]:
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return {
            'modularity': float('nan'),
            'active_token_count': 0,
            'total_importance': 0.0,
            'mean_token_specialization': float('nan'),
            'mean_participation': float('nan'),
            'max_participation': float('nan'),
        }

    values = np.where(np.isfinite(scores), np.maximum(scores, 0.0), 0.0)
    if membership_mask is not None:
        mask = np.asarray(membership_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError('membership_mask must match score_matrix shape')
        values = np.where(mask, values, 0.0)

    mass = np.sum(values, axis=0)
    active = mass > eps
    if not np.any(active):
        return {
            'modularity': float('nan'),
            'active_token_count': 0,
            'total_importance': 0.0,
            'mean_token_specialization': float('nan'),
            'mean_participation': float('nan'),
            'max_participation': float('nan'),
        }

    p = values[:, active] / mass[None, active]
    concentration = np.sum(p * p, axis=0)
    participation = 1.0 - concentration
    rule_count = values.shape[0]
    if rule_count <= 1:
        specialization = np.ones_like(concentration)
        max_participation = 0.0
    else:
        baseline = 1.0 / float(rule_count)
        specialization = (concentration - baseline) / (1.0 - baseline)
        specialization = np.clip(specialization, 0.0, 1.0)
        max_participation = 1.0 - baseline

    weights = mass[active]
    total_importance = float(np.sum(weights))
    modularity = float(np.sum(weights * specialization) / total_importance)
    return {
        'modularity': modularity,
        'active_token_count': int(np.sum(active)),
        'total_importance': total_importance,
        'mean_token_specialization': float(np.mean(specialization)),
        'mean_participation': float(np.mean(participation)),
        'max_participation': float(max_participation),
    }


def importance_weighted_specialization(
    score_matrix: np.ndarray,
    membership_mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    return float(importance_weighted_specialization_summary(score_matrix, membership_mask, eps=eps)['modularity'])


def sparse_bipartite_modularity_summary(
    score_matrix: np.ndarray,
    membership_mask: np.ndarray | None = None,
    eps: float = 1e-12,
    sparsity_power: float = 1.0,
) -> dict[str, float | int]:
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0:
        return {
            'modularity': float('nan'),
            'bipartite_q': float('nan'),
            'normalized_bipartite_q': float('nan'),
            'active_token_count': 0,
            'active_token_fraction': 0.0,
            'sparsity_factor': float('nan'),
            'total_importance': 0.0,
        }

    values = np.where(np.isfinite(scores), np.maximum(scores, 0.0), 0.0)
    if membership_mask is not None:
        mask = np.asarray(membership_mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError('membership_mask must match score_matrix shape')
        values = np.where(mask, values, 0.0)

    token_mass_all = np.sum(values, axis=0)
    active = token_mass_all > eps
    active_count = int(np.sum(active))
    active_fraction = float(active_count / max(values.shape[1], 1))
    if active_count == 0:
        return {
            'modularity': float('nan'),
            'bipartite_q': float('nan'),
            'normalized_bipartite_q': float('nan'),
            'active_token_count': 0,
            'active_token_fraction': active_fraction,
            'sparsity_factor': float('nan'),
            'total_importance': 0.0,
        }

    w = values[:, active]
    total_importance = float(np.sum(w))
    if total_importance <= eps:
        return {
            'modularity': float('nan'),
            'bipartite_q': float('nan'),
            'normalized_bipartite_q': float('nan'),
            'active_token_count': active_count,
            'active_token_fraction': active_fraction,
            'sparsity_factor': float('nan'),
            'total_importance': total_importance,
        }

    rule_count = int(w.shape[0])
    token_strength = np.sum(w, axis=0)
    rule_strength = np.sum(w, axis=1)
    assignment = np.argmax(w, axis=0)
    observed = float(np.sum(w[assignment, np.arange(w.shape[1])] ))
    expected = float(np.sum(rule_strength[assignment] * token_strength / max(total_importance, eps)))
    q = float((observed - expected) / max(total_importance, eps))
    max_balanced_q = 1.0 - 1.0 / float(rule_count) if rule_count > 1 else 1.0
    normalized_q = float(np.clip(q / max(max_balanced_q, eps), 0.0, 1.0))
    sparsity_factor = float(np.clip(1.0 - active_fraction, 0.0, 1.0) ** max(float(sparsity_power), 0.0))
    modularity = float(np.clip(normalized_q * sparsity_factor, 0.0, 1.0))
    return {
        'modularity': modularity,
        'bipartite_q': q,
        'normalized_bipartite_q': normalized_q,
        'active_token_count': active_count,
        'active_token_fraction': active_fraction,
        'sparsity_factor': sparsity_factor,
        'total_importance': total_importance,
    }


def sparse_bipartite_modularity(
    score_matrix: np.ndarray,
    membership_mask: np.ndarray | None = None,
    eps: float = 1e-12,
    sparsity_power: float = 1.0,
) -> float:
    return float(sparse_bipartite_modularity_summary(score_matrix, membership_mask, eps=eps, sparsity_power=sparsity_power)['modularity'])


def exclusive_modularity_from_counts(counts: np.ndarray, combo_sizes: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    combo_sizes = np.maximum(np.asarray(combo_sizes, dtype=float), 1.0)
    if counts.size != combo_sizes.size:
        return float('nan')
    total = float(np.sum(counts))
    if total <= 0:
        return float('nan')
    return float(np.sum(counts / combo_sizes) / total)


def _layer_importance_profile(layer: dict) -> np.ndarray | None:
    if 'score_ratio' in layer:
        return np.asarray(layer['score_ratio'], dtype=float)
    value_name = layer.get('mask_value_name')
    if value_name in layer:
        return np.asarray(layer[value_name], dtype=float)
    for key in ('ablation_scores', 'taylor_scores', 'legacy_hessian_scores', 'weight_scores'):
        if key in layer:
            return np.asarray(layer[key], dtype=float)
    return None


def modularity_from_combo_result(combo_result: dict) -> np.ndarray:
    layers = combo_result.get('layers', [])
    if not layers:
        return np.zeros(0, dtype=float)
    combo_sizes = np.asarray(combo_result.get('combo_sizes', []), dtype=float)
    scores = []
    summaries = []
    for layer in layers:
        matrix = _layer_importance_profile(layer)
        if matrix is None:
            counts = np.asarray(layer.get('exclusive_counts', []), dtype=float)
            score = exclusive_modularity_from_counts(counts, combo_sizes)
            summary = {
                'modularity': score,
                'active_token_count': int(np.nansum(counts)) if counts.size else 0,
                'total_importance': float(np.nansum(counts)) if counts.size else 0.0,
                'mean_token_specialization': float('nan'),
                'mean_participation': float('nan'),
                'max_participation': float('nan'),
            }
        else:
            summary = importance_weighted_specialization_summary(matrix, membership_mask=layer.get('important_mask'))
            score = float(summary['modularity'])
        scores.append(score)
        summaries.append(summary)
    combo_result['modularity_metric'] = 'importance_weighted_normalized_inverse_participation'
    combo_result['layer_specialization_summaries'] = summaries
    return np.asarray(scores, dtype=float)

def forward_with_layer_keep(state: dict[str, np.ndarray], x: np.ndarray, keep_by_layer: dict[int, list[int]]):
    if 'transformer_input_proj' in state:
        input_proj = state['transformer_input_proj']
        token_bias = state['transformer_token_bias']
        half_dim = x.shape[1] // 2
        model_dim = input_proj.shape[1]
        left = x[:, :half_dim] @ input_proj
        right = x[:, half_dim:] @ input_proj
        tokens = np.stack([left, right], axis=1) + token_bias[None, :, :]
        activations = []
        for idx in sorted_transformer_layer_ids(state):
            q = tokens @ state['transformer_layer_{}_wq'.format(idx)]
            k = tokens @ state['transformer_layer_{}_wk'.format(idx)]
            v = tokens @ state['transformer_layer_{}_wv'.format(idx)]
            attention = softmax(q @ np.swapaxes(k, 1, 2) / np.sqrt(model_dim), axis=-1)
            tokens = layer_norm(tokens + (attention @ v) @ state['transformer_layer_{}_wo'.format(idx)])
            ff = relu(tokens @ state['transformer_layer_{}_ff1'.format(idx)] + state['transformer_layer_{}_b1'.format(idx)])
            tokens = layer_norm(tokens + ff @ state['transformer_layer_{}_ff2'.format(idx)] + state['transformer_layer_{}_b2'.format(idx)])
            if idx in keep_by_layer:
                keep_idx = keep_by_layer[idx]
                tokens = tokens.copy()
                mask = np.ones(tokens.shape[2], dtype=bool)
                mask[keep_idx] = False
                tokens[:, :, mask] = 0.0
            activations.append(tokens.mean(axis=1))
        features = tokens.mean(axis=1)
        out = features @ state['output_block_linear1_weight'].T + state['output_block_linear1_bias']
        activations.append(out)
        return out, activations

    current = x
    activations = []
    for idx in sorted_mlp_layer_ids(state):
        w = state['modules_list_{}_linear1_weight'.format(idx)]
        b = state['modules_list_{}_linear1_bias'.format(idx)]
        z = current @ w.T + b
        if idx in keep_by_layer:
            keep_idx = keep_by_layer[idx]
            z = z.copy()
            mask = np.ones(z.shape[1], dtype=bool)
            mask[keep_idx] = False
            z[:, mask] = 0.0
        activations.append(z)
        current = relu(z)
    out = current @ state['output_block_linear1_weight'].T + state['output_block_linear1_bias']
    activations.append(out)
    return out, activations


def rule_output_change_to_baseline(pred: np.ndarray, baseline: np.ndarray, pattern_size: int, rule_indices: list[int]) -> np.ndarray:
    values = []
    for rule_pos, _rule_id in enumerate(rule_indices):
        start = rule_pos * pattern_size
        end = start + pattern_size
        diff = pred[:, start:end] - baseline[:, start:end]
        values.append(float(np.mean(np.sum(diff * diff, axis=1))))
    return np.asarray(values, dtype=float)


def rule_similarity_to_baseline(pred: np.ndarray, baseline: np.ndarray, pattern_size: int, rule_indices: list[int]) -> np.ndarray:
    return 1.0 - rule_output_change_to_baseline(pred, baseline, pattern_size, rule_indices) / max(pattern_size, 1)


def rule_accuracy_vector(pred: np.ndarray, target: np.ndarray, pattern_size: int, rule_indices: list[int]) -> np.ndarray:
    acc = rule_accuracy(pred, target, pattern_size, rule_indices)
    return np.asarray([acc[r] for r in rule_indices], dtype=float)


def combo_list(num_rules: int, max_combo_size: int | None = None) -> list[tuple[int, ...]]:
    import itertools
    max_size = max_combo_size or num_rules
    combos = []
    for size in range(1, min(max_size, num_rules) + 1):
        combos.extend(list(itertools.combinations(range(num_rules), size)))
    return combos


def combo_label(combo: tuple[int, ...], rule_indices: list[int]) -> str:
    return '-'.join(str(rule_indices[i]) for i in combo)


def combo_keep_mask(
    important: np.ndarray,
    combo: tuple[int, ...],
    membership_mask: np.ndarray | None = None,
) -> np.ndarray:
    _ = membership_mask  # kept for call-site compatibility; important already includes extraction membership.
    important = np.asarray(important, dtype=bool)
    return np.any(important[list(combo), :], axis=0)


def combo_exclusive_mask(
    important: np.ndarray,
    combo: tuple[int, ...],
    membership_mask: np.ndarray | None = None,
) -> np.ndarray:
    _ = membership_mask  # kept for call-site compatibility; exclusivity is defined by the important mask.
    rule_membership = np.asarray(important, dtype=bool)
    in_combo = np.all(rule_membership[list(combo), :], axis=0)
    other = [i for i in range(rule_membership.shape[0]) if i not in combo]
    if other:
        return in_combo & np.all(~rule_membership[other, :], axis=0)
    return in_combo

def validate_mask_method(mask_method: str) -> str:
    method = mask_method.lower().strip()
    if method not in {'ablation', 'taylor', 'legacy_hessian'}:
        raise ValueError("mask_method must be 'ablation', 'taylor', or 'legacy_hessian'")
    return method


def validate_subnet_target(subnet_target: str) -> str:
    target = subnet_target.lower().strip()
    if target not in {'neuron', 'weight', 'both'}:
        raise ValueError("subnet_target must be 'neuron', 'weight', or 'both'")
    return target


def validate_importance_mask_mode(mask_mode: str) -> str:
    mode = str(mask_mode).lower().strip()
    aliases = {
        'score_ratio': 'score_ratio',
        'neuron_score_ratio': 'score_ratio',
        'weight_score_ratio': 'score_ratio',
        'functional_ratio': 'score_ratio',
        'functional_score_ratio': 'score_ratio',
        'output_energy_ratio': 'score_ratio',
        'baseline_ratio': 'score_ratio',
        'baseline_energy_ratio': 'score_ratio',
        'energy_ratio': 'score_ratio',
        'neuron_energy_ratio': 'score_ratio',
        'weight_energy_ratio': 'score_ratio',
    }
    mode = aliases.get(mode, mode)
    if mode != 'score_ratio':
        raise ValueError("importance_mask_mode now only supports 'score_ratio'")
    return mode


def validate_neuron_importance_mask_mode(mask_mode: str) -> str:
    return validate_importance_mask_mode(mask_mode)


def rule_output_energy(baseline_pred: np.ndarray, pattern_size: int, rule_indices: list[int], eps: float = 1e-8) -> np.ndarray:
    energies = []
    for rule_pos, _rule_id in enumerate(rule_indices):
        start = rule_pos * pattern_size
        end = start + pattern_size
        segment = baseline_pred[:, start:end]
        energies.append(float(np.mean(np.sum(segment * segment, axis=1)) + eps))
    return np.asarray(energies, dtype=float)


def score_ratio(scores: np.ndarray, output_energy: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    energies = np.asarray(output_energy, dtype=float).reshape(-1, 1)
    if energies.shape[0] != scores.shape[0]:
        raise ValueError('output_energy length must match the number of rules in scores')
    return scores / np.maximum(energies, eps)


def neuron_score_ratio(scores: np.ndarray, output_energy: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return score_ratio(scores, output_energy, eps=eps)


def importance_score_tables(scores: np.ndarray, output_energy: np.ndarray, output_eps: float = 1e-8) -> np.ndarray:
    return score_ratio(scores, output_energy, eps=output_eps)


def scoring_threshold_table(scores: np.ndarray, output_energy: np.ndarray, output_eps: float = 1e-8, mask_method: str = 'ablation') -> np.ndarray:
    if validate_mask_method(mask_method) == 'legacy_hessian':
        return np.asarray(scores, dtype=float)
    return importance_score_tables(scores, output_energy, output_eps=output_eps)


def validate_token_extraction_method(method: str | None) -> str:
    value = str(method or DEFAULT_TOKEN_EXTRACTION_METHOD).lower().strip().replace('-', '_')
    aliases = {
        'legacy': 'pt',
        'p_t': 'pt',
        'pt_threshold': 'pt',
        'mean': 'score_mean_fraction',
        'mean_fraction': 'score_mean_fraction',
        'average_fraction': 'score_mean_fraction',
        'score_average_fraction': 'score_mean_fraction',
        'median': 'score_median_fraction',
        'median_fraction': 'score_median_fraction',
        'quantile': 'score_quantile',
        'score_percentile': 'score_quantile',
        'percentile': 'score_quantile',
    }
    value = aliases.get(value, value)
    if value not in TOKEN_EXTRACTION_METHOD_CHOICES:
        raise ValueError('token_extraction_method must be one of {}'.format(', '.join(TOKEN_EXTRACTION_METHOD_CHOICES)))
    return value


def resolve_token_extraction(
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
    rule_score_average_fraction: float | None = None,
) -> tuple[str, float]:
    if rule_score_average_fraction is not None:
        token_extraction_method = 'score_mean_fraction'
        token_extraction_fraction = float(rule_score_average_fraction)
    return validate_token_extraction_method(token_extraction_method), float(token_extraction_fraction)


def token_extraction_membership(
    scores: np.ndarray,
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
) -> np.ndarray:
    method = validate_token_extraction_method(token_extraction_method)
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return np.zeros_like(scores, dtype=bool)
    finite = np.isfinite(scores)
    if method == 'pt':
        return finite
    fraction = float(token_extraction_fraction)
    if fraction <= 0.0:
        return finite
    if method == 'score_mean_fraction':
        sums = np.sum(np.where(finite, scores, 0.0), axis=1)
        counts = np.sum(finite, axis=1)
        reference = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)
    elif method == 'score_quantile':
        quantile = min(max(fraction, 0.0), 1.0)
        masked = np.where(finite, scores, np.nan)
        with np.errstate(all='ignore'):
            reference = np.nanquantile(masked, quantile, axis=1)
        reference = np.where(np.isfinite(reference), reference, 0.0)
        return finite & (scores >= reference[:, None])
    else:
        masked = np.where(finite, scores, np.nan)
        with np.errstate(all='ignore'):
            reference = np.nanmedian(masked, axis=1)
        reference = np.where(np.isfinite(reference), reference, 0.0)
    return finite & (scores >= reference[:, None] * fraction)


def rule_score_average_membership(
    scores: np.ndarray,
    average_fraction: float | None = DEFAULT_RULE_SCORE_AVERAGE_FRACTION,
) -> np.ndarray:
    if average_fraction is None:
        return token_extraction_membership(scores, token_extraction_method='pt')
    return token_extraction_membership(
        scores,
        token_extraction_method='score_mean_fraction',
        token_extraction_fraction=float(average_fraction),
    )


def importance_mask_from_tables(
    scores: np.ndarray,
    ratio: np.ndarray,
    min_score: float = 1e-9,
    score_ratio_threshold: float | str = 1e-4,
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
    rule_score_average_fraction: float | None = None,
) -> np.ndarray:
    token_extraction_method, token_extraction_fraction = resolve_token_extraction(
        token_extraction_method,
        token_extraction_fraction,
        rule_score_average_fraction=rule_score_average_fraction,
    )
    scores = np.asarray(scores, dtype=float)
    ratio = np.asarray(ratio, dtype=float)
    if scores.size == 0:
        return np.zeros_like(scores, dtype=bool)
    if token_extraction_method != 'pt':
        membership = token_extraction_membership(scores, token_extraction_method, token_extraction_fraction)
        return (scores > min_score) & membership
    base = (scores >= min_score) & np.isfinite(ratio)
    if isinstance(score_ratio_threshold, str) and score_ratio_threshold.lower().strip() == 'median':
        thresholds = np.full((ratio.shape[0], 1), np.inf, dtype=float)
        for rule_pos in range(ratio.shape[0]):
            valid = base[rule_pos] & (ratio[rule_pos] > 0.0)
            if np.any(valid):
                thresholds[rule_pos, 0] = float(np.nanmedian(ratio[rule_pos, valid]))
        return base & (ratio >= thresholds)
    return base & (ratio > float(score_ratio_threshold))


def importance_mask(
    scores: np.ndarray,
    min_score: float = 1e-9,
    mask_mode: str = 'score_ratio',
    output_energy: np.ndarray | None = None,
    score_ratio_threshold: float = 1e-4,
    output_eps: float = 1e-8,
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
    rule_score_average_fraction: float | None = None,
) -> np.ndarray:
    validate_importance_mask_mode(mask_mode)
    if output_energy is None:
        raise ValueError('output_energy is required for score_ratio masks')
    ratio = importance_score_tables(scores, output_energy, output_eps=output_eps)
    return importance_mask_from_tables(
        scores,
        ratio,
        min_score=min_score,
        score_ratio_threshold=score_ratio_threshold,
        token_extraction_method=token_extraction_method,
        token_extraction_fraction=token_extraction_fraction,
        rule_score_average_fraction=rule_score_average_fraction,
    )


def combo_membership(
    important: np.ndarray,
    combos: list[tuple[int, ...]],
    item_indices: np.ndarray | None = None,
    membership_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    important = np.asarray(important, dtype=bool)
    if membership_mask is not None:
        membership_mask = np.asarray(membership_mask, dtype=bool)
        if membership_mask.shape != important.shape:
            raise ValueError('membership_mask shape must match important')
    if item_indices is None:
        item_indices = np.arange(important.shape[1], dtype=int)
    else:
        item_indices = np.asarray(item_indices, dtype=int)
    keep_indices = []
    exclusive_indices = []
    keep_counts = np.zeros(len(combos), dtype=int)
    exclusive_counts = np.zeros(len(combos), dtype=int)
    for combo_idx, combo in enumerate(combos):
        keep_mask = combo_keep_mask(important, combo, membership_mask=membership_mask)
        exclusive_mask = combo_exclusive_mask(important, combo, membership_mask=membership_mask)
        keep_idx = item_indices[keep_mask].astype(int).tolist()
        exact_idx = item_indices[exclusive_mask].astype(int).tolist()
        keep_indices.append(keep_idx)
        exclusive_indices.append(exact_idx)
        keep_counts[combo_idx] = len(keep_idx)
        exclusive_counts[combo_idx] = len(exact_idx)
    return {
        'keep_indices': keep_indices,
        'exclusive_indices': exclusive_indices,
        'keep_counts': keep_counts,
        'exclusive_counts': exclusive_counts,
    }

def combo_subnetwork_analysis(
    state: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    pattern_size: int,
    rule_indices: list[int],
    max_combo_size: int | None = None,
    mask_method: str = 'ablation',
    importance_min_score: float = 1e-9,
    importance_mask_mode: str = 'score_ratio',
    importance_score_ratio_threshold: float = 1e-4,
    importance_output_eps: float = 1e-8,
    perturbation_batch_size: int = 64,
    analysis_device: str = 'auto',
    taylor_probes: int = 4,
    taylor_scores: dict[str, Any] | None = None,
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
    rule_score_average_fraction: float | None = None,
) -> dict:
    mask_method = validate_mask_method(mask_method)
    importance_mask_mode = validate_importance_mask_mode(importance_mask_mode)
    token_extraction_method, token_extraction_fraction = resolve_token_extraction(
        token_extraction_method,
        token_extraction_fraction,
        rule_score_average_fraction=rule_score_average_fraction,
    )
    baseline_pred, _activations = forward_from_state(state, x)
    output_energy = rule_output_energy(baseline_pred, pattern_size, rule_indices, eps=importance_output_eps)
    baseline_acc = rule_accuracy_vector(baseline_pred, y, pattern_size, rule_indices)
    combos = combo_list(len(rule_indices), max_combo_size=max_combo_size)
    labels = [combo_label(c, rule_indices) for c in combos]
    combo_sizes = np.asarray([len(c) for c in combos], dtype=int)
    layer_results = []
    ablation_context = torch_ablation_context(state, x, baseline_pred, device=analysis_device) if mask_method == 'ablation' else None
    if mask_method == 'taylor' and taylor_scores is None:
        taylor_scores = taylor_importance_scores(
            state,
            x,
            pattern_size,
            rule_indices,
            batch_size=perturbation_batch_size,
            device=analysis_device,
            probes=taylor_probes,
        )

    for layer_idx, width in enumerate(analysis_layer_sizes(state)):
        if mask_method == 'taylor':
            values = np.asarray(taylor_scores['neuron_layers'][layer_idx], dtype=float)
            value_name = 'taylor_scores'
        elif mask_method == 'legacy_hessian':
            values = legacy_hessian_neuron_scores(
                state,
                x,
                baseline_pred,
                pattern_size,
                rule_indices,
                layer_idx,
                width,
                batch_size=perturbation_batch_size,
                device=analysis_device,
                context=ablation_context,
            )
            value_name = 'legacy_hessian_scores'
        else:
            similarity = torch_neuron_silence_similarity(
                state,
                x,
                baseline_pred,
                pattern_size,
                rule_indices,
                layer_idx,
                width,
                batch_size=perturbation_batch_size,
                device=analysis_device,
                context=ablation_context,
            )
            if similarity is None:
                values = np.zeros((len(rule_indices), width), dtype=float)
                for neuron in range(width):
                    silenced_pred, _ = forward_from_state(state, x, silence=(layer_idx, [neuron]))
                    values[:, neuron] = rule_output_change_to_baseline(silenced_pred, baseline_pred, pattern_size, rule_indices)
            else:
                values = (1.0 - similarity) * max(pattern_size, 1)
            value_name = 'ablation_scores'
        ratio = scoring_threshold_table(values, output_energy, output_eps=importance_output_eps, mask_method=mask_method)
        important = importance_mask_from_tables(
            values,
            ratio,
            min_score=importance_min_score,
            score_ratio_threshold=importance_score_ratio_threshold,
            token_extraction_method=token_extraction_method,
            token_extraction_fraction=token_extraction_fraction,
        )
        membership_mask = token_extraction_membership(values, token_extraction_method, token_extraction_fraction)
        membership = combo_membership(important, combos, membership_mask=membership_mask)
        task_specialization = importance_weighted_specialization_summary(ratio, membership_mask=important)

        retained = np.zeros((len(rule_indices), len(combos)), dtype=float)
        for combo_idx, keep_idx in enumerate(membership['keep_indices']):
            kept_pred, _ = forward_with_layer_keep(state, x, {layer_idx: keep_idx})
            acc = rule_accuracy_vector(kept_pred, y, pattern_size, rule_indices)
            retained[:, combo_idx] = acc / np.maximum(baseline_acc, 1e-12)

        layer_results.append({
            'layer': layer_idx,
            'mask_method': mask_method,
            'importance_mask_mode': importance_mask_mode,
            'importance_min_score': importance_min_score,
            'importance_score_ratio_threshold': importance_score_ratio_threshold,
            'importance_neuron_threshold': importance_score_ratio_threshold,
            'importance_output_eps': importance_output_eps,
            'token_extraction_method': token_extraction_method,
            'token_extraction_fraction': token_extraction_fraction,
            'output_energy': output_energy,
            'score_ratio': ratio,
            'neuron_score_ratio': ratio,
            'mask_value_name': value_name,
            value_name: values,
            'important_mask': important,
            'combo_labels': labels,
            'combo_sizes': combo_sizes,
            'membership_mode': 'union_combo',
            'modularity_metric': 'importance_weighted_normalized_inverse_participation',
            'task_specialization': task_specialization,
            'keep_indices': membership['keep_indices'],
            'exclusive_indices': membership['exclusive_indices'],
            'keep_counts': membership['keep_counts'],
            'exclusive_counts': membership['exclusive_counts'],
            'retained_accuracy': retained,
        })

    network_retained = np.zeros((len(rule_indices), len(combos)), dtype=float)
    network_keep_counts = np.zeros((len(layer_results), len(combos)), dtype=int)
    network_exclusive_counts = np.zeros((len(layer_results), len(combos)), dtype=int)
    for combo_idx, _combo in enumerate(combos):
        keep_by_layer = {}
        for layer_pos, layer in enumerate(layer_results):
            keep_idx = layer['keep_indices'][combo_idx]
            keep_by_layer[layer['layer']] = keep_idx
            network_keep_counts[layer_pos, combo_idx] = len(keep_idx)
            network_exclusive_counts[layer_pos, combo_idx] = len(layer['exclusive_indices'][combo_idx])
        kept_pred, _ = forward_with_layer_keep(state, x, keep_by_layer)
        acc = rule_accuracy_vector(kept_pred, y, pattern_size, rule_indices)
        network_retained[:, combo_idx] = acc / np.maximum(baseline_acc, 1e-12)

    resource_usage = neuron_subnet_resource_usage(state, layer_results, combos, labels)

    return {
        'mask_method': mask_method,
        'importance_min_score': importance_min_score,
        'importance_mask_mode': importance_mask_mode,
        'importance_score_ratio_threshold': importance_score_ratio_threshold,
        'importance_neuron_threshold': importance_score_ratio_threshold,
        'importance_output_eps': importance_output_eps,
        'token_extraction_method': token_extraction_method,
        'token_extraction_fraction': token_extraction_fraction,
        'baseline_accuracy': baseline_acc,
        'combo_labels': labels,
        'combo_sizes': combo_sizes,
        'layers': layer_results,
        'network_retained_accuracy': network_retained,
        'network_keep_counts': network_keep_counts,
        'network_exclusive_counts': network_exclusive_counts,
        'resource_usage': resource_usage,
    }

def mlp_neuron_subnet_weight_count(state: dict[str, np.ndarray], keep_by_layer: dict[int, list[int]]) -> tuple[int, int, str]:
    layer_ids = sorted_mlp_layer_ids(state)
    total = int(sum(state['modules_list_{}_linear1_weight'.format(idx)].size for idx in layer_ids) + state['output_block_linear1_weight'].size)
    if not layer_ids:
        return 0, total, 'mlp_exact_weight_matrices_excluding_bias'
    retained = 0
    prev_count = None
    for pos, layer_idx in enumerate(layer_ids):
        weight = state['modules_list_{}_linear1_weight'.format(layer_idx)]
        keep_count = len(keep_by_layer.get(layer_idx, []))
        if pos == 0:
            retained += keep_count * int(weight.shape[1])
        else:
            retained += keep_count * int(prev_count or 0)
        prev_count = keep_count
    retained += int(state['output_block_linear1_weight'].shape[0]) * int(prev_count or 0)
    return int(retained), total, 'mlp_exact_weight_matrices_excluding_bias'


def transformer_neuron_subnet_weight_count(state: dict[str, np.ndarray], keep_by_layer: dict[int, list[int]]) -> tuple[int, int, str]:
    specs = weight_block_specs(state)
    total = int(sum(state[spec['key']].size for spec in specs))
    layer_ids = sorted_transformer_layer_ids(state)
    if not layer_ids:
        return 0, total, 'transformer_dimension_estimate_weight_matrices_excluding_bias'
    model_dim = int(state['transformer_input_proj'].shape[1])
    half_dim = int(state['transformer_input_proj'].shape[0])
    retained = half_dim * len(keep_by_layer.get(layer_ids[0], []))
    prev_count = len(keep_by_layer.get(layer_ids[0], []))
    for pos, layer_idx in enumerate(layer_ids):
        curr_count = len(keep_by_layer.get(layer_idx, []))
        source_count = model_dim if pos == 0 else prev_count
        ff_dim = int(state['transformer_layer_{}_ff1'.format(layer_idx)].shape[1])
        for name in ['wq', 'wk', 'wv', 'wo']:
            key = 'transformer_layer_{}_{}'.format(layer_idx, name)
            if key in state:
                retained += int(source_count) * int(curr_count)
        if 'transformer_layer_{}_ff1'.format(layer_idx) in state:
            retained += int(source_count) * ff_dim
        if 'transformer_layer_{}_ff2'.format(layer_idx) in state:
            retained += ff_dim * int(curr_count)
        prev_count = curr_count
    retained += int(state['output_block_linear1_weight'].shape[0]) * int(prev_count)
    return int(retained), total, 'transformer_dimension_estimate_weight_matrices_excluding_bias'


def neuron_subnet_resource_usage(state: dict[str, np.ndarray], layer_results: list[dict[str, Any]], combos: list[tuple[int, ...]], combo_labels: list[str]) -> dict[str, Any]:
    layer_ids = [int(layer['layer']) for layer in layer_results]
    layer_widths = np.asarray([np.asarray(layer['important_mask']).shape[1] for layer in layer_results], dtype=float)
    total_neurons = int(np.sum(layer_widths)) if layer_widths.size else 0
    retained_neuron_counts = np.zeros(len(combos), dtype=int)
    retained_neuron_ratios = np.zeros(len(combos), dtype=float)
    retained_weight_counts = np.zeros(len(combos), dtype=int)
    retained_weight_ratios = np.zeros(len(combos), dtype=float)
    layer_neuron_counts = np.zeros((len(layer_results), len(combos)), dtype=int)
    layer_neuron_ratios = np.zeros((len(layer_results), len(combos)), dtype=float)
    total_weights = int(sum(state[spec['key']].size for spec in weight_block_specs(state)))
    weight_count_mode = 'weight_matrices_excluding_bias'

    for combo_idx, combo in enumerate(combos):
        keep_by_layer: dict[int, list[int]] = {}
        for layer_pos, layer in enumerate(layer_results):
            if 'keep_indices' in layer:
                keep_idx = [int(value) for value in layer['keep_indices'][combo_idx]]
            else:
                keep_mask = combo_keep_mask(np.asarray(layer['important_mask'], dtype=bool), combo)
                keep_idx = np.where(keep_mask)[0].astype(int).tolist()
            layer_id = int(layer['layer'])
            keep_by_layer[layer_id] = keep_idx
            layer_neuron_counts[layer_pos, combo_idx] = len(keep_idx)
            layer_neuron_ratios[layer_pos, combo_idx] = len(keep_idx) / max(float(layer_widths[layer_pos]), 1.0)
        retained_neuron_counts[combo_idx] = int(np.sum(layer_neuron_counts[:, combo_idx]))
        retained_neuron_ratios[combo_idx] = retained_neuron_counts[combo_idx] / max(float(total_neurons), 1.0)
        if 'transformer_input_proj' in state:
            retained_weights, total_weights, weight_count_mode = transformer_neuron_subnet_weight_count(state, keep_by_layer)
        else:
            retained_weights, total_weights, weight_count_mode = mlp_neuron_subnet_weight_count(state, keep_by_layer)
        retained_weight_counts[combo_idx] = int(retained_weights)
        retained_weight_ratios[combo_idx] = retained_weights / max(float(total_weights), 1.0)

    return {
        'combo_labels': combo_labels,
        'layer_ids': np.asarray(layer_ids, dtype=int),
        'layer_widths': layer_widths.astype(int),
        'total_neurons': total_neurons,
        'total_weights': int(total_weights),
        'weight_count_mode': weight_count_mode,
        'layer_neuron_counts': layer_neuron_counts,
        'layer_neuron_ratios': layer_neuron_ratios,
        'retained_neuron_counts': retained_neuron_counts,
        'retained_neuron_ratios': retained_neuron_ratios,
        'retained_weight_counts': retained_weight_counts,
        'retained_weight_ratios': retained_weight_ratios,
    }

def combo_retained_stack(combo_result: dict) -> np.ndarray:
    if not combo_result['layers']:
        return np.zeros((0, 0, 0), dtype=float)
    return np.stack([layer['retained_accuracy'] for layer in combo_result['layers']], axis=0)


def combo_keep_count_stack(combo_result: dict, key: str = 'keep_counts') -> np.ndarray:
    if not combo_result['layers']:
        return np.zeros((0, 0), dtype=int)
    return np.stack([layer[key] for layer in combo_result['layers']], axis=0)



def weight_block_specs(state: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    specs = []
    if 'transformer_input_proj' in state:
        specs.append({'key': 'transformer_input_proj', 'label': 'input_proj', 'shape': list(state['transformer_input_proj'].shape)})
        for layer_idx in sorted_transformer_layer_ids(state):
            for name in ['wq', 'wk', 'wv', 'wo', 'ff1', 'ff2']:
                key = 'transformer_layer_{}_{}'.format(layer_idx, name)
                if key in state:
                    specs.append({'key': key, 'label': 'layer{}_{}'.format(layer_idx, name), 'shape': list(state[key].shape)})
        specs.append({'key': 'output_block_linear1_weight', 'label': 'output', 'shape': list(state['output_block_linear1_weight'].shape)})
        return specs

    for layer_idx in sorted_mlp_layer_ids(state):
        key = 'modules_list_{}_linear1_weight'.format(layer_idx)
        specs.append({'key': key, 'label': 'hidden{}'.format(layer_idx), 'shape': list(state[key].shape)})
    specs.append({'key': 'output_block_linear1_weight', 'label': 'output', 'shape': list(state['output_block_linear1_weight'].shape)})
    return specs


def select_weight_candidates(weight: np.ndarray, max_elements: int | None) -> np.ndarray:
    flat = np.abs(weight.reshape(-1))
    if max_elements is None or max_elements >= flat.size:
        return np.arange(flat.size, dtype=int)
    if max_elements <= 0:
        return np.zeros(0, dtype=int)
    take = min(max_elements, flat.size)
    idx = np.argpartition(-flat, take - 1)[:take]
    return np.sort(idx.astype(int))


def state_with_zeroed_weight(state: dict[str, np.ndarray], key: str, flat_index: int) -> dict[str, np.ndarray]:
    modified = dict(state)
    weight = state[key].copy()
    weight.reshape(-1)[flat_index] = 0.0
    modified[key] = weight
    return modified


def state_with_kept_weights(state: dict[str, np.ndarray], keep_by_key: dict[str, list[int]]) -> dict[str, np.ndarray]:
    modified = dict(state)
    for key, keep_idx in keep_by_key.items():
        weight = state[key].copy()
        flat = weight.reshape(-1)
        mask = np.ones(flat.size, dtype=bool)
        if keep_idx:
            mask[np.asarray(keep_idx, dtype=int)] = False
        flat[mask] = 0.0
        modified[key] = weight
    return modified


def weight_combo_subnetwork_analysis(
    state: dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    pattern_size: int,
    rule_indices: list[int],
    max_combo_size: int | None = None,
    max_elements_per_block: int | None = None,
    mask_method: str = 'ablation',
    importance_min_score: float = 1e-9,
    importance_mask_mode: str = 'score_ratio',
    importance_score_ratio_threshold: float = 1e-6,
    importance_output_eps: float = 1e-8,
    perturbation_batch_size: int = 64,
    analysis_device: str = 'auto',
    taylor_probes: int = 4,
    taylor_scores: dict[str, Any] | None = None,
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
    rule_score_average_fraction: float | None = None,
) -> dict:
    mask_method = validate_mask_method(mask_method)
    if mask_method == 'legacy_hessian':
        raise ValueError("legacy_hessian currently supports neuron subnet analysis only")
    importance_mask_mode = validate_importance_mask_mode(importance_mask_mode)
    token_extraction_method, token_extraction_fraction = resolve_token_extraction(
        token_extraction_method,
        token_extraction_fraction,
        rule_score_average_fraction=rule_score_average_fraction,
    )
    baseline_pred, _activations = forward_from_state(state, x)
    output_energy = rule_output_energy(baseline_pred, pattern_size, rule_indices, eps=importance_output_eps)
    baseline_acc = rule_accuracy_vector(baseline_pred, y, pattern_size, rule_indices)
    combos = combo_list(len(rule_indices), max_combo_size=max_combo_size)
    labels = [combo_label(c, rule_indices) for c in combos]
    combo_sizes = np.asarray([len(c) for c in combos], dtype=int)
    block_results = []
    ablation_context = torch_ablation_context(state, x, baseline_pred, device=analysis_device) if mask_method == 'ablation' else None
    if mask_method == 'taylor' and taylor_scores is None:
        taylor_scores = taylor_importance_scores(
            state,
            x,
            pattern_size,
            rule_indices,
            batch_size=perturbation_batch_size,
            device=analysis_device,
            probes=taylor_probes,
        )

    for block_idx, spec in enumerate(weight_block_specs(state)):
        key = spec['key']
        candidates = select_weight_candidates(state[key], max_elements_per_block)
        if mask_method == 'taylor':
            flat_scores = np.asarray(taylor_scores['weights'][key], dtype=float).reshape(len(rule_indices), -1)
            values = flat_scores[:, candidates]
            value_name = 'taylor_scores'
        else:
            similarity = torch_weight_zero_similarity(
                state,
                x,
                baseline_pred,
                pattern_size,
                rule_indices,
                key,
                candidates,
                batch_size=perturbation_batch_size,
                device=analysis_device,
                context=ablation_context,
            )
            if similarity is None:
                values = np.zeros((len(rule_indices), len(candidates)), dtype=float)
                for pos, flat_index in enumerate(candidates):
                    zeroed_state = state_with_zeroed_weight(state, key, int(flat_index))
                    zeroed_pred, _ = forward_from_state(zeroed_state, x)
                    values[:, pos] = rule_output_change_to_baseline(zeroed_pred, baseline_pred, pattern_size, rule_indices)
            else:
                values = (1.0 - similarity) * max(pattern_size, 1)
            value_name = 'ablation_scores'
        ratio = importance_score_tables(values, output_energy, output_eps=importance_output_eps)
        important = importance_mask_from_tables(
            values,
            ratio,
            min_score=importance_min_score,
            score_ratio_threshold=importance_score_ratio_threshold,
            token_extraction_method=token_extraction_method,
            token_extraction_fraction=token_extraction_fraction,
        )
        membership_mask = token_extraction_membership(values, token_extraction_method, token_extraction_fraction)
        membership = combo_membership(important, combos, candidates, membership_mask=membership_mask)
        task_specialization = importance_weighted_specialization_summary(ratio, membership_mask=important)

        block_results.append({
            'block': block_idx,
            'key': key,
            'label': spec['label'],
            'shape': spec['shape'],
            'mask_method': mask_method,
            'importance_mask_mode': importance_mask_mode,
            'importance_min_score': importance_min_score,
            'importance_score_ratio_threshold': importance_score_ratio_threshold,
            'importance_output_eps': importance_output_eps,
            'token_extraction_method': token_extraction_method,
            'token_extraction_fraction': token_extraction_fraction,
            'output_energy': output_energy,
            'score_ratio': ratio,
            'weight_score_ratio': ratio,
            'mask_value_name': value_name,
            'candidate_flat_indices': candidates.astype(int),
            value_name: values,
            'important_mask': important,
            'combo_labels': labels,
            'combo_sizes': combo_sizes,
            'membership_mode': 'union_combo',
            'modularity_metric': 'importance_weighted_normalized_inverse_participation',
            'task_specialization': task_specialization,
            'keep_flat_indices': membership['keep_indices'],
            'exclusive_flat_indices': membership['exclusive_indices'],
            'keep_counts': membership['keep_counts'],
            'exclusive_counts': membership['exclusive_counts'],
        })

    network_retained = np.zeros((len(rule_indices), len(combos)), dtype=float)
    network_keep_counts = np.zeros((len(block_results), len(combos)), dtype=int)
    network_exclusive_counts = np.zeros((len(block_results), len(combos)), dtype=int)
    for combo_idx, _combo in enumerate(combos):
        keep_by_key = {}
        for block_pos, block in enumerate(block_results):
            keep_idx = block['keep_flat_indices'][combo_idx]
            keep_by_key[block['key']] = keep_idx
            network_keep_counts[block_pos, combo_idx] = len(keep_idx)
            network_exclusive_counts[block_pos, combo_idx] = len(block['exclusive_flat_indices'][combo_idx])
        kept_state = state_with_kept_weights(state, keep_by_key)
        kept_pred, _ = forward_from_state(kept_state, x)
        acc = rule_accuracy_vector(kept_pred, y, pattern_size, rule_indices)
        network_retained[:, combo_idx] = acc / np.maximum(baseline_acc, 1e-12)

    return {
        'mask_method': mask_method,
        'importance_min_score': importance_min_score,
        'importance_mask_mode': importance_mask_mode,
        'importance_score_ratio_threshold': importance_score_ratio_threshold,
        'importance_output_eps': importance_output_eps,
        'token_extraction_method': token_extraction_method,
        'token_extraction_fraction': token_extraction_fraction,
        'max_elements_per_block': max_elements_per_block,
        'baseline_accuracy': baseline_acc,
        'combo_labels': labels,
        'combo_sizes': combo_sizes,
        'blocks': block_results,
        'network_retained_accuracy': network_retained,
        'network_keep_counts': network_keep_counts,
        'network_exclusive_counts': network_exclusive_counts,
    }

def load_run_npz(run_dir: Path) -> dict[str, Any]:
    raw = np.load(run_dir / 'run_data.npz', allow_pickle=False)
    state = {key: raw[key] for key in raw.files if key not in ['x_test', 'y_test', 'pattern_size', 'rule_indices', 'rule_names']}
    return {
        'state': state,
        'x_test': raw['x_test'],
        'y_test': raw['y_test'],
        'pattern_size': int(raw['pattern_size']),
        'rule_indices': raw['rule_indices'].astype(int).tolist(),
        'rule_names': [str(x) for x in raw['rule_names'].tolist()],
    }


def default_analysis_dir(experiment_dir: Path, analysis_root: str | Path | None = None) -> Path:
    if analysis_root is not None:
        return Path(analysis_root) / experiment_dir.name
    if experiment_dir.parent.name == 'simulation':
        return experiment_dir.parent.parent / 'analysis' / experiment_dir.name
    return experiment_dir.parent / 'analysis' / experiment_dir.name


def run_analysis_dir(analysis_dir: Path, run_dir: Path) -> Path:
    return analysis_dir / run_dir.name


def empty_combo_result(
    mask_method: str,
    importance_min_score: float,
    importance_mask_mode: str = 'score_ratio',
    importance_score_ratio_threshold: float | None = None,
    importance_output_eps: float = 1e-8,
    max_elements_per_block: int | None = None,
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
    rule_score_average_fraction: float | None = None,
) -> dict:
    return {
        'mask_method': mask_method,
        'importance_min_score': importance_min_score,
        'importance_mask_mode': importance_mask_mode,
        'importance_score_ratio_threshold': importance_score_ratio_threshold,
        'importance_output_eps': importance_output_eps,
        'max_elements_per_block': max_elements_per_block,
        'token_extraction_method': token_extraction_method,
        'token_extraction_fraction': token_extraction_fraction,
        'baseline_accuracy': np.zeros(0, dtype=float),
        'combo_labels': [],
        'combo_sizes': np.zeros(0, dtype=int),
        'layers': [],
        'blocks': [],
        'network_retained_accuracy': np.zeros((0, 0), dtype=float),
        'network_keep_counts': np.zeros((0, 0), dtype=int),
        'network_exclusive_counts': np.zeros((0, 0), dtype=int),
        'resource_usage': {
            'combo_labels': [],
            'layer_ids': np.zeros(0, dtype=int),
            'layer_widths': np.zeros(0, dtype=int),
            'total_neurons': 0,
            'total_weights': 0,
            'weight_count_mode': 'empty',
            'layer_neuron_counts': np.zeros((0, 0), dtype=int),
            'layer_neuron_ratios': np.zeros((0, 0), dtype=float),
            'retained_neuron_counts': np.zeros(0, dtype=int),
            'retained_neuron_ratios': np.zeros(0, dtype=float),
            'retained_weight_counts': np.zeros(0, dtype=int),
            'retained_weight_ratios': np.zeros(0, dtype=float),
        },
    }

def analyze_run(
    run_dir: str | Path,
    max_weight_elements_per_block: int | None = None,
    mask_method: str = 'ablation',
    importance_min_score: float = 1e-9,
    importance_mask_mode: str = 'score_ratio',
    importance_mask_mode_neuron: str | None = None,
    importance_mask_mode_weight: str | None = None,
    importance_neuron_threshold: float = 1e-4,
    importance_score_ratio_threshold_weight: float = 1e-6,
    importance_output_eps: float = 1e-8,
    perturbation_batch_size: int = 64,
    analysis_device: str = 'auto',
    output_dir: str | Path | None = None,
    subnet_target: str = 'both',
    taylor_probes: int = 4,
    correlation_sample_size: int = 5000,
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
    rule_score_average_fraction: float | None = None,
) -> dict:
    mask_method = validate_mask_method(mask_method)
    importance_mask_mode = validate_importance_mask_mode(importance_mask_mode)
    token_extraction_method, token_extraction_fraction = resolve_token_extraction(
        token_extraction_method,
        token_extraction_fraction,
        rule_score_average_fraction=rule_score_average_fraction,
    )
    importance_mask_mode_neuron = validate_importance_mask_mode(importance_mask_mode_neuron or importance_mask_mode)
    importance_mask_mode_weight = validate_importance_mask_mode(importance_mask_mode_weight or importance_mask_mode)
    subnet_target = validate_subnet_target(subnet_target)
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_run_npz(run_dir)
    state = loaded['state']
    x_test = loaded['x_test']
    y_test = loaded['y_test']
    pattern_size = loaded['pattern_size']
    rule_indices = loaded['rule_indices']

    pred, activations = forward_from_state(state, x_test)
    acc = rule_accuracy(pred, y_test, pattern_size, rule_indices)
    cpa = cpa_analysis(activations)
    entropy = pca_entropy(activations)
    corr_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(run_dir.name)) % (2 ** 32)
    neuron_corr = neuron_correlation_distribution(activations, sample_size=correlation_sample_size, seed=corr_seed)

    do_neuron = subnet_target in {'neuron', 'both'}
    do_weight = subnet_target in {'weight', 'both'}
    taylor_scores = None
    if mask_method == 'taylor' and (do_neuron or do_weight):
        taylor_scores = taylor_importance_scores(
            state,
            x_test,
            pattern_size,
            rule_indices,
            batch_size=perturbation_batch_size,
            device=analysis_device,
            probes=taylor_probes,
        )

    if do_neuron:
        combo_subnet = combo_subnetwork_analysis(
            state,
            x_test,
            y_test,
            pattern_size,
            rule_indices,
            mask_method=mask_method,
            importance_min_score=importance_min_score,
            importance_mask_mode=importance_mask_mode_neuron,
            importance_score_ratio_threshold=importance_neuron_threshold,
            importance_output_eps=importance_output_eps,
            perturbation_batch_size=perturbation_batch_size,
            analysis_device=analysis_device,
            taylor_probes=taylor_probes,
            taylor_scores=taylor_scores,
            token_extraction_method=token_extraction_method,
            token_extraction_fraction=token_extraction_fraction,
        )
        modularity = modularity_from_combo_result(combo_subnet)
        combo_retained = combo_retained_stack(combo_subnet)
        combo_keep_counts = combo_keep_count_stack(combo_subnet, key='keep_counts')
        combo_exclusive_counts = combo_keep_count_stack(combo_subnet, key='exclusive_counts')
        combo_network_retained = combo_subnet['network_retained_accuracy']
        combo_network_keep_counts = combo_subnet['network_keep_counts']
        combo_network_exclusive_counts = combo_subnet['network_exclusive_counts']
    else:
        combo_subnet = empty_combo_result(
            mask_method,
            importance_min_score,
            importance_mask_mode=importance_mask_mode_neuron,
            importance_score_ratio_threshold=importance_neuron_threshold,
            importance_output_eps=importance_output_eps,
            token_extraction_method=token_extraction_method,
            token_extraction_fraction=token_extraction_fraction,
        )
        modularity = np.zeros(0, dtype=float)
        combo_retained = np.zeros((0, 0, 0), dtype=float)
        combo_keep_counts = np.zeros((0, 0), dtype=int)
        combo_exclusive_counts = np.zeros((0, 0), dtype=int)
        combo_network_retained = np.zeros((0, 0), dtype=float)
        combo_network_keep_counts = np.zeros((0, 0), dtype=int)
        combo_network_exclusive_counts = np.zeros((0, 0), dtype=int)

    if do_weight:
        weight_combo = weight_combo_subnetwork_analysis(
            state,
            x_test,
            y_test,
            pattern_size,
            rule_indices,
            max_elements_per_block=max_weight_elements_per_block,
            mask_method=mask_method,
            importance_min_score=importance_min_score,
            importance_mask_mode=importance_mask_mode_weight,
            importance_score_ratio_threshold=importance_score_ratio_threshold_weight,
            importance_output_eps=importance_output_eps,
            perturbation_batch_size=perturbation_batch_size,
            analysis_device=analysis_device,
            taylor_probes=taylor_probes,
            taylor_scores=taylor_scores,
            token_extraction_method=token_extraction_method,
            token_extraction_fraction=token_extraction_fraction,
        )
        weight_combo_retained = weight_combo['network_retained_accuracy']
        weight_combo_keep_counts = weight_combo['network_keep_counts']
        weight_combo_exclusive_counts = weight_combo['network_exclusive_counts']
    else:
        weight_combo = empty_combo_result(
            mask_method,
            importance_min_score,
            importance_mask_mode=importance_mask_mode_weight,
            importance_score_ratio_threshold=importance_score_ratio_threshold_weight,
            importance_output_eps=importance_output_eps,
            max_elements_per_block=max_weight_elements_per_block,
            token_extraction_method=token_extraction_method,
            token_extraction_fraction=token_extraction_fraction,
        )
        weight_combo_retained = np.zeros((0, 0), dtype=float)
        weight_combo_keep_counts = np.zeros((0, 0), dtype=int)
        weight_combo_exclusive_counts = np.zeros((0, 0), dtype=int)

    neuron_resource = combo_subnet.get('resource_usage', {})
    result = {
        'accuracy': acc,
        'subnet_target': subnet_target,
        'cpa': cpa,
        'pca_entropy': entropy,
        'neuron_correlation': neuron_corr,
        'modularity': modularity,
        'combo_mask_method': mask_method,
        'importance_min_score': importance_min_score,
        'importance_mask_mode': importance_mask_mode,
        'importance_mask_mode_neuron': importance_mask_mode_neuron,
        'importance_mask_mode_weight': importance_mask_mode_weight,
        'importance_neuron_threshold': importance_neuron_threshold,
        'importance_score_ratio_threshold_weight': importance_score_ratio_threshold_weight,
        'importance_output_eps': importance_output_eps,
        'taylor_probes': taylor_probes,
        'combo_subnetwork': combo_subnet,
        'neuron_subnet_resource_usage': neuron_resource,
        'combo_retained_accuracy': combo_retained,
        'combo_keep_counts': combo_keep_counts,
        'combo_exclusive_counts': combo_exclusive_counts,
        'combo_network_retained_accuracy': combo_network_retained,
        'combo_network_keep_counts': combo_network_keep_counts,
        'combo_network_exclusive_counts': combo_network_exclusive_counts,
        'weight_combo_subnetwork': weight_combo,
        'weight_combo_retained_accuracy': weight_combo_retained,
        'weight_combo_keep_counts': weight_combo_keep_counts,
        'weight_combo_exclusive_counts': weight_combo_exclusive_counts,
    }
    save_json(output_dir / 'analysis.json', result)
    savemat(output_dir / 'analysis.mat', {
        'subnet_target': np.asarray(subnet_target),
        'combo_mask_method': np.asarray(mask_method),
        'importance_min_score': np.asarray(importance_min_score),
        'importance_mask_mode': np.asarray(importance_mask_mode),
        'importance_mask_mode_neuron': np.asarray(importance_mask_mode_neuron),
        'importance_mask_mode_weight': np.asarray(importance_mask_mode_weight),
        'importance_neuron_threshold': np.asarray(importance_neuron_threshold),
        'importance_score_ratio_threshold_weight': np.asarray(importance_score_ratio_threshold_weight),
        'importance_output_eps': np.asarray(importance_output_eps),
        'taylor_probes': np.asarray(taylor_probes),
        'pca_entropy': entropy,
        'neuron_correlation_mean_abs': neuron_correlation_stat_matrix(neuron_corr, 'mean_abs_corr'),
        'neuron_correlation_p95_abs': neuron_correlation_stat_matrix(neuron_corr, 'p95_abs_corr'),
        'neuron_correlation_max_abs': neuron_correlation_stat_matrix(neuron_corr, 'max_abs_corr'),
        'neuron_correlation_pair_count': neuron_correlation_stat_matrix(neuron_corr, 'pair_count'),
        'neuron_correlation_abs_hist': neuron_correlation_hist_stack(neuron_corr, 'abs_hist'),
        'neuron_correlation_abs_hist_bins': np.asarray(neuron_corr['layers'][0]['abs_hist_bins']) if neuron_corr.get('layers') else np.zeros(0, dtype=float),
        'modularity': modularity,
        'cpa_linear_cka': cpa['linear_cka'],
        'cpa_rbf_cka': cpa['rbf_cka'],
        'combo_retained_accuracy': combo_retained,
        'combo_keep_counts': combo_keep_counts,
        'combo_exclusive_counts': combo_exclusive_counts,
        'combo_network_retained_accuracy': combo_network_retained,
        'combo_network_keep_counts': combo_network_keep_counts,
        'combo_network_exclusive_counts': combo_network_exclusive_counts,
        'neuron_subnet_retained_neuron_counts': np.asarray(neuron_resource.get('retained_neuron_counts', np.zeros(0, dtype=int))),
        'neuron_subnet_retained_neuron_ratios': np.asarray(neuron_resource.get('retained_neuron_ratios', np.zeros(0, dtype=float))),
        'neuron_subnet_retained_weight_counts': np.asarray(neuron_resource.get('retained_weight_counts', np.zeros(0, dtype=int))),
        'neuron_subnet_retained_weight_ratios': np.asarray(neuron_resource.get('retained_weight_ratios', np.zeros(0, dtype=float))),
        'neuron_subnet_layer_neuron_ratios': np.asarray(neuron_resource.get('layer_neuron_ratios', np.zeros((0, 0), dtype=float))),
        'weight_combo_retained_accuracy': weight_combo_retained,
        'weight_combo_keep_counts': weight_combo_keep_counts,
        'weight_combo_exclusive_counts': weight_combo_exclusive_counts,
    })
    return result

def plot_accuracy(experiment_dir: Path, histories: list[dict], fig_dir: Path) -> None:
    curves = np.asarray([h['test_mean_acc'] for h in histories], dtype=float)
    mean = curves.mean(axis=0)
    std = curves.std(axis=0)
    epoch_values = [h.get('epoch') for h in histories]
    if epoch_values and all(v is not None and len(v) == len(mean) for v in epoch_values):
        x = np.asarray(epoch_values[0], dtype=float)
    else:
        x = np.arange(1, len(mean) + 1)
    plt.figure(figsize=(NATURE_SINGLE_COLUMN_IN, 2.35))
    plt.plot(x, mean, color=_nature_color(0), label='test mean accuracy')
    plt.fill_between(x, mean - std, mean + std, color=_nature_color(0), alpha=0.2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1.02)
    _style_publication_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(fig_dir / 'accuracy_curve.png', dpi=PUBLICATION_DPI)
    plt.close()


def plot_matrix(mat: np.ndarray, path: Path, title: str) -> None:
    plt.figure(figsize=(3.25, 2.9))
    plt.imshow(mat, vmin=0, vmax=1, cmap='viridis')
    cbar = plt.colorbar(label='CKA', fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel('layer')
    plt.ylabel('layer')
    _style_publication_axes(plt.gca())
    cbar.ax.tick_params(direction='out', length=3.0, width=0.8, labelsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=PUBLICATION_DPI)
    plt.close()


def plot_vector(values: np.ndarray, path: Path, ylabel: str, title: str) -> None:
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    x = np.arange(len(mean))
    plt.figure(figsize=(NATURE_SINGLE_COLUMN_IN, 2.4))
    plt.bar(x, mean, yerr=std, color=_nature_color(2), alpha=0.9, capsize=2)
    plt.xlabel('layer')
    plt.ylabel(ylabel)
    plt.title(title)
    _style_publication_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(path, dpi=PUBLICATION_DPI)
    plt.close()



def _violin_values(values: np.ndarray, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    center = float(np.nanmedian(values))
    sample_span = float(np.nanmax(values) - np.nanmin(values)) if values.size > 1 else 0.0
    if sample_span <= epsilon:
        anchors = np.asarray([center - epsilon, center + epsilon], dtype=float)
        return np.concatenate([values, anchors])
    return values


def _retained_ratio_ylim(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.02
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))
    lower = max(0.0, float(np.floor(ymin * 10.0) / 10.0))
    upper = max(1.02, float(np.ceil((ymax + 0.02) * 10.0) / 10.0))
    if upper <= lower:
        upper = lower + 0.1
    return lower, upper


def _demo_combo_count_until_rule_set(combo_labels: list[str], cutoff_label: str = '25') -> int:
    for idx, label in enumerate(combo_labels):
        if _one_based_compact_combo_label(label) == cutoff_label:
            return idx + 1
    return len(combo_labels)


def _one_based_compact_combo_label(label: str) -> str:
    try:
        return ''.join(str(int(part) + 1) for part in str(label).split('-'))
    except ValueError:
        return str(label)


def _one_based_hyphen_combo_label(label: str) -> str:
    try:
        return '-'.join(str(int(part) + 1) for part in str(label).split('-'))
    except ValueError:
        return str(label)


def _demo_rule_label(label: str) -> str:
    value = _one_based_hyphen_combo_label(label)
    return value if value.lower().startswith('rule ') else 'rule {}'.format(value)


def plot_combo_retained_violin(
    retained: np.ndarray,
    fig_dir: Path,
    rule_labels: list[str],
    combo_labels: list[str],
    filename: str,
    title: str,
    publication_style: bool = False,
) -> None:
    retained = np.asarray(retained, dtype=float)
    if retained.size == 0 or retained.ndim != 3:
        return
    num_runs, num_rules, num_combos = retained.shape
    if num_runs == 0 or num_rules == 0 or num_combos == 0:
        return

    finite = retained[np.isfinite(retained)]
    if finite.size == 0:
        return
    value_span = float(np.nanmax(finite) - np.nanmin(finite))
    epsilon = max(0.006, value_span * 0.012)

    if publication_style:
        inner_gap = 0.22
        group_gap = max(1.65, num_rules * inner_gap + 0.75)
        violin_width = 0.18
    else:
        inner_gap = 0.30
        group_gap = max(2.05, num_rules * inner_gap + 1.0)
        violin_width = 0.24
    data = []
    positions = []
    rule_ids = []
    for combo_idx in range(num_combos):
        base_x = combo_idx * group_gap
        for rule_idx in range(num_rules):
            values = _violin_values(retained[:, rule_idx, combo_idx], epsilon)
            if values.size:
                data.append(values)
                positions.append(base_x + rule_idx * inner_gap)
                rule_ids.append(rule_idx)
    if not data:
        return
    xticks = [combo_idx * group_gap + inner_gap * (num_rules - 1) / 2.0 for combo_idx in range(num_combos)]

    if publication_style:
        fig_width = NATURE_DOUBLE_COLUMN_IN
        fig_height = 3.6
    else:
        fig_width = max(NATURE_DOUBLE_COLUMN_IN, min(NATURE_SUPPLEMENT_MAX_IN, 0.38 * num_combos + 3.6))
        fig_height = 4.2 if num_combos <= 10 else 4.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    parts = ax.violinplot(data, positions=positions, widths=violin_width, showmeans=True, showmedians=True, showextrema=False)
    colors = [_nature_color(i) for i in range(num_rules)]
    for body, rule_idx in zip(parts['bodies'], rule_ids):
        color = colors[rule_idx]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.65)
        body.set_linewidth(1.0)
    for key, color, width in [('cmeans', '#222222', 1.1), ('cmedians', '#b23a48', 1.45)]:
        if key in parts:
            parts[key].set_color(color)
            parts[key].set_linewidth(width)

    label_rotation = 0 if publication_style else (35 if num_combos <= 8 else 45)
    label_ha = 'center' if publication_style else 'right'
    ax.set_xticks(xticks)
    ax.set_xticklabels(combo_labels, rotation=label_rotation, ha=label_ha)
    axis_label_size = 16 if publication_style else None
    ax.set_xlabel('rule set', fontsize=axis_label_size, labelpad=4 if publication_style else None)
    ax.set_ylabel('relative accuracy', fontsize=axis_label_size, labelpad=4 if publication_style else None)
    if title and not publication_style:
        ax.set_title(title, pad=12)
    lower, upper = _retained_ratio_ylim(finite)
    ax.set_ylim(lower, upper)
    tick_step = 0.1 if upper - lower <= 1.2 else 0.2
    ax.set_yticks(np.round(np.arange(lower, upper + tick_step * 0.5, tick_step), 2))
    if lower <= 1.0 <= upper:
        ax.axhline(1.0, color='#666666', linewidth=0.9, linestyle='--', alpha=0.45, zorder=0)
    _style_publication_axes(ax)
    ax.set_axisbelow(True)
    ax.margins(x=0.03)
    if publication_style:
        ax.tick_params(axis='both', labelsize=14, length=3.5, width=0.9, pad=2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    handles = [plt.Line2D([0], [0], color=colors[i], linewidth=5, alpha=0.75, label=rule_labels[i]) for i in range(num_rules)]
    if publication_style:
        ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.03),
            ncol=min(num_rules, 5),
            frameon=False,
            fontsize=13,
            handlelength=1.1,
            handletextpad=0.35,
            columnspacing=0.8,
            borderaxespad=0.0,
        )
        fig.tight_layout(pad=0.35)
        fig.savefig(fig_dir / filename, dpi=PUBLICATION_DPI, bbox_inches='tight', pad_inches=0.02)
    else:
        ax.legend(handles=handles, title='evaluated rule', loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, borderaxespad=0.0)
        fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
        fig.savefig(fig_dir / filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_combo_retained_violin_demo(retained: np.ndarray, fig_dir: Path, rule_labels: list[str], combo_labels: list[str], filename: str, title: str) -> None:
    retained = np.asarray(retained, dtype=float)
    if retained.size == 0 or retained.ndim != 3:
        return
    available_count = min(retained.shape[2], len(combo_labels))
    keep_count = _demo_combo_count_until_rule_set([str(label) for label in combo_labels[:available_count]], cutoff_label='25')
    if keep_count <= 0:
        return
    demo_labels = [_one_based_hyphen_combo_label(label) for label in combo_labels[:keep_count]]
    demo_rule_labels = [_demo_rule_label(label) for label in rule_labels]
    plot_combo_retained_violin(
        retained[:, :, :keep_count],
        fig_dir,
        demo_rule_labels,
        demo_labels,
        filename,
        title,
        publication_style=True,
    )


def plot_combo_retained_boxplot(retained: np.ndarray, fig_dir: Path, rule_labels: list[str], combo_labels: list[str], filename: str, title: str) -> None:
    plot_combo_retained_violin(retained, fig_dir, rule_labels, combo_labels, filename, title)

def plot_combo_retained_layers(retained: np.ndarray, fig_dir: Path, rule_labels: list[str], combo_labels: list[str], prefix: str, title_prefix: str) -> None:
    retained = np.asarray(retained, dtype=float)
    if retained.size == 0 or retained.ndim != 4:
        return
    for layer_idx in range(retained.shape[1]):
        plot_combo_retained_violin(
            retained[:, layer_idx, :, :],
            fig_dir,
            rule_labels,
            combo_labels,
            '{}_layer_{}.png'.format(prefix, layer_idx),
            '{} - layer {}'.format(title_prefix, layer_idx),
        )

def _score_matrix_name(item: dict) -> str | None:
    value_name = item.get('mask_value_name')
    if value_name and value_name in item:
        return str(value_name)
    for fallback in ('ablation_scores', 'taylor_scores'):
        if fallback in item:
            return fallback
    return None



def _collect_importance_score_samples_by_rule(
    results: list[dict],
    token_type: str,
    rule_count: int,
    max_points_per_rule: int | None = 200000,
    seed: int = 20260721,
) -> tuple[list[np.ndarray], list[int], list[int]]:
    if token_type == 'neuron':
        result_key = 'combo_subnetwork'
        collection_key = 'layers'
    elif token_type == 'weight':
        result_key = 'weight_combo_subnetwork'
        collection_key = 'blocks'
    else:
        raise ValueError("token_type must be 'neuron' or 'weight'")

    limit = None if max_points_per_rule is None else max(1, int(max_points_per_rule))
    rng = np.random.default_rng(seed)
    total_counts = [0 for _ in range(rule_count)]
    plot_counts = [0 for _ in range(rule_count)]
    if limit is None:
        chunks: list[list[np.ndarray]] = [[] for _ in range(rule_count)]
    else:
        samples = [np.zeros(0, dtype=float) for _ in range(rule_count)]
        priorities = [np.zeros(0, dtype=float) for _ in range(rule_count)]

    for result in results:
        collection = result.get(result_key, {}).get(collection_key, [])
        for item in collection:
            value_name = _score_matrix_name(item)
            if value_name is None:
                continue
            values = np.asarray(item.get(value_name, []), dtype=float)
            if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
                continue
            for rule_idx in range(min(rule_count, values.shape[0])):
                row = values[rule_idx].reshape(-1)
                row = row[np.isfinite(row)]
                if row.size == 0:
                    continue
                row = np.log(row + 1e-12)
                row = row[np.isfinite(row)]
                if row.size == 0:
                    continue
                total_counts[rule_idx] += int(row.size)
                if limit is None:
                    chunks[rule_idx].append(row.astype(float, copy=False))
                    continue
                row_priority = rng.random(row.size)
                combined_values = np.concatenate([samples[rule_idx], row]) if samples[rule_idx].size else row
                combined_priority = np.concatenate([priorities[rule_idx], row_priority]) if priorities[rule_idx].size else row_priority
                if combined_values.size > limit:
                    keep = np.argpartition(combined_priority, -limit)[-limit:]
                    samples[rule_idx] = combined_values[keep]
                    priorities[rule_idx] = combined_priority[keep]
                else:
                    samples[rule_idx] = combined_values
                    priorities[rule_idx] = combined_priority

    if limit is None:
        data = [np.concatenate(rule_chunks).astype(float, copy=False) if rule_chunks else np.zeros(0, dtype=float) for rule_chunks in chunks]
    else:
        data = samples
    for rule_idx, values in enumerate(data):
        plot_counts[rule_idx] = int(values.size)
    return data, total_counts, plot_counts


def plot_importance_score_distribution(
    results: list[dict],
    fig_dir: Path,
    rule_labels: list[str],
    token_type: str,
    suffix: str,
    max_points_per_rule: int = 200000,
) -> None:
    rule_count = len(rule_labels)
    if rule_count == 0:
        return
    sampled, total_counts, plot_counts = _collect_importance_score_samples_by_rule(
        results,
        token_type,
        rule_count,
        max_points_per_rule=max_points_per_rule,
    )
    data = []
    labels = []
    positions = []
    for rule_idx, values in enumerate(sampled):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if values.size == 1:
            values = np.repeat(values, 2)
        data.append(values)
        positions.append(rule_idx + 1)
        label = str(rule_labels[rule_idx])
        total_count = int(total_counts[rule_idx])
        plot_count = int(plot_counts[rule_idx])
        if plot_count == total_count:
            label = '{}\nn={:,}'.format(label, total_count)
        else:
            label = '{}\nn={:,}\nplot={:,}'.format(label, total_count, plot_count)
        labels.append(label)
    if not data:
        return

    method = str(results[0].get('combo_mask_method', '')) if results else ''
    method_title = ' ({})'.format(method) if method else ''
    fig_width = max(NATURE_SINGLE_COLUMN_IN, min(NATURE_DOUBLE_COLUMN_IN, 0.75 * max(rule_count, len(data)) + 2.4))
    fig, ax = plt.subplots(figsize=(fig_width, 3.8))
    parts = ax.violinplot(data, positions=positions, widths=0.82, showmeans=True, showmedians=True, showextrema=False)
    for idx, body in enumerate(parts['bodies']):
        color = _nature_color(idx)
        body.set_facecolor(color)
        body.set_edgecolor('#333333')
        body.set_alpha(0.65)
        body.set_linewidth(0.8)
    for key, color, width in [('cmeans', '#222222', 1.2), ('cmedians', '#b23a48', 1.6)]:
        if key in parts:
            parts[key].set_color(color)
            parts[key].set_linewidth(width)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel('rule r')
    ax.set_ylabel('$\\log(S_{r,q} + 10^{-12})$')
    ax.set_title('{} importance score distribution{}'.format(token_type.capitalize(), method_title), pad=12)
    _style_publication_axes(ax)
    ax.margins(x=0.08)
    mean_handle = plt.Line2D([0], [0], color='#222222', linewidth=1.2, label='mean')
    median_handle = plt.Line2D([0], [0], color='#b23a48', linewidth=1.6, label='median')
    ax.legend(handles=[mean_handle, median_handle], frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(fig_dir / '{}_importance_score_distribution{}.png'.format(token_type, suffix), dpi=PUBLICATION_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_neuron_correlation_distribution(results: list[dict], fig_dir: Path) -> None:
    layer_count = max((len(r.get('neuron_correlation', {}).get('layers', [])) for r in results), default=0)
    if layer_count == 0:
        return
    data = []
    labels = []
    pair_counts = []
    for layer_idx in range(layer_count):
        values = []
        layer_pair_counts = []
        for result in results:
            layers = result.get('neuron_correlation', {}).get('layers', [])
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                values.extend(np.asarray(layer.get('sampled_abs_correlations', []), dtype=float).tolist())
                layer_pair_counts.append(float(layer.get('pair_count', np.nan)))
        values = [v for v in values if np.isfinite(v)]
        if values:
            data.append(values)
            mean_pairs = float(np.nanmean(layer_pair_counts)) if layer_pair_counts else float('nan')
            labels.append('L{}\npairs={:,.0f}'.format(layer_idx, mean_pairs) if np.isfinite(mean_pairs) else 'L{}'.format(layer_idx))
            pair_counts.append(mean_pairs)
    if not data:
        return
    positions = np.arange(1, len(data) + 1)
    fig, ax = plt.subplots(figsize=(max(NATURE_SINGLE_COLUMN_IN, min(NATURE_DOUBLE_COLUMN_IN, 0.75 * len(data) + 2.8)), 3.6))
    parts = ax.violinplot(data, positions=positions, widths=0.82, showmeans=True, showmedians=True, showextrema=False)
    for idx, body in enumerate(parts['bodies']):
        color = _nature_color(idx)
        body.set_facecolor(color)
        body.set_edgecolor('#333333')
        body.set_alpha(0.65)
        body.set_linewidth(0.8)
    for key, color, width in [('cmeans', '#222222', 1.2), ('cmedians', '#b23a48', 1.6)]:
        if key in parts:
            parts[key].set_color(color)
            parts[key].set_linewidth(width)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel('layer')
    ax.set_ylabel('abs pairwise neuron correlation')
    ax.set_title('Within-layer neuron correlation violin', pad=12)
    ax.set_ylim(0, 1.02)
    _style_publication_axes(ax)
    ax.margins(x=0.08)
    mean_handle = plt.Line2D([0], [0], color='#222222', linewidth=1.2, label='mean')
    median_handle = plt.Line2D([0], [0], color='#b23a48', linewidth=1.6, label='median')
    ax.legend(handles=[mean_handle, median_handle], frameon=False, loc='upper right')
    fig.tight_layout()
    fig.savefig(fig_dir / 'neuron_correlation_abs_distribution.png', dpi=PUBLICATION_DPI, bbox_inches='tight')
    plt.close(fig)


def plot_neuron_correlation_pair_histogram(results: list[dict], fig_dir: Path) -> None:
    layer_count = max((len(r.get('neuron_correlation', {}).get('layers', [])) for r in results), default=0)
    if layer_count == 0:
        return
    plotted = False
    plt.figure(figsize=(NATURE_DOUBLE_COLUMN_IN, 3.4))
    for layer_idx in range(layer_count):
        hist_rows = []
        pair_counts = []
        bins = None
        for result in results:
            layers = result.get('neuron_correlation', {}).get('layers', [])
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                hist = np.asarray(layer.get('abs_hist', []), dtype=float)
                layer_bins = np.asarray(layer.get('abs_hist_bins', []), dtype=float)
                if hist.size and layer_bins.size == hist.size + 1:
                    hist_rows.append(hist)
                    bins = layer_bins
                    pair_counts.append(float(layer.get('pair_count', np.nansum(hist))))
        if not hist_rows or bins is None:
            continue
        arr = np.asarray(hist_rows, dtype=float)
        centers = 0.5 * (bins[:-1] + bins[1:])
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(np.sum(np.isfinite(arr), axis=0), 1))
        color = _nature_color(layer_idx)
        mean_pairs = float(np.nanmean(pair_counts)) if pair_counts else float(np.nansum(mean))
        plt.step(centers, mean, where='mid', linewidth=1.2, color=color, label='L{} pairs/run={:,.0f}'.format(layer_idx, mean_pairs))
        plt.fill_between(centers, np.maximum(mean - sem, 0.0), mean + sem, step='mid', color=color, alpha=0.16)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel('abs pairwise neuron correlation bin')
    plt.ylabel('neuron pair count per run')
    plt.title('Neuron pair counts by correlation bin')
    plt.xlim(0, 1.0)
    _style_publication_axes(plt.gca())
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / 'neuron_correlation_abs_pair_counts.png', dpi=PUBLICATION_DPI)
    plt.close()


def plot_neuron_correlation_summary(results: list[dict], fig_dir: Path) -> None:
    layer_count = max((len(r.get('neuron_correlation', {}).get('layers', [])) for r in results), default=0)
    if layer_count == 0:
        return
    metric_defs = [
        ('mean_abs_corr', 'mean |corr|', _nature_color(2)),
        ('p95_abs_corr', 'p95 |corr|', _nature_color(5)),
        ('max_abs_corr', 'max |corr|', _nature_color(3)),
    ]
    x = np.arange(layer_count)
    plt.figure(figsize=(NATURE_SINGLE_COLUMN_IN, 2.45))
    for key, label, color in metric_defs:
        rows = []
        for result in results:
            layers = result.get('neuron_correlation', {}).get('layers', [])
            row = [float(layers[i].get(key, np.nan)) if i < len(layers) else np.nan for i in range(layer_count)]
            rows.append(row)
        arr = np.asarray(rows, dtype=float)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(np.sum(np.isfinite(arr), axis=0), 1))
        plt.errorbar(x, mean, yerr=sem, marker='o', linewidth=1.2, capsize=2, color=color, label=label)
    plt.xlabel('layer')
    plt.ylabel('abs pairwise neuron correlation')
    plt.title('Within-layer neuron correlation summary')
    plt.ylim(0, 1.02)
    _style_publication_axes(plt.gca())
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / 'neuron_correlation_summary.png', dpi=PUBLICATION_DPI)
    plt.close()


def plot_resource_ratio_violin(
    ratios: np.ndarray,
    fig_dir: Path,
    combo_labels: list[str],
    filename: str,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    ratios = np.asarray(ratios, dtype=float)
    if ratios.size == 0 or ratios.ndim != 2:
        return
    finite = ratios[np.isfinite(ratios)]
    if finite.size == 0:
        return
    value_span = float(np.nanmax(finite) - np.nanmin(finite))
    epsilon = max(0.006, value_span * 0.012)
    labels = [_one_based_hyphen_combo_label(label) for label in combo_labels[: ratios.shape[1]]]
    data = []
    positions = []
    for combo_idx in range(ratios.shape[1]):
        values = _violin_values(ratios[:, combo_idx], epsilon)
        if values.size:
            data.append(values)
            positions.append(combo_idx)
    if not data:
        return

    fig_width = max(NATURE_DOUBLE_COLUMN_IN, min(NATURE_SUPPLEMENT_MAX_IN, 0.32 * len(labels) + 3.2))
    fig, ax = plt.subplots(figsize=(fig_width, 3.4))
    parts = ax.violinplot(data, positions=positions, widths=0.72, showmeans=True, showmedians=True, showextrema=False)
    for body in parts['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.65)
        body.set_linewidth(1.0)
    for key, line_color, width in [('cmeans', '#222222', 1.1), ('cmedians', '#b23a48', 1.45)]:
        if key in parts:
            parts[key].set_color(line_color)
            parts[key].set_linewidth(width)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('rule set')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    lower, upper = _retained_ratio_ylim(finite)
    ax.set_ylim(lower, upper)
    tick_step = 0.1 if upper - lower <= 1.2 else 0.2
    ax.set_yticks(np.round(np.arange(lower, upper + tick_step * 0.5, tick_step), 2))
    _style_publication_axes(ax, grid=True)
    fig.tight_layout()
    fig.savefig(fig_dir / filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
    plt.close(fig)


def _weight_subnet_total_weights(weight_combo: dict) -> float:
    blocks = weight_combo.get('blocks', [])
    total = 0
    for block in blocks:
        shape = block.get('shape', [])
        if shape:
            total += int(np.prod(np.asarray(shape, dtype=int)))
        elif 'candidate_flat_indices' in block:
            total += len(block.get('candidate_flat_indices', []))
    if total > 0:
        return float(total)
    keep_counts = np.asarray(weight_combo.get('network_keep_counts', np.zeros((0, 0))), dtype=float)
    return float(np.nanmax(np.nansum(keep_counts, axis=0))) if keep_counts.size else 0.0


def _weight_subnet_retained_weight_ratios(weight_combos: list[dict]) -> np.ndarray:
    rows = []
    for weight_combo in weight_combos:
        keep_counts = np.asarray(weight_combo.get('network_keep_counts', np.zeros((0, 0))), dtype=float)
        if keep_counts.size == 0:
            continue
        total = _weight_subnet_total_weights(weight_combo)
        if total <= 0:
            continue
        rows.append(np.nansum(keep_counts, axis=0) / total)
    return np.asarray(rows, dtype=float) if rows else np.zeros((0, 0), dtype=float)


def plot_neuron_subnet_resource_usage(resources: list[dict], fig_dir: Path, combo_labels: list[str], suffix: str) -> None:
    if not resources or not combo_labels:
        return
    neuron = np.asarray([r.get('retained_neuron_ratios', np.zeros(len(combo_labels), dtype=float)) for r in resources], dtype=float)
    if neuron.size == 0:
        return
    plot_resource_ratio_violin(
        neuron,
        fig_dir,
        combo_labels,
        'neuron_subnet_retained_neuron_ratio{}.png'.format(suffix),
        'Neuron-defined subnet retained neuron ratio',
        'retained neuron ratio',
        _nature_color(2),
    )
    demo_count = min(5, neuron.shape[1], len(combo_labels))
    if demo_count > 0:
        plot_resource_ratio_violin(
            neuron[:, :demo_count],
            fig_dir,
            combo_labels[:demo_count],
            'neuron_subnet_retained_neuron_ratio{}_demo.png'.format(suffix),
            'Neuron-defined subnet retained neuron ratio demo',
            'retained neuron ratio',
            _nature_color(2),
        )

    layer_values = [np.asarray(r.get('layer_neuron_ratios', np.zeros((0, len(combo_labels)), dtype=float)), dtype=float) for r in resources]
    layer_values = [v for v in layer_values if v.size]
    if layer_values:
        mean_layer = np.nanmean(np.stack(layer_values, axis=0), axis=0)
        plt.figure(figsize=(max(NATURE_DOUBLE_COLUMN_IN, min(NATURE_SUPPLEMENT_MAX_IN, 0.30 * len(combo_labels) + 3.2)), max(3.2, 0.38 * mean_layer.shape[0] + 1.8)))
        im = plt.imshow(mean_layer, vmin=0, vmax=1, cmap='viridis', aspect='auto')
        cbar = plt.colorbar(im, label='retained neuron ratio', fraction=0.046, pad=0.04)
        plt.xticks(np.arange(len(combo_labels)), [_one_based_hyphen_combo_label(label) for label in combo_labels], rotation=45, ha='right')
        plt.yticks(np.arange(mean_layer.shape[0]), np.arange(mean_layer.shape[0]))
        plt.xlabel('rule set')
        plt.ylabel('layer')
        plt.title('Neuron-defined subnet retained neuron ratio by layer')
        _style_publication_axes(plt.gca())
        cbar.ax.tick_params(direction='out', length=3.0, width=0.8, labelsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / 'neuron_subnet_layer_neuron_ratio{}.png'.format(suffix), dpi=PUBLICATION_DPI)
        plt.close()


def plot_weight_subnet_resource_usage(weight_combos: list[dict], fig_dir: Path, combo_labels: list[str], suffix: str) -> None:
    if not weight_combos or not combo_labels:
        return
    weight = _weight_subnet_retained_weight_ratios(weight_combos)
    if weight.size == 0:
        return
    plot_resource_ratio_violin(
        weight,
        fig_dir,
        combo_labels,
        'weight_subnet_retained_weight_ratio{}.png'.format(suffix),
        'Weight-defined subnet retained weight ratio',
        'retained weight ratio',
        _nature_color(5),
    )
    demo_count = min(5, weight.shape[1], len(combo_labels))
    if demo_count > 0:
        plot_resource_ratio_violin(
            weight[:, :demo_count],
            fig_dir,
            combo_labels[:demo_count],
            'weight_subnet_retained_weight_ratio{}_demo.png'.format(suffix),
            'Weight-defined subnet retained weight ratio demo',
            'retained weight ratio',
            _nature_color(5),
        )

def _normal_cached_path(value: str | Path) -> Path:
    return Path(str(value).replace('\\', '/'))


def _load_cached_analysis_summary(analysis_dir: Path) -> dict:
    path = analysis_dir / 'analysis_summary.json'
    return load_json(path) if path.is_file() else {}


def _load_cached_analysis_results(analysis_dir: Path) -> list[dict]:
    run_dirs = sorted(p for p in analysis_dir.glob('run_*') if (p / 'analysis.json').is_file())
    return [load_json(run_dir / 'analysis.json') for run_dir in run_dirs]


def _infer_cached_rule_labels(results: list[dict]) -> list[str]:
    if not results:
        return []
    first = results[0]
    for result_key, value_key in [
        ('combo_subnetwork', 'combo_network_retained_accuracy'),
        ('weight_combo_subnetwork', 'weight_combo_retained_accuracy'),
    ]:
        values = np.asarray(first.get(value_key, []), dtype=float)
        if values.ndim == 2 and values.shape[0] > 0:
            labels = first.get(result_key, {}).get('combo_labels', [])
            if len(labels) >= values.shape[0]:
                return [str(label) for label in labels[: values.shape[0]]]
            return [str(i) for i in range(values.shape[0])]
    return []


def _cached_method_parts(summary: dict, results: list[dict]) -> tuple[str, str]:
    method = str(summary.get('mask_method') or (results[0].get('combo_mask_method', '') if results else '')).strip()
    suffix = '' if method in {'', 'ablation'} else '_{}'.format(method)
    method_title = ' ({})'.format(method) if method else ''
    return suffix, method_title


def _candidate_cached_simulation_dirs(analysis_dir: Path, summary: dict) -> list[Path]:
    raw = summary.get('simulation_dir')
    if not raw:
        return []
    path = _normal_cached_path(raw)
    candidates = [path] if path.is_absolute() else []
    if not path.is_absolute():
        candidates.append(Path.cwd() / path)
        for parent in analysis_dir.parents:
            candidates.append(parent / path)
    unique = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _load_cached_histories(simulation_dir: Path | None) -> list[dict]:
    if simulation_dir is None:
        return []
    run_dirs = sorted(p for p in simulation_dir.glob('run_*') if (p / 'history.json').is_file())
    return [load_json(run_dir / 'history.json') for run_dir in run_dirs]


def _resolve_cached_simulation_dir(analysis_dir: Path, summary: dict, explicit: Path | None) -> Path | None:
    candidates = [explicit] if explicit is not None else []
    candidates.extend(_candidate_cached_simulation_dirs(analysis_dir, summary))
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate
    return None


def _cached_results_have(results: list[dict], key: str) -> bool:
    return bool(results) and all(key in result for result in results)


def plot_cached_analysis(
    analysis_dir: str | Path,
    simulation_dir: str | Path | None = None,
    subnet_target: str | None = None,
    plot_accuracy_curve: bool = True,
) -> dict:
    analysis_dir = Path(analysis_dir)
    fig_dir = analysis_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_cached_analysis_summary(analysis_dir)
    results = _load_cached_analysis_results(analysis_dir)
    if not results:
        raise FileNotFoundError('No cached run_*/analysis.json files found under {}'.format(analysis_dir))

    explicit_sim = _normal_cached_path(simulation_dir) if simulation_dir is not None else None
    resolved_sim = _resolve_cached_simulation_dir(analysis_dir, summary, explicit_sim)
    histories = _load_cached_histories(resolved_sim) if plot_accuracy_curve else []
    if histories:
        plot_accuracy(resolved_sim or analysis_dir, histories, fig_dir)

    target = subnet_target or summary.get('subnet_target') or results[0].get('subnet_target') or 'both'
    target = str(target).lower().strip()
    if target not in {'neuron', 'weight', 'both'}:
        raise ValueError("subnet_target must be 'neuron', 'weight', or 'both'")
    do_neuron = target in {'neuron', 'both'}
    do_weight = target in {'weight', 'both'}
    suffix, method_title = _cached_method_parts(summary, results)
    rule_labels = _infer_cached_rule_labels(results)

    if _cached_results_have(results, 'pca_entropy'):
        pca = np.asarray([r['pca_entropy'] for r in results], dtype=float)
        plot_vector(pca, fig_dir / 'pca_entropy.png', 'PCA entropy', 'PCA spectral entropy')

    if all('cpa' in r and 'linear_cka' in r['cpa'] for r in results):
        cka = np.asarray([r['cpa']['linear_cka'] for r in results], dtype=float)
        plot_matrix(np.nanmean(cka, axis=0), fig_dir / 'cpa_linear_cka.png', 'Mean linear CKA')

    plot_neuron_correlation_distribution(results, fig_dir)
    plot_neuron_correlation_pair_histogram(results, fig_dir)
    plot_neuron_correlation_summary(results, fig_dir)

    if do_neuron and _cached_results_have(results, 'combo_network_retained_accuracy'):
        if _cached_results_have(results, 'modularity'):
            mod = np.asarray([r['modularity'] for r in results], dtype=float)
            plot_vector(mod, fig_dir / 'modularity.png', 'modularity', 'Weighted modularity')
        combo_labels = results[0].get('combo_subnetwork', {}).get('combo_labels', [])
        neuron_resources = [r.get('neuron_subnet_resource_usage', {}) for r in results]
        plot_neuron_subnet_resource_usage(neuron_resources, fig_dir, combo_labels, suffix)
        if rule_labels:
            plot_importance_score_distribution(results, fig_dir, rule_labels, 'neuron', suffix)
            neuron_combo_network = np.asarray([r['combo_network_retained_accuracy'] for r in results], dtype=float)
            plot_combo_retained_violin(
                neuron_combo_network,
                fig_dir,
                rule_labels,
                combo_labels,
                'neuron_retained_combo_accuracy{}.png'.format(suffix),
                'Neuron combo retained accuracy{}'.format(method_title),
            )
            plot_combo_retained_violin_demo(
                neuron_combo_network,
                fig_dir,
                rule_labels,
                combo_labels,
                'neuron_retained_combo_accuracy{}_demo.png'.format(suffix),
                'Neuron combo retained accuracy demo{}'.format(method_title),
            )
        if _cached_results_have(results, 'combo_retained_accuracy') and rule_labels:
            neuron_combo_layers = np.asarray([r['combo_retained_accuracy'] for r in results], dtype=float)
            plot_combo_retained_layers(
                neuron_combo_layers,
                fig_dir,
                rule_labels,
                combo_labels,
                'neuron_retained_combo_accuracy{}'.format(suffix),
                'Neuron combo retained accuracy{}'.format(method_title),
            )

    if do_weight and _cached_results_have(results, 'weight_combo_retained_accuracy'):
        weight_combo_labels = results[0].get('weight_combo_subnetwork', {}).get('combo_labels', [])
        weight_combos = [r.get('weight_combo_subnetwork', {}) for r in results]
        plot_weight_subnet_resource_usage(weight_combos, fig_dir, weight_combo_labels, suffix)
        if rule_labels:
            plot_importance_score_distribution(results, fig_dir, rule_labels, 'weight', suffix)
            weight_combo_network = np.asarray([r['weight_combo_retained_accuracy'] for r in results], dtype=float)
            plot_combo_retained_violin(
                weight_combo_network,
                fig_dir,
                rule_labels,
                weight_combo_labels,
                'weight_retained_combo_accuracy{}.png'.format(suffix),
                'Weight combo retained accuracy{}'.format(method_title),
            )
            plot_combo_retained_violin_demo(
                weight_combo_network,
                fig_dir,
                rule_labels,
                weight_combo_labels,
                'weight_retained_combo_accuracy{}_demo.png'.format(suffix),
                'Weight combo retained accuracy demo{}'.format(method_title),
            )

    figure_names = sorted(p.name for p in fig_dir.glob('*.png'))
    visualization_summary = {
        'analysis_dir': str(analysis_dir),
        'num_runs': len(results),
        'subnet_target': target,
        'simulation_dir': str(resolved_sim) if resolved_sim is not None else None,
        'accuracy_curve_requested': bool(plot_accuracy_curve),
        'accuracy_curve_plotted': bool(histories),
        'figures': figure_names,
    }
    save_json(analysis_dir / 'visualization_summary.json', visualization_summary)
    return visualization_summary


def analyze_experiment(
    experiment_dir: str | Path,
    max_weight_elements_per_block: int | None = None,
    mask_method: str = 'ablation',
    importance_min_score: float = 1e-9,
    importance_mask_mode: str = 'score_ratio',
    importance_mask_mode_neuron: str | None = None,
    importance_mask_mode_weight: str | None = None,
    importance_neuron_threshold: float = 1e-4,
    importance_score_ratio_threshold_weight: float = 1e-6,
    importance_output_eps: float = 1e-8,
    perturbation_batch_size: int = 64,
    analysis_device: str = 'auto',
    analysis_root: str | Path | None = None,
    analysis_dir: str | Path | None = None,
    subnet_target: str = 'both',
    taylor_probes: int = 4,
    correlation_sample_size: int = 5000,
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
    rule_score_average_fraction: float | None = None,
) -> dict:
    mask_method = validate_mask_method(mask_method)
    importance_mask_mode = validate_importance_mask_mode(importance_mask_mode)
    token_extraction_method, token_extraction_fraction = resolve_token_extraction(
        token_extraction_method,
        token_extraction_fraction,
        rule_score_average_fraction=rule_score_average_fraction,
    )
    importance_mask_mode_neuron = validate_importance_mask_mode(importance_mask_mode_neuron or importance_mask_mode)
    importance_mask_mode_weight = validate_importance_mask_mode(importance_mask_mode_weight or importance_mask_mode)
    subnet_target = validate_subnet_target(subnet_target)
    experiment_dir = Path(experiment_dir)
    analysis_dir = Path(analysis_dir) if analysis_dir is not None else default_analysis_dir(experiment_dir, analysis_root=analysis_root)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = analysis_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    run_dirs = sorted([p for p in experiment_dir.glob('run_*') if (p / 'run_data.npz').is_file()])
    histories = []
    results = []
    for run_dir in run_dirs:
        histories.append(load_json(run_dir / 'history.json'))
        results.append(analyze_run(
            run_dir,
            max_weight_elements_per_block=max_weight_elements_per_block,
            mask_method=mask_method,
            importance_min_score=importance_min_score,
            importance_mask_mode=importance_mask_mode,
            importance_mask_mode_neuron=importance_mask_mode_neuron,
            importance_mask_mode_weight=importance_mask_mode_weight,
            importance_neuron_threshold=importance_neuron_threshold,
            importance_score_ratio_threshold_weight=importance_score_ratio_threshold_weight,
            importance_output_eps=importance_output_eps,
            perturbation_batch_size=perturbation_batch_size,
            analysis_device=analysis_device,
            output_dir=run_analysis_dir(analysis_dir, run_dir),
            subnet_target=subnet_target,
            taylor_probes=taylor_probes,
            correlation_sample_size=correlation_sample_size,
            token_extraction_method=token_extraction_method,
            token_extraction_fraction=token_extraction_fraction,
        ))
    if histories:
        plot_accuracy(experiment_dir, histories, fig_dir)
    if results:
        do_neuron = subnet_target in {'neuron', 'both'}
        do_weight = subnet_target in {'weight', 'both'}
        pca = np.asarray([r['pca_entropy'] for r in results], dtype=float)
        cka = np.asarray([r['cpa']['linear_cka'] for r in results], dtype=float)
        first_loaded = load_run_npz(run_dirs[0])
        rule_labels = first_loaded.get('rule_names') or [str(r) for r in first_loaded['rule_indices']]
        suffix = '' if mask_method == 'ablation' else '_{}'.format(mask_method)
        method_title = ' ({})'.format(mask_method)
        plot_vector(pca, fig_dir / 'pca_entropy.png', 'PCA entropy', 'PCA spectral entropy')
        plot_matrix(np.nanmean(cka, axis=0), fig_dir / 'cpa_linear_cka.png', 'Mean linear CKA')
        plot_neuron_correlation_distribution(results, fig_dir)
        plot_neuron_correlation_pair_histogram(results, fig_dir)
        plot_neuron_correlation_summary(results, fig_dir)
        if do_neuron:
            mod = np.asarray([r['modularity'] for r in results], dtype=float)
            neuron_combo_layers = np.asarray([r['combo_retained_accuracy'] for r in results], dtype=float)
            neuron_combo_network = np.asarray([r['combo_network_retained_accuracy'] for r in results], dtype=float)
            combo_labels = results[0]['combo_subnetwork']['combo_labels']
            neuron_resources = [r.get('neuron_subnet_resource_usage', {}) for r in results]
            plot_neuron_subnet_resource_usage(neuron_resources, fig_dir, combo_labels, suffix)
            plot_importance_score_distribution(results, fig_dir, rule_labels, 'neuron', suffix)
            plot_vector(mod, fig_dir / 'modularity.png', 'modularity', 'Weighted modularity')
            plot_combo_retained_violin(
                neuron_combo_network,
                fig_dir,
                rule_labels,
                combo_labels,
                'neuron_retained_combo_accuracy{}.png'.format(suffix),
                'Neuron combo retained accuracy{}'.format(method_title),
            )
            plot_combo_retained_violin_demo(
                neuron_combo_network,
                fig_dir,
                rule_labels,
                combo_labels,
                'neuron_retained_combo_accuracy{}_demo.png'.format(suffix),
                'Neuron combo retained accuracy demo{}'.format(method_title),
            )
            plot_combo_retained_layers(
                neuron_combo_layers,
                fig_dir,
                rule_labels,
                combo_labels,
                'neuron_retained_combo_accuracy{}'.format(suffix),
                'Neuron combo retained accuracy{}'.format(method_title),
            )
        if do_weight:
            weight_combo_network = np.asarray([r['weight_combo_retained_accuracy'] for r in results], dtype=float)
            weight_combo_labels = results[0]['weight_combo_subnetwork']['combo_labels']
            weight_combos = [r.get('weight_combo_subnetwork', {}) for r in results]
            plot_weight_subnet_resource_usage(weight_combos, fig_dir, weight_combo_labels, suffix)
            plot_importance_score_distribution(results, fig_dir, rule_labels, 'weight', suffix)
            plot_combo_retained_violin(
                weight_combo_network,
                fig_dir,
                rule_labels,
                weight_combo_labels,
                'weight_retained_combo_accuracy{}.png'.format(suffix),
                'Weight combo retained accuracy{}'.format(method_title),
            )
            plot_combo_retained_violin_demo(
                weight_combo_network,
                fig_dir,
                rule_labels,
                weight_combo_labels,
                'weight_retained_combo_accuracy{}_demo.png'.format(suffix),
                'Weight combo retained accuracy demo{}'.format(method_title),
            )
    summary = {
        'simulation_dir': str(experiment_dir),
        'analysis_dir': str(analysis_dir),
        'num_runs': len(run_dirs),
        'mask_method': mask_method,
        'subnet_target': subnet_target,
        'importance_min_score': importance_min_score,
        'importance_mask_mode': importance_mask_mode,
        'importance_mask_mode_neuron': importance_mask_mode_neuron,
        'importance_mask_mode_weight': importance_mask_mode_weight,
        'importance_neuron_threshold': importance_neuron_threshold,
        'importance_score_ratio_threshold_weight': importance_score_ratio_threshold_weight,
        'importance_output_eps': importance_output_eps,
        'token_extraction_method': token_extraction_method,
        'token_extraction_fraction': token_extraction_fraction,
        'max_weight_elements_per_block': max_weight_elements_per_block,
        'perturbation_batch_size': perturbation_batch_size,
        'analysis_device': analysis_device,
        'taylor_probes': taylor_probes,
        'correlation_sample_size': correlation_sample_size,
        'figures': sorted([p.name for p in fig_dir.glob('*.png')]),
    }
    save_json(analysis_dir / 'analysis_summary.json', summary)
    return summary

def parse_optional_int(value: str) -> int | None:
    if value.lower() in {'none', 'all', 'full'}:
        return None
    return int(value)



def weight_combo_protocol_matches(combo, *, mask_method='ablation',
                                  min_score=1e-9, output_eps=1e-8,
                                  extraction_method='score_quantile', fraction=0.75,
                                  max_elements_per_block=None, state=None):
    """Reject incompatible/incomplete saved masks; no defaults stand in for metadata."""
    required = {'mask_method': mask_method, 'importance_min_score': min_score,
                'importance_output_eps': output_eps,
                'token_extraction_method': extraction_method,
                'token_extraction_fraction': fraction,
                'max_elements_per_block': max_elements_per_block}
    if any(key not in combo or combo[key] != value for key, value in required.items()):
        return False
    blocks = combo.get('blocks', [])
    if not blocks or any(not str(b.get('key', '')).endswith('_weight') for b in blocks):
        return False
    if state is not None:
        expected = {b['key']: tuple(b['shape']) for b in weight_block_specs(state)}
        if {b['key']: tuple(b['shape']) for b in blocks} != expected:
            return False
    for block in blocks:
        if any(block.get(k) != v for k, v in required.items() if k != 'max_elements_per_block'):
            return False
        n = int(np.prod(block['shape']))
        candidates = np.asarray(block.get('candidate_flat_indices', []), dtype=int)
        if max_elements_per_block is None and not np.array_equal(candidates, np.arange(n)):
            return False
        mask = np.asarray(block.get('important_mask', []))
        if mask.ndim != 2 or mask.shape[1] != len(candidates):
            return False
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir')
    parser.add_argument('--max-weight-elements-per-block', type=parse_optional_int, default=None)
    parser.add_argument('--mask-method', choices=['ablation', 'taylor', 'legacy_hessian'], default='ablation')
    parser.add_argument('--subnet-target', choices=['neuron', 'weight', 'both'], default='both')
    parser.add_argument('--importance-min-score', type=float, default=1e-9)
    mask_choices = ['score_ratio', 'neuron_score_ratio', 'weight_score_ratio']
    parser.add_argument('--importance-mask-mode', choices=mask_choices, default='score_ratio')
    parser.add_argument('--importance-mask-mode-neuron', choices=mask_choices, default=None)
    parser.add_argument('--importance-mask-mode-weight', choices=mask_choices, default=None)
    parser.add_argument('--importance-neuron-threshold', type=float, default=1e-4)
    parser.add_argument('--importance-score-ratio-threshold-weight', type=float, default=1e-6)
    parser.add_argument('--importance-output-eps', type=float, default=1e-8)
    parser.add_argument('--perturbation-batch-size', type=int, default=64)
    parser.add_argument('--taylor-probes', type=int, default=4)
    parser.add_argument('--correlation-sample-size', type=int, default=5000)
    parser.add_argument('--token-extraction-method', choices=TOKEN_EXTRACTION_METHOD_CHOICES, default=DEFAULT_TOKEN_EXTRACTION_METHOD)
    parser.add_argument('--token-extraction-fraction', '--rule-score-average-fraction', dest='token_extraction_fraction', type=float, default=DEFAULT_TOKEN_EXTRACTION_FRACTION)
    parser.add_argument('--analysis-device', default='auto')
    parser.add_argument('--analysis-root', default=None)
    parser.add_argument('--analysis-dir', default=None)
    args = parser.parse_args()
    analyze_experiment(
        args.experiment_dir,
        max_weight_elements_per_block=args.max_weight_elements_per_block,
        mask_method=args.mask_method,
        importance_min_score=args.importance_min_score,
        importance_mask_mode=args.importance_mask_mode,
        importance_mask_mode_neuron=args.importance_mask_mode_neuron,
        importance_mask_mode_weight=args.importance_mask_mode_weight,
        importance_neuron_threshold=args.importance_neuron_threshold,
        importance_score_ratio_threshold_weight=args.importance_score_ratio_threshold_weight,
        importance_output_eps=args.importance_output_eps,
        perturbation_batch_size=args.perturbation_batch_size,
        taylor_probes=args.taylor_probes,
        correlation_sample_size=args.correlation_sample_size,
        token_extraction_method=args.token_extraction_method,
        token_extraction_fraction=args.token_extraction_fraction,
        analysis_device=args.analysis_device,
        analysis_root=args.analysis_root,
        analysis_dir=args.analysis_dir,
        subnet_target=args.subnet_target,
    )
