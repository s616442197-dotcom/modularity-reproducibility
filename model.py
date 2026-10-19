from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-12)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def lgamma_regularization_grad(
    weight: np.ndarray,
    l1: float = 0.0,
    gamma: float = 1.0,
    regularization_epsilon: float = 1e-8,
) -> np.ndarray:
    if not l1:
        return np.zeros_like(weight)
    abs_weight = np.abs(weight) + regularization_epsilon
    grad = l1 * gamma * np.power(abs_weight, gamma - 1.0) * np.sign(weight)
    return (grad / weight.size).astype(np.float32)


@dataclass
class ForwardCache:
    layer_inputs: list[np.ndarray]
    pre_activations: list[np.ndarray]
    hidden_activations: list[np.ndarray]


class ModularMLP:
    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        hidden_dim: int = 64,
        num_modules: int = 3,
        hidden_dims: list[int] | None = None,
        init_scale: float = 1.0,
        seed: int = 0,
    ):
        self.params = {'hidden_dim': hidden_dim, 'num_modules': num_modules, 'hidden_dims': hidden_dims, 'init_scale': init_scale}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers: list[dict[str, np.ndarray]] = []
        self.output_layer: dict[str, np.ndarray] = {}
        if input_dim is not None and output_dim is not None:
            self._init_weights(input_dim, output_dim, seed)

    def build(self, input_dim: int, output_dim: int, seed: int):
        return ModularMLP(input_dim=input_dim, output_dim=output_dim, seed=seed, **self.params)

    def _init_weights(self, input_dim: int, output_dim: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        dims = self.params['hidden_dims'] or [self.params['hidden_dim']] * self.params['num_modules']
        prev_dim = input_dim
        for dim in dims:
            scale = self.params['init_scale'] / np.sqrt(prev_dim)
            self.layers.append({'weight': rng.normal(0.0, scale, size=(dim, prev_dim)).astype(np.float32), 'bias': np.zeros(dim, dtype=np.float32)})
            prev_dim = dim
        scale = self.params['init_scale'] / np.sqrt(prev_dim)
        self.output_layer = {'weight': rng.normal(0.0, scale, size=(output_dim, prev_dim)).astype(np.float32), 'bias': np.zeros(output_dim, dtype=np.float32)}

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
        current = x
        layer_inputs = []
        pre_activations = []
        hidden_activations = []
        for layer in self.layers:
            layer_inputs.append(current)
            z = current @ layer['weight'].T + layer['bias']
            h = relu(z)
            pre_activations.append(z)
            hidden_activations.append(h)
            current = h
        out = current @ self.output_layer['weight'].T + self.output_layer['bias']
        return out, ForwardCache(layer_inputs, pre_activations, hidden_activations)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def activations(self, x: np.ndarray, include_output: bool = True) -> list[np.ndarray]:
        out, cache = self.forward(x)
        acts = list(cache.pre_activations)
        if include_output:
            acts.append(out)
        return acts

    def train_step(self, x: np.ndarray, grad_output: np.ndarray, lr: float, l1: float = 0.0, gamma: float = 1.0, regularization_epsilon: float = 1e-8) -> None:
        _, cache = self.forward(x)
        self.train_step_from_cache(x, grad_output, cache, lr=lr, l1=l1, gamma=gamma, regularization_epsilon=regularization_epsilon)

    def train_step_from_cache(self, x: np.ndarray, grad_output: np.ndarray, cache: ForwardCache, lr: float, l1: float = 0.0, gamma: float = 1.0, regularization_epsilon: float = 1e-8) -> None:
        last_hidden = cache.hidden_activations[-1] if cache.hidden_activations else x
        grad_w_out = grad_output.T @ last_hidden
        grad_b_out = grad_output.sum(axis=0)
        grad_hidden = grad_output @ self.output_layer['weight']
        layer_grads = []
        for layer_idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_idx]
            grad_z = grad_hidden * relu_grad(cache.pre_activations[layer_idx])
            layer_grads.append({'weight': grad_z.T @ cache.layer_inputs[layer_idx], 'bias': grad_z.sum(axis=0)})
            grad_hidden = grad_z @ layer['weight']
        out_lgamma = lgamma_regularization_grad(self.output_layer['weight'], l1=l1, gamma=gamma, regularization_epsilon=regularization_epsilon)
        self.output_layer['weight'] -= lr * (grad_w_out + out_lgamma)
        self.output_layer['bias'] -= lr * grad_b_out
        for layer, grad in zip(reversed(self.layers), layer_grads):
            layer_lgamma = lgamma_regularization_grad(layer['weight'], l1=l1, gamma=gamma, regularization_epsilon=regularization_epsilon)
            layer['weight'] -= lr * (grad['weight'] + layer_lgamma)
            layer['bias'] -= lr * grad['bias']

    def analysis_layer_sizes(self) -> list[int]:
        return [layer['weight'].shape[0] for layer in self.layers]

    def predict_with_silenced_neurons(self, x: np.ndarray, layer_idx: int, neuron_indices: list[int]) -> np.ndarray:
        current = x
        for idx, layer in enumerate(self.layers):
            z = current @ layer['weight'].T + layer['bias']
            if idx == layer_idx:
                z = z.copy()
                z[:, neuron_indices] = 0.0
            current = relu(z)
        return current @ self.output_layer['weight'].T + self.output_layer['bias']

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {}
        for idx, layer in enumerate(self.layers):
            state['modules_list_{}_linear1_weight'.format(idx)] = layer['weight']
            state['modules_list_{}_linear1_bias'.format(idx)] = layer['bias']
        state['output_block_linear1_weight'] = self.output_layer['weight']
        state['output_block_linear1_bias'] = self.output_layer['bias']
        return state


class TinyTransformer:
    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        model_dim: int = 32,
        num_layers: int = 2,
        ff_dim: int | None = None,
        init_scale: float = 0.6,
        seed: int = 0,
    ):
        self.params = {'model_dim': model_dim, 'num_layers': num_layers, 'ff_dim': ff_dim, 'init_scale': init_scale}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.blocks: list[dict[str, np.ndarray]] = []
        self.output_layer: dict[str, np.ndarray] = {}
        if input_dim is not None and output_dim is not None:
            self._init_weights(input_dim, output_dim, seed)

    def build(self, input_dim: int, output_dim: int, seed: int):
        return TinyTransformer(input_dim=input_dim, output_dim=output_dim, seed=seed, **self.params)

    def _init_weights(self, input_dim: int, output_dim: int, seed: int) -> None:
        if input_dim % 2 != 0:
            raise ValueError('TinyTransformer expects an even input_dim')
        rng = np.random.default_rng(seed)
        self.half_dim = input_dim // 2
        self.model_dim = self.params['model_dim']
        self.ff_dim = self.params['ff_dim'] or self.model_dim * 2
        init_scale = self.params['init_scale']
        self.input_proj = rng.normal(0.0, init_scale / np.sqrt(self.half_dim), size=(self.half_dim, self.model_dim)).astype(np.float32)
        self.token_bias = rng.normal(0.0, 0.02, size=(2, self.model_dim)).astype(np.float32)
        for _ in range(self.params['num_layers']):
            scale = init_scale / np.sqrt(self.model_dim)
            self.blocks.append({
                'wq': rng.normal(0.0, scale, size=(self.model_dim, self.model_dim)).astype(np.float32),
                'wk': rng.normal(0.0, scale, size=(self.model_dim, self.model_dim)).astype(np.float32),
                'wv': rng.normal(0.0, scale, size=(self.model_dim, self.model_dim)).astype(np.float32),
                'wo': rng.normal(0.0, scale, size=(self.model_dim, self.model_dim)).astype(np.float32),
                'ff1': rng.normal(0.0, scale, size=(self.model_dim, self.ff_dim)).astype(np.float32),
                'ff2': rng.normal(0.0, scale, size=(self.ff_dim, self.model_dim)).astype(np.float32),
                'b1': np.zeros(self.ff_dim, dtype=np.float32),
                'b2': np.zeros(self.model_dim, dtype=np.float32),
            })
        out_scale = init_scale / np.sqrt(self.model_dim)
        self.output_layer = {'weight': rng.normal(0.0, out_scale, size=(output_dim, self.model_dim)).astype(np.float32), 'bias': np.zeros(output_dim, dtype=np.float32)}

    def _tokens(self, x: np.ndarray) -> np.ndarray:
        left = x[:, : self.half_dim] @ self.input_proj
        right = x[:, self.half_dim :] @ self.input_proj
        return np.stack([left, right], axis=1) + self.token_bias[None, :, :]

    def _encode(self, x: np.ndarray, silence: tuple[int, list[int]] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
        tokens = self._tokens(x)
        pooled_layers = []
        for layer_idx, block in enumerate(self.blocks):
            q = tokens @ block['wq']
            k = tokens @ block['wk']
            v = tokens @ block['wv']
            attention = softmax(q @ np.swapaxes(k, 1, 2) / np.sqrt(self.model_dim), axis=-1)
            tokens = layer_norm(tokens + (attention @ v) @ block['wo'])
            tokens = layer_norm(tokens + relu(tokens @ block['ff1'] + block['b1']) @ block['ff2'] + block['b2'])
            if silence is not None and silence[0] == layer_idx:
                tokens = tokens.copy()
                tokens[:, :, silence[1]] = 0.0
            pooled_layers.append(tokens.mean(axis=1))
        return tokens.mean(axis=1), pooled_layers

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
        features, pooled_layers = self._encode(x)
        out = features @ self.output_layer['weight'].T + self.output_layer['bias']
        return out, ForwardCache([], pooled_layers, [features])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def activations(self, x: np.ndarray, include_output: bool = True) -> list[np.ndarray]:
        out, cache = self.forward(x)
        acts = list(cache.pre_activations)
        if include_output:
            acts.append(out)
        return acts

    def train_step(self, x: np.ndarray, grad_output: np.ndarray, lr: float, l1: float = 0.0, gamma: float = 1.0, regularization_epsilon: float = 1e-8) -> None:
        features, _ = self._encode(x)
        cache = ForwardCache([], [], [features])
        self.train_step_from_cache(x, grad_output, cache, lr=lr, l1=l1, gamma=gamma, regularization_epsilon=regularization_epsilon)

    def train_step_from_cache(self, x: np.ndarray, grad_output: np.ndarray, cache: ForwardCache, lr: float, l1: float = 0.0, gamma: float = 1.0, regularization_epsilon: float = 1e-8) -> None:
        features = cache.hidden_activations[-1] if cache.hidden_activations else self._encode(x)[0]
        grad_w = grad_output.T @ features
        grad_b = grad_output.sum(axis=0)
        out_lgamma = lgamma_regularization_grad(self.output_layer['weight'], l1=l1, gamma=gamma, regularization_epsilon=regularization_epsilon)
        self.output_layer['weight'] -= lr * (grad_w + out_lgamma)
        self.output_layer['bias'] -= lr * grad_b

    def analysis_layer_sizes(self) -> list[int]:
        return [self.model_dim for _ in self.blocks]

    def predict_with_silenced_neurons(self, x: np.ndarray, layer_idx: int, neuron_indices: list[int]) -> np.ndarray:
        features, _ = self._encode(x, silence=(layer_idx, neuron_indices))
        return features @ self.output_layer['weight'].T + self.output_layer['bias']

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {
            'transformer_input_proj': self.input_proj,
            'transformer_token_bias': self.token_bias,
            'output_block_linear1_weight': self.output_layer['weight'],
            'output_block_linear1_bias': self.output_layer['bias'],
        }
        for idx, block in enumerate(self.blocks):
            for key, value in block.items():
                state['transformer_layer_{}_{}'.format(idx, key)] = value
        return state


try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    TORCH_AVAILABLE = False


def torch_device_from_arg(device: str = 'auto'):
    if not TORCH_AVAILABLE:
        raise ImportError('PyTorch is required for Torch models')
    if device == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


class TorchModularMLP(nn.Module if TORCH_AVAILABLE else object):
    is_torch_model = True

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        hidden_dim: int = 64,
        num_modules: int = 3,
        hidden_dims: list[int] | None = None,
        init_scale: float = 1.0,
        seed: int = 0,
        device: str = 'auto',
    ):
        if not TORCH_AVAILABLE:
            raise ImportError('PyTorch is required for TorchModularMLP')
        super().__init__()
        self.params = {'hidden_dim': hidden_dim, 'num_modules': num_modules, 'hidden_dims': hidden_dims, 'init_scale': init_scale, 'device': device}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch_device_from_arg(device)
        self.layers = nn.ModuleList()
        self.output_layer = None
        if input_dim is not None and output_dim is not None:
            self._init_weights(input_dim, output_dim, seed)
            self.to(self.device)

    def build(self, input_dim: int, output_dim: int, seed: int):
        return TorchModularMLP(input_dim=input_dim, output_dim=output_dim, seed=seed, **self.params)

    def _init_weights(self, input_dim: int, output_dim: int, seed: int) -> None:
        torch.manual_seed(seed)
        dims = self.params['hidden_dims'] or [self.params['hidden_dim']] * self.params['num_modules']
        prev_dim = input_dim
        for dim in dims:
            layer = nn.Linear(prev_dim, dim)
            scale = self.params['init_scale'] / np.sqrt(prev_dim)
            with torch.no_grad():
                layer.weight.normal_(0.0, scale)
                layer.bias.zero_()
            self.layers.append(layer)
            prev_dim = dim
        self.output_layer = nn.Linear(prev_dim, output_dim)
        scale = self.params['init_scale'] / np.sqrt(prev_dim)
        with torch.no_grad():
            self.output_layer.weight.normal_(0.0, scale)
            self.output_layer.bias.zero_()

    def forward_tensor(self, x):
        current = x
        for layer in self.layers:
            current = torch.relu(layer(current))
        return self.output_layer(current)

    def forward(self, x: np.ndarray):
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            pred = self.forward_tensor(xt).detach().cpu().numpy()
        return pred, None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def activations(self, x: np.ndarray, include_output: bool = True) -> list[np.ndarray]:
        acts = []
        with torch.no_grad():
            current = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            for layer in self.layers:
                z = layer(current)
                acts.append(z.detach().cpu().numpy())
                current = torch.relu(z)
            out = self.output_layer(current)
            if include_output:
                acts.append(out.detach().cpu().numpy())
        return acts

    def analysis_layer_sizes(self) -> list[int]:
        return [layer.out_features for layer in self.layers]

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {}
        for idx, layer in enumerate(self.layers):
            state['modules_list_{}_linear1_weight'.format(idx)] = layer.weight.detach().cpu().numpy().astype(np.float32)
            state['modules_list_{}_linear1_bias'.format(idx)] = layer.bias.detach().cpu().numpy().astype(np.float32)
        state['output_block_linear1_weight'] = self.output_layer.weight.detach().cpu().numpy().astype(np.float32)
        state['output_block_linear1_bias'] = self.output_layer.bias.detach().cpu().numpy().astype(np.float32)
        return state


class TorchTinyTransformer(nn.Module if TORCH_AVAILABLE else object):
    is_torch_model = True

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        model_dim: int = 32,
        num_layers: int = 2,
        ff_dim: int | None = None,
        init_scale: float = 0.6,
        seed: int = 0,
        device: str = 'auto',
        train_encoder: bool = False,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError('PyTorch is required for TorchTinyTransformer')
        super().__init__()
        self.params = {'model_dim': model_dim, 'num_layers': num_layers, 'ff_dim': ff_dim, 'init_scale': init_scale, 'device': device, 'train_encoder': train_encoder}
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch_device_from_arg(device)
        self.blocks = nn.ParameterList()
        self.block_names = []
        self.output_layer = None
        if input_dim is not None and output_dim is not None:
            self._init_weights(input_dim, output_dim, seed)
            self.to(self.device)

    def build(self, input_dim: int, output_dim: int, seed: int):
        return TorchTinyTransformer(input_dim=input_dim, output_dim=output_dim, seed=seed, **self.params)

    def _param(self, value, trainable: bool):
        return nn.Parameter(torch.as_tensor(value, dtype=torch.float32), requires_grad=trainable)

    def _init_weights(self, input_dim: int, output_dim: int, seed: int) -> None:
        if input_dim % 2 != 0:
            raise ValueError('TorchTinyTransformer expects an even input_dim')
        rng = np.random.default_rng(seed)
        self.half_dim = input_dim // 2
        self.model_dim = self.params['model_dim']
        self.ff_dim = self.params['ff_dim'] or self.model_dim * 2
        init_scale = self.params['init_scale']
        train_encoder = self.params['train_encoder']
        self.input_proj = self._param(rng.normal(0.0, init_scale / np.sqrt(self.half_dim), size=(self.half_dim, self.model_dim)).astype(np.float32), train_encoder)
        self.token_bias = self._param(rng.normal(0.0, 0.02, size=(2, self.model_dim)).astype(np.float32), train_encoder)
        self.blocks = nn.ParameterList()
        self.block_names = []
        for layer_idx in range(self.params['num_layers']):
            scale = init_scale / np.sqrt(self.model_dim)
            for name, shape in [
                ('wq', (self.model_dim, self.model_dim)),
                ('wk', (self.model_dim, self.model_dim)),
                ('wv', (self.model_dim, self.model_dim)),
                ('wo', (self.model_dim, self.model_dim)),
                ('ff1', (self.model_dim, self.ff_dim)),
                ('ff2', (self.ff_dim, self.model_dim)),
            ]:
                self.blocks.append(self._param(rng.normal(0.0, scale, size=shape).astype(np.float32), train_encoder))
                self.block_names.append((layer_idx, name))
            self.blocks.append(self._param(np.zeros(self.ff_dim, dtype=np.float32), train_encoder))
            self.block_names.append((layer_idx, 'b1'))
            self.blocks.append(self._param(np.zeros(self.model_dim, dtype=np.float32), train_encoder))
            self.block_names.append((layer_idx, 'b2'))
        self.output_layer = nn.Linear(self.model_dim, output_dim)
        out_scale = init_scale / np.sqrt(self.model_dim)
        with torch.no_grad():
            self.output_layer.weight.normal_(0.0, out_scale)
            self.output_layer.bias.zero_()

    def _block_dict(self, layer_idx: int):
        return {name: param for (idx, name), param in zip(self.block_names, self.blocks) if idx == layer_idx}

    def _encode_tensor(self, x):
        left = x[:, : self.half_dim] @ self.input_proj
        right = x[:, self.half_dim :] @ self.input_proj
        tokens = torch.stack([left, right], dim=1) + self.token_bias[None, :, :]
        pooled = []
        for layer_idx in range(self.params['num_layers']):
            block = self._block_dict(layer_idx)
            q = tokens @ block['wq']
            k = tokens @ block['wk']
            v = tokens @ block['wv']
            attention = torch.softmax(q @ k.transpose(-1, -2) / np.sqrt(self.model_dim), dim=-1)
            tokens = torch.nn.functional.layer_norm(tokens + (attention @ v) @ block['wo'], (self.model_dim,))
            ff = torch.relu(tokens @ block['ff1'] + block['b1'])
            tokens = torch.nn.functional.layer_norm(tokens + ff @ block['ff2'] + block['b2'], (self.model_dim,))
            pooled.append(tokens.mean(dim=1))
        return tokens.mean(dim=1), pooled

    def forward_tensor(self, x):
        features, _ = self._encode_tensor(x)
        return self.output_layer(features)

    def forward(self, x: np.ndarray):
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            pred = self.forward_tensor(xt).detach().cpu().numpy()
        return pred, None

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    def activations(self, x: np.ndarray, include_output: bool = True) -> list[np.ndarray]:
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            features, pooled = self._encode_tensor(xt)
            out = self.output_layer(features)
            acts = [p.detach().cpu().numpy() for p in pooled]
            if include_output:
                acts.append(out.detach().cpu().numpy())
        return acts

    def analysis_layer_sizes(self) -> list[int]:
        return [self.model_dim for _ in range(self.params['num_layers'])]

    def state_dict(self) -> dict[str, np.ndarray]:
        state = {
            'transformer_input_proj': self.input_proj.detach().cpu().numpy().astype(np.float32),
            'transformer_token_bias': self.token_bias.detach().cpu().numpy().astype(np.float32),
            'output_block_linear1_weight': self.output_layer.weight.detach().cpu().numpy().astype(np.float32),
            'output_block_linear1_bias': self.output_layer.bias.detach().cpu().numpy().astype(np.float32),
        }
        for (idx, name), param in zip(self.block_names, self.blocks):
            state['transformer_layer_{}_{}'.format(idx, name)] = param.detach().cpu().numpy().astype(np.float32)
        return state
