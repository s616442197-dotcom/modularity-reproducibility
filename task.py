from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from dataset import DatasetBundle, PairData, int_to_bits


@dataclass
class ExperimentData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    pattern_size: int
    rule_indices: list[int]
    rule_names: list[str]


def last_bits(values: np.ndarray, width: int) -> np.ndarray:
    mask = (1 << width) - 1
    return int_to_bits(values.astype(np.int64) & mask, width)


class BinaryArithmeticTask:
    rule_names = ['x+y', 'x*y', 'x^2', 'y^2', 'x*(x+y)', 'y*(x+y)']

    def __init__(self, rule_indices: list[int] | None = None):
        self.rule_indices = rule_indices or [0, 1, 2, 3, 4, 5]
        self.rule_functions: list[Callable[[np.ndarray, np.ndarray], np.ndarray]] = [
            lambda x, y: x + y,
            lambda x, y: x * y,
            lambda x, y: x * x,
            lambda x, y: y * y,
            lambda x, y: x * (x + y),
            lambda x, y: y * (x + y),
        ]

    def targets(self, data: PairData, pattern_size: int) -> np.ndarray:
        chunks = [last_bits(self.rule_functions[i](data.left_int, data.right_int), pattern_size) for i in self.rule_indices]
        return np.concatenate(chunks, axis=1).astype(np.float32)

    def prepare(self, bundle: DatasetBundle) -> ExperimentData:
        return ExperimentData(
            x_train=bundle.train.x,
            y_train=self.targets(bundle.train, bundle.pattern_size),
            x_test=bundle.test.x,
            y_test=self.targets(bundle.test, bundle.pattern_size),
            pattern_size=bundle.pattern_size,
            rule_indices=list(self.rule_indices),
            rule_names=[self.rule_names[i] for i in self.rule_indices],
        )


class RandomParityTask:
    """Random linear parity targets over the concatenated binary input bits."""

    def __init__(
        self,
        rule_indices: list[int] | None = None,
        num_rules: int = 5,
        mask_seed: int = 0,
        mask_weight: int | None = 4,
    ):
        self.rule_indices = rule_indices or list(range(num_rules))
        self.num_rules = max(num_rules, max(self.rule_indices) + 1 if self.rule_indices else 0)
        self.mask_seed = int(mask_seed)
        self.mask_weight = mask_weight
        self.rule_names = ['parity_{}'.format(i) for i in range(self.num_rules)]
        self._mask_cache: dict[tuple[int, int], np.ndarray] = {}

    def _masks(self, pattern_size: int) -> np.ndarray:
        key = (int(pattern_size), int(self.num_rules))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        input_dim = 2 * int(pattern_size)
        total_outputs = int(self.num_rules) * int(pattern_size)
        if total_outputs >= (1 << input_dim):
            raise ValueError('Need fewer than 2^input_dim nonzero parity masks')
        rng = np.random.default_rng(self.mask_seed + 1009 * int(pattern_size) + 9176 * int(self.num_rules))
        masks = []
        seen: set[tuple[int, ...]] = set()
        while len(masks) < total_outputs:
            if self.mask_weight is None:
                mask = rng.integers(0, 2, size=input_dim, dtype=np.uint8)
                if not np.any(mask):
                    continue
            else:
                weight = int(self.mask_weight)
                if weight <= 0 or weight > input_dim:
                    raise ValueError('mask_weight must be in [1, input_dim], got {}'.format(self.mask_weight))
                mask = np.zeros(input_dim, dtype=np.uint8)
                mask[rng.choice(input_dim, size=weight, replace=False)] = 1
            encoded = tuple(int(x) for x in mask)
            if encoded in seen:
                continue
            seen.add(encoded)
            masks.append(mask)
        mask_array = np.stack(masks, axis=0).reshape(self.num_rules, int(pattern_size), input_dim)
        self._mask_cache[key] = mask_array
        return mask_array

    def targets(self, data: PairData, pattern_size: int) -> np.ndarray:
        x_bits = np.asarray(data.x, dtype=np.uint8)
        masks = self._masks(pattern_size)
        chunks = []
        for rule_idx in self.rule_indices:
            projected = x_bits @ masks[int(rule_idx)].T
            chunks.append((projected & 1).astype(np.float32))
        return np.concatenate(chunks, axis=1).astype(np.float32)

    def prepare(self, bundle: DatasetBundle) -> ExperimentData:
        return ExperimentData(
            x_train=bundle.train.x,
            y_train=self.targets(bundle.train, bundle.pattern_size),
            x_test=bundle.test.x,
            y_test=self.targets(bundle.test, bundle.pattern_size),
            pattern_size=bundle.pattern_size,
            rule_indices=list(self.rule_indices),
            rule_names=[self.rule_names[i] for i in self.rule_indices],
        )


class MNISTArithmeticTask(BinaryArithmeticTask):
    """Arithmetic rules on MNIST digit labels; targets keep the lowest output_bits bits."""
    rule_names = ['digit_x+digit_y', 'digit_x*digit_y', 'digit_x^2', 'digit_y^2', 'digit_x*(digit_x+digit_y)', 'digit_y*(digit_x+digit_y)']

    def __init__(self, rule_indices: list[int] | None = None, output_bits: int | None = None):
        super().__init__(rule_indices=rule_indices)
        self.output_bits = output_bits

    def prepare(self, bundle: DatasetBundle) -> ExperimentData:
        pattern_size = int(self.output_bits if self.output_bits is not None else bundle.pattern_size)
        return ExperimentData(
            x_train=bundle.train.x,
            y_train=self.targets(bundle.train, pattern_size),
            x_test=bundle.test.x,
            y_test=self.targets(bundle.test, pattern_size),
            pattern_size=pattern_size,
            rule_indices=list(self.rule_indices),
            rule_names=[self.rule_names[i] for i in self.rule_indices],
        )


class PatternLogicTask:
    rule_names = ['and', 'or', 'xor', 'not_x', 'x', 'y']

    def __init__(self, rule_indices: list[int] | None = None):
        self.rule_indices = rule_indices or [0, 1, 2, 3, 4, 5]

    def targets(self, data: PairData, pattern_size: int) -> np.ndarray:
        rule_outputs = [
            data.left_bits * data.right_bits,
            np.maximum(data.left_bits, data.right_bits),
            np.abs(data.left_bits - data.right_bits),
            1.0 - data.left_bits,
            data.left_bits,
            data.right_bits,
        ]
        return np.concatenate([rule_outputs[i] for i in self.rule_indices], axis=1).astype(np.float32)

    def prepare(self, bundle: DatasetBundle) -> ExperimentData:
        return ExperimentData(
            x_train=bundle.train.x,
            y_train=self.targets(bundle.train, bundle.pattern_size),
            x_test=bundle.test.x,
            y_test=self.targets(bundle.test, bundle.pattern_size),
            pattern_size=bundle.pattern_size,
            rule_indices=list(self.rule_indices),
            rule_names=[self.rule_names[i] for i in self.rule_indices],
        )