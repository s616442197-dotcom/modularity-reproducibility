from __future__ import annotations

import numpy as np

from task import ExperimentData


class MSELoss:
    def __init__(self, l1: float = 0.0, gamma: float = 1.0, regularization_epsilon: float = 1e-8):
        if gamma <= 0.0:
            raise ValueError('gamma must be positive for L_gamma regularization')
        self.l1 = l1
        self.gamma = gamma
        self.regularization_epsilon = regularization_epsilon

    def loss_and_grad(self, pred: np.ndarray, target: np.ndarray, data: ExperimentData) -> tuple[float, np.ndarray]:
        diff = pred - target
        loss = float(np.mean(diff * diff))
        grad = (2.0 / diff.size) * diff
        return loss, grad.astype(np.float32)

    def weight_regularization_grad(self, weight: np.ndarray) -> np.ndarray:
        if not self.l1:
            return np.zeros_like(weight)
        abs_weight = np.abs(weight) + self.regularization_epsilon
        grad = self.l1 * self.gamma * np.power(abs_weight, self.gamma - 1.0) * np.sign(weight)
        return (grad / weight.size).astype(np.float32)

    def to_config(self) -> dict:
        return {
            'l1': float(self.l1),
            'gamma': float(self.gamma),
            'regularization_epsilon': float(self.regularization_epsilon),
        }


class WeightedMSELoss(MSELoss):
    def __init__(
        self,
        weighted_rule: int | None = None,
        weighted_loss: float = 1.0,
        rule_weights: list[float] | None = None,
        l1: float = 0.0,
        gamma: float = 1.0,
        regularization_epsilon: float = 1e-8,
    ):
        super().__init__(l1=l1, gamma=gamma, regularization_epsilon=regularization_epsilon)
        self.weighted_rule = weighted_rule
        self.weighted_loss = weighted_loss
        self.rule_weights = rule_weights

    def to_config(self) -> dict:
        config = super().to_config()
        config.update({
            'weighted_rule': self.weighted_rule,
            'weighted_loss': float(self.weighted_loss),
            'rule_weights': self.rule_weights,
        })
        return config

    def output_weights(self, data: ExperimentData) -> np.ndarray:
        weights = np.ones(len(data.rule_indices), dtype=np.float32)
        if self.rule_weights is not None:
            weights = np.asarray(self.rule_weights, dtype=np.float32)
            if len(weights) != len(data.rule_indices):
                raise ValueError('rule_weights length must match active rules')
        if self.weighted_rule in data.rule_indices:
            weights[data.rule_indices.index(self.weighted_rule)] *= self.weighted_loss
        return np.repeat(weights, data.pattern_size)[None, :]

    def loss_and_grad(self, pred: np.ndarray, target: np.ndarray, data: ExperimentData) -> tuple[float, np.ndarray]:
        diff = pred - target
        weights = self.output_weights(data)
        loss = float(np.mean(weights * diff * diff))
        grad = (2.0 / diff.size) * weights * diff
        return loss, grad.astype(np.float32)
