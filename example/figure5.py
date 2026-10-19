from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(EXAMPLE_ROOT))

import analysis as ana
import modularity_time as mt


SWEEP_SUMMARY = ROOT / "example/results/simulation/lasso_l_gamma_sweep/lasso_l_gamma_sweep_summary.json"
OUTPUT_ROOT = ROOT / "example/results/figure5"
DEFAULT_LASSO_A = "0,1,5"
DEFAULT_LASSO_B = "0,1,5,10"
DEFAULT_GAMMAS = "0.5,1,2"
DEFAULT_GAMMA_LASSO = 5.0
DEFAULT_FIG_L_SCALE = 5.0
DEFAULT_GAMMA_FOR_LASSO = 1.0
DEFAULT_T_MIN = 3.0
DEFAULT_T_MAX = 7.0
DEFAULT_THRESHOLD_COUNT = 25
DEFAULT_TOKEN_TYPE = "weight"
DEFAULT_MASK_METHOD = "ablation"
DEFAULT_CUTOFF_MODE = "fixed"
DEFAULT_TOKEN_EXTRACTION_METHOD = ana.DEFAULT_TOKEN_EXTRACTION_METHOD
DEFAULT_TOKEN_EXTRACTION_FRACTION = ana.DEFAULT_TOKEN_EXTRACTION_FRACTION
DEFAULT_NEURON_T_FOR_PANEL_A = 4.0
DEFAULT_WEIGHT_T_FOR_PANEL_A = 6.0
DEFAULT_SCORE_DISTRIBUTION_LASSO = DEFAULT_LASSO_B
DEFAULT_SCORE_DISTRIBUTION_TOKEN_TYPES = "weight"
DEFAULT_SCORE_DISTRIBUTION_VALUE = "score_ratio"
DEFAULT_SCORE_DISTRIBUTION_MAX_VALUES = 800
DEFAULT_SCORE_DISTRIBUTION_FLOOR = 1e-12
LAYER_COLORS = ["#4DBEEE", "#A2142F", "#0072BD"]
RULE_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30"]
EXPORT_EXTENSIONS = ("png", "pdf", "svg")
MODULARITY_METRIC_TAG = "qcount"
STRUCTURAL_SCORE_QUANTILE = 0.75


def normal_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def parse_float_list(value: str) -> list[float]:
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def parse_token_types(value: str) -> list[str]:
    aliases = {"neurons": "neuron", "weights": "weight"}
    out = []
    for item in str(value).split(","):
        item = aliases.get(item.strip().lower(), item.strip().lower())
        if not item:
            continue
        if item not in {"neuron", "weight"}:
            raise ValueError("score distribution token types must be neuron and/or weight")
        if item not in out:
            out.append(item)
    return out


def format_float(value: float) -> str:
    return "{:g}".format(float(value))


def method_suffix(mask_method: str) -> str:
    method = str(mask_method).lower().strip()
    return "" if method == "ablation" else "_{}".format(method)


def cutoff_suffix(cutoff_mode: str) -> str:
    mode = str(cutoff_mode).lower().strip()
    return "" if mode == "fixed" else "_{}_cutoff".format(mode)


def panel_a_threshold_value(args: argparse.Namespace) -> float:
    if str(args.cutoff_mode).lower().strip() == "median":
        return float("nan")
    return float(10.0 ** (-float(args.panel_a_t)))


def panel_a_threshold_arg(args: argparse.Namespace) -> float | str:
    if str(args.cutoff_mode).lower().strip() == "median":
        return "median"
    return panel_a_threshold_value(args)


def method_cache_tag(args: argparse.Namespace) -> str:
    method = str(args.mask_method).lower().strip()
    if method == "legacy_hessian":
        tag = "{}_eps{}".format(method, mt.tag_float(args.legacy_hessian_epsilon))
    elif method == "taylor":
        tag = "{}_p{}".format(method, int(args.taylor_probes))
    else:
        tag = method
    return "{}_{}".format(tag, MODULARITY_METRIC_TAG)


def configure_style() -> None:
    ana.configure_publication_style()
    matplotlib.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "svg.fonttype": "none",
        }
    )


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def export_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / "{}.{}".format(stem, fmt)
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = ana.PUBLICATION_DPI
        fig.savefig(path, **kwargs)
        paths.append(path)
    return paths


def parse_formats(value: str) -> list[str]:
    formats = []
    for item in str(value).split(","):
        item = item.strip().lower().lstrip(".")
        if item and item not in formats:
            formats.append(item)
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def selected_for_gamma(
    summary_path: Path,
    gamma: float,
    fig_lasso_values: list[float],
    fig_l_scale: float,
) -> tuple[dict[float, tuple[float, Path]], list[float]]:
    return mt.selected_conditions(summary_path, gamma, fig_lasso_values, fig_l_scale)


def resolve_epoch(condition_dir: Path, epoch_arg: str) -> int:
    epoch_arg = str(epoch_arg).strip().lower()
    epoch_dirs = sorted((condition_dir / "snapshots").glob("epoch_*"), key=mt.snapshot_epoch)
    if not epoch_dirs:
        raise FileNotFoundError("No snapshots/epoch_* directories under {}".format(condition_dir))
    if epoch_arg in {"latest", "last", ""}:
        return mt.snapshot_epoch(epoch_dirs[-1])
    target = int(float(epoch_arg))
    if not (condition_dir / "snapshots" / "epoch_{:06d}".format(target)).is_dir():
        available = ", ".join(str(mt.snapshot_epoch(path)) for path in epoch_dirs)
        raise FileNotFoundError("Epoch {} not found under {}; available: {}".format(target, condition_dir, available))
    return target


def run_dirs_for_epoch(condition_dir: Path, epoch: int, max_runs: int | None) -> list[Path]:
    epoch_dir = condition_dir / "snapshots" / "epoch_{:06d}".format(int(epoch))
    run_dirs = sorted(path for path in epoch_dir.glob("run_*") if (path / "run_data.npz").is_file())
    if max_runs is not None:
        run_dirs = run_dirs[: max(0, int(max_runs))]
    if not run_dirs:
        raise FileNotFoundError("No run_*/run_data.npz files under {}".format(epoch_dir))
    return run_dirs


def cache_path(
    output_root: Path,
    gamma: float,
    experiment_lasso: float,
    epoch: int,
    run_name: str,
    args: argparse.Namespace,
) -> Path:
    sample_tag = "full" if args.max_samples is None else str(int(args.max_samples))
    sweep_tag = "{}_{}_{}_extract{}_frac{}_demoacc_t{}_{}_n{}_panel{}".format(
        args.token_type,
        method_cache_tag(args),
        str(args.cutoff_mode).lower().strip(),
        ana.validate_token_extraction_method(args.token_extraction_method),
        mt.tag_float(args.token_extraction_fraction),
        mt.tag_float(args.t_min),
        mt.tag_float(args.t_max),
        int(args.threshold_count),
        mt.tag_float(args.panel_a_t),
    )
    return (
        output_root
        / "cache"
        / "samples_{}".format(sample_tag)
        / sweep_tag
        / "gamma{}".format(mt.tag_float(gamma))
        / "experiment_l{}".format(mt.tag_float(experiment_lasso))
        / "epoch_{:06d}".format(int(epoch))
        / "{}.json".format(run_name)
    )


def score_distribution_cache_path(
    output_root: Path,
    gamma: float,
    experiment_lasso: float,
    epoch: int,
    run_name: str,
    args: argparse.Namespace,
) -> Path:
    sample_tag = "full" if args.max_samples is None else str(int(args.max_samples))
    token_tag = "_".join(args.score_distribution_token_types) or "none"
    score_tag = "{}_{}_max{}_floor{}".format(
        method_cache_tag(args),
        args.score_distribution_value,
        int(args.score_distribution_max_values),
        mt.tag_float(args.score_distribution_floor),
    )
    return (
        output_root
        / "cache"
        / "score_distribution"
        / "samples_{}".format(sample_tag)
        / token_tag
        / score_tag
        / "gamma{}".format(mt.tag_float(gamma))
        / "experiment_l{}".format(mt.tag_float(experiment_lasso))
        / "epoch_{:06d}".format(int(epoch))
        / "{}.json".format(run_name)
    )


def subset_loaded_data(loaded: dict[str, Any], max_samples: int | None, seed: int) -> dict[str, Any]:
    return mt.subset_loaded_data(loaded, max_samples=max_samples, seed=seed)


def layer_score_tables(
    state: dict[str, np.ndarray],
    x_test: np.ndarray,
    baseline_pred: np.ndarray,
    pattern_size: int,
    rule_indices: list[int],
    output_energy: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    mask_method = ana.validate_mask_method(args.mask_method)
    context = ana.torch_ablation_context(state, x_test, baseline_pred, device=args.analysis_device)
    layer_ids = ana.sorted_transformer_layer_ids(state) if "transformer_input_proj" in state else ana.sorted_mlp_layer_ids(state)
    widths = ana.analysis_layer_sizes(state)
    taylor_scores = None
    if mask_method == "taylor":
        taylor_scores = ana.taylor_importance_scores(
            state,
            x_test,
            pattern_size,
            rule_indices,
            batch_size=args.perturbation_batch_size,
            device=args.analysis_device,
            probes=args.taylor_probes,
        )
    out = []
    for layer_pos, layer_id in enumerate(layer_ids):
        width = int(widths[layer_pos])
        if mask_method == "taylor":
            scores = np.asarray(taylor_scores["neuron_layers"][layer_pos], dtype=float)
        elif mask_method == "legacy_hessian":
            scores = ana.legacy_hessian_neuron_scores(
                state,
                x_test,
                baseline_pred,
                pattern_size,
                rule_indices,
                int(layer_id),
                width,
                batch_size=args.perturbation_batch_size,
                epsilon=args.legacy_hessian_epsilon,
                device=args.analysis_device,
                context=context,
            )
        else:
            similarity = ana.torch_neuron_silence_similarity(
                state,
                x_test,
                baseline_pred,
                pattern_size,
                rule_indices,
                int(layer_id),
                width,
                batch_size=args.perturbation_batch_size,
                device=args.analysis_device,
                context=context,
            )
            if similarity is None:
                scores = np.zeros((len(rule_indices), width), dtype=float)
                for neuron_idx in range(width):
                    zeroed_pred, _ = ana.forward_from_state(state, x_test, silence=(int(layer_id), [neuron_idx]))
                    scores[:, neuron_idx] = ana.rule_output_change_to_baseline(zeroed_pred, baseline_pred, pattern_size, rule_indices)
            else:
                scores = (1.0 - similarity) * max(pattern_size, 1)
        ratio = ana.scoring_threshold_table(scores, output_energy, output_eps=args.importance_output_eps, mask_method=mask_method)
        out.append({"layer_id": int(layer_id), "layer_index": int(layer_pos + 1), "scores": scores, "ratio": ratio, "mask_method": mask_method})
    return out


def single_rule_combo_indices(combo_labels: list[str], rule_indices: list[int]) -> list[int]:
    labels = [str(label) for label in combo_labels]
    indices = []
    for rule_pos, rule_id in enumerate(rule_indices):
        wanted = str(rule_id)
        if wanted in labels:
            indices.append(labels.index(wanted))
        else:
            indices.append(rule_pos)
    return indices



def fixed_mask_modularity_summary(
    score_matrix: np.ndarray,
    ratio_matrix: np.ndarray,
    min_score: float,
    token_extraction_method: str,
    token_extraction_fraction: float,
) -> dict[str, float | int | str]:
    scores = np.asarray(score_matrix, dtype=float)
    rule_count = int(scores.shape[0]) if scores.ndim == 2 else 0
    total_tokens = int(scores.shape[1]) if scores.ndim == 2 else 0
    metric_name = "quantile_rule_count_purity"
    empty = {
        "modularity": float("nan"),
        "modularity_q": float("nan"),
        "normalized_modularity_q": float("nan"),
        "active_token_count": 0,
        "active_token_fraction": 0.0,
        "sparsity_factor": 1.0,
        "exclusive_token_count": 0,
        "structural_exclusivity": float("nan"),
        "single_rule_fraction": float("nan"),
        "mean_rule_hit_count": float("nan"),
        "total_importance": 0.0,
        "mean_token_specialization": float("nan"),
        "mean_participation": float("nan"),
        "modularity_metric": metric_name,
    }
    for idx in range(1, max(rule_count, 5) + 1):
        empty["rule_count_{}_fraction".format(idx)] = float("nan")
    if scores.ndim != 2 or rule_count == 0 or total_tokens == 0:
        return empty

    finite = np.isfinite(scores)
    thresholds = np.full(rule_count, np.nan, dtype=float)
    for rule_idx in range(rule_count):
        values = scores[rule_idx, finite[rule_idx]]
        if values.size:
            thresholds[rule_idx] = float(np.quantile(values, STRUCTURAL_SCORE_QUANTILE))
    hits = finite & (scores > thresholds[:, None])
    hit_counts = np.sum(hits, axis=0).astype(int)
    active_hit_counts = hit_counts[hit_counts > 0]
    active_count = int(active_hit_counts.size)
    if active_count == 0:
        out = dict(empty)
        out["active_token_fraction"] = 0.0
        return out

    weights = (rule_count - active_hit_counts + 1).astype(float) / float(max(rule_count, 1))
    modularity = float(np.mean(weights))
    exclusive_count = int(np.sum(active_hit_counts == 1))
    single_rule_fraction = float(exclusive_count / active_count)
    mean_hit_count = float(np.mean(active_hit_counts))
    out = {
        "modularity": modularity,
        "modularity_q": single_rule_fraction,
        "normalized_modularity_q": single_rule_fraction,
        "active_token_count": active_count,
        "active_token_fraction": float(active_count / max(total_tokens, 1)),
        "sparsity_factor": 1.0,
        "exclusive_token_count": exclusive_count,
        "structural_exclusivity": single_rule_fraction,
        "single_rule_fraction": single_rule_fraction,
        "mean_rule_hit_count": mean_hit_count,
        "total_importance": float(active_count),
        "mean_token_specialization": single_rule_fraction,
        "mean_participation": mean_hit_count,
        "modularity_metric": metric_name,
    }
    for idx in range(1, max(rule_count, 5) + 1):
        out["rule_count_{}_fraction".format(idx)] = float(np.mean(active_hit_counts == idx))
    return out


def compute_neuron_run_metrics(
    run_dir: Path,
    gamma: float,
    fig_lasso: float,
    experiment_lasso: float,
    epoch: int,
    t_values: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(run_dir))) % (2**32)
    loaded = subset_loaded_data(ana.load_run_npz(run_dir), args.max_samples, seed)
    state = loaded["state"]
    x_test = loaded["x_test"]
    y_test = loaded["y_test"]
    pattern_size = int(loaded["pattern_size"])
    rule_indices = [int(x) for x in loaded["rule_indices"]]
    baseline_pred, _activations = ana.forward_from_state(state, x_test)
    baseline_acc = ana.rule_accuracy_vector(baseline_pred, y_test, pattern_size, rule_indices)
    output_energy = ana.rule_output_energy(baseline_pred, pattern_size, rule_indices, eps=args.importance_output_eps)
    layer_tables = layer_score_tables(state, x_test, baseline_pred, pattern_size, rule_indices, output_energy, args)
    combos = ana.combo_list(len(rule_indices))
    combo_sizes = np.asarray([len(combo) for combo in combos], dtype=float)

    panel_a_threshold = panel_a_threshold_value(args)
    panel_a_threshold_for_mask = panel_a_threshold_arg(args)
    total_neurons = int(sum(table["scores"].shape[1] for table in layer_tables))
    important_by_layer = []
    for table in layer_tables:
        important_by_layer.append(
            ana.importance_mask_from_tables(
                table["scores"],
                table["ratio"],
                min_score=args.importance_min_score,
                score_ratio_threshold=panel_a_threshold_for_mask,
                token_extraction_method=args.token_extraction_method,
                token_extraction_fraction=args.token_extraction_fraction,
            )
        )

    retained_counts_by_layer_rule = np.zeros((len(layer_tables), len(rule_indices)), dtype=float)
    single_accuracy_matrix = np.full((len(rule_indices), len(rule_indices)), np.nan, dtype=float)
    single_rows = []
    for rule_pos, _rule_id in enumerate(rule_indices):
        keep_by_layer = {}
        retained_count = 0
        for layer_pos, (table, important) in enumerate(zip(layer_tables, important_by_layer)):
            keep_idx = np.where(important[rule_pos])[0].astype(int).tolist()
            keep_by_layer[int(table["layer_id"])] = keep_idx
            retained_count += len(keep_idx)
            retained_counts_by_layer_rule[layer_pos, rule_pos] = len(keep_idx)
        kept_pred, _ = ana.forward_with_layer_keep(state, x_test, keep_by_layer)
        kept_acc = ana.rule_accuracy_vector(kept_pred, y_test, pattern_size, rule_indices)
        for eval_pos, _eval_rule_id in enumerate(rule_indices):
            relative_accuracy = float(kept_acc[eval_pos] / max(float(baseline_acc[eval_pos]), 1e-12))
            single_accuracy_matrix[rule_pos, eval_pos] = relative_accuracy
            single_rows.append(
                {
                    "gamma": float(gamma),
                    "lasso": float(fig_lasso),
                    "experiment_lasso": float(experiment_lasso),
                    "mask_method": str(args.mask_method),
                    "epoch": int(epoch),
                    "run": run_dir.name,
                    "rule_index": int(rule_pos + 1),
                    "rule_label": "rule {}".format(rule_pos + 1),
                    "retained_rule_index": int(rule_pos + 1),
                    "retained_rule_label": "rule {}".format(rule_pos + 1),
                    "evaluated_rule_index": int(eval_pos + 1),
                    "evaluated_rule_label": "rule {}".format(eval_pos + 1),
                    "relative_accuracy": relative_accuracy,
                    "panel_a_t": float(args.panel_a_t),
                    "panel_a_threshold": float(panel_a_threshold),
                    "panel_a_cutoff_mode": str(args.cutoff_mode),
                    "token_extraction_method": str(args.token_extraction_method),
                    "token_extraction_fraction": float(args.token_extraction_fraction),
                    "retained_neuron_count": int(retained_count),
                    "retained_neuron_ratio": float(retained_count / max(total_neurons, 1)),
                    "baseline_accuracy": float(baseline_acc[eval_pos]),
                    "num_samples": int(len(x_test)),
                }
            )

    structural_by_layer = []
    for table in layer_tables:
        structural_by_layer.append(fixed_mask_modularity_summary(table["scores"], table["ratio"], args.importance_min_score, args.token_extraction_method, args.token_extraction_fraction))

    threshold_rows = []
    for table in layer_tables:
        layer_idx = int(table["layer_index"])
        modularity_summary = structural_by_layer[layer_idx - 1] if 0 <= layer_idx - 1 < len(structural_by_layer) else fixed_mask_modularity_summary(np.zeros_like(table["scores"], dtype=float), np.zeros_like(table["scores"], dtype=float), args.importance_min_score, args.token_extraction_method, args.token_extraction_fraction)
        for t_value in t_values:
            threshold = float(10.0 ** (-float(t_value)))
            threshold_rows.append(
                {
                    "gamma": float(gamma),
                    "lasso": float(fig_lasso),
                    "experiment_lasso": float(experiment_lasso),
                    "mask_method": str(args.mask_method),
                    "epoch": int(epoch),
                    "run": run_dir.name,
                    "layer_index": layer_idx,
                    "layer_label": "layer{}".format(layer_idx),
                    "t": float(t_value),
                    "threshold": threshold,
                    "modularity": float(modularity_summary["modularity"]),
                    "active_token_count": int(modularity_summary["active_token_count"]),
                    "modularity_q": float(modularity_summary["modularity_q"]),
                    "normalized_modularity_q": float(modularity_summary["normalized_modularity_q"]),
                    "active_token_fraction": float(modularity_summary["active_token_fraction"]),
                    "sparsity_factor": float(modularity_summary["sparsity_factor"]),
                    "exclusive_token_count": int(modularity_summary["exclusive_token_count"]),
                    "structural_exclusivity": float(modularity_summary["structural_exclusivity"]),
                    "single_rule_fraction": float(modularity_summary["single_rule_fraction"]),
                    "mean_rule_hit_count": float(modularity_summary["mean_rule_hit_count"]),
                    "rule_count_1_fraction": float(modularity_summary["rule_count_1_fraction"]),
                    "rule_count_2_fraction": float(modularity_summary["rule_count_2_fraction"]),
                    "rule_count_3_fraction": float(modularity_summary["rule_count_3_fraction"]),
                    "rule_count_4_fraction": float(modularity_summary["rule_count_4_fraction"]),
                    "rule_count_5_fraction": float(modularity_summary["rule_count_5_fraction"]),
                    "total_importance": float(modularity_summary["total_importance"]),
                    "mean_token_specialization": float(modularity_summary["mean_token_specialization"]),
                    "mean_participation": float(modularity_summary["mean_participation"]),
                    "modularity_metric": str(modularity_summary["modularity_metric"]),
                    "num_samples": int(len(x_test)),
                }
            )
    return single_rows, threshold_rows


def compute_weight_run_metrics(
    run_dir: Path,
    gamma: float,
    fig_lasso: float,
    experiment_lasso: float,
    epoch: int,
    t_values: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(run_dir))) % (2**32)
    loaded = subset_loaded_data(ana.load_run_npz(run_dir), args.max_samples, seed)
    state = loaded["state"]
    x_test = loaded["x_test"]
    y_test = loaded["y_test"]
    pattern_size = int(loaded["pattern_size"])
    rule_indices = [int(x) for x in loaded["rule_indices"]]
    combos = ana.combo_list(len(rule_indices))
    combo_sizes = np.asarray([len(combo) for combo in combos], dtype=float)

    panel_a_threshold = panel_a_threshold_value(args)
    panel_a_threshold_for_mask = panel_a_threshold_arg(args)
    weight_combo = ana.weight_combo_subnetwork_analysis(
        state,
        x_test,
        y_test,
        pattern_size,
        rule_indices,
        max_elements_per_block=args.max_weight_elements_per_block,
        mask_method=args.mask_method,
        importance_min_score=args.importance_min_score,
        importance_mask_mode="score_ratio",
        importance_score_ratio_threshold=panel_a_threshold_for_mask,
        importance_output_eps=args.importance_output_eps,
        perturbation_batch_size=args.perturbation_batch_size,
        analysis_device=args.analysis_device,
        token_extraction_method=args.token_extraction_method,
        token_extraction_fraction=args.token_extraction_fraction,
    )
    baseline_acc = np.asarray(weight_combo.get("baseline_accuracy", np.ones(len(rule_indices))), dtype=float)
    retained_network = np.asarray(weight_combo.get("network_retained_accuracy", np.zeros((len(rule_indices), 0))), dtype=float)
    keep_counts = np.asarray(weight_combo.get("network_keep_counts", np.zeros((0, retained_network.shape[1]))), dtype=float)
    total_weights = float(ana._weight_subnet_total_weights(weight_combo))
    combo_labels = [str(label) for label in weight_combo.get("combo_labels", [])]
    single_indices = single_rule_combo_indices(combo_labels, rule_indices)

    single_rows = []
    for rule_pos, combo_idx in enumerate(single_indices):
        retained_weight_count = int(np.nansum(keep_counts[:, combo_idx])) if keep_counts.size and combo_idx < keep_counts.shape[1] else 0
        retained_weight_ratio = float(retained_weight_count / max(total_weights, 1.0))
        for eval_pos, _eval_rule_id in enumerate(rule_indices):
            relative_accuracy = float(retained_network[eval_pos, combo_idx]) if retained_network.size and combo_idx < retained_network.shape[1] else float("nan")
            single_rows.append(
                {
                    "gamma": float(gamma),
                    "lasso": float(fig_lasso),
                    "experiment_lasso": float(experiment_lasso),
                    "mask_method": str(args.mask_method),
                    "epoch": int(epoch),
                    "run": run_dir.name,
                    "rule_index": int(rule_pos + 1),
                    "rule_label": "rule {}".format(rule_pos + 1),
                    "retained_rule_index": int(rule_pos + 1),
                    "retained_rule_label": "rule {}".format(rule_pos + 1),
                    "evaluated_rule_index": int(eval_pos + 1),
                    "evaluated_rule_label": "rule {}".format(eval_pos + 1),
                    "relative_accuracy": relative_accuracy,
                    "panel_a_t": float(args.panel_a_t),
                    "panel_a_threshold": float(panel_a_threshold),
                    "panel_a_cutoff_mode": str(args.cutoff_mode),
                    "token_extraction_method": str(args.token_extraction_method),
                    "token_extraction_fraction": float(args.token_extraction_fraction),
                    "retained_weight_count": int(retained_weight_count),
                    "retained_weight_ratio": retained_weight_ratio,
                    "baseline_accuracy": float(baseline_acc[eval_pos]) if eval_pos < baseline_acc.size else float("nan"),
                    "num_samples": int(len(x_test)),
                }
            )

    layer_blocks = mt.weight_layer_blocks(weight_combo)
    structural_by_layer = []
    for block in layer_blocks:
        structural_by_layer.append(fixed_mask_modularity_summary(mt.weight_block_score_matrix(block), mt.weight_block_ratio_matrix(block), args.importance_min_score, args.token_extraction_method, args.token_extraction_fraction))

    threshold_rows = []
    thresholds = np.asarray([10.0 ** (-float(t_value)) for t_value in t_values], dtype=float)
    for layer_pos, block in enumerate(layer_blocks, start=1):
        modularity_summary = structural_by_layer[layer_pos - 1] if 0 <= layer_pos - 1 < len(structural_by_layer) else fixed_mask_modularity_summary(np.zeros((len(rule_indices), 0), dtype=float), np.zeros((len(rule_indices), 0), dtype=float), args.importance_min_score, args.token_extraction_method, args.token_extraction_fraction)
        for t_value, threshold in zip(t_values, thresholds):
            threshold_rows.append(
                {
                    "gamma": float(gamma),
                    "lasso": float(fig_lasso),
                    "experiment_lasso": float(experiment_lasso),
                    "mask_method": str(args.mask_method),
                    "epoch": int(epoch),
                    "run": run_dir.name,
                    "layer_index": int(layer_pos),
                    "layer_label": "layer{}".format(int(layer_pos)),
                    "weight_block": str(block.get("label", block.get("key", layer_pos))),
                    "t": float(t_value),
                    "threshold": float(threshold),
                    "modularity": float(modularity_summary["modularity"]),
                    "active_token_count": int(modularity_summary["active_token_count"]),
                    "modularity_q": float(modularity_summary["modularity_q"]),
                    "normalized_modularity_q": float(modularity_summary["normalized_modularity_q"]),
                    "active_token_fraction": float(modularity_summary["active_token_fraction"]),
                    "sparsity_factor": float(modularity_summary["sparsity_factor"]),
                    "exclusive_token_count": int(modularity_summary["exclusive_token_count"]),
                    "structural_exclusivity": float(modularity_summary["structural_exclusivity"]),
                    "single_rule_fraction": float(modularity_summary["single_rule_fraction"]),
                    "mean_rule_hit_count": float(modularity_summary["mean_rule_hit_count"]),
                    "rule_count_1_fraction": float(modularity_summary["rule_count_1_fraction"]),
                    "rule_count_2_fraction": float(modularity_summary["rule_count_2_fraction"]),
                    "rule_count_3_fraction": float(modularity_summary["rule_count_3_fraction"]),
                    "rule_count_4_fraction": float(modularity_summary["rule_count_4_fraction"]),
                    "rule_count_5_fraction": float(modularity_summary["rule_count_5_fraction"]),
                    "total_importance": float(modularity_summary["total_importance"]),
                    "mean_token_specialization": float(modularity_summary["mean_token_specialization"]),
                    "mean_participation": float(modularity_summary["mean_participation"]),
                    "modularity_metric": str(modularity_summary["modularity_metric"]),
                    "num_samples": int(len(x_test)),
                }
            )
    return single_rows, threshold_rows


def compute_run_metrics(
    run_dir: Path,
    gamma: float,
    fig_lasso: float,
    experiment_lasso: float,
    epoch: int,
    t_values: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.token_type == "weight":
        return compute_weight_run_metrics(run_dir, gamma, fig_lasso, experiment_lasso, epoch, t_values, args)
    return compute_neuron_run_metrics(run_dir, gamma, fig_lasso, experiment_lasso, epoch, t_values, args)


def run_metric_cache_is_current(cached: dict[str, Any]) -> bool:
    threshold_rows = list(cached.get("threshold_rows", []))
    if not threshold_rows:
        return False
    wanted = "quantile_rule_count_purity"
    metrics = {str(row.get("modularity_metric", "") or "").lower().strip() for row in threshold_rows}
    required = ("active_token_count", "exclusive_token_count", "mean_rule_hit_count", "rule_count_1_fraction")
    return metrics == {wanted} and all(all(key in row for key in required) for row in threshold_rows)


def compute_or_load_run_metrics(
    run_dir: Path,
    gamma: float,
    fig_lasso: float,
    experiment_lasso: float,
    epoch: int,
    t_values: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = cache_path(args.output_root, gamma, experiment_lasso, epoch, run_dir.name, args)
    if path.is_file() and not args.recompute:
        cached = load_json(path)
        if run_metric_cache_is_current(cached):
            return list(cached["single_rule_rows"]), list(cached["threshold_rows"])
    single_rows, threshold_rows = compute_run_metrics(run_dir, gamma, fig_lasso, experiment_lasso, epoch, t_values, args)
    save_json(
        path,
        {
            "run_dir": str(run_dir),
            "gamma": float(gamma),
            "lasso": float(fig_lasso),
            "experiment_lasso": float(experiment_lasso),
            "mask_method": str(args.mask_method),
            "cutoff_mode": str(args.cutoff_mode),
            "token_extraction_method": str(args.token_extraction_method),
            "token_extraction_fraction": float(args.token_extraction_fraction),
            "legacy_hessian_epsilon": float(args.legacy_hessian_epsilon),
            "epoch": int(epoch),
            "single_rule_rows": single_rows,
            "threshold_rows": threshold_rows,
        },
    )
    return single_rows, threshold_rows


def sampled_positive_log_values(values: np.ndarray, max_values: int, floor: float, seed: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    flat = np.asarray(values, dtype=float).ravel()
    finite = flat[np.isfinite(flat)]
    positive = finite[finite > 0.0]
    positive_total = int(positive.size)
    if positive_total == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float), int(finite.size), positive_total
    if max_values > 0 and positive.size > int(max_values):
        rng = np.random.default_rng(int(seed) % (2**32))
        keep = np.sort(rng.choice(positive.size, size=int(max_values), replace=False))
        positive = positive[keep]
    sampled = np.maximum(positive, float(floor))
    return sampled, np.log10(sampled), int(finite.size), positive_total


def append_score_distribution_rows(
    rows: list[dict[str, Any]],
    values: np.ndarray,
    token_type: str,
    layer_index: int,
    layer_label: str,
    gamma: float,
    fig_lasso: float,
    experiment_lasso: float,
    epoch: int,
    run_name: str,
    args: argparse.Namespace,
    seed: int,
) -> None:
    sampled, log_values, total_values, positive_values = sampled_positive_log_values(
        values,
        max_values=int(args.score_distribution_max_values),
        floor=float(args.score_distribution_floor),
        seed=seed,
    )
    for sample_index, (value, log_value) in enumerate(zip(sampled, log_values)):
        rows.append(
            {
                "gamma": float(gamma),
                "lasso": float(fig_lasso),
                "experiment_lasso": float(experiment_lasso),
                "mask_method": str(args.mask_method),
                "epoch": int(epoch),
                "run": run_name,
                "token_type": str(token_type),
                "layer_index": int(layer_index),
                "layer_label": str(layer_label),
                "value_kind": str(args.score_distribution_value),
                "sample_index": int(sample_index),
                "value": float(value),
                "log10_value": float(log_value),
                "num_total_values": int(total_values),
                "num_positive_values": int(positive_values),
                "num_sampled_values": int(sampled.size),
            }
        )


def compute_score_distribution_run_rows(
    run_dir: Path,
    gamma: float,
    fig_lasso: float,
    experiment_lasso: float,
    epoch: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(run_dir))) % (2**32)
    loaded = subset_loaded_data(ana.load_run_npz(run_dir), args.max_samples, seed)
    state = loaded["state"]
    x_test = loaded["x_test"]
    y_test = loaded["y_test"]
    pattern_size = int(loaded["pattern_size"])
    rule_indices = [int(x) for x in loaded["rule_indices"]]
    rows: list[dict[str, Any]] = []

    if "neuron" in args.score_distribution_token_types:
        baseline_pred, _activations = ana.forward_from_state(state, x_test)
        output_energy = ana.rule_output_energy(baseline_pred, pattern_size, rule_indices, eps=args.importance_output_eps)
        layer_tables = layer_score_tables(state, x_test, baseline_pred, pattern_size, rule_indices, output_energy, args)
        for table in layer_tables:
            matrix = table["ratio"] if args.score_distribution_value == "score_ratio" else table["scores"]
            layer_index = int(table["layer_index"])
            append_score_distribution_rows(
                rows,
                matrix,
                token_type="neuron",
                layer_index=layer_index,
                layer_label="layer{}".format(layer_index),
                gamma=gamma,
                fig_lasso=fig_lasso,
                experiment_lasso=experiment_lasso,
                epoch=epoch,
                run_name=run_dir.name,
                args=args,
                seed=seed + layer_index * 1009,
            )

    if "weight" in args.score_distribution_token_types:
        if args.mask_method == "legacy_hessian":
            raise ValueError("legacy_hessian is a neuron scoring mode; remove weight from --score-token-types")
        weight_combo = ana.weight_combo_subnetwork_analysis(
            state,
            x_test,
            y_test,
            pattern_size,
            rule_indices,
            max_elements_per_block=args.max_weight_elements_per_block,
            mask_method=args.mask_method,
            importance_min_score=args.importance_min_score,
            importance_mask_mode="score_ratio",
            importance_score_ratio_threshold=float(10.0 ** (-DEFAULT_WEIGHT_T_FOR_PANEL_A)),
            importance_output_eps=args.importance_output_eps,
            perturbation_batch_size=args.perturbation_batch_size,
            analysis_device=args.analysis_device,
        )
        for layer_index, block_data in enumerate(mt.weight_layer_blocks(weight_combo), start=1):
            matrix = mt.weight_block_ratio_matrix(block_data) if args.score_distribution_value == "score_ratio" else mt.weight_block_score_matrix(block_data)
            append_score_distribution_rows(
                rows,
                matrix,
                token_type="weight",
                layer_index=layer_index,
                layer_label="layer{}".format(layer_index),
                gamma=gamma,
                fig_lasso=fig_lasso,
                experiment_lasso=experiment_lasso,
                epoch=epoch,
                run_name=run_dir.name,
                args=args,
                seed=seed + 1000003 + layer_index * 1009,
            )
    return rows


def compute_or_load_score_distribution_rows(
    run_dir: Path,
    gamma: float,
    fig_lasso: float,
    experiment_lasso: float,
    epoch: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    path = score_distribution_cache_path(args.output_root, gamma, experiment_lasso, epoch, run_dir.name, args)
    if path.is_file() and not args.recompute:
        cached = load_json(path)
        return list(cached["score_distribution_rows"])
    rows = compute_score_distribution_run_rows(run_dir, gamma, fig_lasso, experiment_lasso, epoch, args)
    save_json(
        path,
        {
            "run_dir": str(run_dir),
            "gamma": float(gamma),
            "lasso": float(fig_lasso),
            "experiment_lasso": float(experiment_lasso),
            "mask_method": str(args.mask_method),
            "epoch": int(epoch),
            "score_distribution_value": str(args.score_distribution_value),
            "score_distribution_token_types": list(args.score_distribution_token_types),
            "score_distribution_rows": rows,
        },
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in (
                "gamma",
                "lasso",
                "experiment_lasso",
                "relative_accuracy",
                "retained_neuron_ratio",
                "retained_weight_ratio",
                "baseline_accuracy",
                "panel_a_t",
                "panel_a_threshold",
                "t",
                "threshold",
                "modularity",
                "modularity_q",
                "normalized_modularity_q",
                "active_token_fraction",
                "sparsity_factor",
                "structural_exclusivity",
                "single_rule_fraction",
                "mean_rule_hit_count",
                "rule_count_1_fraction",
                "rule_count_2_fraction",
                "rule_count_3_fraction",
                "rule_count_4_fraction",
                "rule_count_5_fraction",
                "functional_within_accuracy",
                "functional_between_accuracy",
                "functional_separation",
                "retained_token_ratio",
                "compression_factor",
                "mean_token_specialization",
                "mean_participation",
                "value",
                "log10_value",
            ):
                if key in row and row[key] != "":
                    row[key] = float(row[key])
            for key in (
                "epoch",
                "rule_index",
                "retained_rule_index",
                "evaluated_rule_index",
                "retained_neuron_count",
                "retained_weight_count",
                "layer_index",
                "active_token_count",
                "exclusive_token_count",
                "num_samples",
                "sample_index",
                "num_total_values",
                "num_positive_values",
                "num_sampled_values",
            ):
                if key in row and row[key] != "":
                    row[key] = int(float(row[key]))
            rows.append(row)
    return rows


def validate_modularity_cache(rows: list[dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    metrics = {str(row.get("modularity_metric", "") or "").lower().strip() for row in rows}
    wanted = "quantile_rule_count_purity"
    if metrics != {wanted}:
        raise RuntimeError(
            "{} was not computed with modularity_metric={}. Re-run without --only-plot to regenerate structural M.".format(
                csv_path, wanted
            )
        )

def validate_panel_a_cache(rows: list[dict[str, Any]], args: argparse.Namespace, csv_path: Path) -> None:
    if not rows:
        return
    modes = {str(row.get("panel_a_cutoff_mode", "fixed") or "fixed").lower().strip() for row in rows}
    wanted_mode = str(args.cutoff_mode).lower().strip()
    if modes != {wanted_mode}:
        raise RuntimeError(
            "{} was computed with cutoff mode {}, but the current request uses {}. Re-run without --only-plot or use the matching --cutoff-mode.".format(
                csv_path, sorted(modes), wanted_mode
            )
        )
    methods = {str(row.get("token_extraction_method", "pt") or "pt").lower().strip() for row in rows}
    wanted_method = ana.validate_token_extraction_method(args.token_extraction_method)
    if methods != {wanted_method}:
        raise RuntimeError(
            "{} was computed with token_extraction_method={}, but the current request uses {}. Re-run without --only-plot or use the matching --token-extraction-method.".format(
                csv_path, sorted(methods), wanted_method
            )
        )
    fractions = [float(row["token_extraction_fraction"]) for row in rows if "token_extraction_fraction" in row and row.get("token_extraction_fraction") not in {"", None}]
    if fractions:
        unique_fractions = sorted(set(round(value, 10) for value in fractions))
        wanted_fraction = round(float(args.token_extraction_fraction), 10)
        if unique_fractions != [wanted_fraction]:
            raise RuntimeError(
                "{} was computed with token_extraction_fraction={}, but the current request uses {}. Re-run without --only-plot or use the matching --token-extraction-fraction.".format(
                    csv_path, unique_fractions, args.token_extraction_fraction
                )
            )
    if wanted_mode == "median":
        return
    values = [float(row["panel_a_t"]) for row in rows if "panel_a_t" in row and row.get("panel_a_t") not in {"", None}]
    if values:
        unique_values = sorted(set(round(value, 10) for value in values))
        if len(unique_values) != 1 or not math.isclose(unique_values[0], float(args.panel_a_t), abs_tol=1e-9):
            raise RuntimeError(
                "{} was computed with panel_a_t={}, but the current request uses panel_a_t={}. Re-run without --only-plot or use the matching --panel-a-t.".format(
                    csv_path, unique_values, args.panel_a_t
                )
            )
        return
    if args.token_type == "weight" and not math.isclose(float(args.panel_a_t), 4.0, abs_tol=1e-9):
        raise RuntimeError(
            "{} has no panel_a_t metadata and is likely the old weight t=4 cache. Re-run without --only-plot to compute weight tokens with panel_a_t={}.".format(
                csv_path, args.panel_a_t
            )
        )


def retained_rule_index(row: dict[str, Any]) -> int:
    return int(row.get("retained_rule_index", row.get("rule_index")))


def evaluated_rule_index(row: dict[str, Any]) -> int:
    return int(row.get("evaluated_rule_index", retained_rule_index(row)))


def samples_for_group_hue(
    rows: list[dict[str, Any]],
    groups: list[float],
    hues: list[int],
    value_key: str,
    *,
    gamma: float,
) -> list[list[np.ndarray]]:
    samples = []
    own_rule_only = value_key.startswith(("retained_neuron", "retained_weight"))
    for group in groups:
        group_samples = []
        for hue in hues:
            values = []
            for row in rows:
                if not (
                    math.isclose(float(row["gamma"]), float(gamma), abs_tol=1e-9)
                    and math.isclose(float(row["lasso"]), float(group), abs_tol=1e-9)
                    and retained_rule_index(row) == int(hue)
                ):
                    continue
                if own_rule_only and evaluated_rule_index(row) != retained_rule_index(row):
                    continue
                values.append(float(row[value_key]))
            group_samples.append(np.asarray(values, dtype=float))
        samples.append(group_samples)
    return samples


def demo_accuracy_samples_for_lasso(
    rows: list[dict[str, Any]],
    lasso_value: float,
    retained_rules: list[int],
    evaluated_rules: list[int],
    *,
    gamma: float,
) -> list[list[np.ndarray]]:
    samples = []
    for retained_rule in retained_rules:
        group_samples = []
        for evaluated_rule in evaluated_rules:
            values = []
            for row in rows:
                if not (
                    math.isclose(float(row["gamma"]), float(gamma), abs_tol=1e-9)
                    and math.isclose(float(row["lasso"]), float(lasso_value), abs_tol=1e-9)
                    and retained_rule_index(row) == int(retained_rule)
                    and evaluated_rule_index(row) == int(evaluated_rule)
                ):
                    continue
                values.append(float(row["relative_accuracy"]))
            group_samples.append(np.asarray(values, dtype=float))
        samples.append(group_samples)
    return samples


def samples_for_gamma_hue(
    rows: list[dict[str, Any]],
    gamma_values: list[float],
    hues: list[int],
    value_key: str,
    *,
    lasso: float,
) -> list[list[np.ndarray]]:
    samples = []
    own_rule_only = value_key.startswith(("retained_neuron", "retained_weight"))
    for gamma_value in gamma_values:
        group_samples = []
        for hue in hues:
            values = []
            for row in rows:
                if not (
                    math.isclose(float(row["gamma"]), float(gamma_value), abs_tol=1e-9)
                    and math.isclose(float(row["lasso"]), float(lasso), abs_tol=1e-9)
                    and retained_rule_index(row) == int(hue)
                ):
                    continue
                if own_rule_only and evaluated_rule_index(row) != retained_rule_index(row):
                    continue
                values.append(float(row[value_key]))
            group_samples.append(np.asarray(values, dtype=float))
        samples.append(group_samples)
    return samples


def demo_accuracy_samples_for_gamma(
    rows: list[dict[str, Any]],
    gamma_value: float,
    lasso_value: float,
    retained_rules: list[int],
    evaluated_rules: list[int],
) -> list[list[np.ndarray]]:
    samples = []
    for retained_rule in retained_rules:
        group_samples = []
        for evaluated_rule in evaluated_rules:
            values = []
            for row in rows:
                if not (
                    math.isclose(float(row["gamma"]), float(gamma_value), abs_tol=1e-9)
                    and math.isclose(float(row["lasso"]), float(lasso_value), abs_tol=1e-9)
                    and retained_rule_index(row) == int(retained_rule)
                    and evaluated_rule_index(row) == int(evaluated_rule)
                ):
                    continue
                values.append(float(row["relative_accuracy"]))
            group_samples.append(np.asarray(values, dtype=float))
        samples.append(group_samples)
    return samples


def rule_handles(rule_labels: list[str]) -> list[plt.Line2D]:
    return [
        plt.Line2D([0], [0], color=RULE_COLORS[idx % len(RULE_COLORS)], linewidth=5, alpha=0.78, label=label)
        for idx, label in enumerate(rule_labels)
    ]


def finite_values_from_samples(samples: list[list[np.ndarray]]) -> np.ndarray:
    chunks = []
    for group in samples:
        for values in group:
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                chunks.append(arr)
    if not chunks:
        return np.zeros(0, dtype=float)
    return np.concatenate(chunks)


def count_ylim(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    upper = float(np.nanmax(values))
    if upper <= 10:
        step = 1.0
    elif upper <= 50:
        step = 5.0
    elif upper <= 200:
        step = 20.0
    elif upper <= 800:
        step = 100.0
    else:
        step = 250.0
    return 0.0, float(np.ceil((upper + step * 0.08) / step) * step)


def ratio_ylim_ticks(values: np.ndarray) -> tuple[tuple[float, float], np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0), np.linspace(0.0, 1.0, 6)
    upper = max(0.0, float(np.nanmax(values)))
    if upper <= 0.1:
        step = 0.02
    elif upper <= 0.25:
        step = 0.05
    elif upper <= 0.5:
        step = 0.1
    else:
        step = 0.2
    ymax = float(np.ceil((upper + step * 0.45) / step) * step)
    ymax = min(1.0, max(step, ymax))
    ticks = np.round(np.arange(0.0, ymax + step * 0.5, step), 2)
    return (0.0, ymax), ticks


def style_violin_parts(parts: dict[str, Any], body_ids: list[int], colors: list[str]) -> None:
    for body, idx in zip(parts["bodies"], body_ids):
        color = colors[idx % len(colors)]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.68)
        body.set_linewidth(0.85)
    for key, color, width in (("cmeans", "#222222", 0.9), ("cmedians", "#B23A48", 1.15)):
        if key in parts:
            parts[key].set_color(color)
            parts[key].set_linewidth(width)


def finite_sample_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def visible_violin_epsilon(
    values: np.ndarray,
    *,
    accuracy_axis: bool = False,
    modularity_axis: bool = False,
    ratio_axis: bool = False,
) -> float:
    finite = finite_sample_values(values)
    if finite.size == 0:
        return 1e-6
    span = float(np.nanmax(finite) - np.nanmin(finite))
    floor = 0.006 if (accuracy_axis or modularity_axis or ratio_axis) else 0.02
    return max(floor, span * 0.012)


def overlay_violin_observations(
    ax: plt.Axes,
    raw_samples: list[np.ndarray],
    positions: list[float],
    body_ids: list[int],
    colors: list[str],
    violin_width: float,
    *,
    seed: int = 20260819,
) -> None:
    rng = np.random.default_rng(seed)
    max_points = 18
    jitter = min(violin_width * 0.22, 0.026)
    half_line = max(violin_width * 0.34, 0.025)
    for raw, position, idx in zip(raw_samples, positions, body_ids):
        values = finite_sample_values(raw)
        if values.size == 0:
            continue
        shown = values
        if values.size > max_points:
            shown = rng.choice(values, size=max_points, replace=False)
        if shown.size:
            xs = np.full(shown.size, position, dtype=float)
            if jitter > 0.0:
                xs = xs + rng.uniform(-jitter, jitter, size=shown.size)
            ax.scatter(
                xs,
                shown,
                s=5.5,
                color=colors[idx % len(colors)],
                alpha=0.34,
                edgecolors="none",
                linewidths=0.0,
                zorder=3,
            )
        ax.hlines(float(np.nanmean(values)), position - half_line, position + half_line, color="#222222", linewidth=0.85, zorder=4)
        ax.hlines(float(np.nanmedian(values)), position - half_line, position + half_line, color="#B23A48", linewidth=1.05, zorder=4)


def draw_grouped_violin_samples(
    ax: plt.Axes,
    samples: list[list[np.ndarray]],
    group_labels: list[str],
    hue_labels: list[str],
    colors: list[str],
    *,
    xlabel: str,
    ylabel: str,
    legend_title: str | None = None,
    panel_label: str | None = None,
    accuracy_axis: bool = False,
    modularity_axis: bool = False,
    ratio_axis: bool = False,
    rotate: float = 0.0,
    show_legend: bool = True,
    legend_anchor: tuple[float, float] = (0.5, 1.18),
    defer_ylim: bool = False,
) -> None:
    num_groups = len(samples)
    num_hues = len(hue_labels)
    finite = finite_values_from_samples(samples)
    if finite.size == 0:
        raise ValueError("No finite values to plot")
    epsilon = visible_violin_epsilon(
        finite,
        accuracy_axis=accuracy_axis,
        modularity_axis=modularity_axis,
        ratio_axis=ratio_axis,
    )
    group_gap = 1.55 if num_hues <= 3 else 1.75
    group_width = 0.50 if num_hues <= 3 else 0.68
    offsets = np.linspace(-group_width / 2.0, group_width / 2.0, num_hues) if num_hues > 1 else np.asarray([0.0])
    violin_width = min(0.18, group_width / max(num_hues + 0.95, 1.0))
    if num_hues <= 3:
        violin_width = min(0.22, violin_width * 1.15)
    centers = np.arange(num_groups, dtype=float) * group_gap
    data = []
    raw_samples = []
    positions = []
    body_ids = []
    for group_idx, group in enumerate(samples):
        for hue_idx, values in enumerate(group):
            raw = finite_sample_values(values)
            display_values = ana._violin_values(raw, epsilon)
            if display_values.size:
                data.append(display_values)
                raw_samples.append(raw)
                positions.append(float(centers[group_idx] + offsets[hue_idx]))
                body_ids.append(hue_idx)
    if data:
        parts = ax.violinplot(data, positions=positions, widths=violin_width, showmeans=False, showmedians=False, showextrema=False)
        style_violin_parts(parts, body_ids, colors)
        overlay_violin_observations(ax, raw_samples, positions, body_ids, colors, violin_width)
    xpad = 0.62 if num_hues > 3 else 0.54
    ax.set_xlim(float(centers[0] - xpad), float(centers[-1] + xpad))
    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels, rotation=rotate, ha="right" if rotate else "center")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if accuracy_axis:
        lower, upper = ana._retained_ratio_ylim(finite)
        ax.set_ylim(lower, upper)
        tick_step = 0.1 if upper - lower <= 1.2 else 0.2
        ax.set_yticks(np.round(np.arange(lower, upper + tick_step * 0.5, tick_step), 2))
        if lower <= 1.0 <= upper:
            ax.axhline(1.0, color="#666666", linewidth=0.75, linestyle="--", alpha=0.42, zorder=0)
    elif modularity_axis:
        ax.set_ylim(0.0, 1.02)
        ax.set_yticks(np.linspace(0, 1.0, 6))
    elif ratio_axis:
        limits, ticks = ratio_ylim_ticks(finite)
        ax.set_ylim(*limits)
        ax.set_yticks(ticks)
    elif not defer_ylim:
        lower, upper = count_ylim(finite)
        ax.set_ylim(lower, upper)
    ana._style_publication_axes(ax, grid=True)
    if panel_label:
        ax.text(-0.06, 1.045, panel_label, transform=ax.transAxes, fontweight="bold", fontsize=14, ha="left", va="bottom")
    if show_legend:
        handles = [plt.Line2D([0], [0], color=colors[idx % len(colors)], linewidth=5, alpha=0.78, label=hue_labels[idx]) for idx in range(num_hues)]
        ax.legend(
            handles=handles,
            title=legend_title,
            loc="upper center",
            bbox_to_anchor=legend_anchor,
            ncol=min(num_hues, 5),
            frameon=False,
            handlelength=0.95,
            handletextpad=0.35,
            columnspacing=0.62,
            borderaxespad=0.0,
        )


def threshold_curve_stats(rows: list[dict[str, Any]], gamma: float, lasso: float, layer_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_t: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if (
            math.isclose(float(row["gamma"]), float(gamma), abs_tol=1e-9)
            and math.isclose(float(row["lasso"]), float(lasso), abs_tol=1e-9)
            and int(row["layer_index"]) == int(layer_index)
        ):
            value = float(row["modularity"])
            if np.isfinite(value):
                by_t[float(row["t"])].append(value)
    ts = np.asarray(sorted(by_t), dtype=float)
    if ts.size == 0:
        return ts, np.zeros(0, dtype=float), np.zeros(0, dtype=float)
    means = np.asarray([np.mean(by_t[t]) for t in ts], dtype=float)
    sems = np.asarray([np.std(by_t[t], ddof=1) / math.sqrt(len(by_t[t])) if len(by_t[t]) > 1 else 0.0 for t in ts], dtype=float)
    return ts, means, sems


def modularity_samples(rows: list[dict[str, Any]], gamma: float, lasso: float, layer_index: int) -> np.ndarray:
    by_run: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if (
            math.isclose(float(row["gamma"]), float(gamma), abs_tol=1e-9)
            and math.isclose(float(row["lasso"]), float(lasso), abs_tol=1e-9)
            and int(row["layer_index"]) == int(layer_index)
        ):
            value = float(row["modularity"])
            if np.isfinite(value):
                by_run[str(row["run"])].append(value)
    return np.asarray([float(np.mean(values)) for values in by_run.values() if values], dtype=float)


def set_shared_accuracy_axis(axes: list[plt.Axes], samples_list: list[list[list[np.ndarray]]]) -> None:
    chunks = []
    for samples in samples_list:
        values = finite_values_from_samples(samples)
        if values.size:
            chunks.append(values)
    if not chunks:
        return
    finite = np.concatenate(chunks)
    lower, upper = ana._retained_ratio_ylim(finite)
    tick_step = 0.1 if upper - lower <= 1.2 else 0.2
    yticks = np.round(np.arange(lower, upper + tick_step * 0.5, tick_step), 2)
    for ax in axes:
        ax.set_ylim(lower, upper)
        ax.set_yticks(yticks)
        if lower <= 1.0 <= upper:
            ax.axhline(1.0, color="#666666", linewidth=0.75, linestyle="--", alpha=0.42, zorder=0)


def token_ratio_key(token_type: str) -> str:
    return "retained_weight_ratio" if token_type == "weight" else "retained_neuron_ratio"


def token_ratio_ylabel(token_type: str) -> str:
    return "retained weight ratio" if token_type == "weight" else "retained neuron ratio"


def extraction_suffix(token_extraction_method: str, token_extraction_fraction: float) -> str:
    method = ana.validate_token_extraction_method(token_extraction_method)
    if method == "pt":
        return ""
    return "_{}_frac{}".format(method, mt.tag_float(token_extraction_fraction))


def fig5a_stem(token_type: str, mask_method: str, cutoff_mode: str, token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD, token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION) -> str:
    suffix = "{}{}{}".format(method_suffix(mask_method), cutoff_suffix(cutoff_mode), extraction_suffix(token_extraction_method, token_extraction_fraction))
    if token_type == "weight":
        return "fig5a_lasso_accuracy_retained_weight_tokens{}".format(suffix)
    return "fig5a_lasso_accuracy_retained_tokens{}".format(suffix)


def fig5a_gamma_stem(token_type: str, mask_method: str, cutoff_mode: str, token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD, token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION) -> str:
    suffix = "{}{}{}".format(method_suffix(mask_method), cutoff_suffix(cutoff_mode), extraction_suffix(token_extraction_method, token_extraction_fraction))
    token_name = "weight" if token_type == "weight" else "neuron"
    return "fig5a_gamma_accuracy_retained_{}_ratio{}".format(token_name, suffix)


def metric_csv_paths(output_root: Path, token_type: str, mask_method: str, cutoff_mode: str, token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD, token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION) -> tuple[Path, Path]:
    suffix = "{}{}{}".format(method_suffix(mask_method), cutoff_suffix(cutoff_mode), extraction_suffix(token_extraction_method, token_extraction_fraction))
    if token_type == "weight":
        return output_root / "figure5_weight_single_rule_metrics{}.csv".format(suffix), output_root / "figure5_weight_threshold_metrics{}.csv".format(suffix)
    return output_root / "figure5_single_rule_metrics{}.csv".format(suffix), output_root / "figure5_threshold_metrics{}.csv".format(suffix)


def score_distribution_csv_path(output_root: Path, mask_method: str) -> Path:
    return output_root / "figure5_score_distribution_samples{}.csv".format(method_suffix(mask_method))


def score_distribution_stem(value_kind: str, mask_method: str) -> str:
    prefix = "score_ratio" if value_kind == "score_ratio" else "score"
    return "fig5d_{}_distribution{}".format(prefix, method_suffix(mask_method))


def score_distribution_ylabel(value_kind: str) -> str:
    return "log10 score ratio" if value_kind == "score_ratio" else "log10 score"


def threshold_mask_note(token_type: str, mask_method: str, cutoff_mode: str, token_extraction_method: str) -> str:
    if token_extraction_method != "pt":
        return (
            "Fig. 5a: single-rule functional tokens use {} {} masks; relative accuracy is stored as a retained rule set x evaluated rule matrix, each entry divided by the corresponding full-network rule accuracy."
        ).format(token_type, token_extraction_method)
    if mask_method == "legacy_hessian":
        return (
            "Fig. 5a: single-rule functional tokens use {} raw legacy-Hessian score masks at "
            "the configured fixed cutoff; relative accuracy is stored as a retained rule set x evaluated "
            "rule matrix, each entry divided by the corresponding full-network rule accuracy."
        ).format(token_type)
    return (
        "Fig. 5a: single-rule functional tokens use {} legacy pt masks with {} cutoff; "
        "relative accuracy is stored as a retained rule set x evaluated rule matrix, each entry "
        "divided by the corresponding full-network rule accuracy."
    ).format(token_type, "median positive normalized-score" if cutoff_mode == "median" else "the configured fixed cutoff")


def plot_fig5a(single_rows: list[dict[str, Any]], lasso_values: list[float], output_dir: Path, formats: list[str], token_type: str, mask_method: str, cutoff_mode: str, token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD, token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION) -> list[Path]:
    rule_ids = [1, 2, 3, 4, 5]
    rule_set_labels = [str(idx) for idx in rule_ids]
    lasso_labels = ["l={}".format(format_float(value)) for value in lasso_values]
    rule_labels = ["rule {}".format(idx) for idx in rule_ids]
    ratio_samples = samples_for_group_hue(single_rows, lasso_values, rule_ids, token_ratio_key(token_type), gamma=DEFAULT_GAMMA_FOR_LASSO)

    fig = plt.figure(figsize=(7.5, 4.75))
    grid = fig.add_gridspec(2, len(lasso_values), height_ratios=[1.16, 0.84], hspace=0.50, wspace=0.30)
    accuracy_axes = []
    accuracy_samples_by_lasso = []
    for idx, lasso_value in enumerate(lasso_values):
        ax = fig.add_subplot(grid[0, idx])
        samples = demo_accuracy_samples_for_lasso(single_rows, lasso_value, rule_ids, rule_ids, gamma=DEFAULT_GAMMA_FOR_LASSO)
        accuracy_samples_by_lasso.append(samples)
        accuracy_axes.append(ax)
        draw_grouped_violin_samples(
            ax,
            samples,
            rule_set_labels,
            rule_labels,
            RULE_COLORS,
            xlabel="rule set",
            ylabel="relative accuracy" if idx == 0 else "",
            panel_label="a" if idx == 0 else None,
            accuracy_axis=True,
            show_legend=False,
        )
        ax.set_title(lasso_labels[idx], loc="right", fontsize=14, pad=2.2)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)
    set_shared_accuracy_axis(accuracy_axes, accuracy_samples_by_lasso)

    ax_ratio = fig.add_subplot(grid[1, :])
    draw_grouped_violin_samples(
        ax_ratio,
        ratio_samples,
        lasso_labels,
        rule_labels,
        RULE_COLORS,
        xlabel="Lasso strength l",
        ylabel=token_ratio_ylabel(token_type),
        ratio_axis=True,
        show_legend=False,
    )
    fig.legend(
        handles=rule_handles(rule_labels),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        handlelength=0.95,
        handletextpad=0.35,
        columnspacing=0.62,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.80, bottom=0.15)
    paths = export_figure(fig, output_dir, fig5a_stem(token_type, mask_method, cutoff_mode, token_extraction_method, token_extraction_fraction), formats)
    plt.close(fig)
    return paths


def plot_fig5a_gamma(
    single_rows: list[dict[str, Any]],
    gamma_values: list[float],
    gamma_lasso: float,
    output_dir: Path,
    formats: list[str],
    token_type: str,
    mask_method: str,
    cutoff_mode: str,
    token_extraction_method: str = DEFAULT_TOKEN_EXTRACTION_METHOD,
    token_extraction_fraction: float = DEFAULT_TOKEN_EXTRACTION_FRACTION,
) -> list[Path]:
    rule_ids = [1, 2, 3, 4, 5]
    rule_set_labels = [str(idx) for idx in rule_ids]
    gamma_labels = [r"$\gamma={}$".format(format_float(value)) for value in gamma_values]
    rule_labels = ["rule {}".format(idx) for idx in rule_ids]
    ratio_samples = samples_for_gamma_hue(single_rows, gamma_values, rule_ids, token_ratio_key(token_type), lasso=gamma_lasso)

    fig = plt.figure(figsize=(7.5, 4.75))
    grid = fig.add_gridspec(2, len(gamma_values), height_ratios=[1.16, 0.84], hspace=0.50, wspace=0.30)
    accuracy_axes = []
    accuracy_samples_by_gamma = []
    for idx, gamma_value in enumerate(gamma_values):
        ax = fig.add_subplot(grid[0, idx])
        samples = demo_accuracy_samples_for_gamma(single_rows, gamma_value, gamma_lasso, rule_ids, rule_ids)
        accuracy_samples_by_gamma.append(samples)
        accuracy_axes.append(ax)
        draw_grouped_violin_samples(
            ax,
            samples,
            rule_set_labels,
            rule_labels,
            RULE_COLORS,
            xlabel="rule set",
            ylabel="relative accuracy" if idx == 0 else "",
            panel_label="a" if idx == 0 else None,
            accuracy_axis=True,
            show_legend=False,
        )
        ax.set_title(gamma_labels[idx], loc="right", fontsize=14, pad=2.2)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)
    set_shared_accuracy_axis(accuracy_axes, accuracy_samples_by_gamma)

    ax_ratio = fig.add_subplot(grid[1, :])
    draw_grouped_violin_samples(
        ax_ratio,
        ratio_samples,
        gamma_labels,
        rule_labels,
        RULE_COLORS,
        xlabel=r"regularizer $L_\gamma$",
        ylabel=token_ratio_ylabel(token_type),
        ratio_axis=True,
        show_legend=False,
    )
    fig.legend(
        handles=rule_handles(rule_labels),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        handlelength=0.95,
        handletextpad=0.35,
        columnspacing=0.62,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.80, bottom=0.15)
    paths = export_figure(fig, output_dir, fig5a_gamma_stem(token_type, mask_method, cutoff_mode, token_extraction_method, token_extraction_fraction), formats)
    plt.close(fig)
    return paths


def plot_fig5b(threshold_rows: list[dict[str, Any]], lasso_values: list[float], output_dir: Path, formats: list[str], mask_method: str) -> list[Path]:
    layers = [1, 2, 3]
    samples = [
        [modularity_samples(threshold_rows, DEFAULT_GAMMA_FOR_LASSO, lasso, layer) for layer in layers]
        for lasso in lasso_values
    ]
    fig, ax = plt.subplots(figsize=(4.15, 2.55))
    draw_grouped_violin_samples(
        ax,
        samples,
        ["l={}".format(format_float(value)) for value in lasso_values],
        ["layer{}".format(idx) for idx in layers],
        LAYER_COLORS,
        xlabel="Lasso strength l",
        ylabel=r"$M$",
        panel_label="b",
        modularity_axis=True,
        show_legend=True,
        legend_anchor=(0.5, 1.105),
    )
    fig.subplots_adjust(left=0.20, right=0.985, top=0.76, bottom=0.27)
    paths = export_figure(fig, output_dir, "fig5b_lasso_m{}".format(method_suffix(mask_method)), formats)
    plt.close(fig)
    return paths


def plot_fig5c(threshold_rows: list[dict[str, Any]], gamma_values: list[float], gamma_lasso: float, output_dir: Path, formats: list[str], mask_method: str) -> list[Path]:
    layers = [1, 2, 3]
    samples = []
    for gamma in gamma_values:
        group = []
        for layer in layers:
            group.append(modularity_samples(threshold_rows, gamma, gamma_lasso, layer))
        samples.append(group)
    fig, ax = plt.subplots(figsize=(4.15, 2.55))
    draw_grouped_violin_samples(
        ax,
        samples,
        [r"$\gamma={}$".format(format_float(value)) for value in gamma_values],
        ["layer{}".format(idx) for idx in layers],
        LAYER_COLORS,
        xlabel=r"regularizer $L_\gamma$",
        ylabel=r"$M$",
        legend_title=None,
        panel_label="c",
        modularity_axis=True,
        show_legend=True,
        legend_anchor=(0.5, 1.105),
    )
    ax.set_title("l={}".format(format_float(gamma_lasso)), loc="right", fontsize=14, pad=2.2)
    fig.subplots_adjust(left=0.20, right=0.985, top=0.76, bottom=0.27)
    paths = export_figure(fig, output_dir, "fig5c_gamma_m{}".format(method_suffix(mask_method)), formats)
    plt.close(fig)
    return paths


def score_distribution_samples(
    score_rows: list[dict[str, Any]],
    token_type: str,
    lasso_values: list[float],
    layers: list[int],
    gamma: float = DEFAULT_GAMMA_FOR_LASSO,
) -> list[list[np.ndarray]]:
    samples: list[list[np.ndarray]] = []
    for lasso_value in lasso_values:
        group = []
        for layer_index in layers:
            values = []
            for row in score_rows:
                if (
                    str(row.get("token_type", "")) == str(token_type)
                    and math.isclose(float(row.get("gamma", float("nan"))), float(gamma), abs_tol=1e-9)
                    and math.isclose(float(row.get("lasso", float("nan"))), float(lasso_value), abs_tol=1e-9)
                    and int(row.get("layer_index", -1)) == int(layer_index)
                ):
                    value = float(row.get("log10_value", float("nan")))
                    if np.isfinite(value):
                        values.append(value)
            group.append(np.asarray(values, dtype=float))
        samples.append(group)
    return samples


def score_log_ylim(score_rows: list[dict[str, Any]], value_kind: str) -> tuple[float, float]:
    values = [float(row["log10_value"]) for row in score_rows if "log10_value" in row and np.isfinite(float(row["log10_value"]))]
    if value_kind == "score_ratio":
        values.extend([-DEFAULT_NEURON_T_FOR_PANEL_A, -DEFAULT_WEIGHT_T_FOR_PANEL_A])
    if not values:
        return -8.0, 0.0
    finite = np.asarray(values, dtype=float)
    lower = float(np.floor(np.nanmin(finite) - 0.25))
    upper = float(np.ceil(np.nanmax(finite) + 0.25))
    if upper <= lower:
        upper = lower + 1.0
    return lower, upper


def score_log_ticks(limits: tuple[float, float]) -> np.ndarray:
    lower, upper = limits
    span = float(upper - lower)
    if span <= 8.0:
        return np.arange(math.ceil(lower), math.floor(upper) + 1, 1.0)
    return np.round(np.linspace(lower, upper, 6), 1)


def add_score_cutoff_reference(ax: plt.Axes, token_type: str, value_kind: str, limits: tuple[float, float]) -> None:
    if value_kind != "score_ratio":
        return
    threshold_t = DEFAULT_WEIGHT_T_FOR_PANEL_A if token_type == "weight" else DEFAULT_NEURON_T_FOR_PANEL_A
    y_value = -float(threshold_t)
    if limits[0] <= y_value <= limits[1]:
        ax.axhline(y_value, color="#555555", linewidth=0.72, linestyle="--", alpha=0.55, zorder=0)
        ax.text(
            0.992,
            y_value,
            r"$10^{-%g}$" % float(threshold_t),
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=14,
            color="#555555",
        )


def plot_score_distribution(
    score_rows: list[dict[str, Any]],
    lasso_values: list[float],
    output_dir: Path,
    formats: list[str],
    value_kind: str,
    mask_method: str,
) -> list[Path]:
    token_order = [token_type for token_type in ("neuron", "weight") if any(str(row.get("token_type", "")) == token_type for row in score_rows)]
    if not token_order or not lasso_values:
        return []
    layers = [1, 2, 3]
    lasso_labels = ["l={}".format(format_float(value)) for value in lasso_values]
    layer_labels = ["layer{}".format(idx) for idx in layers]
    y_limits = score_log_ylim(score_rows, value_kind)
    fig, axes_grid = plt.subplots(len(token_order), 1, figsize=(7.5, 2.55 * len(token_order) + 0.3), sharex=True)
    axes = np.atleast_1d(axes_grid).ravel()
    for idx, (ax, token_type) in enumerate(zip(axes, token_order)):
        samples = score_distribution_samples(score_rows, token_type, lasso_values, layers)
        draw_grouped_violin_samples(
            ax,
            samples,
            lasso_labels,
            layer_labels,
            LAYER_COLORS,
            xlabel="Lasso strength l",
            ylabel=score_distribution_ylabel(value_kind),
            panel_label="d" if idx == 0 else None,
            show_legend=False,
            defer_ylim=True,
        )
        ax.set_ylim(*y_limits)
        ax.set_yticks(score_log_ticks(y_limits))
        add_score_cutoff_reference(ax, token_type, value_kind, y_limits)
        ax.set_title("{} tokens".format(token_type), loc="left", fontweight="bold", pad=3)
        if idx < len(token_order) - 1:
            ax.set_xlabel("")
    handles = [plt.Line2D([0], [0], color=LAYER_COLORS[idx], linewidth=5.0, alpha=0.82, label=layer_labels[idx]) for idx in range(len(layers))]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=3, handlelength=1.0, columnspacing=0.65)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.80, bottom=0.15, hspace=0.38)
    paths = export_figure(fig, output_dir, score_distribution_stem(value_kind, mask_method), formats)
    plt.close(fig)
    return paths


def plot_combined_fig5(
    single_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    lasso_a: list[float],
    lasso_b: list[float],
    gamma_values: list[float],
    gamma_lasso: float,
    output_dir: Path,
    formats: list[str],
    token_type: str,
    mask_method: str,
    cutoff_mode: str,
) -> list[Path]:
    fig = plt.figure(figsize=(7.8, 8.95))
    outer = fig.add_gridspec(
        6,
        2,
        height_ratios=[1.0, 0.46, 1.0, 0.46, 0.16, 0.82],
        width_ratios=[1.0, 1.0],
        hspace=0.50,
        wspace=0.38,
    )
    rule_ids = [1, 2, 3, 4, 5]
    rule_set_labels = [str(idx) for idx in rule_ids]
    lasso_labels = ["l={}".format(format_float(value)) for value in lasso_a]
    gamma_labels = [r"$\gamma={}$".format(format_float(value)) for value in gamma_values]
    rule_labels = ["rule {}".format(idx) for idx in rule_ids]
    layer_labels = ["layer{}".format(idx) for idx in [1, 2, 3]]
    accuracy_axes = []
    accuracy_samples_all = []

    a_grid = outer[0, :].subgridspec(1, len(lasso_a), wspace=0.30)
    for idx, lasso_value in enumerate(lasso_a):
        ax = fig.add_subplot(a_grid[0, idx])
        samples = demo_accuracy_samples_for_lasso(single_rows, lasso_value, rule_ids, rule_ids, gamma=DEFAULT_GAMMA_FOR_LASSO)
        accuracy_samples_all.append(samples)
        accuracy_axes.append(ax)
        draw_grouped_violin_samples(
            ax,
            samples,
            rule_set_labels,
            rule_labels,
            RULE_COLORS,
            xlabel="rule set",
            ylabel="relative accuracy" if idx == 0 else "",
            panel_label="a" if idx == 0 else None,
            accuracy_axis=True,
            show_legend=False,
        )
        ax.set_title(lasso_labels[idx], loc="right", fontsize=14, pad=2.0)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    ax_a2 = fig.add_subplot(outer[1, :])
    draw_grouped_violin_samples(
        ax_a2,
        samples_for_group_hue(single_rows, lasso_a, rule_ids, token_ratio_key(token_type), gamma=DEFAULT_GAMMA_FOR_LASSO),
        lasso_labels,
        rule_labels,
        RULE_COLORS,
        xlabel="Lasso strength l",
        ylabel=token_ratio_ylabel(token_type),
        ratio_axis=True,
        show_legend=False,
    )

    b_grid = outer[2, :].subgridspec(1, len(gamma_values), wspace=0.30)
    for idx, gamma_value in enumerate(gamma_values):
        ax = fig.add_subplot(b_grid[0, idx])
        samples = demo_accuracy_samples_for_gamma(single_rows, gamma_value, gamma_lasso, rule_ids, rule_ids)
        accuracy_samples_all.append(samples)
        accuracy_axes.append(ax)
        draw_grouped_violin_samples(
            ax,
            samples,
            rule_set_labels,
            rule_labels,
            RULE_COLORS,
            xlabel="rule set",
            ylabel="relative accuracy" if idx == 0 else "",
            panel_label="b" if idx == 0 else None,
            accuracy_axis=True,
            show_legend=False,
        )
        ax.set_title(gamma_labels[idx], loc="right", fontsize=14, pad=2.0)
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)
    set_shared_accuracy_axis(accuracy_axes, accuracy_samples_all)

    ax_b2 = fig.add_subplot(outer[3, :])
    draw_grouped_violin_samples(
        ax_b2,
        samples_for_gamma_hue(single_rows, gamma_values, rule_ids, token_ratio_key(token_type), lasso=gamma_lasso),
        gamma_labels,
        rule_labels,
        RULE_COLORS,
        xlabel=r"regularizer $L_\gamma$",
        ylabel=token_ratio_ylabel(token_type),
        ratio_axis=True,
        show_legend=False,
    )

    fig.legend(
        handles=rule_handles(rule_labels),
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        handlelength=0.95,
        handletextpad=0.35,
        columnspacing=0.62,
    )

    ax_c = fig.add_subplot(outer[5, 0])
    lasso_b_samples = [[modularity_samples(threshold_rows, DEFAULT_GAMMA_FOR_LASSO, lasso, layer) for layer in [1, 2, 3]] for lasso in lasso_b]
    draw_grouped_violin_samples(
        ax_c,
        lasso_b_samples,
        ["l={}".format(format_float(value)) for value in lasso_b],
        layer_labels,
        LAYER_COLORS,
        xlabel="Lasso strength l",
        ylabel=r"$M$",
        panel_label="c",
        modularity_axis=True,
        show_legend=False,
    )

    ax_d = fig.add_subplot(outer[5, 1])
    gamma_samples = [[modularity_samples(threshold_rows, gamma, gamma_lasso, layer) for layer in [1, 2, 3]] for gamma in gamma_values]
    draw_grouped_violin_samples(
        ax_d,
        gamma_samples,
        gamma_labels,
        layer_labels,
        LAYER_COLORS,
        xlabel=r"regularizer $L_\gamma$",
        ylabel=r"$M$",
        panel_label="d",
        modularity_axis=True,
        show_legend=False,
    )
    ax_layer_legend = fig.add_subplot(outer[4, :])
    ax_layer_legend.axis("off")
    layer_handles = [
        plt.Line2D([0], [0], color=LAYER_COLORS[idx], linewidth=5.0, alpha=0.78, label=layer_labels[idx])
        for idx in range(len(layer_labels))
    ]
    fig.legend(
        handles=layer_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.268),
        ncol=3,
        handlelength=1.0,
        handletextpad=0.35,
        columnspacing=0.72,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.105, right=0.995, top=0.895, bottom=0.078)
    paths = export_figure(fig, output_dir, "fig5{}{}".format(method_suffix(mask_method), cutoff_suffix(cutoff_mode)), formats)
    plt.close(fig)
    return paths


def required_conditions(
    args: argparse.Namespace,
    lasso_a: list[float],
    lasso_b: list[float],
    gamma_values: list[float],
    score_lasso: list[float],
) -> tuple[dict[tuple[float, float], tuple[float, Path]], list[dict[str, Any]]]:
    wanted: dict[tuple[float, float], tuple[float, Path]] = {}
    missing_rows: list[dict[str, Any]] = []
    gamma_one_lasso = sorted(set(lasso_a + lasso_b + score_lasso + [float(args.gamma_lasso)]))
    selected, missing = selected_for_gamma(args.summary, DEFAULT_GAMMA_FOR_LASSO, gamma_one_lasso, args.fig_l_scale)
    for fig_lasso, value in selected.items():
        wanted[(DEFAULT_GAMMA_FOR_LASSO, float(fig_lasso))] = value
    for value in missing:
        missing_rows.append({"gamma": DEFAULT_GAMMA_FOR_LASSO, "lasso": float(value)})

    for gamma in gamma_values:
        selected_g, missing_g = selected_for_gamma(args.summary, float(gamma), [float(args.gamma_lasso)], args.fig_l_scale)
        for fig_lasso, value in selected_g.items():
            wanted[(float(gamma), float(fig_lasso))] = value
        for value in missing_g:
            missing_rows.append({"gamma": float(gamma), "lasso": float(value)})
    return dict(sorted(wanted.items())), missing_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and draw article Fig. 5 from lasso/gamma sweep snapshots.")
    parser.add_argument("--summary", type=normal_path, default=SWEEP_SUMMARY)
    parser.add_argument("--output-root", type=normal_path, default=OUTPUT_ROOT)
    parser.add_argument("--epoch", default="latest", help="Snapshot epoch such as 5000, or 'latest'.")
    parser.add_argument("--lasso-a", default=DEFAULT_LASSO_A, help="Figure/text l values for Fig. 5a.")
    parser.add_argument("--lasso-b", default=DEFAULT_LASSO_B, help="Figure/text l values for Fig. 5b M violin plot.")
    parser.add_argument("--gamma-values", default=DEFAULT_GAMMAS, help="Gamma values for Fig. 5c and the gamma accuracy plot.")
    parser.add_argument("--gamma-lasso", type=float, default=DEFAULT_GAMMA_LASSO, help="Figure/text l used for gamma comparison.")
    parser.add_argument("--fig-l-scale", type=float, default=DEFAULT_FIG_L_SCALE, help="Figure/text l = experiment l * this scale.")
    parser.add_argument("--t-min", type=float, default=DEFAULT_T_MIN)
    parser.add_argument("--t-max", type=float, default=DEFAULT_T_MAX)
    parser.add_argument("--threshold-count", type=int, default=DEFAULT_THRESHOLD_COUNT)
    parser.add_argument("--panel-a-t", type=float, default=None, help="Threshold t for Fig. 5a, used only by the legacy pt extraction mode. Defaults to 6 for weight tokens.")
    parser.add_argument("--token-type", choices=["neuron", "weight"], default=DEFAULT_TOKEN_TYPE, help="Elementary token type used for Fig. 5 metrics.")
    parser.add_argument("--cutoff-mode", choices=["fixed", "median"], default=DEFAULT_CUTOFF_MODE, help="Cutoff used for Fig. 5a retained functional tokens. fixed uses the configured legacy pt cutoff; median uses the per-rule positive normalized-score median within each layer/block.")
    parser.add_argument("--token-extraction-method", choices=ana.TOKEN_EXTRACTION_METHOD_CHOICES, default=DEFAULT_TOKEN_EXTRACTION_METHOD, help="Functional-token extraction method: pt uses the legacy cutoff mode; score_quantile keeps rule r when S(r, token) is at or above the per-rule quantile set by --token-extraction-fraction; score_median_fraction and score_mean_fraction keep tokens above fraction times the per-rule median or mean.")
    parser.add_argument("--token-extraction-fraction", "--rule-score-average-fraction", dest="token_extraction_fraction", type=float, default=DEFAULT_TOKEN_EXTRACTION_FRACTION, help="Fraction used by score_quantile as the per-rule quantile, or by score_median_fraction/score_mean_fraction as the scale factor.")
    parser.add_argument("--mask-method", choices=["ablation", "taylor", "legacy_hessian"], default=DEFAULT_MASK_METHOD, help="Scoring mode for functional tokens.")
    parser.add_argument("--legacy-hessian-epsilon", type=float, default=ana.LEGACY_HESSIAN_EPSILON)
    parser.add_argument("--taylor-probes", type=int, default=4)
    parser.add_argument("--max-weight-elements-per-block", type=int, default=None, help="Optional cap for weight ablation candidates per weight block.")
    parser.add_argument("--max-runs", type=int, default=None, help="Use only first N runs per condition for quick checks.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional deterministic subsample of x_test/y_test.")
    parser.add_argument("--importance-min-score", type=float, default=0)
    parser.add_argument("--importance-output-eps", type=float, default=1e-24)
    parser.add_argument("--perturbation-batch-size", type=int, default=64)
    parser.add_argument("--analysis-device", default="auto")
    parser.add_argument("--formats", default="png,svg,pdf")
    parser.add_argument("--score-lasso", default=DEFAULT_SCORE_DISTRIBUTION_LASSO, help="Figure/text l values used for the extra score-distribution plot.")
    parser.add_argument("--score-token-types", default=DEFAULT_SCORE_DISTRIBUTION_TOKEN_TYPES, help="Comma-separated token types for the score-distribution plot: neuron,weight.")
    parser.add_argument("--score-distribution-value", choices=["score", "score_ratio"], default=DEFAULT_SCORE_DISTRIBUTION_VALUE, help="Value plotted in the extra score-distribution figure.")
    parser.add_argument("--score-distribution-max-values", type=int, default=DEFAULT_SCORE_DISTRIBUTION_MAX_VALUES, help="Maximum positive score values sampled per run/layer/token type.")
    parser.add_argument("--score-distribution-floor", type=float, default=DEFAULT_SCORE_DISTRIBUTION_FLOOR, help="Smallest positive value used before log10 in the score-distribution plot.")
    parser.add_argument("--skip-score-distribution", action="store_true")
    parser.add_argument("--only-score-distribution", action="store_true", help="Only compute/regenerate the extra score-distribution plot.")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--only-plot", action="store_true", help="Read existing CSV files and only regenerate figures.")
    parser.add_argument("--dry-run", action="store_true", help="Only report selected conditions and run counts.")
    parser.add_argument("--skip-combined", action="store_true")
    args = parser.parse_args()
    args.mask_method = ana.validate_mask_method(args.mask_method)
    args.cutoff_mode = str(args.cutoff_mode).lower().strip()
    args.token_extraction_method = ana.validate_token_extraction_method(args.token_extraction_method)
    args.score_distribution_token_types = parse_token_types(args.score_token_types)
    if args.only_score_distribution and args.skip_score_distribution:
        raise ValueError("--only-score-distribution cannot be combined with --skip-score-distribution")
    if args.token_type == "weight" and args.mask_method == "legacy_hessian" and not args.only_score_distribution:
        raise ValueError("legacy_hessian is a neuron scoring mode; use --token-type neuron")
    if "weight" in args.score_distribution_token_types and args.mask_method == "legacy_hessian":
        raise ValueError("legacy_hessian is a neuron scoring mode; remove weight from --score-token-types")
    if args.panel_a_t is None:
        args.panel_a_t = DEFAULT_WEIGHT_T_FOR_PANEL_A if args.token_type == "weight" else DEFAULT_NEURON_T_FOR_PANEL_A

    args.output_root.mkdir(parents=True, exist_ok=True)
    configure_style()
    formats = parse_formats(args.formats)
    lasso_a = parse_float_list(args.lasso_a)
    lasso_b = parse_float_list(args.lasso_b)
    score_lasso = parse_float_list(args.score_lasso)
    gamma_values = parse_float_list(args.gamma_values)
    t_values = np.linspace(float(args.t_min), float(args.t_max), int(args.threshold_count))
    conditions, missing = required_conditions(args, lasso_a, lasso_b, gamma_values, score_lasso)

    print("Figure/text l = experiment l * {:g}".format(float(args.fig_l_scale)))
    if missing:
        print("Missing requested condition(s): {}".format(", ".join("gamma={:g}, l={:g}".format(row["gamma"], row["lasso"]) for row in missing)))
    for (gamma, fig_lasso), (experiment_lasso, condition_dir) in conditions.items():
        epoch = resolve_epoch(condition_dir, args.epoch)
        run_dirs = run_dirs_for_epoch(condition_dir, epoch, args.max_runs)
        print(
            "gamma={:g}, l={:g} (experiment l={:g}) epoch {}: {} runs".format(
                gamma,
                fig_lasso,
                experiment_lasso,
                epoch,
                len(run_dirs),
            )
        )
    if args.dry_run:
        return

    single_csv, threshold_csv = metric_csv_paths(args.output_root, args.token_type, args.mask_method, args.cutoff_mode, args.token_extraction_method, args.token_extraction_fraction)
    score_csv = score_distribution_csv_path(args.output_root, args.mask_method)
    single_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    written: list[Path] = []

    if not args.only_score_distribution:
        if args.only_plot:
            single_rows = read_csv(single_csv)
            validate_panel_a_cache(single_rows, args, single_csv)
            threshold_rows = read_csv(threshold_csv)
            validate_modularity_cache(threshold_rows, threshold_csv)
        else:
            for (gamma, fig_lasso), (experiment_lasso, condition_dir) in conditions.items():
                epoch = resolve_epoch(condition_dir, args.epoch)
                for run_dir in run_dirs_for_epoch(condition_dir, epoch, args.max_runs):
                    print("Computing gamma={:g}, l={:g} epoch {} {}".format(gamma, fig_lasso, epoch, run_dir.name))
                    run_single, run_threshold = compute_or_load_run_metrics(run_dir, gamma, fig_lasso, experiment_lasso, epoch, t_values, args)
                    single_rows.extend(run_single)
                    threshold_rows.extend(run_threshold)
            write_csv(single_csv, single_rows)
            write_csv(threshold_csv, threshold_rows)

        written.extend(plot_fig5a(single_rows, lasso_a, args.output_root, formats, args.token_type, args.mask_method, args.cutoff_mode, args.token_extraction_method, args.token_extraction_fraction))
        written.extend(plot_fig5a_gamma(single_rows, gamma_values, float(args.gamma_lasso), args.output_root, formats, args.token_type, args.mask_method, args.cutoff_mode, args.token_extraction_method, args.token_extraction_fraction))
        written.extend(plot_fig5b(threshold_rows, lasso_b, args.output_root, formats, args.mask_method))
        written.extend(plot_fig5c(threshold_rows, gamma_values, float(args.gamma_lasso), args.output_root, formats, args.mask_method))
        if not args.skip_combined:
            written.extend(plot_combined_fig5(single_rows, threshold_rows, lasso_a, lasso_b, gamma_values, float(args.gamma_lasso), args.output_root, formats, args.token_type, args.mask_method, args.cutoff_mode))

    if not args.skip_score_distribution:
        if args.only_plot:
            if score_csv.is_file():
                score_rows = read_csv(score_csv)
            else:
                print("Score-distribution CSV not found; skipping {}".format(score_csv))
        else:
            wanted_score_lasso = {round(float(value), 10) for value in score_lasso}
            for (gamma, fig_lasso), (experiment_lasso, condition_dir) in conditions.items():
                if not math.isclose(float(gamma), DEFAULT_GAMMA_FOR_LASSO, abs_tol=1e-9):
                    continue
                if round(float(fig_lasso), 10) not in wanted_score_lasso:
                    continue
                epoch = resolve_epoch(condition_dir, args.epoch)
                for run_dir in run_dirs_for_epoch(condition_dir, epoch, args.max_runs):
                    print("Computing score distribution gamma={:g}, l={:g} epoch {} {}".format(gamma, fig_lasso, epoch, run_dir.name))
                    score_rows.extend(compute_or_load_score_distribution_rows(run_dir, gamma, fig_lasso, experiment_lasso, epoch, args))
            write_csv(score_csv, score_rows)
        if score_rows:
            written.extend(plot_score_distribution(score_rows, score_lasso, args.output_root, formats, args.score_distribution_value, args.mask_method))

    csv_outputs = []
    if not args.only_score_distribution:
        csv_outputs.extend([str(single_csv), str(threshold_csv)])
    if not args.skip_score_distribution:
        csv_outputs.append(str(score_csv))

    save_json(
        args.output_root / "figure5_summary{}{}.json".format(method_suffix(args.mask_method), extraction_suffix(args.token_extraction_method, args.token_extraction_fraction)),
        {
            "summary": str(args.summary),
            "output_root": str(args.output_root),
            "epoch": args.epoch,
            "lasso_a": lasso_a,
            "lasso_b": lasso_b,
            "score_lasso": score_lasso,
            "gamma_values": gamma_values,
            "gamma_lasso": float(args.gamma_lasso),
            "fig_l_scale": float(args.fig_l_scale),
            "score_distribution_token_types": list(args.score_distribution_token_types),
            "score_distribution_value": str(args.score_distribution_value),
            "score_distribution_max_values": int(args.score_distribution_max_values),
            "score_distribution_floor": float(args.score_distribution_floor),
            "token_type": args.token_type,
            "cutoff_mode": args.cutoff_mode,
            "token_extraction_method": str(args.token_extraction_method),
            "token_extraction_fraction": float(args.token_extraction_fraction),
            "mask_method": args.mask_method,
            "legacy_hessian_epsilon": float(args.legacy_hessian_epsilon),
            "taylor_probes": int(args.taylor_probes),
            "modularity_metric": "quantile_rule_count_purity",
            "missing_conditions": missing,
            "single_rule_rows": len(single_rows),
            "threshold_rows": len(threshold_rows),
            "score_distribution_rows": len(score_rows),
            "metric_notes": [
                threshold_mask_note(args.token_type, args.mask_method, args.cutoff_mode, args.token_extraction_method),
                "Fig. 5b: gamma comparison uses the same relative-accuracy and retained-ratio plotting style as Fig. 5a at fixed l=gamma_lasso.",
                "Fig. 5c: M uses the fixed-mask, importance-weighted normalized inverse-participation score from analysis.py.",
                "Fig. 5d: M uses the same fixed-mask, importance-weighted score-profile definition for each gamma/layer.",
                "Extra score-distribution plot: sampled positive {} values are plotted as log10 distributions across l and layer for the selected token type(s).".format(args.score_distribution_value),
            ],
            "outputs": [str(path) for path in written],
            "csv": csv_outputs,
        },
    )
    verb = "Read" if args.only_plot else "Wrote"
    if not args.only_score_distribution:
        print("{} {}".format(verb, single_csv))
        print("{} {}".format(verb, threshold_csv))
    if not args.skip_score_distribution:
        print("{} {}".format(verb, score_csv))
    print("Wrote {} figure files under {}".format(len(written), args.output_root))


if __name__ == "__main__":
    main()
