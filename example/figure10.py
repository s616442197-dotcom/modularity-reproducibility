from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis as ana


DEFAULT_INPUT = ROOT / "example/results/simulation/lasso_l_gamma_sweep"
DEFAULT_ANALYSIS_ROOT = ROOT / "example/results/score_ratio_analysis"
DEFAULT_OUTPUT_ROOT = ROOT / "example/results/figure10"
DEFAULT_CACHE_ROOT = DEFAULT_OUTPUT_ROOT / "cache"
DEFAULT_COST_SUMMARY = ROOT / "example/composite/results/multitask_vs_separate_cost/summary.json"
EXPORT_EXTENSIONS = ("png", "pdf", "svg")

ADD_RULE = 0
MULT_RULE = 1
SQUARE_X_RULE = 2
TARGET_RULE = 4
EPS = 1e-8

CONDITION_LABELS = ["control", "Type 1", "Type 2"]
CONDITION_COLORS = {
    "control": "#7E2F8E",
    "Type 1": "#77AC30",
    "Type 2": "#4DBEEE",
}


@dataclass(frozen=True)
class ConditionDir:
    path: Path
    label: str
    experiment_lasso: float
    figure_lasso: float
    gamma: float | None


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


def format_number(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    value = float(value)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    text = "{:.4g}".format(value)
    return text.rstrip("0").rstrip(".") if "." in text else text


def parse_lasso_values(value: str) -> set[float] | None:
    text = str(value).strip().lower()
    if text in {"", "all", "available", "*"}:
        return None
    return {float(part.strip()) for part in text.split(",") if part.strip()}


def lasso_from_name(name: str) -> float | None:
    match = re.search(r"_l([0-9]+(?:p[0-9]+)?)_repeat", name)
    if not match:
        return None
    return float(match.group(1).replace("p", "."))


def condition_config(condition_dir: Path) -> dict[str, Any]:
    run_dirs = sorted(path.parent for path in condition_dir.glob("run_*/config.json"))
    if not run_dirs:
        return {}
    try:
        return json.loads((run_dirs[0] / "config.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def condition_from_dir(condition_dir: Path, fig_l_scale: float) -> ConditionDir:
    config = condition_config(condition_dir)
    loss_config = config.get("loss_config") if isinstance(config.get("loss_config"), dict) else {}
    gamma = loss_config.get("gamma")
    experiment_lasso = loss_config.get("l1")
    if experiment_lasso is None:
        experiment_lasso = lasso_from_name(condition_dir.name)
    if experiment_lasso is None:
        experiment_lasso = 0.0
    experiment_lasso = float(experiment_lasso)
    figure_lasso = experiment_lasso * float(fig_l_scale)
    return ConditionDir(
        path=condition_dir,
        label="l={}".format(format_number(figure_lasso)),
        experiment_lasso=experiment_lasso,
        figure_lasso=figure_lasso,
        gamma=float(gamma) if gamma is not None else None,
    )


def has_run_data(path: Path) -> bool:
    return any(path.glob("run_*/run_data.npz"))


def discover_condition_dirs(
    input_path: Path,
    gamma: float | None,
    include_all_gamma: bool,
    fig_l_scale: float,
    requested_lasso: set[float] | None,
) -> list[ConditionDir]:
    input_path = input_path.resolve()
    if has_run_data(input_path):
        condition = condition_from_dir(input_path, fig_l_scale)
        if requested_lasso is not None and not any(
            math.isclose(condition.figure_lasso, value, rel_tol=1e-9, abs_tol=1e-12) for value in requested_lasso
        ):
            raise FileNotFoundError(
                "{} has figure/text l={}, which is not in --lasso".format(input_path, format_number(condition.figure_lasso))
            )
        return [condition]

    conditions = []
    for child in sorted(input_path.iterdir()):
        if not child.is_dir() or not has_run_data(child):
            continue
        condition = condition_from_dir(child, fig_l_scale)
        if not include_all_gamma and gamma is not None and condition.gamma is not None:
            if not math.isclose(condition.gamma, float(gamma), rel_tol=1e-9, abs_tol=1e-12):
                continue
        if requested_lasso is not None and not any(
            math.isclose(condition.figure_lasso, value, rel_tol=1e-9, abs_tol=1e-12) for value in requested_lasso
        ):
            continue
        conditions.append(condition)
    conditions.sort(key=lambda item: (item.figure_lasso, item.path.name))
    if not conditions:
        raise FileNotFoundError("No run_*/run_data.npz directories found under {}".format(input_path))
    return conditions


def relative_analysis_candidates(condition_dir: Path, run_dir: Path, analysis_root: Path) -> list[Path]:
    candidates = [analysis_root / condition_dir.name / run_dir.name]
    simulation_root = ROOT / "example/results/simulation"
    try:
        rel_condition = condition_dir.resolve().relative_to(simulation_root.resolve())
        candidates.append(analysis_root / rel_condition / run_dir.name)
    except ValueError:
        pass
    deduped = []
    for path in candidates:
        if path not in deduped:
            deduped.append(path)
    return deduped


def compute_minimal_token_analysis(run_dir: Path, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    loaded = ana.load_run_npz(run_dir)
    weight_combo = ana.weight_combo_subnetwork_analysis(
        loaded["state"],
        loaded["x_test"],
        loaded["y_test"],
        int(loaded["pattern_size"]),
        [int(x) for x in loaded["rule_indices"]],
        max_elements_per_block=args.max_weight_elements_per_block,
        mask_method=args.mask_method,
        importance_min_score=args.importance_min_score,
        importance_mask_mode="score_ratio",
        importance_score_ratio_threshold=args.importance_weight_threshold,
        importance_output_eps=args.importance_output_eps,
        perturbation_batch_size=args.perturbation_batch_size,
        analysis_device=args.analysis_device,
        taylor_probes=args.taylor_probes,
    )
    result = {
        "combo_mask_method": args.mask_method,
        "importance_min_score": args.importance_min_score,
        "importance_mask_mode": "score_ratio",
        "importance_mask_mode_weight": "score_ratio",
        "importance_weight_threshold": args.importance_weight_threshold,
        "importance_output_eps": args.importance_output_eps,
        "taylor_probes": args.taylor_probes,
        "subnet_target": "weight",
        "weight_combo_subnetwork": weight_combo,
        "weight_combo_retained_accuracy": weight_combo.get("network_retained_accuracy", []),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    ana.save_json(output_dir / "analysis.json", result)
    return result


def analysis_has_weight_subnetwork(result: dict[str, Any]) -> bool:
    return bool(result.get("weight_combo_subnetwork", {}).get("blocks"))


def load_or_compute_analysis(
    run_dir: Path,
    condition_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    analysis_candidates = relative_analysis_candidates(condition_dir, run_dir, args.analysis_root)
    cache_candidates = relative_analysis_candidates(condition_dir, run_dir, args.cache_root)
    candidates = analysis_candidates + [path for path in cache_candidates if path not in analysis_candidates]
    if not args.recompute_analysis:
        for candidate in candidates:
            path = candidate / "analysis.json"
            if path.is_file():
                result = json.loads(path.read_text(encoding="utf-8"))
                if analysis_has_weight_subnetwork(result):
                    return result

    return compute_minimal_token_analysis(run_dir, cache_candidates[-1], args)


def rule_block(values: np.ndarray, rule_pos: int, pattern_size: int) -> np.ndarray:
    start = int(rule_pos) * int(pattern_size)
    end = start + int(pattern_size)
    return np.asarray(values[:, start:end], dtype=float)


def binary_accuracy(pred_block: np.ndarray, target_block: np.ndarray) -> float:
    return float(np.mean((np.asarray(pred_block) > 0.5) == (np.asarray(target_block) > 0.5)))


def single_rule_combo_index(combo_result: dict[str, Any], rule_pos: int, rule_indices: list[int]) -> int:
    labels = [str(label) for label in combo_result.get("combo_labels", [])]
    wanted = str(rule_indices[rule_pos]) if rule_pos < len(rule_indices) else str(rule_pos)
    for idx, label in enumerate(labels):
        if label == wanted:
            return idx
    for idx, label in enumerate(labels):
        if label == str(rule_pos):
            return idx
    if rule_pos < len(labels):
        return int(rule_pos)
    raise IndexError("Could not find single-rule combo index for rule {}".format(rule_pos + 1))


def keep_by_key_for_rule(analysis_result: dict[str, Any], rule_pos: int, rule_indices: list[int]) -> dict[str, list[int]]:
    combo_result = analysis_result.get("weight_combo_subnetwork", {})
    combo_idx = single_rule_combo_index(combo_result, rule_pos, rule_indices)
    keep_by_key: dict[str, list[int]] = {}
    for block in combo_result.get("blocks", []):
        key = str(block["key"])
        keep_by_key[key] = [int(idx) for idx in block["keep_flat_indices"][combo_idx]]
    if not keep_by_key:
        raise ValueError("No weight keep indices found in cached analysis")
    return keep_by_key


def forward_token(state: dict[str, np.ndarray], x: np.ndarray, keep_by_key: dict[str, list[int]]) -> np.ndarray:
    masked_state = ana.state_with_kept_weights(state, keep_by_key)
    pred, _ = ana.forward_from_state(masked_state, x)
    return pred


def retained_weight_fraction(state: dict[str, np.ndarray], keep_by_key: dict[str, list[int]]) -> float:
    total = 0
    retained = 0
    for spec in ana.weight_block_specs(state):
        key = str(spec["key"])
        total += int(np.prod(np.asarray(spec["shape"], dtype=int)))
        retained += len(keep_by_key.get(key, []))
    return retained / max(float(total), 1.0)


def evaluate_recomposition_run(run_dir: Path, condition: ConditionDir, analysis_result: dict[str, Any]) -> list[dict[str, Any]]:
    loaded = ana.load_run_npz(run_dir)
    state = loaded["state"]
    x_test = loaded["x_test"]
    y_test = loaded["y_test"]
    pattern_size = int(loaded["pattern_size"])
    rule_indices = [int(x) for x in loaded["rule_indices"]]
    if len(rule_indices) <= TARGET_RULE:
        raise ValueError("{} does not contain rule 5 output blocks".format(run_dir))

    target = rule_block(y_test, TARGET_RULE, pattern_size)
    full_pred, _ = ana.forward_from_state(state, x_test)
    full_rule5_accuracy = binary_accuracy(rule_block(full_pred, TARGET_RULE, pattern_size), target)

    keep_add = keep_by_key_for_rule(analysis_result, ADD_RULE, rule_indices)
    keep_mult = keep_by_key_for_rule(analysis_result, MULT_RULE, rule_indices)
    keep_square = keep_by_key_for_rule(analysis_result, SQUARE_X_RULE, rule_indices)
    keep_rule5 = keep_by_key_for_rule(analysis_result, TARGET_RULE, rule_indices)
    s5 = retained_weight_fraction(state, keep_rule5)

    control_pred = forward_token(state, x_test, keep_rule5)
    control_acc = binary_accuracy(rule_block(control_pred, TARGET_RULE, pattern_size), target)

    add_pred = forward_token(state, x_test, keep_add)
    add_out = (rule_block(add_pred, ADD_RULE, pattern_size) > 0.5).astype(np.float32)
    type1_input = np.concatenate([add_out, x_test[:, :pattern_size]], axis=1)
    type1_pred = forward_token(state, type1_input, keep_mult)
    type1_acc = binary_accuracy(rule_block(type1_pred, MULT_RULE, pattern_size), target)

    square_pred = forward_token(state, x_test, keep_square)
    mult_pred = forward_token(state, x_test, keep_mult)
    square_out = (rule_block(square_pred, SQUARE_X_RULE, pattern_size) > 0.5).astype(np.float32)
    mult_out = (rule_block(mult_pred, MULT_RULE, pattern_size) > 0.5).astype(np.float32)
    type2_input = np.concatenate([square_out, mult_out], axis=1)
    type2_pred = forward_token(state, type2_input, keep_add)
    type2_acc = binary_accuracy(rule_block(type2_pred, ADD_RULE, pattern_size), target)

    rows = []
    for label, accuracy in (
        ("control", control_acc),
        ("Type 1", type1_acc),
        ("Type 2", type2_acc),
    ):
        relative = accuracy / max(full_rule5_accuracy, EPS)
        rows.append(
            {
                "source_dir": str(condition.path),
                "run": run_dir.name,
                "condition": label,
                "experiment_lasso": condition.experiment_lasso,
                "figure_lasso": condition.figure_lasso,
                "lasso_label": condition.label,
                "gamma": condition.gamma if condition.gamma is not None else "",
                "accuracy": accuracy,
                "full_rule5_accuracy": full_rule5_accuracy,
                "relative_accuracy": relative,
                "rule5_retained_weight_fraction": s5,
                "rule5_retained_parameter_fraction": s5,
                "reconstruction_efficiency": relative / max(s5, EPS),
            }
        )
    return rows


def finite_epsilon(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1e-6
    span = float(np.nanmax(values) - np.nanmin(values))
    return max(0.006, span * 0.012)


def style_violin_parts(parts: dict[str, Any], labels: list[str]) -> None:
    for body, label in zip(parts["bodies"], labels):
        color = CONDITION_COLORS[label]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.68)
        body.set_linewidth(0.95)
    for key, color, width in (("cmeans", "#222222", 1.0), ("cmedians", "#B23A48", 1.25)):
        if key in parts:
            parts[key].set_color(color)
            parts[key].set_linewidth(width)


def violin_axis(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    show_reference: bool,
) -> None:
    lasso_values = sorted({float(row["figure_lasso"]) for row in rows})
    group_gap = 1.75
    offsets = np.linspace(-0.24, 0.24, len(CONDITION_LABELS))
    width = 0.18
    all_values = np.asarray([float(row[metric]) for row in rows], dtype=float)
    epsilon = finite_epsilon(all_values)
    data = []
    positions = []
    body_labels = []
    for group_idx, lasso_value in enumerate(lasso_values):
        base = group_idx * group_gap
        for offset, condition_label in zip(offsets, CONDITION_LABELS):
            values = np.asarray(
                [
                    float(row[metric])
                    for row in rows
                    if math.isclose(float(row["figure_lasso"]), lasso_value)
                    and str(row["condition"]) == condition_label
                ],
                dtype=float,
            )
            values = ana._violin_values(values, epsilon)
            if values.size:
                data.append(values)
                positions.append(base + float(offset))
                body_labels.append(condition_label)

    if data:
        parts = ax.violinplot(data, positions=positions, widths=width, showmeans=True, showmedians=True, showextrema=False)
        style_violin_parts(parts, body_labels)

    centers = np.arange(len(lasso_values), dtype=float) * group_gap
    ax.set_xticks(centers)
    ax.set_xticklabels(["l={}".format(format_number(value)) for value in lasso_values])
    ax.set_xlim(centers[0] - 0.72, centers[-1] + 0.72)
    ax.set_xlabel("Lasso strength l")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    if show_reference:
        lower, upper = ana._retained_ratio_ylim(all_values)
        ax.set_ylim(lower, upper)
        tick_step = 0.1 if upper - lower <= 1.2 else 0.2
        ax.set_yticks(np.round(np.arange(lower, upper + tick_step * 0.5, tick_step), 2))
        if lower <= 1.0 <= upper:
            ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.42, zorder=0)
    else:
        finite = all_values[np.isfinite(all_values)]
        upper = max(1.02, float(np.ceil((np.nanmax(finite) + 0.05) * 10.0) / 10.0)) if finite.size else 1.02
        ax.set_ylim(0.0, upper)
    ana._style_publication_axes(ax, grid=True)


def legend_handles() -> list[plt.Line2D]:
    return [
        plt.Line2D([0], [0], color=CONDITION_COLORS[label], linewidth=5, alpha=0.8, label=label)
        for label in CONDITION_LABELS
    ]


def export_figure(fig: plt.Figure, output_root: Path, stem: str) -> list[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in EXPORT_EXTENSIONS:
        path = output_root / "{}.{}".format(stem, ext)
        fig.savefig(path, dpi=ana.PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.03)
        written.append(str(path))
    return written


def plot_metric(rows: list[dict[str, Any]], output_root: Path, metric: str, ylabel: str, stem: str, title: str, show_reference: bool) -> list[str]:
    fig, ax = plt.subplots(figsize=(ana.NATURE_SINGLE_COLUMN_IN, 2.65))
    violin_axis(ax, rows, metric, ylabel, title, show_reference)
    ax.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, frameon=False)
    fig.tight_layout(pad=0.35)
    written = export_figure(fig, output_root, stem)
    plt.close(fig)
    return written


def load_cost_summary(path: Path, enabled: bool = True) -> dict[str, Any] | None:
    if not enabled:
        return None
    path = Path(path)
    if not path.is_file():
        print("Cost summary not found; skipping Fig. 10d: {}".format(path), flush=True)
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["_source_path"] = str(path)
    return data


def cost_proxy_arrays(cost_summary: dict[str, Any]) -> tuple[list[str], list[np.ndarray]]:
    rows = list(cost_summary.get("per_repeat", []))
    labels = ["multitask", "multitask+\nextraction", "5 separate"]
    keys = ["multitask_train_proxy", "multitask_plus_extraction_proxy", "separate_train_proxy"]
    if rows:
        values = []
        for key in keys:
            current = []
            for row in rows:
                denom = float(row.get("multitask_plus_extraction_proxy", np.nan))
                value = float(row.get(key, np.nan))
                if np.isfinite(value) and np.isfinite(denom) and denom > 0:
                    current.append(value / denom)
            values.append(np.asarray(current, dtype=float))
        return labels, values

    mean = cost_summary.get("mean", {})
    denom = float(mean.get("multitask_plus_extraction_proxy", np.nan))
    values = []
    for key in keys:
        value = float(mean.get(key, np.nan))
        if np.isfinite(value) and np.isfinite(denom) and denom > 0:
            values.append(np.asarray([value / denom], dtype=float))
        else:
            values.append(np.zeros(0, dtype=float))
    return labels, values


def plot_cost_axis(ax: plt.Axes, cost_summary: dict[str, Any], title: str = "d") -> None:
    labels, values_by_group = cost_proxy_arrays(cost_summary)
    colors = ["#0072B2", "#009E73", "#D55E00"]
    means = [float(np.nanmean(values)) if values.size else np.nan for values in values_by_group]
    sems = [
        float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        for values in values_by_group
    ]
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.78, capsize=2.5, width=0.62)
    rng = np.random.default_rng(20260807)
    for idx, values in enumerate(values_by_group):
        finite = values[np.isfinite(values)]
        if finite.size:
            jitter = rng.normal(0.0, 0.035, size=finite.size)
            ax.scatter(np.full(finite.size, x[idx]) + jitter, finite, s=12, color="#222222", alpha=0.62, linewidths=0, zorder=3)
    finite_means = np.asarray([value for value in means if np.isfinite(value)], dtype=float)
    if finite_means.size:
        ymax = max(1.2, float(np.nanmax(finite_means) * 1.22))
        ax.set_ylim(0.0, ymax)
        if np.isfinite(means[-1]):
            ax.text(x[-1], means[-1] + ymax * 0.035, "{:.1f}x".format(means[-1]), ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.42, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_ylabel("relative cost")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ana._style_publication_axes(ax, grid=True)


def plot_cost_metric(cost_summary: dict[str, Any], output_root: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(ana.NATURE_SINGLE_COLUMN_IN, 2.65))
    plot_cost_axis(ax, cost_summary, "Cost comparison")
    fig.tight_layout(pad=0.35)
    written = export_figure(fig, output_root, "fig10_cost_comparison")
    plt.close(fig)
    return written


def plot_combined(rows: list[dict[str, Any]], output_root: Path, cost_summary: dict[str, Any] | None = None) -> list[str]:
    if cost_summary is None:
        fig, axes = plt.subplots(1, 2, figsize=(ana.NATURE_DOUBLE_COLUMN_IN, 2.85))
        violin_axis(axes[0], rows, "relative_accuracy", "relative accuracy", "b", show_reference=True)
        violin_axis(axes[1], rows, "reconstruction_efficiency", r"reconstruction efficiency $E_w$", "c", show_reference=False)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(ana.NATURE_DOUBLE_COLUMN_IN, 2.85), gridspec_kw={"width_ratios": [1.08, 1.08, 0.92]})
        violin_axis(axes[0], rows, "relative_accuracy", "relative accuracy", "b", show_reference=True)
        violin_axis(axes[1], rows, "reconstruction_efficiency", r"reconstruction efficiency $E_w$", "c", show_reference=False)
        plot_cost_axis(axes[2], cost_summary, "d")
    fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93), pad=0.35)
    written = export_figure(fig, output_root, "fig10_recomposition")
    plt.close(fig)
    return written


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_dir",
        "run",
        "condition",
        "experiment_lasso",
        "figure_lasso",
        "lasso_label",
        "gamma",
        "accuracy",
        "full_rule5_accuracy",
        "relative_accuracy",
        "rule5_retained_weight_fraction",
        "rule5_retained_parameter_fraction",
        "reconstruction_efficiency",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(
    rows: list[dict[str, Any]],
    conditions: list[ConditionDir],
    output_root: Path,
    written: list[str],
    cost_summary: dict[str, Any] | None = None,
) -> None:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = "{}::{}".format(row["lasso_label"], row["condition"])
        grouped.setdefault(
            key,
            {
                "lasso_label": row["lasso_label"],
                "figure_lasso": row["figure_lasso"],
                "experiment_lasso": row["experiment_lasso"],
                "condition": row["condition"],
                "n": 0,
                "relative_accuracy_mean": 0.0,
                "reconstruction_efficiency_mean": 0.0,
            },
        )
        item = grouped[key]
        item["n"] += 1
        item["relative_accuracy_mean"] += float(row["relative_accuracy"])
        item["reconstruction_efficiency_mean"] += float(row["reconstruction_efficiency"])
    for item in grouped.values():
        count = max(int(item["n"]), 1)
        item["relative_accuracy_mean"] /= count
        item["reconstruction_efficiency_mean"] /= count

    summary = {
        "conditions": [
            {
                "path": str(condition.path),
                "label": condition.label,
                "experiment_lasso": condition.experiment_lasso,
                "figure_lasso": condition.figure_lasso,
                "gamma": condition.gamma,
            }
            for condition in conditions
        ],
        "rows": len(rows),
        "runs_per_condition": {
            condition.label: len({row["run"] for row in rows if row["source_dir"] == str(condition.path)})
            for condition in conditions
        },
        "means": sorted(grouped.values(), key=lambda item: (item["figure_lasso"], CONDITION_LABELS.index(item["condition"]))),
        "written": written,
    }
    if cost_summary is not None:
        summary["cost_comparison"] = {
            "source_path": cost_summary.get("_source_path", ""),
            "repeat_count": cost_summary.get("repeat_count"),
            "mean": cost_summary.get("mean", {}),
        }
    (output_root / "fig10_recomposition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[ConditionDir]]:
    requested_lasso = parse_lasso_values(args.lasso)
    conditions = discover_condition_dirs(args.input, args.gamma, args.include_all_gamma, args.fig_l_scale, requested_lasso)
    all_rows: list[dict[str, Any]] = []
    for condition_idx, condition in enumerate(conditions, start=1):
        run_dirs = sorted(path.parent for path in condition.path.glob("run_*/run_data.npz"))
        if args.max_runs is not None:
            run_dirs = run_dirs[: max(0, int(args.max_runs))]
        print(
            "Condition {}/{}: {} ({}) with {} runs".format(
                condition_idx,
                len(conditions),
                condition.label,
                condition.path.name,
                len(run_dirs),
            ),
            flush=True,
        )
        for run_idx, run_dir in enumerate(run_dirs, start=1):
            print("  run {}/{} {}".format(run_idx, len(run_dirs), run_dir.name), flush=True)
            analysis_result = load_or_compute_analysis(run_dir, condition.path, args)
            all_rows.extend(evaluate_recomposition_run(run_dir, condition, analysis_result))
    if not all_rows:
        raise RuntimeError("No recomposition rows were produced")
    return all_rows, conditions


def main() -> None:
    parser = argparse.ArgumentParser(description="Figure 10 compositional reuse analysis for binary MLP functional tokens.")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT, help="Simulation directory or a parent containing lasso-condition directories.")
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT, help="Cached analysis root used for functional-token masks.")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT, help="Figure-10-only cache for computed weight-token masks.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cost-summary", type=Path, default=DEFAULT_COST_SUMMARY, help="Composite cost summary JSON for Fig. 10d.")
    parser.add_argument("--no-cost-panel", action="store_true", help="Do not add the multitask-vs-separate cost panel.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma filter when input is a lasso sweep parent.")
    parser.add_argument("--include-all-gamma", action="store_true", help="Do not filter lasso sweep children by gamma.")
    parser.add_argument("--lasso", default="0,1,5,10", help="Figure/text Lasso l values, comma list or 'available'.")
    parser.add_argument("--fig-l-scale", type=float, default=5.0, help="Figure/text l = experiment l * this scale.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit repeats for a quick check.")
    parser.add_argument("--recompute-analysis", action="store_true", help="Recompute weight masks even when analysis.json exists.")
    parser.add_argument("--mask-method", choices=["ablation", "taylor"], default="ablation")
    parser.add_argument("--importance-neuron-threshold", type=float, default=1e-5, help="Deprecated; kept for old commands.")
    parser.add_argument("--importance-weight-threshold", type=float, default=1e-6)
    parser.add_argument("--max-weight-elements-per-block", type=int, default=None)
    parser.add_argument("--importance-min-score", type=float, default=1e-9)
    parser.add_argument("--importance-output-eps", type=float, default=1e-8)
    parser.add_argument("--perturbation-batch-size", type=int, default=64)
    parser.add_argument("--analysis-device", default="auto")
    parser.add_argument("--taylor-probes", type=int, default=4)
    args = parser.parse_args()

    configure_style()
    args.input = args.input.resolve()
    args.analysis_root = args.analysis_root.resolve()
    args.cache_root = args.cache_root.resolve()
    args.output_root = args.output_root.resolve()
    args.cost_summary = args.cost_summary.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    cost_summary = load_cost_summary(args.cost_summary, enabled=not args.no_cost_panel)
    rows, conditions = build_rows(args)
    csv_path = args.output_root / "fig10_recomposition_metrics.csv"
    write_csv(rows, csv_path)
    written = [str(csv_path)]
    written.extend(plot_combined(rows, args.output_root, cost_summary=cost_summary))
    written.extend(
        plot_metric(
            rows,
            args.output_root,
            "relative_accuracy",
            "relative accuracy",
            "fig10_relative_accuracy",
            "Relative accuracy",
            show_reference=True,
        )
    )
    written.extend(
        plot_metric(
            rows,
            args.output_root,
            "reconstruction_efficiency",
            r"reconstruction efficiency $E_w$",
            "fig10_reconstruction_efficiency",
            "Reconstruction efficiency",
            show_reference=False,
        )
    )
    if cost_summary is not None:
        written.extend(plot_cost_metric(cost_summary, args.output_root))
    save_summary(rows, conditions, args.output_root, written, cost_summary=cost_summary)
    print("Wrote {} metric rows to {}".format(len(rows), csv_path))
    for path in written[1:]:
        print(path)


if __name__ == "__main__":
    main()
