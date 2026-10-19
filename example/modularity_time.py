from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis as ana


SWEEP_SUMMARY = ROOT / "example/results/simulation/lasso_l_gamma_sweep/lasso_l_gamma_sweep_summary.json"
OUTPUT_ROOT = ROOT / "example/results/modularity_time"
DEFAULT_GAMMA = 1.0
DEFAULT_LASSO = "0,1,5,10"
DEFAULT_FIG_L_SCALE = 5.0
DEFAULT_MASK_METHOD = "taylor"
DEFAULT_TAYLOR_PROBES = 4
DEFAULT_ARTICLE_FIG9 = ROOT / "article/figs/modularity_time.pdf"
DEFAULT_THRESHOLDS = np.logspace(-7, -3, 25)
LAYER_COLORS = ["#4DBEEE", "#A2142F", "#0072BD"]
LASSO_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_lasso_values(value: str) -> list[float] | None:
    value = str(value).strip().lower()
    if value in {"available", "all"}:
        return None
    out = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def parse_epochs(value: str) -> set[int] | None:
    value = str(value).strip().lower()
    if value in {"available", "all", ""}:
        return None
    return {int(float(item.strip())) for item in value.split(",") if item.strip()}


def lasso_label(value: float) -> str:
    return "l={:g}".format(float(value))


def tag_float(value: float) -> str:
    return "{:g}".format(float(value)).replace("-", "m").replace(".", "p")


def resolve_condition_dir(summary_path: Path, row: dict[str, Any]) -> Path:
    output_dir = Path(str(row.get("output_dir", "")).replace("\\", "/"))
    candidates = []
    if output_dir.is_absolute():
        candidates.append(output_dir)
    candidates.append(summary_path.parent / str(row["experiment_name"]))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[-1]


def available_conditions(summary_path: Path, gamma: float) -> dict[float, Path]:
    data = load_json(summary_path)
    conditions: dict[float, Path] = {}
    for row in data.get("conditions", []):
        if math.isclose(float(row["gamma"]), float(gamma), abs_tol=1e-9):
            conditions[float(row["l"])] = resolve_condition_dir(summary_path, row)
    return dict(sorted(conditions.items()))


def selected_conditions(
    summary_path: Path,
    gamma: float,
    lasso_values: list[float] | None,
    fig_l_scale: float,
) -> tuple[dict[float, tuple[float, Path]], list[float]]:
    available = available_conditions(summary_path, gamma)
    if lasso_values is None:
        return {value * fig_l_scale: (value, path) for value, path in available.items()}, []
    selected = {}
    missing = []
    for fig_lasso in lasso_values:
        experiment_lasso = float(fig_lasso) / float(fig_l_scale)
        match = None
        for value in available:
            if math.isclose(value, experiment_lasso, abs_tol=1e-9):
                match = value
                break
        if match is None:
            missing.append(float(fig_lasso))
        else:
            selected[float(fig_lasso)] = (match, available[match])
    return dict(sorted(selected.items())), missing


def snapshot_epoch(path: Path) -> int:
    match = re.search(r"epoch_(\d+)", path.name)
    if not match:
        raise ValueError("Cannot parse epoch from {}".format(path))
    return int(match.group(1))


def snapshot_run_dirs(condition_dir: Path, epochs: set[int] | None, max_runs: int | None) -> list[tuple[int, Path]]:
    rows = []
    for epoch_dir in sorted((condition_dir / "snapshots").glob("epoch_*"), key=snapshot_epoch):
        epoch = snapshot_epoch(epoch_dir)
        if epochs is not None and epoch not in epochs:
            continue
        run_dirs = sorted(path for path in epoch_dir.glob("run_*") if (path / "run_data.npz").is_file())
        if max_runs is not None:
            run_dirs = run_dirs[: max(0, int(max_runs))]
        rows.extend((epoch, run_dir) for run_dir in run_dirs)
    return rows


def subset_loaded_data(loaded: dict[str, Any], max_samples: int | None, seed: int) -> dict[str, Any]:
    if max_samples is None or max_samples <= 0 or len(loaded["x_test"]) <= max_samples:
        return loaded
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(loaded["x_test"]), size=int(max_samples), replace=False))
    out = dict(loaded)
    out["x_test"] = loaded["x_test"][idx]
    out["y_test"] = loaded["y_test"][idx]
    return out


def fixed_mask_modularity_summary(
    scores: np.ndarray,
    ratios: np.ndarray,
    min_score: float,
    token_extraction_method: str,
    token_extraction_fraction: float,
) -> dict[str, float | int]:
    important = ana.importance_mask_from_tables(
        scores,
        ratios,
        min_score=min_score,
        token_extraction_method=token_extraction_method,
        token_extraction_fraction=token_extraction_fraction,
    )
    return ana.importance_weighted_specialization_summary(ratios, membership_mask=important)

def weight_block_score_matrix(block: dict[str, Any]) -> np.ndarray:
    value_name = str(block.get("mask_value_name", ""))
    if value_name and value_name in block:
        return np.asarray(block[value_name], dtype=float)
    for key in ("ablation_scores", "taylor_scores"):
        if key in block:
            return np.asarray(block[key], dtype=float)
    raise KeyError("weight block is missing ablation_scores/taylor_scores")


def weight_block_ratio_matrix(block: dict[str, Any]) -> np.ndarray:
    for key in ("weight_score_ratio", "score_ratio"):
        if key in block:
            return np.asarray(block[key], dtype=float)
    raise KeyError("weight block is missing score_ratio")


def weight_layer_blocks(weight_combo: dict[str, Any], max_layers: int | None = 3) -> list[dict[str, Any]]:
    blocks = []
    for block in weight_combo.get("blocks", []):
        key = str(block.get("key", ""))
        label = str(block.get("label", "")).lower()
        if key == "output_block_linear1_weight" or label == "output":
            continue
        blocks.append(block)
    if not blocks:
        blocks = list(weight_combo.get("blocks", []))
    if max_layers is not None:
        blocks = blocks[: max(0, int(max_layers))]
    return blocks


def compute_run_rows(
    run_dir: Path,
    fig_lasso_value: float,
    experiment_lasso_value: float,
    epoch: int,
    max_samples: int | None,
    min_score: float,
    output_eps: float,
    perturbation_batch_size: int,
    analysis_device: str,
    max_weight_elements_per_block: int | None,
    mask_method: str,
    taylor_probes: int,
    token_extraction_method: str,
    token_extraction_fraction: float,
) -> list[dict[str, Any]]:
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(run_dir))) % (2**32)
    loaded = subset_loaded_data(ana.load_run_npz(run_dir), max_samples=max_samples, seed=seed)
    state = loaded["state"]
    x_test = loaded["x_test"]
    y_test = loaded["y_test"]
    pattern_size = int(loaded["pattern_size"])
    rule_indices = [int(x) for x in loaded["rule_indices"]]
    weight_combo = ana.weight_combo_subnetwork_analysis(
        state,
        x_test,
        y_test,
        pattern_size,
        rule_indices,
        max_elements_per_block=max_weight_elements_per_block,
        mask_method=mask_method,
        importance_min_score=min_score,
        importance_mask_mode="score_ratio",
        importance_output_eps=output_eps,
        perturbation_batch_size=perturbation_batch_size,
        analysis_device=analysis_device,
        taylor_probes=taylor_probes,
        token_extraction_method=token_extraction_method,
        token_extraction_fraction=token_extraction_fraction,
    )

    rows = []
    for layer_pos, block in enumerate(weight_layer_blocks(weight_combo), start=1):
        scores = weight_block_score_matrix(block)
        ratios = weight_block_ratio_matrix(block)
        summary = block.get("task_specialization")
        if not isinstance(summary, dict):
            summary = fixed_mask_modularity_summary(
                scores,
                ratios,
                min_score=min_score,
                token_extraction_method=token_extraction_method,
                token_extraction_fraction=token_extraction_fraction,
            )
        rows.append(
            {
                "lasso": float(fig_lasso_value),
                "experiment_lasso": float(experiment_lasso_value),
                "epoch": int(epoch),
                "run": run_dir.name,
                "layer_index": int(layer_pos),
                "layer_label": "L{}".format(layer_pos),
                "weight_block": str(block.get("label", block.get("key", layer_pos))),
                "modularity": float(summary.get("modularity", float("nan"))),
                "active_token_count": int(summary.get("active_token_count", 0)),
                "total_importance": float(summary.get("total_importance", float("nan"))),
                "mean_token_specialization": float(summary.get("mean_token_specialization", float("nan"))),
                "mean_participation": float(summary.get("mean_participation", float("nan"))),
                "num_samples": int(len(x_test)),
                "mask_method": str(mask_method),
                "modularity_metric": "importance_weighted_normalized_inverse_participation",
            }
        )
    return rows

def cache_path(output_root: Path, gamma: float, experiment_lasso_value: float, epoch: int, run_name: str, mask_method: str) -> Path:
    cache_dir = "cache_weight_fixed_m_{}".format(str(mask_method).lower().strip())
    return output_root / cache_dir / "gamma{}".format(tag_float(gamma)) / "experiment_l{}".format(tag_float(experiment_lasso_value)) / "epoch_{:06d}".format(epoch) / "{}.json".format(run_name)


def row_cache_is_current(rows: list[dict[str, Any]], mask_method: str) -> bool:
    if not rows:
        return False
    wanted = "importance_weighted_normalized_inverse_participation"
    metrics = {str(row.get("modularity_metric", "") or "").lower().strip() for row in rows}
    methods = {str(row.get("mask_method", "") or "").lower().strip() for row in rows}
    required = ("active_token_count", "total_importance", "mean_token_specialization", "mean_participation")
    return metrics == {wanted} and methods == {str(mask_method).lower().strip()} and all(all(key in row for key in required) for row in rows)


def compute_or_load_rows(
    run_dir: Path,
    fig_lasso_value: float,
    experiment_lasso_value: float,
    epoch: int,
    gamma: float,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    path = cache_path(args.output_root, gamma, experiment_lasso_value, epoch, run_dir.name, args.mask_method)
    if path.is_file() and not args.recompute:
        cached_rows = list(load_json(path)["rows"])
        if row_cache_is_current(cached_rows, args.mask_method):
            return cached_rows
    rows = compute_run_rows(
        run_dir,
        fig_lasso_value,
        experiment_lasso_value,
        epoch,
        max_samples=args.max_samples,
        min_score=args.importance_min_score,
        output_eps=args.importance_output_eps,
        perturbation_batch_size=args.perturbation_batch_size,
        analysis_device=args.analysis_device,
        max_weight_elements_per_block=args.max_weight_elements_per_block,
        mask_method=args.mask_method,
        taylor_probes=args.taylor_probes,
        token_extraction_method=args.token_extraction_method,
        token_extraction_fraction=args.token_extraction_fraction,
    )
    save_json(path, {"run_dir": str(run_dir), "rows": rows})
    return rows

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
    out = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["lasso"] = float(row["lasso"])
            row["experiment_lasso"] = float(row.get("experiment_lasso", row["lasso"]))
            row["epoch"] = int(float(row["epoch"]))
            row["layer_index"] = int(float(row["layer_index"]))
            row["modularity"] = float(row["modularity"])
            row["active_token_count"] = int(float(row.get("active_token_count", 0)))
            row["num_samples"] = int(float(row["num_samples"]))
            out.append(row)
    return out

def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def _violin_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    center = float(np.nanmedian(values))
    epsilon = 0.006
    sample_span = float(np.nanmax(values) - np.nanmin(values)) if values.size > 1 else 0.0
    if sample_span <= epsilon:
        anchors = np.asarray([center - epsilon, center + epsilon], dtype=float)
        return np.concatenate([values, anchors])
    return values


def _style_violin_parts(parts: dict[str, Any], colors: list[str], layer_ids: list[int]) -> None:
    for body, layer_idx in zip(parts["bodies"], layer_ids):
        color = colors[(layer_idx - 1) % len(colors)]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.66)
        body.set_linewidth(1.0)
    for key, color, width in [("cmeans", "#222222", 1.0), ("cmedians", "#B23A48", 1.25)]:
        if key in parts:
            parts[key].set_color(color)
            parts[key].set_linewidth(width)


def plot_rows(rows: list[dict[str, Any]], output_root: Path, missing_lasso: list[float]) -> None:
    if not rows:
        raise RuntimeError("No modularity-time rows to plot")
    layers = sorted({int(row["layer_index"]) for row in rows})
    lasso_values = sorted({float(row["lasso"]) for row in rows})
    epochs = sorted({int(row["epoch"]) for row in rows})
    ncols = 2 if len(lasso_values) > 1 else 1
    nrows = int(math.ceil(len(lasso_values) / ncols))
    fig_width = 7.4 if ncols == 2 else 4.6
    fig_height = 2.55 * nrows + 0.55
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), sharey=True)
    axes = np.atleast_1d(axes_grid).ravel()

    offsets = np.linspace(-0.26, 0.26, len(layers)) if len(layers) > 1 else np.asarray([0.0])
    width = 0.20 if len(layers) > 1 else 0.42
    group_gap = 1.55
    epoch_centers = np.arange(len(epochs), dtype=float) * group_gap
    for ax, lasso_value in zip(axes, lasso_values):
        data = []
        positions = []
        layer_ids = []
        for epoch_pos, epoch in enumerate(epochs):
            for offset, layer_idx in zip(offsets, layers):
                vals = np.asarray(
                    [
                        float(row["modularity"])
                        for row in rows
                        if math.isclose(float(row["lasso"]), lasso_value)
                        and int(row["epoch"]) == epoch
                        and int(row["layer_index"]) == layer_idx
                    ],
                    dtype=float,
                )
                vals = _violin_values(vals)
                if vals.size:
                    data.append(vals)
                    positions.append(float(epoch_centers[epoch_pos] + offset))
                    layer_ids.append(layer_idx)
        if data:
            parts = ax.violinplot(data, positions=positions, widths=width, showmeans=True, showmedians=True, showextrema=False)
            _style_violin_parts(parts, LAYER_COLORS, layer_ids)
        ax.set_title(lasso_label(lasso_value))
        ax.set_xlabel("training generation")
        ax.set_xticks(epoch_centers)
        ax.set_xticklabels([str(epoch) for epoch in epochs], rotation=35, ha="right")
        ax.set_xlim(float(epoch_centers[0] - 0.65), float(epoch_centers[-1] + 0.65))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.7)
        ax.set_axisbelow(True)
    for empty_ax in axes[len(lasso_values) :]:
        empty_ax.set_visible(False)
    for idx, ax in enumerate(axes[: len(lasso_values)]):
        if idx % ncols == 0:
            ax.set_ylabel(r"$M$")
    handles = [
        plt.Line2D([0], [0], color=LAYER_COLORS[(layer_idx - 1) % len(LAYER_COLORS)], linewidth=5, label="Layer {}".format(layer_idx))
        for layer_idx in layers
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=len(handles))
    if missing_lasso:
        fig.suptitle("Missing requested Lasso strengths: {}".format(", ".join(lasso_label(x) for x in missing_lasso)), fontsize=8, y=1.05)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(output_root / "modularity_time.{}".format(ext), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and plot temporal evolution of fixed-mask importance-weighted modularity from lasso sweep snapshots.")
    parser.add_argument("--summary", type=Path, default=SWEEP_SUMMARY)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--lasso", default=DEFAULT_LASSO, help="Figure/text Lasso l values, such as 0,1,5,10, or 'available'.")
    parser.add_argument("--fig-l-scale", type=float, default=DEFAULT_FIG_L_SCALE, help="Figure/text l = experiment l * this scale.")
    parser.add_argument("--epochs", default="available", help="Comma list such as 1000,2000,5000, or 'available'.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit repeats per epoch for quick checks; default uses all runs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional deterministic subsample of test examples per run.")
    parser.add_argument("--max-weight-elements-per-block", type=int, default=None, help="Optional cap for weight ablation candidates per weight block.")
    parser.add_argument("--importance-min-score", type=float, default=1e-9)
    parser.add_argument("--importance-output-eps", type=float, default=1e-8)
    parser.add_argument("--mask-method", choices=["ablation", "taylor"], default=DEFAULT_MASK_METHOD)
    parser.add_argument("--taylor-probes", type=int, default=DEFAULT_TAYLOR_PROBES)
    parser.add_argument("--token-extraction-method", choices=ana.TOKEN_EXTRACTION_METHOD_CHOICES, default=ana.DEFAULT_TOKEN_EXTRACTION_METHOD)
    parser.add_argument("--token-extraction-fraction", type=float, default=ana.DEFAULT_TOKEN_EXTRACTION_FRACTION)
    parser.add_argument("--perturbation-batch-size", type=int, default=64)
    parser.add_argument("--analysis-device", default="auto")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--only-plot", action="store_true", help="Use existing CSV/cache and do not compute missing runs.")
    parser.add_argument("--dry-run", action="store_true", help="Only report selected/missing conditions and snapshot counts.")
    parser.add_argument("--article-fig9", type=Path, default=DEFAULT_ARTICLE_FIG9)
    parser.add_argument("--no-replace-article", action="store_true")
    args = parser.parse_args()
    args.mask_method = ana.validate_mask_method(args.mask_method)
    args.output_root.mkdir(parents=True, exist_ok=True)
    configure_style()

    lasso_values = parse_lasso_values(args.lasso)
    selected, missing = selected_conditions(args.summary, args.gamma, lasso_values, args.fig_l_scale)
    available = available_conditions(args.summary, args.gamma)
    epochs = parse_epochs(args.epochs)
    print("Available experiment l for gamma={:g}: {}".format(args.gamma, ", ".join(lasso_label(x) for x in available)))
    print("Available figure/text l after x{:g} scaling: {}".format(args.fig_l_scale, ", ".join(lasso_label(x * args.fig_l_scale) for x in available)))
    if missing:
        print("Missing requested figure/text l: {}".format(", ".join(lasso_label(x) for x in missing)))
    print("Selected figure/text l: {}".format(", ".join(lasso_label(x) for x in selected) if selected else "none"))

    for fig_lasso_value, (experiment_lasso_value, condition_dir) in selected.items():
        rows = snapshot_run_dirs(condition_dir, epochs, args.max_runs)
        by_epoch = defaultdict(int)
        for epoch, _run_dir in rows:
            by_epoch[epoch] += 1
        print(
            "{} (experiment {}): {}".format(
                lasso_label(fig_lasso_value),
                lasso_label(experiment_lasso_value),
                ", ".join("epoch {}: {} runs".format(k, by_epoch[k]) for k in sorted(by_epoch)),
            )
        )
    if args.dry_run:
        return

    csv_path = args.output_root / "modularity_time_metrics.csv"
    if args.only_plot:
        rows = read_csv(csv_path)
    else:
        rows = []
        for fig_lasso_value, (experiment_lasso_value, condition_dir) in selected.items():
            for epoch, run_dir in snapshot_run_dirs(condition_dir, epochs, args.max_runs):
                print(
                    "Computing {} (experiment {}) epoch {} {}".format(
                        lasso_label(fig_lasso_value),
                        lasso_label(experiment_lasso_value),
                        epoch,
                        run_dir.name,
                    )
                )
                rows.extend(compute_or_load_rows(run_dir, fig_lasso_value, experiment_lasso_value, epoch, args.gamma, args))
        write_csv(csv_path, rows)

    plot_rows(rows, args.output_root, missing)
    if not args.no_replace_article:
        args.article_fig9.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.output_root / "modularity_time.pdf", args.article_fig9)
        print("Replaced {}".format(args.article_fig9))
    save_json(
        args.output_root / "modularity_time_summary.json",
        {
            "summary": str(args.summary),
            "gamma": float(args.gamma),
            "requested_lasso": args.lasso,
            "fig_l_scale": float(args.fig_l_scale),
            "lasso_mapping": "figure/text l = experiment l * fig_l_scale",
            "missing_lasso": [float(x) for x in missing],
            "rows": len(rows),
            "mask_method": str(args.mask_method),
            "taylor_probes": int(args.taylor_probes),
            "token_extraction_method": str(args.token_extraction_method),
            "token_extraction_fraction": float(args.token_extraction_fraction),
            "metric": "Layer-wise fixed-mask importance-weighted modularity M from masked normalized score profiles over scalar weight tokens.",
            "outputs": [
                str(args.output_root / "modularity_time_metrics.csv"),
                str(args.output_root / "modularity_time.png"),
                str(args.output_root / "modularity_time.pdf"),
                str(args.output_root / "modularity_time.svg"),
            ],
        },
    )
    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(args.output_root / "modularity_time.png"))


if __name__ == "__main__":
    main()
