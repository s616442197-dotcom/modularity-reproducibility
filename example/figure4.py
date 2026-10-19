from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis as ana


OUTPUT_ROOT = ROOT / "example/results/figure4"
FIG4_SUPP_DIR = ROOT / "example/results/mlp300_binary_fig4_supp"
TASK_DIRS = {
    "Binary Sequence\nCalculation": ROOT / "example/results/score_ratio_analysis/mlp300_binary_repeat30",
    "Sparse Pattern\nLogic": ROOT / "example/results/score_ratio_analysis/mlp300_pattern_repeat30",
    "MNIST Digit-Pair\nArithmetic": ROOT / "example/results/score_ratio_analysis/mlp_mnist_arithmetic_repeat30",
}
TAYLOR_MLP_DIR = ROOT / "example/results/score_ratio_taylor_analysis/mlp300_binary_repeat30"
TRANSFORMER_DIR = ROOT / "example/results/score_ratio_taylor_analysis/transformer_binary_repeat30"
TASK_SIMULATION_DIRS = {
    "Binary Sequence\nCalculation": ROOT / "example/results/simulation/mlp300_binary_repeat30",
    "Sparse Pattern\nLogic": ROOT / "example/results/simulation/mlp300_pattern_repeat30",
    "MNIST Digit-Pair\nArithmetic": ROOT / "example/results/simulation/mlp_mnist_arithmetic_repeat30",
}
TRANSFORMER_SIMULATION_DIR = ROOT / "example/results/simulation/transformer_binary_repeat30"

RULE_COUNT = 5
DEMO_CUTOFF_LABEL = "25"
EXPORT_EXTENSIONS = ("png", "pdf", "svg")
MM_PER_INCH = 25.4
NCOMMS_DOUBLE_WIDTH_IN = 180.0 / MM_PER_INCH
NATURE_MAX_HEIGHT_IN = 166.0 / MM_PER_INCH
NATURE_FONT_PT = 7.0
NATURE_TICK_PT = 6.0
NATURE_SMALL_PT = 5.5
RULE_COLORS = [
    "#0072BD",  # Rule 1: blue
    "#D95319",  # Rule 2: orange
    "#EDB120",  # Rule 3: yellow
    "#7E2F8E",  # Rule 4: purple
    "#77AC30",  # Rule 5: green
]


def configure_style() -> None:
    ana.configure_publication_style()
    matplotlib.rcParams.update(
        {
            "font.size": NATURE_TICK_PT,
            "axes.titlesize": NATURE_FONT_PT,
            "axes.labelsize": NATURE_FONT_PT,
            "xtick.labelsize": NATURE_TICK_PT,
            "ytick.labelsize": NATURE_TICK_PT,
            "legend.fontsize": NATURE_TICK_PT,
            "legend.title_fontsize": NATURE_TICK_PT,
            "svg.fonttype": "none",
        }
    )


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def export_figure(fig: plt.Figure, path_stem: Path) -> list[str]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in EXPORT_EXTENSIONS:
        path = path_stem.with_suffix("." + ext)
        fig.savefig(path, dpi=ana.PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.03)
        written.append(str(path))
    return written


def analysis_mat_paths(analysis_dir: Path) -> list[Path]:
    paths = sorted(analysis_dir.glob("run_*/analysis.mat"))
    if not paths:
        raise FileNotFoundError("No analysis.mat files found under {}".format(analysis_dir))
    return paths




def validate_analysis_source(analysis_dir: Path, simulation_dir: Path) -> list[str]:
    simulation_runs = sorted(path.parent.name for path in simulation_dir.glob("run_*/run_data.npz"))
    analysis_runs = sorted(path.parent.name for path in analysis_dir.glob("run_*/analysis.mat"))
    if not simulation_runs:
        raise FileNotFoundError("No run_*/run_data.npz files found under {}".format(simulation_dir))
    missing = sorted(set(simulation_runs) - set(analysis_runs))
    if missing:
        raise FileNotFoundError(
            "Analysis cache {} is missing simulation run(s): {}".format(analysis_dir, ", ".join(missing))
        )
    return simulation_runs


def load_mat_stack(analysis_dir: Path, key: str) -> np.ndarray:
    arrays = []
    for path in analysis_mat_paths(analysis_dir):
        mat = loadmat(path, variable_names=[key])
        if key not in mat:
            raise KeyError("{} missing from {}".format(key, path))
        arrays.append(np.asarray(mat[key], dtype=float))
    return np.stack(arrays, axis=0)


def zero_based_combo_labels(rule_count: int = RULE_COUNT) -> list[str]:
    return [ana.combo_label(combo, list(range(rule_count))) for combo in ana.combo_list(rule_count)]


def demo_combo_labels(rule_count: int = RULE_COUNT, cutoff_label: str = DEMO_CUTOFF_LABEL) -> tuple[int, list[str]]:
    raw_labels = zero_based_combo_labels(rule_count)
    keep_count = ana._demo_combo_count_until_rule_set(raw_labels, cutoff_label=cutoff_label)
    labels = [ana._one_based_hyphen_combo_label(label) for label in raw_labels[:keep_count]]
    return keep_count, labels


def rule_labels(rule_count: int = RULE_COUNT) -> list[str]:
    return ["rule {}".format(idx + 1) for idx in range(rule_count)]


def task_title(label: str) -> str:
    return label.replace("\n", " ")


def safe_name(label: str) -> str:
    return label.replace("\n", "_").replace(" ", "_").replace("-", "_").lower()


def finite_epsilon(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1e-6
    span = float(np.nanmax(finite) - np.nanmin(finite))
    return max(0.006, span * 0.012)


def style_violin_parts(parts: dict[str, Any], body_rule_ids: list[int], colors: list[str]) -> None:
    for body, rule_idx in zip(parts["bodies"], body_rule_ids):
        color = colors[rule_idx % len(colors)]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.66)
        body.set_linewidth(0.95)
    for key, color, width in (("cmeans", "#222222", 1.0), ("cmedians", "#B23A48", 1.25)):
        if key in parts:
            parts[key].set_color(color)
            parts[key].set_linewidth(width)


def apply_accuracy_axis_style(ax: plt.Axes, values: np.ndarray) -> None:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    lower, upper = ana._retained_ratio_ylim(finite)
    tick_step = 0.1 if upper - lower <= 1.2 else 0.2
    ax.set_ylim(lower, upper)
    ax.set_yticks(np.round(np.arange(lower, upper + tick_step * 0.5, tick_step), 2))
    if lower <= 1.0 <= upper:
        ax.axhline(1.0, color="#666666", linewidth=0.8, linestyle="--", alpha=0.42, zorder=0)
    ana._style_publication_axes(ax, grid=True)


def plot_grouped_violin(
    ax: plt.Axes,
    retained: np.ndarray,
    group_labels: list[str],
    series_labels: list[str],
    title: str,
    xlabel: str,
    ylabel: str = "relative accuracy",
    rotate: int = 0,
    show_legend: bool = True,
    tick_size: float | None = None,
) -> None:
    retained = np.asarray(retained, dtype=float)
    if retained.ndim != 3:
        raise ValueError("retained must have shape runs x series x groups")
    num_runs, num_series, num_groups = retained.shape
    if num_runs == 0 or num_series == 0 or num_groups == 0:
        raise ValueError("retained has an empty dimension")

    colors = [RULE_COLORS[idx % len(RULE_COLORS)] for idx in range(num_series)]
    group_gap = 1.45 if num_series <= 3 else 1.65
    group_width = 0.50 if num_series <= 3 else 0.68
    offsets = np.linspace(-group_width / 2.0, group_width / 2.0, num_series) if num_series > 1 else np.asarray([0.0])
    width = min(0.17, group_width / max(num_series + 0.95, 1.0))
    epsilon = finite_epsilon(retained)
    centers = np.arange(num_groups, dtype=float) * group_gap

    data = []
    positions = []
    body_rule_ids = []
    for group_idx in range(num_groups):
        for series_idx, offset in enumerate(offsets):
            values = ana._violin_values(retained[:, series_idx, group_idx], epsilon)
            if values.size:
                data.append(values)
                positions.append(float(centers[group_idx] + offset))
                body_rule_ids.append(series_idx)

    if data:
        parts = ax.violinplot(
            data,
            positions=positions,
            widths=width,
            showmeans=True,
            showmedians=True,
            showextrema=False,
        )
        style_violin_parts(parts, body_rule_ids, colors)

    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(float(centers[0] - 0.68), float(centers[-1] + 0.68))
    ax.set_xticks(centers)
    ax.set_xticklabels(group_labels, rotation=rotate, ha="right" if rotate else "center")
    if tick_size is not None:
        ax.tick_params(axis="x", labelsize=tick_size)
    apply_accuracy_axis_style(ax, retained)

    if show_legend:
        handles = [
            plt.Line2D([0], [0], color=colors[idx], linewidth=5, alpha=0.75, label=series_labels[idx])
            for idx in range(num_series)
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.16),
            ncol=min(num_series, 5),
            frameon=False,
            handlelength=1.1,
            handletextpad=0.35,
            columnspacing=0.75,
            borderaxespad=0.0,
        )


def set_demo_typography(ax: plt.Axes, compact: bool) -> None:
    ax.xaxis.label.set_size(NATURE_FONT_PT)
    ax.yaxis.label.set_size(NATURE_FONT_PT)
    ax.title.set_size(NATURE_FONT_PT)
    if compact:
        ax.tick_params(axis="both", labelsize=NATURE_SMALL_PT, length=2.8, width=1.0, pad=1.6)
    else:
        ax.tick_params(axis="both", labelsize=NATURE_TICK_PT, length=3.0, width=1.0, pad=1.8)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(NATURE_FONT_PT)
        for line in legend.get_lines():
            line.set_linewidth(4.2)


def plot_retained_demo_axes(
    ax: plt.Axes,
    retained: np.ndarray,
    title: str,
    xlabel: str = "rule set",
    show_legend: bool = True,
    compact: bool = False,
    tick_size: float | None = None,
    rotate: int = 0,
) -> None:
    keep_count, labels = demo_combo_labels()
    plot_grouped_violin(
        ax,
        retained[:, :, :keep_count],
        labels,
        rule_labels(),
        title,
        xlabel=xlabel,
        rotate=rotate,
        show_legend=show_legend,
        tick_size=tick_size,
    )
    set_demo_typography(ax, compact=compact)


def build_task_retained_data(task_dirs: dict[str, Path]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    retained_by_task = {}
    run_counts = {}
    for label, analysis_dir in task_dirs.items():
        retained = load_mat_stack(analysis_dir, "weight_combo_retained_accuracy")
        retained_by_task[label] = retained
        run_counts[task_title(label)] = int(retained.shape[0])
    return retained_by_task, run_counts


def taylor_sensitivity_matrix(profile_path: Path, fallback_retained: np.ndarray) -> tuple[np.ndarray, str]:
    if profile_path.is_file():
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        matrix = np.asarray(data.get("matrix", []), dtype=float)
        source = str(profile_path)
    else:
        matrix = np.nanmean(fallback_retained[:, :, :RULE_COUNT], axis=0).T
        source = "fallback from Taylor weight-retained single-rule accuracy"
    if matrix.shape != (RULE_COUNT, RULE_COUNT):
        raise ValueError("Taylor sensitivity matrix must be {}x{}, got {}".format(RULE_COUNT, RULE_COUNT, matrix.shape))
    row_max = np.nanmax(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(row_max > 0, row_max, 1.0)
    return matrix, source


def plot_taylor_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    with_colorbar: bool = True,
) -> None:
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    ticks = np.arange(RULE_COUNT)
    labels = [str(idx + 1) for idx in range(RULE_COUNT)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("evaluated rule")
    ax.set_ylabel("Taylor-token rule")
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    ax.xaxis.label.set_size(NATURE_FONT_PT)
    ax.yaxis.label.set_size(NATURE_FONT_PT)
    ax.title.set_size(NATURE_FONT_PT)
    ax.tick_params(axis="both", labelsize=NATURE_TICK_PT, length=2.8, width=1.0, pad=1.6)
    for row in range(RULE_COUNT):
        for col in range(RULE_COUNT):
            value = float(matrix[row, col])
            color = "white" if value > 0.52 else "#222222"
            ax.text(col, row, "{:.2f}".format(value), ha="center", va="center", color=color, fontsize=NATURE_SMALL_PT)
    ana._style_publication_axes(ax)
    for side in ("top", "right"):
        ax.spines[side].set_visible(True)
    if with_colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
        cbar.set_label("relative Taylor score", fontsize=NATURE_TICK_PT)
        cbar.ax.tick_params(direction="out", length=2.6, width=1.0, labelsize=NATURE_SMALL_PT)


def plot_fig4a_tasks(retained_by_task: dict[str, np.ndarray], output_root: Path) -> list[str]:
    fig, axes = plt.subplots(1, len(retained_by_task), figsize=(NCOMMS_DOUBLE_WIDTH_IN, 2.28), sharey=True, constrained_layout=True)
    if len(retained_by_task) == 1:
        axes = [axes]
    for idx, (ax, (label, retained)) in enumerate(zip(axes, retained_by_task.items())):
        title = "a {}".format(task_title(label)) if idx == 0 else task_title(label)
        plot_retained_demo_axes(ax, retained, title, show_legend=False, compact=True, tick_size=NATURE_SMALL_PT, rotate=42)
        if idx > 0:
            ax.set_ylabel("")
    handles = [
        plt.Line2D([0], [0], color=RULE_COLORS[idx], linewidth=5, alpha=0.75, label=rule_labels()[idx])
        for idx in range(RULE_COUNT)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=RULE_COUNT, frameon=False, fontsize=NATURE_FONT_PT)
    written = export_figure(fig, output_root / "fig4a_retained_combo_accuracy_demo")
    plt.close(fig)
    return written


def plot_fig4b(matrix: np.ndarray, output_root: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(3.15, 2.85))
    plot_taylor_heatmap(ax, matrix, "b Taylor sensitivity profile", with_colorbar=True)
    fig.tight_layout()
    written = export_figure(fig, output_root / "fig4b_taylor_sensitivity_profile")
    plt.close(fig)
    return written


def plot_fig4c(taylor_retained: np.ndarray, output_root: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(NCOMMS_DOUBLE_WIDTH_IN, 2.28))
    plot_retained_demo_axes(ax, taylor_retained, "c Taylor MLP weight subnetworks", show_legend=True, compact=False)
    written = export_figure(fig, output_root / "fig4c_taylor_mlp_retained_combo_accuracy_demo")
    plt.close(fig)
    return written


def plot_fig4d(transformer_retained: np.ndarray, output_root: Path) -> list[str]:
    fig, ax = plt.subplots(figsize=(NCOMMS_DOUBLE_WIDTH_IN, 2.28))
    plot_retained_demo_axes(ax, transformer_retained, "d TinyTransformer weight-token subnetworks", show_legend=True, compact=False)
    written = export_figure(fig, output_root / "fig4d_transformer_weight_retained_combo_accuracy_demo")
    plt.close(fig)
    return written


def plot_combined(
    retained_by_task: dict[str, np.ndarray],
    taylor_matrix: np.ndarray,
    taylor_retained: np.ndarray,
    transformer_retained: np.ndarray,
    output_root: Path,
) -> list[str]:
    fig = plt.figure(figsize=(NCOMMS_DOUBLE_WIDTH_IN, NATURE_MAX_HEIGHT_IN), constrained_layout=True)
    grid = fig.add_gridspec(3, 6, height_ratios=(1.0, 1.08, 1.02))
    task_axes = [fig.add_subplot(grid[0, idx * 2 : idx * 2 + 2]) for idx in range(len(retained_by_task))]
    ax_b = fig.add_subplot(grid[1, 0:2])
    ax_c = fig.add_subplot(grid[1, 2:6])
    ax_d = fig.add_subplot(grid[2, 0:6])

    for idx, (ax, (label, retained)) in enumerate(zip(task_axes, retained_by_task.items())):
        title = "a {}".format(task_title(label)) if idx == 0 else task_title(label)
        plot_retained_demo_axes(ax, retained, title, show_legend=False, compact=True, tick_size=5.0, rotate=48)
        if idx > 0:
            ax.set_ylabel("")
    plot_taylor_heatmap(ax_b, taylor_matrix, "b Taylor sensitivity profile", with_colorbar=True)
    plot_retained_demo_axes(ax_c, taylor_retained, "c Taylor MLP weight subnetworks", show_legend=False, compact=True, tick_size=NATURE_SMALL_PT)
    plot_retained_demo_axes(ax_d, transformer_retained, "d TinyTransformer weight-token subnetworks", show_legend=False, compact=True, tick_size=NATURE_SMALL_PT)

    handles = [
        plt.Line2D([0], [0], color=RULE_COLORS[idx], linewidth=5, alpha=0.75, label=rule_labels()[idx])
        for idx in range(RULE_COUNT)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.018), ncol=RULE_COUNT, frameon=False, fontsize=NATURE_FONT_PT)
    written = export_figure(fig, output_root / "fig4")
    plt.close(fig)
    return written

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw Fig. 4 panels from cached example/results analysis outputs.")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--binary-dir", type=Path, default=TASK_DIRS["Binary Sequence\nCalculation"])
    parser.add_argument("--pattern-dir", type=Path, default=TASK_DIRS["Sparse Pattern\nLogic"])
    parser.add_argument("--mnist-dir", type=Path, default=TASK_DIRS["MNIST Digit-Pair\nArithmetic"])
    parser.add_argument("--taylor-mlp-dir", type=Path, default=TAYLOR_MLP_DIR)
    parser.add_argument("--transformer-dir", type=Path, default=TRANSFORMER_DIR)
    parser.add_argument("--taylor-profile", type=Path, default=FIG4_SUPP_DIR / "taylor_sensitivity_profile.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_style()
    args.output_root.mkdir(parents=True, exist_ok=True)

    task_dirs = {
        "Binary Sequence\nCalculation": args.binary_dir,
        "Sparse Pattern\nLogic": args.pattern_dir,
        "MNIST Digit-Pair\nArithmetic": args.mnist_dir,
    }
    for label, analysis_dir in task_dirs.items():
        validate_analysis_source(analysis_dir, TASK_SIMULATION_DIRS[label])
    validate_analysis_source(args.taylor_mlp_dir, TASK_SIMULATION_DIRS["Binary Sequence\nCalculation"])
    validate_analysis_source(args.transformer_dir, TRANSFORMER_SIMULATION_DIR)

    retained_by_task, task_run_counts = build_task_retained_data(task_dirs)
    taylor_retained = load_mat_stack(args.taylor_mlp_dir, "weight_combo_retained_accuracy")
    transformer_retained = load_mat_stack(args.transformer_dir, "weight_combo_retained_accuracy")
    keep_count, combo_labels = demo_combo_labels()
    taylor_matrix, taylor_matrix_source = taylor_sensitivity_matrix(args.taylor_profile, taylor_retained)

    outputs = []
    outputs.extend(plot_fig4a_tasks(retained_by_task, args.output_root))
    outputs.extend(plot_fig4b(taylor_matrix, args.output_root))
    outputs.extend(plot_fig4c(taylor_retained, args.output_root))
    outputs.extend(plot_fig4d(transformer_retained, args.output_root))
    outputs.extend(plot_combined(retained_by_task, taylor_matrix, taylor_retained, transformer_retained, args.output_root))
    plt.close("all")

    summary = {
        "description": "Figure 4 panels generated from cached analysis.mat outputs; no model analysis is rerun. Main combined figure is sized for double-column journal output with editable sans-serif text.",
        "functional_token_plot_format": "All weight-based functional-token retained-accuracy panels use the retained_combo_accuracy_demo style: rule-set x-axis, rule 1-5 legend, and combo labels kept through {}.".format(combo_labels[-1]),
        "panel_a": {
            "metric": "weight_combo_retained_accuracy demo-style functional-token plots for each task family",
            "simulation_dirs": {task_title(label): str(TASK_SIMULATION_DIRS[label]) for label in task_dirs},
            "task_dirs": {task_title(label): str(path) for label, path in task_dirs.items()},
            "run_counts": task_run_counts,
        },
        "panel_b": {
            "metric": "row-normalized task-specific Taylor sensitivity profile",
            "source": taylor_matrix_source,
        },
        "panel_c": {
            "metric": "Taylor MLP weight_combo_retained_accuracy demo-style functional-token plot",
            "simulation_dir": str(TASK_SIMULATION_DIRS["Binary Sequence\nCalculation"]),
            "source": str(args.taylor_mlp_dir),
            "runs": int(taylor_retained.shape[0]),
        },
        "panel_d": {
            "metric": "TinyTransformer Taylor weight_combo_retained_accuracy demo-style functional-token plot",
            "simulation_dir": str(TRANSFORMER_SIMULATION_DIR),
            "source": str(args.transformer_dir),
            "runs": int(transformer_retained.shape[0]),
        },
        "demo_rule_set_labels": combo_labels,
        "demo_rule_set_count": int(keep_count),
        "outputs": outputs,
    }
    save_json(args.output_root / "figure4_summary.json", summary)
    print("Wrote Figure 4 outputs to {}".format(args.output_root))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
