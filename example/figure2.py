from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis as ana


EXAMPLE_ROOT = Path(__file__).resolve().parent
DEFAULT_SIMULATION_DIR = EXAMPLE_ROOT / "results/simulation/mlp300_binary_repeat30"
DEFAULT_ANALYSIS_DIR = EXAMPLE_ROOT / "results/score_ratio_analysis/mlp300_binary_repeat30"
DEFAULT_OUTPUT_DIR = EXAMPLE_ROOT / "results/figure2"
LAYER_COLORS = ["#4DBEEE", "#A2142F", "#0072BD"]
RULE_COLORS = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30"]
PUBLICATION_DPI = int(getattr(ana, "PUBLICATION_DPI", 600))
NATURE_BODY_FONT_PT = 10.0
NATURE_PANEL_LABEL_FONT_PT = 10.5


def normal_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def configure_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "DejaVu Sans"],
            "font.size": NATURE_BODY_FONT_PT,
            "axes.titlesize": NATURE_BODY_FONT_PT,
            "axes.labelsize": NATURE_BODY_FONT_PT,
            "xtick.labelsize": NATURE_BODY_FONT_PT,
            "ytick.labelsize": NATURE_BODY_FONT_PT,
            "legend.fontsize": NATURE_BODY_FONT_PT,
            "legend.title_fontsize": NATURE_BODY_FONT_PT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def analysis_mat_paths(analysis_dir: Path, max_runs: int | None = None) -> list[Path]:
    paths = sorted(analysis_dir.glob("run_*/analysis.mat"), key=lambda item: item.parent.name)
    if not paths:
        raise FileNotFoundError(f"no run_*/analysis.mat files found under {analysis_dir}")
    if max_runs is not None:
        paths = paths[: max(0, max_runs)]
    if not paths:
        raise ValueError("max_runs selected zero cached runs")
    return paths


def validate_analysis_source(analysis_dir: Path, simulation_dir: Path) -> list[str]:
    simulation_runs = sorted(path.parent.name for path in simulation_dir.glob("run_*/run_data.npz"))
    analysis_runs = sorted(path.parent.name for path in analysis_dir.glob("run_*/analysis.mat"))
    if not simulation_runs:
        raise FileNotFoundError("no run_*/run_data.npz files found under {}".format(simulation_dir))
    missing = sorted(set(simulation_runs) - set(analysis_runs))
    if missing:
        raise FileNotFoundError(
            "analysis cache {} is missing simulation run(s): {}".format(analysis_dir, ", ".join(missing))
        )
    return simulation_runs


def load_figure2_arrays(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    exclusive_counts = []
    retained_accuracy = []
    for path in paths:
        data = loadmat(path)
        missing = [key for key in ("weight_combo_exclusive_counts", "weight_combo_retained_accuracy") if key not in data]
        if missing:
            raise KeyError(f"{path} is missing cached field(s): {', '.join(missing)}")
        exclusive = np.asarray(data["weight_combo_exclusive_counts"], dtype=float)
        retained = np.asarray(data["weight_combo_retained_accuracy"], dtype=float)
        if exclusive.ndim != 2:
            raise ValueError(f"{path}: weight_combo_exclusive_counts must be 2-D, got shape {exclusive.shape}")
        if retained.ndim != 2:
            raise ValueError(f"{path}: weight_combo_retained_accuracy must be 2-D, got shape {retained.shape}")
        if exclusive.shape[0] > len(LAYER_COLORS):
            exclusive = exclusive[: len(LAYER_COLORS), :]
        exclusive_counts.append(exclusive)
        retained_accuracy.append(retained)
    return np.stack(exclusive_counts, axis=0), np.stack(retained_accuracy, axis=0)


def combo_labels(num_rules: int, num_combos: int) -> list[str]:
    rule_indices = list(range(num_rules))
    combos = ana.combo_list(num_rules)
    if len(combos) < num_combos:
        raise ValueError(f"need {num_combos} rule-set labels, but only generated {len(combos)}")
    zero_based = [ana.combo_label(combo, rule_indices) for combo in combos[:num_combos]]
    return [ana._one_based_hyphen_combo_label(label) for label in zero_based]


def finite_values(values: np.ndarray) -> np.ndarray:
    finite = np.asarray(values, dtype=float)
    return finite[np.isfinite(finite)]


def count_ylim(values: np.ndarray) -> tuple[float, float]:
    finite = finite_values(values)
    if finite.size == 0:
        return 0.0, 1.0
    upper = float(np.nanmax(finite))
    if upper <= 10:
        step = 1.0
    elif upper <= 50:
        step = 5.0
    elif upper <= 200:
        step = 20.0
    else:
        step = 50.0
    return 0.0, float(np.ceil((upper + step * 0.12) / step) * step)


def set_count_ticks(ax: plt.Axes, upper: float) -> None:
    if upper <= 10:
        step = 1.0
    elif upper <= 50:
        step = 5.0
    elif upper <= 200:
        step = 20.0
    else:
        step = 50.0
    ax.set_yticks(np.arange(0.0, upper + step * 0.5, step))


def apply_axis_style(ax: plt.Axes) -> None:
    if hasattr(ana, "_style_publication_axes"):
        ana._style_publication_axes(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=3.2, width=0.8, pad=2)


def add_panel_label(ax: plt.Axes, label: str | None) -> None:
    if label:
        ax.text(
            -0.055,
            1.04,
            label,
            transform=ax.transAxes,
            fontsize=NATURE_PANEL_LABEL_FONT_PT,
            fontweight="bold",
            ha="left",
            va="bottom",
        )


def visible_violin_epsilon(values: np.ndarray, *, accuracy_axis: bool = False, log1p_y: bool = False) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return 1e-6
    span = float(np.nanmax(finite) - np.nanmin(finite))
    if accuracy_axis:
        floor = 0.006
    elif log1p_y:
        floor = 0.025
    else:
        floor = 1e-6
    return max(floor, span * 0.012)


def overlay_violin_observations(
    ax: plt.Axes,
    raw_samples: list[np.ndarray],
    positions: list[float],
    hue_ids: list[int],
    colors: list[str],
    violin_width: float,
    *,
    show_points: bool = False,
    seed: int = 20260819,
) -> None:
    rng = np.random.default_rng(seed)
    max_points = 18
    jitter = min(violin_width * 0.22, 0.026)
    half_line = max(violin_width * 0.34, 0.025)
    for raw, position, hue_idx in zip(raw_samples, positions, hue_ids):
        values = finite_values(raw)
        if values.size == 0:
            continue
        shown = values
        if show_points and values.size > max_points:
            shown = rng.choice(values, size=max_points, replace=False)
        if show_points and shown.size:
            xs = np.full(shown.size, position, dtype=float)
            if jitter > 0.0:
                xs = xs + rng.uniform(-jitter, jitter, size=shown.size)
            ax.scatter(
                xs,
                shown,
                s=5.5,
                color=colors[hue_idx % len(colors)],
                alpha=0.34,
                edgecolors="none",
                linewidths=0.0,
                zorder=3,
            )
        ax.hlines(float(np.nanmean(values)), position - half_line, position + half_line, color="#222222", linewidth=0.85, zorder=4)
        ax.hlines(float(np.nanmedian(values)), position - half_line, position + half_line, color="#b23a48", linewidth=1.05, zorder=4)


def draw_grouped_violin(
    ax: plt.Axes,
    values: np.ndarray,
    x_labels: list[str],
    hue_labels: list[str],
    hue_colors: list[str],
    *,
    xlabel: str,
    ylabel: str,
    legend_title: str,
    panel_label: str | None = None,
    accuracy_axis: bool = False,
    label_rotation: float = 45.0,
    legend_anchor: tuple[float, float] = (0.5, 1.16),
    legend_columns: int | None = None,
    log1p_y: bool = False,
    show_points: bool = False,
) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"grouped violin values must have shape run x hue x group, got {values.shape}")
    num_runs, num_hues, num_groups = values.shape
    if num_runs == 0 or num_hues == 0 or num_groups == 0:
        raise ValueError("grouped violin values are empty")
    if len(x_labels) != num_groups:
        raise ValueError(f"got {len(x_labels)} x labels for {num_groups} groups")
    if len(hue_labels) != num_hues:
        raise ValueError(f"got {len(hue_labels)} hue labels for {num_hues} hue levels")

    if log1p_y:
        finite_raw = finite_values(values)
        if finite_raw.size and float(np.nanmin(finite_raw)) < 0.0:
            raise ValueError("log1p_y requires non-negative values")
        values = np.log1p(values)

    finite = finite_values(values)
    if finite.size == 0:
        raise ValueError("grouped violin values contain no finite numbers")
    epsilon = visible_violin_epsilon(finite, accuracy_axis=accuracy_axis, log1p_y=log1p_y)

    group_gap = 1.45 if num_hues <= 3 else 1.65
    group_width = 0.50 if num_hues <= 3 else 0.68
    offsets = np.linspace(-group_width / 2.0, group_width / 2.0, num_hues) if num_hues > 1 else np.asarray([0.0])
    violin_width = min(0.18, group_width / max(num_hues + 0.95, 1.0))
    if num_hues <= 3:
        violin_width = min(0.22, violin_width * 1.15)

    data = []
    raw_samples = []
    positions = []
    hue_ids = []
    x_positions = np.arange(num_groups, dtype=float) * group_gap
    for group_idx in range(num_groups):
        for hue_idx in range(num_hues):
            raw = finite_values(values[:, hue_idx, group_idx])
            sample = ana._violin_values(raw, epsilon)
            if sample.size:
                data.append(sample)
                raw_samples.append(raw)
                positions.append(float(x_positions[group_idx] + offsets[hue_idx]))
                hue_ids.append(hue_idx)
    if not data:
        raise ValueError("no non-empty violin samples were found")

    parts = ax.violinplot(
        data,
        positions=positions,
        widths=violin_width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body, hue_idx in zip(parts["bodies"], hue_ids):
        color = hue_colors[hue_idx % len(hue_colors)]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.68)
        body.set_linewidth(0.8)
    for key, color, width in (("cmeans", "#222222", 0.9), ("cmedians", "#b23a48", 1.15)):
        if key in parts:
            parts[key].set_color(color)
            parts[key].set_linewidth(width)
    overlay_violin_observations(
        ax, raw_samples, positions, hue_ids, hue_colors, violin_width, show_points=show_points
    )

    ax.set_xlim(float(x_positions[0] - 0.68), float(x_positions[-1] + 0.68))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=label_rotation, ha="right" if label_rotation else "center")
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    if accuracy_axis:
        lower, upper = ana._retained_ratio_ylim(finite)
        ax.set_ylim(lower, upper)
        tick_step = 0.1 if upper - lower <= 1.2 else 0.2
        ax.set_yticks(np.round(np.arange(lower, upper + tick_step * 0.5, tick_step), 2))
        if lower <= 1.0 <= upper:
            ax.axhline(1.0, color="#666666", linewidth=0.85, linestyle="--", alpha=0.45, zorder=0)
    elif log1p_y:
        lower = 0.0
        upper = max(1.0, float(np.ceil(float(np.nanmax(finite)) + 0.1)))
        ax.set_ylim(lower, upper)
        ax.set_yticks(np.arange(lower, upper + 0.5, 1.0))
    else:
        lower, upper = count_ylim(finite)
        ax.set_ylim(lower, upper)
        set_count_ticks(ax, upper)
    apply_axis_style(ax)
    add_panel_label(ax, panel_label)

    handles = [
        plt.Line2D([0], [0], color=hue_colors[idx % len(hue_colors)], linewidth=5, alpha=0.78, label=hue_labels[idx])
        for idx in range(num_hues)
    ]
    ax.legend(
        handles=handles,
        title=legend_title,
        loc="upper center",
        bbox_to_anchor=legend_anchor,
        ncol=legend_columns or min(num_hues, 5),
        frameon=False,
        handlelength=1.15,
        handletextpad=0.35,
        columnspacing=0.8,
        borderaxespad=0.0,
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if fmt.lower() in {"png", "jpg", "jpeg", "tif", "tiff"}:
            save_kwargs["dpi"] = PUBLICATION_DPI
        fig.savefig(path, **save_kwargs)
        paths.append(path)
    return paths


def plot_fig2a(exclusive_counts: np.ndarray, labels: list[str], output_dir: Path, formats: list[str]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.0, 3.45))
    draw_grouped_violin(
        ax,
        exclusive_counts,
        labels,
        ["Layer 1", "Layer 2", "Layer 3"],
        LAYER_COLORS,
        xlabel="rule set",
        ylabel="log(weight-token count + 1)",
        legend_title="layer",
        panel_label="a",
        accuracy_axis=False,
        log1p_y=True,
        label_rotation=55,
        legend_anchor=(0.5, 1.15),
        legend_columns=3,
    )
    fig.tight_layout(pad=0.35)
    paths = save_figure(fig, output_dir, "fig2a_exclusive_weight_tokens", formats)
    plt.close(fig)
    return paths


def plot_fig2b(retained_accuracy: np.ndarray, labels: list[str], output_dir: Path, formats: list[str]) -> list[Path]:
    single_count = min(5, retained_accuracy.shape[2], len(labels))
    fig, ax = plt.subplots(figsize=(4.9, 3.25))
    draw_grouped_violin(
        ax,
        retained_accuracy[:, :, :single_count],
        labels[:single_count],
        [f"rule {idx + 1}" for idx in range(retained_accuracy.shape[1])],
        RULE_COLORS,
        xlabel="rule set",
        ylabel="relative accuracy",
        legend_title="evaluated rule",
        panel_label="b",
        accuracy_axis=True,
        label_rotation=0,
        legend_anchor=(0.5, 1.18),
        legend_columns=5,
    )
    fig.tight_layout(pad=0.35)
    paths = save_figure(fig, output_dir, "fig2b_single_rule_accuracy", formats)
    plt.close(fig)
    return paths


def plot_fig2c(retained_accuracy: np.ndarray, labels: list[str], output_dir: Path, formats: list[str]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.4, 3.55))
    draw_grouped_violin(
        ax,
        retained_accuracy,
        labels,
        [f"rule {idx + 1}" for idx in range(retained_accuracy.shape[1])],
        RULE_COLORS,
        xlabel="rule set",
        ylabel="relative accuracy",
        legend_title="evaluated rule",
        panel_label="c",
        accuracy_axis=True,
        label_rotation=55,
        legend_anchor=(0.5, 1.16),
        legend_columns=5,
    )
    fig.tight_layout(pad=0.35)
    paths = save_figure(fig, output_dir, "fig2c_rule_set_accuracy", formats)
    plt.close(fig)
    return paths


def plot_combined_fig2(
    exclusive_counts: np.ndarray,
    retained_accuracy: np.ndarray,
    labels: list[str],
    output_dir: Path,
    formats: list[str],
) -> list[Path]:
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.4, 9.5),
        gridspec_kw={"height_ratios": [1.0, 0.84, 1.05], "hspace": 0.63},
    )
    draw_grouped_violin(
        axes[0],
        exclusive_counts,
        labels,
        ["Layer 1", "Layer 2", "Layer 3"],
        LAYER_COLORS,
        xlabel="rule set",
        ylabel="log(weight-token count + 1)",
        legend_title="layer",
        panel_label="a",
        accuracy_axis=False,
        log1p_y=True,
        label_rotation=55,
        legend_anchor=(0.5, 1.16),
        legend_columns=3,
    )
    single_count = min(5, retained_accuracy.shape[2], len(labels))
    draw_grouped_violin(
        axes[1],
        retained_accuracy[:, :, :single_count],
        labels[:single_count],
        [f"rule {idx + 1}" for idx in range(retained_accuracy.shape[1])],
        RULE_COLORS,
        xlabel="rule set",
        ylabel="relative accuracy",
        legend_title="evaluated rule",
        panel_label="b",
        accuracy_axis=True,
        label_rotation=0,
        legend_anchor=(0.5, 1.21),
        legend_columns=5,
    )
    draw_grouped_violin(
        axes[2],
        retained_accuracy,
        labels,
        [f"rule {idx + 1}" for idx in range(retained_accuracy.shape[1])],
        RULE_COLORS,
        xlabel="rule set",
        ylabel="relative accuracy",
        legend_title="evaluated rule",
        panel_label="c",
        accuracy_axis=True,
        label_rotation=55,
        legend_anchor=(0.5, 1.16),
        legend_columns=5,
    )
    fig.subplots_adjust(left=0.08, right=0.995, top=0.965, bottom=0.075, hspace=0.78)
    paths = save_figure(fig, output_dir, "fig2", formats)
    plt.close(fig)
    return paths


def parse_formats(value: str) -> list[str]:
    formats = []
    for item in value.split(","):
        item = item.strip().lower().lstrip(".")
        if item and item not in formats:
            formats.append(item)
    if not formats:
        raise ValueError("at least one output format is required")
    return formats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw Fig. 2 panels from cached run_*/analysis.mat results without rerunning analysis."
    )
    parser.add_argument(
        "analysis_dir",
        nargs="?",
        default=str(DEFAULT_ANALYSIS_DIR),
        help="Path like example/results/score_ratio_analysis/mlp300_binary_repeat30.",
    )
    parser.add_argument(
        "--simulation-dir",
        default=str(DEFAULT_SIMULATION_DIR),
        help="Raw simulation directory used to verify the cached analysis runs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for figure outputs. Defaults to example/results/figure2.",
    )
    parser.add_argument("--max-runs", type=int, default=None, help="Use only the first N sorted run_* caches.")
    parser.add_argument("--formats", default="png,svg,pdf", help="Comma-separated output formats.")
    parser.add_argument("--skip-combined", action="store_true", help="Only save fig2a/fig2b/fig2c panels.")
    args = parser.parse_args()

    configure_style()
    analysis_dir = normal_path(args.analysis_dir)
    simulation_dir = normal_path(args.simulation_dir)
    output_dir = normal_path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    formats = parse_formats(args.formats)
    validate_analysis_source(analysis_dir, simulation_dir)
    paths = analysis_mat_paths(analysis_dir, max_runs=args.max_runs)
    exclusive_counts, retained_accuracy = load_figure2_arrays(paths)
    labels = combo_labels(retained_accuracy.shape[1], retained_accuracy.shape[2])

    written: list[Path] = []
    written.extend(plot_fig2a(exclusive_counts, labels, output_dir, formats))
    written.extend(plot_fig2b(retained_accuracy, labels, output_dir, formats))
    written.extend(plot_fig2c(retained_accuracy, labels, output_dir, formats))
    if not args.skip_combined:
        written.extend(plot_combined_fig2(exclusive_counts, retained_accuracy, labels, output_dir, formats))

    summary = {
        "simulation_dir": str(simulation_dir),
        "analysis_dir": str(analysis_dir),
        "output_dir": str(output_dir),
        "runs": [str(path.parent.name) for path in paths],
        "run_count": len(paths),
        "combo_count": int(retained_accuracy.shape[2]),
        "rule_labels": [f"rule {idx + 1}" for idx in range(retained_accuracy.shape[1])],
        "combo_labels": labels,
        "fig2a_y_transform": "log1p(weight_token_counts_by_exact_rule_set)",
        "subnetwork_basis": "weight",
        "fig2a_weight_blocks": "first three non-output weight blocks",
        "figures": [str(path) for path in written],
    }
    summary_path = output_dir / "figure2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"loaded {len(paths)} cached runs from {analysis_dir}")
    print(f"wrote {len(written)} figure files under {output_dir}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
