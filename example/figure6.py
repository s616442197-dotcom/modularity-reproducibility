from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
import numpy as np
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis as ana
import ruleweight as rw


DEFAULT_RESULTS_ROOT = ROOT / "example/results/ruleweight"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "figures"
DEFAULT_ARTICLE_FIG6 = ROOT / "article/figs/fig6.pdf"
DEFAULT_FORMATS = "png,pdf,svg"
RULE_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def normal_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def parse_formats(value: str) -> list[str]:
    formats = []
    for item in str(value).split(","):
        fmt = item.strip().lower().lstrip(".")
        if fmt and fmt not in formats:
            formats.append(fmt)
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def configure_style() -> None:
    ana.configure_publication_style()
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("Figure 6 summary not found: {}".format(path))
    with path.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)

    target = str(summary.get("config", {}).get("analysis_subnet_target", "")).lower()
    if target != "weight":
        raise ValueError(
            "Figure 6 requires weight-based subnetwork results; found {!r} in {}".format(
                target or "missing",
                path,
            )
        )
    return summary


def condition_analysis_dir(results_root: Path, condition: dict[str, Any]) -> Path:
    recorded = normal_path(condition.get("analysis_dir", ""))
    if recorded.is_dir():
        return recorded
    candidate = results_root / "analysis" / recorded.name
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError("Rule-weight analysis directory not found for {}".format(condition))


def build_summary_from_ruleweight(results_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = results_root / "ruleweight_conditions.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Rule-weight condition manifest not found: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = list(manifest.get("conditions", []))
    if not conditions:
        raise ValueError("No rule-weight conditions were recorded in {}".format(manifest_path))

    condition_metrics = []
    for condition in conditions:
        upweighted_rule = condition.get("upweighted_rule")
        if upweighted_rule is not None:
            upweighted_rule = int(upweighted_rule)
        analysis_dir = condition_analysis_dir(results_root, condition)
        condition_metrics.append(rw.extract_condition_metrics(analysis_dir, upweighted_rule))

    previous_output_root = rw.OUTPUT_ROOT
    try:
        rw.OUTPUT_ROOT = results_root
        summary = rw.write_summary_tables(condition_metrics)
    finally:
        rw.OUTPUT_ROOT = previous_output_root
    return summary, condition_metrics


def figure_matrices(summary: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrices = summary["matrices"]
    whole_network = np.asarray(matrices["whole_network_accuracy_by_condition"], dtype=float)
    weight_delta = np.asarray(matrices["functional_token_weight_ratio_delta_vs_control"], dtype=float)
    specialization = np.asarray(matrices["specialization_margin_by_upweighted_rule"], dtype=float)

    if whole_network.shape[0] != whole_network.shape[1] + 1:
        raise ValueError(
            "Expected one control row plus one row per upweighted rule; got {}".format(
                whole_network.shape
            )
        )
    accuracy_delta = whole_network[1:, :] - whole_network[0:1, :]
    expected_shape = accuracy_delta.shape
    if weight_delta.shape != expected_shape or specialization.shape != expected_shape:
        raise ValueError(
            "Inconsistent Figure 6 matrix shapes: accuracy={}, weight={}, specialization={}".format(
                accuracy_delta.shape,
                weight_delta.shape,
                specialization.shape,
            )
        )
    return accuracy_delta, weight_delta, specialization


def matched_target_statistics(
    summary: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = summary["rows"]
    rule_count = len(summary["config"]["rule_indices"])
    x_values = []
    y_values = []
    x_errors = []
    y_errors = []

    for rule_id in range(rule_count):
        control = next(
            row
            for row in rows
            if bool(row.get("is_control")) and int(row["target_rule"]) == rule_id
        )
        upweighted = next(
            row
            for row in rows
            if row.get("upweighted_rule") == rule_id and int(row["target_rule"]) == rule_id
        )
        x_values.append(
            float(upweighted["whole_network_accuracy_mean"])
            - float(control["whole_network_accuracy_mean"])
        )
        y_values.append(
            float(upweighted["functional_token_weight_ratio_mean"])
            - float(control["functional_token_weight_ratio_mean"])
        )
        x_errors.append(
            np.hypot(
                float(upweighted["whole_network_accuracy_sem"]),
                float(control["whole_network_accuracy_sem"]),
            )
        )
        y_errors.append(
            np.hypot(
                float(upweighted["functional_token_weight_ratio_sem"]),
                float(control["functional_token_weight_ratio_sem"]),
            )
        )

    return tuple(
        np.asarray(values, dtype=float)
        for values in (x_values, y_values, x_errors, y_errors)
    )


def symmetric_limit(values: np.ndarray, floor: float = 1e-12) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return floor
    return max(floor, float(np.nanmax(np.abs(finite))))


def contrasting_text_color(im, value: float) -> str:
    red, green, blue, _alpha = im.cmap(im.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.48 else "#1A1A1A"


def add_panel_letter(ax, letter: str) -> None:
    ax.text(
        -0.14,
        1.10,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        clip_on=False,
    )


def draw_heatmap(
    fig: plt.Figure,
    ax,
    matrix: np.ndarray,
    panel_letter: str,
    title: str,
    colorbar_label: str,
    value_format: str,
    cmap: str,
    diverging: bool,
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if diverging:
        limit = symmetric_limit(matrix)
        norm = mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    else:
        finite = matrix[np.isfinite(matrix)]
        upper = max(1e-12, float(np.nanmax(finite)) * 1.02)
        norm = mcolors.Normalize(vmin=0.0, vmax=upper)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    count = matrix.shape[0]
    labels = ["Rule {}".format(index + 1) for index in range(count)]
    ax.set_xticks(np.arange(count), labels=labels, rotation=38, ha="right", rotation_mode="anchor")
    ax.set_yticks(np.arange(count), labels=labels)
    ax.set_xlabel("Evaluated rule", labelpad=4)
    ax.set_ylabel("Upweighted rule", labelpad=4)
    ax.set_title(title, loc="left", pad=8, fontweight="bold")
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            ax.text(
                column,
                row,
                value_format.format(value),
                ha="center",
                va="center",
                fontsize=7.2,
                color=contrasting_text_color(im, value),
            )

    for index in range(min(matrix.shape)):
        ax.add_patch(
            Rectangle(
                (index - 0.5, index - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#111111",
                linewidth=1.15,
            )
        )

    colorbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.025, shrink=0.82)
    colorbar.set_label(colorbar_label, fontsize=8.5, labelpad=5)
    colorbar.ax.tick_params(labelsize=7.5, length=2.5, pad=2)
    colorbar.outline.set_linewidth(0.6)
    add_panel_letter(ax, panel_letter)


def padded_limits(
    values: np.ndarray,
    errors: np.ndarray,
    include_zero: bool = True,
    pad_fraction: float = 0.12,
) -> tuple[float, float]:
    lower = float(np.nanmin(values - errors))
    upper = float(np.nanmax(values + errors))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = max(1e-12, upper - lower)
    return lower - pad_fraction * span, upper + pad_fraction * span


def draw_matched_scatter(fig: plt.Figure, ax, summary: dict[str, Any]) -> None:
    x_values, y_values, x_errors, y_errors = matched_target_statistics(summary)
    y_scale = 1e4
    y_plot = y_values * y_scale
    y_error_plot = y_errors * y_scale

    for index, (x_value, y_value, x_error, y_error) in enumerate(
        zip(x_values, y_plot, x_errors, y_error_plot)
    ):
        color = RULE_COLORS[index % len(RULE_COLORS)]
        ax.errorbar(
            x_value,
            y_value,
            xerr=x_error,
            yerr=y_error,
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.58,
            zorder=2,
        )
        ax.scatter(
            x_value,
            y_value,
            s=52,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
            label="Rule {}".format(index + 1),
        )

    finite = np.isfinite(x_values) & np.isfinite(y_plot)
    if int(np.count_nonzero(finite)) >= 2 and float(np.ptp(x_values[finite])) > 0.0:
        slope, intercept = np.polyfit(x_values[finite], y_plot[finite], 1)
        fit_x = np.linspace(float(np.min(x_values[finite])), float(np.max(x_values[finite])), 100)
        ax.plot(
            fit_x,
            slope * fit_x + intercept,
            color="#333333",
            linewidth=1.15,
            linestyle="--",
            alpha=0.78,
            zorder=1,
        )
        correlation = float(np.corrcoef(x_values[finite], y_plot[finite])[0, 1])
        ax.text(
            0.04,
            0.96,
            "Pearson r = {:.2f}".format(correlation),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

    ax.axvline(0.0, color="#555555", linewidth=0.8, alpha=0.65, zorder=0)
    ax.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.65, zorder=0)
    ax.set_xlim(*padded_limits(x_values, x_errors))
    ax.set_ylim(*padded_limits(y_plot, y_error_plot))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.set_xlabel("Target-rule accuracy change", labelpad=5)
    ax.set_ylabel(r"Target-rule weight-ratio change ($\times 10^{-4}$)", labelpad=5)
    ax.set_title("Matched target-rule response", loc="left", pad=8, fontweight="bold")
    ax.grid(True, color="#D9D9D9", linewidth=0.55, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", pad=2)
    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.2,
    )
    add_panel_letter(ax, "d")


def paired_trial_rows(condition_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    control = next(metric for metric in condition_metrics if metric["upweighted_rule"] is None)
    rows: list[dict[str, Any]] = []
    for metric in condition_metrics:
        rule_id = metric["upweighted_rule"]
        if rule_id is None:
            continue
        rule_pos = rw.RULE_INDICES.index(int(rule_id))
        accuracy = np.asarray(metric["baseline_acc_runs"], dtype=float)[:, rule_pos]
        control_accuracy = np.asarray(control["baseline_acc_runs"], dtype=float)[:, rule_pos]
        weight_ratio = np.asarray(metric["weight_ratio_runs"], dtype=float)[:, rule_pos]
        control_weight_ratio = np.asarray(control["weight_ratio_runs"], dtype=float)[:, rule_pos]
        if not (accuracy.shape == control_accuracy.shape == weight_ratio.shape == control_weight_ratio.shape):
            raise ValueError("Cannot pair rule {} repeats across weighted and control conditions".format(rule_id))
        for repeat_index in range(len(accuracy)):
            rows.append(
                {
                    "rule": int(rule_id) + 1,
                    "repeat": int(repeat_index),
                    "accuracy_delta": float(accuracy[repeat_index] - control_accuracy[repeat_index]),
                    "weight_ratio_delta": float(weight_ratio[repeat_index] - control_weight_ratio[repeat_index]),
                }
            )
    return rows


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "p = NA"
    if value < 1e-4:
        return r"$p < 10^{-4}$"
    return "p = {:.3f}".format(value)


def draw_trial_scatter(ax, rows: list[dict[str, Any]]) -> tuple[float, float]:
    x_values = np.asarray([float(row["weight_ratio_delta"]) * 1e4 for row in rows], dtype=float)
    y_values = np.asarray([float(row["accuracy_delta"]) for row in rows], dtype=float)
    rule_values = np.asarray([int(row["rule"]) for row in rows], dtype=int)
    finite = np.isfinite(x_values) & np.isfinite(y_values)

    for rule in sorted(set(rule_values.tolist())):
        selected = finite & (rule_values == rule)
        ax.scatter(
            x_values[selected],
            y_values[selected],
            s=18,
            color=RULE_COLORS[(rule - 1) % len(RULE_COLORS)],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.3,
            label="Rule {}".format(rule),
            zorder=3,
        )

    correlation = float("nan")
    p_value = float("nan")
    if int(np.count_nonzero(finite)) >= 3 and float(np.ptp(x_values[finite])) > 0.0:
        correlation, p_value = (float(value) for value in pearsonr(x_values[finite], y_values[finite]))
        slope, intercept = np.polyfit(x_values[finite], y_values[finite], 1)
        fit_x = np.linspace(float(np.min(x_values[finite])), float(np.max(x_values[finite])), 200)
        ax.plot(fit_x, slope * fit_x + intercept, color="#222222", linewidth=1.25, zorder=2)
        ax.text(
            0.02,
            0.97,
            "Pearson r = {:.2f}, {}".format(correlation, format_p_value(p_value)),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )

    ax.axvline(0.0, color="#777777", linewidth=0.75, alpha=0.65, zorder=0)
    ax.axhline(0.0, color="#777777", linewidth=0.75, alpha=0.65, zorder=0)
    ax.set_xlabel(r"Target-rule retained weight-ratio change ($\times 10^{-4}$)")
    ax.set_ylabel("Target-rule accuracy change")
    ax.set_title("Paired trial-level response", loc="left", pad=8, fontweight="bold")
    ax.grid(True, color="#D9D9D9", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", ncol=5, frameon=False, columnspacing=0.8, handletextpad=0.3)
    add_panel_letter(ax, "e")
    return correlation, p_value


def write_trial_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rule", "repeat", "accuracy_delta", "weight_ratio_delta"])
        writer.writeheader()
        writer.writerows(rows)


def generate_figure(
    summary: dict[str, Any],
    condition_metrics: list[dict[str, Any]],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_stem: str = "fig6",
    formats: list[str] | None = None,
) -> tuple[list[Path], list[dict[str, Any]], float, float]:
    configure_style()
    output_dir = normal_path(output_dir)
    formats = list(formats or parse_formats(DEFAULT_FORMATS))
    accuracy_delta, weight_delta, specialization = figure_matrices(summary)
    trial_rows = paired_trial_rows(condition_metrics)

    fig = plt.figure(figsize=(7.45, 8.45))
    grid = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 0.92))
    axes = {
        "a": fig.add_subplot(grid[0, 0]),
        "b": fig.add_subplot(grid[0, 1]),
        "c": fig.add_subplot(grid[1, 0]),
        "d": fig.add_subplot(grid[1, 1]),
        "e": fig.add_subplot(grid[2, :]),
    }
    draw_heatmap(
        fig,
        axes["a"],
        accuracy_delta,
        panel_letter="a",
        title="Whole-network accuracy change",
        colorbar_label=r"$\Delta$ accuracy",
        value_format="{:+.2f}",
        cmap="RdBu_r",
        diverging=True,
    )
    draw_heatmap(
        fig,
        axes["b"],
        weight_delta * 1e4,
        panel_letter="b",
        title="Retained weight-ratio change",
        colorbar_label=r"$\Delta$ weight ratio ($\times 10^{-4}$)",
        value_format="{:+.2f}",
        cmap="RdBu_r",
        diverging=True,
    )
    draw_heatmap(
        fig,
        axes["c"],
        specialization,
        panel_letter="c",
        title="Functional-token specialization",
        colorbar_label="Specialization margin",
        value_format="{:.2f}",
        cmap="viridis",
        diverging=False,
    )
    draw_matched_scatter(fig, axes["d"], summary)
    correlation, p_value = draw_trial_scatter(axes["e"], trial_rows)

    fig.subplots_adjust(left=0.105, right=0.965, bottom=0.075, top=0.965, wspace=0.48, hspace=0.56)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for fmt in formats:
        output_path = output_dir / "{}.{}".format(output_stem, fmt)
        kwargs: dict[str, Any] = {"bbox_inches": "tight", "pad_inches": 0.035}
        if fmt in {"png", "jpg", "jpeg", "tif", "tiff"}:
            kwargs["dpi"] = ana.PUBLICATION_DPI
        fig.savefig(output_path, **kwargs)
        output_paths.append(output_path)
    plt.close(fig)
    return output_paths, trial_rows, correlation, p_value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build article Figure 6 directly from the current rule-weight simulation and analysis outputs."
    )
    parser.add_argument(
        "--results-root",
        type=normal_path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root containing ruleweight_conditions.json plus simulation/ and analysis/.",
    )
    parser.add_argument("--output-dir", type=normal_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default="fig6")
    parser.add_argument("--formats", default=DEFAULT_FORMATS)
    parser.add_argument("--article-fig6", type=normal_path, default=DEFAULT_ARTICLE_FIG6)
    parser.add_argument("--no-replace-article", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary, condition_metrics = build_summary_from_ruleweight(args.results_root)
    output_paths, trial_rows, correlation, p_value = generate_figure(
        summary=summary,
        condition_metrics=condition_metrics,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        formats=parse_formats(args.formats),
    )

    trial_csv = args.output_dir / "figure6_paired_trials.csv"
    write_trial_rows(trial_csv, trial_rows)
    figure_summary = {
        "source_root": str(args.results_root),
        "ruleweight_summary": str(args.results_root / "ruleweight_summary.json"),
        "repeat_count_per_condition": int(summary["config"]["repeat"]),
        "epochs": int(summary["config"]["epochs"]),
        "l1": float(summary["config"]["l1"]),
        "gamma": float(summary["config"]["gamma"]),
        "upweight": float(summary["config"]["upweight"]),
        "token_extraction_method": str(summary["config"]["token_extraction_method"]),
        "token_extraction_fraction": float(summary["config"]["token_extraction_fraction"]),
        "paired_trial_count": len(trial_rows),
        "paired_trial_pearson_r": correlation,
        "paired_trial_pearson_p": p_value,
        "outputs": [str(path) for path in output_paths],
    }
    (args.output_dir / "figure6_summary.json").write_text(
        json.dumps(figure_summary, indent=2),
        encoding="utf-8",
    )

    pdf_path = next((path for path in output_paths if path.suffix.lower() == ".pdf"), None)
    if pdf_path is not None and not args.no_replace_article:
        args.article_fig6.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_path, args.article_fig6)
        print("Replaced {}".format(args.article_fig6))

    print("Figure 6 source: {}".format(args.results_root))
    print("Wrote {}".format(trial_csv))
    for path in output_paths:
        print("Wrote {}".format(path))


if __name__ == "__main__":
    main()
