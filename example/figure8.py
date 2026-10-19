from __future__ import annotations

import argparse
import csv
import json
import math
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
EXAMPLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(EXAMPLE_ROOT))

import analysis as ana
import modularity_time as mt


SWEEP_SUMMARY = ROOT / "example/results/simulation/lasso_l_gamma_sweep/lasso_l_gamma_sweep_summary.json"
OUTPUT_ROOT = ROOT / "example/results/figure8"
ARTICLE_FIG8 = ROOT / "article/figs/fig8.pdf"
DEFAULT_GAMMA = 1.0
DEFAULT_LASSO = "0,1,2,3,5,10,15"
DEFAULT_CORRELATION_LASSO = "0,1,5,10"
DEFAULT_FIG_L_SCALE = 5.0
DEFAULT_CORRELATION_BINS = 100
DEFAULT_CORRELATION_MAX_RUNS = 4
DEFAULT_HIDDEN_MODE = "relu"
LAYER_COLORS = ["#4DBEEE", "#A2142F", "#0072BD"]
LASSO_COLORS = ["#5C677D", "#D55E00", "#009E73", "#7E2F8E", "#CC79A7", "#E69F00"]
PAIR_COLORS = ["#D55E00", "#EDB120", "#7E2F8E"]
OUTPUT_PAIR_COLORS = ["#77AC30", "#4DBEEE", "#A2142F"]
OUTPUT_COLOR = "#D95319"
EXPORT_EXTENSIONS = ("png", "pdf", "svg")


def normal_path(value: str | Path) -> Path:
    return Path(str(value).replace("\\", "/"))


def parse_float_list(value: str) -> list[float]:
    out = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    return out


def parse_formats(value: str) -> list[str]:
    formats = []
    for item in str(value).split(","):
        item = item.strip().lower().lstrip(".")
        if item and item not in formats:
            formats.append(item)
    if not formats:
        raise ValueError("At least one output format is required")
    return formats


def format_float(value: float) -> str:
    return "{:g}".format(float(value))


def configure_style() -> None:
    ana.configure_publication_style()
    matplotlib.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "legend.title_fontsize": 8.2,
            "svg.fonttype": "none",
        }
    )


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"gamma", "lasso", "experiment_lasso", "value", "bin_left", "bin_right", "bin_center", "probability"}:
                    converted[key] = float(value)
                elif key in {"epoch", "layer_index", "bin_index", "total_pairs", "valid_neurons"}:
                    converted[key] = int(float(value))
                else:
                    converted[key] = value
            rows.append(converted)
    return rows


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


def resolve_epoch(condition_dir: Path, epoch_arg: str) -> int:
    epoch_arg = str(epoch_arg).strip().lower()
    snapshot_root = condition_dir / "snapshots"
    if snapshot_root.is_dir():
        epoch_dirs = sorted(snapshot_root.glob("epoch_*"), key=mt.snapshot_epoch)
        if not epoch_dirs:
            raise FileNotFoundError("No snapshots/epoch_* directories under {}".format(condition_dir))
        if epoch_arg in {"latest", "last", ""}:
            return mt.snapshot_epoch(epoch_dirs[-1])
        target = int(float(epoch_arg))
        wanted = snapshot_root / "epoch_{:06d}".format(target)
        if not wanted.is_dir():
            available = ", ".join(str(mt.snapshot_epoch(path)) for path in epoch_dirs)
            raise FileNotFoundError("Epoch {} not found under {}; available: {}".format(target, condition_dir, available))
        return target
    if epoch_arg in {"latest", "last", ""}:
        return -1
    return int(float(epoch_arg))


def run_dirs_for_epoch(condition_dir: Path, epoch: int, max_runs: int | None) -> list[Path]:
    if epoch >= 0:
        root = condition_dir / "snapshots" / "epoch_{:06d}".format(int(epoch))
    else:
        root = condition_dir
    run_dirs = sorted(path for path in root.glob("run_*") if (path / "run_data.npz").is_file())
    if max_runs is not None:
        run_dirs = run_dirs[: max(0, int(max_runs))]
    if not run_dirs:
        raise FileNotFoundError("No run_*/run_data.npz files under {}".format(root))
    return run_dirs


def selected_conditions(summary_path: Path, gamma: float, fig_lasso_values: list[float], fig_l_scale: float) -> tuple[dict[float, tuple[float, Path]], list[float]]:
    return mt.selected_conditions(summary_path, gamma, fig_lasso_values, fig_l_scale)


def seed_from_path(path: Path) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(path))) % (2**32)


def centered_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        x = x.reshape(x.shape[0], -1)
    return x - x.mean(axis=0, keepdims=True)


def pca_spectral_entropy(x: np.ndarray) -> float:
    xc = centered_matrix(x)
    if xc.shape[0] < 2 or xc.shape[1] < 1:
        return float("nan")
    singular = np.linalg.svd(xc, compute_uv=False)
    latent = singular**2
    total = float(np.sum(latent))
    if total <= 0.0:
        return 0.0
    probs = latent / total
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def linear_cka_fast(x: np.ndarray, y: np.ndarray) -> float:
    xc = centered_matrix(x)
    yc = centered_matrix(y)
    if xc.shape[0] != yc.shape[0] or xc.shape[0] < 2:
        return float("nan")
    cross = xc.T @ yc
    xx = xc.T @ xc
    yy = yc.T @ yc
    numerator = float(np.sum(cross * cross))
    denominator = math.sqrt(float(np.sum(xx * xx)) * float(np.sum(yy * yy)))
    if denominator <= 0.0:
        return float("nan")
    return float(numerator / (denominator + 1e-12))


def abs_correlation_hist(x: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, int, int]:
    values = centered_matrix(x)
    neuron_count = int(values.shape[1]) if values.ndim == 2 else 0
    if values.shape[0] < 2 or neuron_count < 2:
        return np.zeros(len(bins) - 1, dtype=float), 0, neuron_count
    std = values.std(axis=0, ddof=1)
    valid = std > 1e-12
    values = values[:, valid]
    valid_neurons = int(values.shape[1])
    if valid_neurons < 2:
        return np.zeros(len(bins) - 1, dtype=float), 0, valid_neurons
    z = values / (std[valid][None, :] + 1e-12)
    corr = (z.T @ z) / max(z.shape[0] - 1, 1)
    corr = np.clip(corr, -1.0, 1.0)
    abs_values = np.abs(corr[np.triu_indices_from(corr, k=1)])
    abs_values = abs_values[np.isfinite(abs_values)]
    hist, _ = np.histogram(abs_values, bins=bins)
    total = int(np.sum(hist))
    probs = hist.astype(float) / max(total, 1)
    return probs, total, valid_neurons


def load_sampled_run(run_dir: Path, max_samples: int | None) -> dict[str, Any]:
    loaded = ana.load_run_npz(run_dir)
    return mt.subset_loaded_data(loaded, max_samples=max_samples, seed=seed_from_path(run_dir))


def compute_run_rows(
    run_dir: Path,
    gamma: float,
    fig_lasso: float,
    experiment_lasso: float,
    epoch: int,
    max_samples: int | None,
    bins: np.ndarray,
    include_correlation: bool,
    hidden_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    loaded = load_sampled_run(run_dir, max_samples=max_samples)
    state = loaded["state"]
    x_test = loaded["x_test"]
    pred, activations = ana.forward_from_state(state, x_test)
    hidden_preactivation = [np.asarray(layer, dtype=float) for layer in activations[:-1]]
    if hidden_mode == "relu":
        hidden = [np.maximum(layer, 0.0) for layer in hidden_preactivation]
    elif hidden_mode == "preactivation":
        hidden = hidden_preactivation
    else:
        raise ValueError("hidden_mode must be relu or preactivation")
    output = np.asarray(pred, dtype=float)
    metric_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    base = {
        "gamma": float(gamma),
        "lasso": float(fig_lasso),
        "experiment_lasso": float(experiment_lasso),
        "epoch": int(epoch),
        "run": run_dir.name,
        "num_samples": int(len(x_test)),
    }
    metric_rows.append({**base, "metric": "pca_entropy", "layer_index": 4, "pair": "out", "value": pca_spectral_entropy(output)})
    for layer_pos, layer in enumerate(hidden, start=1):
        metric_rows.append({**base, "metric": "pca_entropy", "layer_index": layer_pos, "pair": "", "value": pca_spectral_entropy(layer)})
        metric_rows.append({**base, "metric": "cka_output", "layer_index": layer_pos, "pair": "{}-out".format(layer_pos), "value": linear_cka_fast(layer, output)})
        if include_correlation:
            probs, total_pairs, valid_neurons = abs_correlation_hist(hidden_preactivation[layer_pos - 1], bins)
            for bin_idx, probability in enumerate(probs):
                hist_rows.append(
                    {
                        **base,
                        "layer_index": layer_pos,
                        "bin_index": int(bin_idx),
                        "bin_left": float(bins[bin_idx]),
                        "bin_right": float(bins[bin_idx + 1]),
                        "bin_center": float((bins[bin_idx] + bins[bin_idx + 1]) * 0.5),
                        "probability": float(probability),
                        "total_pairs": int(total_pairs),
                        "valid_neurons": int(valid_neurons),
                    }
                )
    for i in range(len(hidden)):
        for j in range(i + 1, len(hidden)):
            pair = "{}-{}".format(i + 1, j + 1)
            metric_rows.append({**base, "metric": "cka_hidden_pair", "layer_index": 0, "pair": pair, "value": linear_cka_fast(hidden[i], hidden[j])})
    return metric_rows, hist_rows


def compute_all(args: argparse.Namespace, lasso_values: list[float], conditions: dict[float, tuple[float, Path]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bins = np.linspace(0.0, 1.0, int(args.correlation_bins) + 1)
    metric_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    for fig_lasso in lasso_values:
        experiment_lasso, condition_dir = conditions[float(fig_lasso)]
        epoch = resolve_epoch(condition_dir, args.epoch)
        run_dirs = run_dirs_for_epoch(condition_dir, epoch, args.max_runs)
        corr_limit = len(run_dirs) if args.correlation_max_runs is None else min(len(run_dirs), max(0, int(args.correlation_max_runs)))
        use_correlation = any(math.isclose(float(fig_lasso), float(value), abs_tol=1e-9) for value in args.correlation_lasso_values)
        shown_corr_limit = corr_limit if use_correlation else 0
        print("Computing Fig. 8 gamma={:g}, l={:g} (experiment l={:g}) epoch {}: {} runs, {} correlation runs".format(args.gamma, fig_lasso, experiment_lasso, epoch, len(run_dirs), shown_corr_limit))
        for run_index, run_dir in enumerate(run_dirs):
            run_metrics, run_hist = compute_run_rows(
                run_dir,
                args.gamma,
                fig_lasso,
                experiment_lasso,
                epoch,
                args.max_samples,
                bins,
                include_correlation=use_correlation and run_index < corr_limit,
                hidden_mode=str(args.hidden_mode),
            )
            metric_rows.extend(run_metrics)
            hist_rows.extend(run_hist)
    return metric_rows, hist_rows


def metric_values(rows: list[dict[str, Any]], metric: str, lasso: float, *, layer: int | None = None, pair: str | None = None) -> np.ndarray:
    values = []
    for row in rows:
        if row.get("metric") != metric:
            continue
        if not math.isclose(float(row.get("lasso", float("nan"))), float(lasso), abs_tol=1e-9):
            continue
        if layer is not None and int(row.get("layer_index", -999)) != int(layer):
            continue
        if pair is not None and str(row.get("pair", "")) != str(pair):
            continue
        value = float(row.get("value", float("nan")))
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def median_iqr(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmedian(values)), float(np.nanpercentile(values, 25)), float(np.nanpercentile(values, 75))


def style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ana._style_publication_axes(ax, grid=grid)



def plot_hist_panel(fig: plt.Figure, outer_slot: Any, hist_rows: list[dict[str, Any]], correlation_lasso_values: list[float]) -> list[plt.Axes]:
    grid = outer_slot.subgridspec(1, len(correlation_lasso_values), wspace=0.30)
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(len(correlation_lasso_values))]
    max_density = 0.0
    density_bin_width = None
    cached: dict[tuple[int, float], tuple[np.ndarray, np.ndarray]] = {}
    for lasso in correlation_lasso_values:
        for layer_idx in [1, 2, 3]:
            by_bin: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for row in hist_rows:
                if int(row["layer_index"]) == layer_idx and math.isclose(float(row["lasso"]), float(lasso), abs_tol=1e-9):
                    bin_width = max(float(row["bin_right"]) - float(row["bin_left"]), 1e-12)
                    density_bin_width = bin_width if density_bin_width is None else min(density_bin_width, bin_width)
                    by_bin[int(row["bin_index"])].append((float(row["bin_center"]), float(row["probability"]) / bin_width))
            centers = []
            probabilities = []
            for bin_idx in sorted(by_bin):
                centers.append(float(np.mean([item[0] for item in by_bin[bin_idx]])))
                probabilities.append(float(np.mean([item[1] for item in by_bin[bin_idx]])))
            centers_arr = np.asarray(centers, dtype=float)
            probs_arr = np.asarray(probabilities, dtype=float)
            cached[(layer_idx, float(lasso))] = (centers_arr, probs_arr)
            if probs_arr.size:
                max_density = max(max_density, float(np.nanmax(probs_arr)))
    y_upper = 0.08 / density_bin_width if density_bin_width else 0.08
    for ax_idx, (ax, lasso) in enumerate(zip(axes, correlation_lasso_values)):
        for layer_idx in [1, 2, 3]:
            centers, probabilities = cached.get((layer_idx, float(lasso)), (np.zeros(0), np.zeros(0)))
            if centers.size:
                color = LAYER_COLORS[layer_idx - 1]
                ax.fill_between(centers, probabilities, color=color, alpha=0.42, linewidth=0.0, label="Layer {}".format(layer_idx))
                ax.plot(centers, probabilities, color=color, linewidth=0.85, alpha=0.95)
        ax.set_title(r"$l={}$".format(format_float(lasso)), pad=3)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, y_upper)
        ax.set_yticks([0.0, y_upper * 0.5, y_upper])
        ax.set_xlabel("Absolute correlation")
        if ax_idx == 0:
            ax.set_ylabel("Probability density")
            add_panel_label(ax, "a", x=-0.43, y=1.11)
            ax.legend(frameon=False, loc="upper right", handlelength=1.0, handletextpad=0.28, borderaxespad=0.1)
        else:
            ax.tick_params(axis="y", labelleft=False)
        style_axis(ax, grid=False)
    return axes


def series_values(rows: list[dict[str, Any]], metric: str, lasso_values: list[float], *, layer: int | None = None, pair: str | None = None) -> list[np.ndarray]:
    return [metric_values(rows, metric, lasso, layer=layer, pair=pair) for lasso in lasso_values]


def draw_grouped_violinplot(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    lasso_values: list[float],
    metric: str,
    series: list[tuple[str, int | None, str | None, str]],
    ylabel: str,
    panel_label: str,
    ylim: tuple[float, float] | None = None,
    legend_cols: int = 1,
) -> None:
    centers = np.arange(len(lasso_values), dtype=float)
    num_series = len(series)
    group_width = 0.72 if num_series >= 4 else 0.60
    offsets = np.linspace(-group_width / 2.0, group_width / 2.0, num_series)
    violin_width = min(0.16, group_width / max(num_series + 0.8, 1.0))
    handles = []
    all_values = []
    for _label, layer, pair, _color in series:
        for values in series_values(rows, metric, lasso_values, layer=layer, pair=pair):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                all_values.append(arr)
    if all_values:
        finite = np.concatenate(all_values)
        span = float(np.nanmax(finite) - np.nanmin(finite))
    else:
        span = 0.0
    epsilon = max(1e-6, span * 1e-4)

    for series_idx, (label, layer, pair, color) in enumerate(series):
        data = []
        positions = []
        for group_idx, values in enumerate(series_values(rows, metric, lasso_values, layer=layer, pair=pair)):
            arr = np.asarray(values, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 1:
                arr = np.asarray([arr[0] - epsilon, arr[0], arr[0] + epsilon], dtype=float)
            if arr.size:
                data.append(arr)
                positions.append(float(centers[group_idx] + offsets[series_idx]))
        if data:
            parts = ax.violinplot(data, positions=positions, widths=violin_width, showmeans=False, showmedians=True, showextrema=False)
            for body in parts.get("bodies", []):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.58)
                body.set_linewidth(0.75)
            if "cmedians" in parts:
                parts["cmedians"].set_color("#222222")
                parts["cmedians"].set_linewidth(0.85)
        handles.append(plt.Line2D([0], [0], marker="s", linestyle="", markersize=4.6, markerfacecolor=color, markeredgecolor=color, label=label))
    ax.set_xlim(-0.65, len(lasso_values) - 0.35)
    ax.set_xticks(centers)
    ax.set_xticklabels([format_float(value) for value in lasso_values], rotation=35, ha="right")
    ax.set_xlabel(r"$l$")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    add_panel_label(ax, panel_label, x=-0.30, y=1.08)
    style_axis(ax, grid=False)
    ax.legend(handles=handles, frameon=False, loc="best", ncol=legend_cols, handletextpad=0.25, borderaxespad=0.1, columnspacing=0.55)

def add_panel_label(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.08) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontweight="bold", fontsize=11.5, ha="left", va="bottom")


def plot_fig8(metric_rows: list[dict[str, Any]], hist_rows: list[dict[str, Any]], lasso_values: list[float], correlation_lasso_values: list[float], output_dir: Path, formats: list[str]) -> list[Path]:
    configure_style()
    fig = plt.figure(figsize=(7.2, 4.35))
    outer = fig.add_gridspec(2, 1, height_ratios=[0.96, 1.32], hspace=0.48)
    plot_hist_panel(fig, outer[0, 0], hist_rows, correlation_lasso_values)

    bottom = outer[1, 0].subgridspec(1, 3, wspace=0.42)
    ax_b = fig.add_subplot(bottom[0, 0])
    b_series = [("layer 1", 1, None, LAYER_COLORS[0]), ("layer 2", 2, None, LAYER_COLORS[1]), ("layer 3", 3, None, LAYER_COLORS[2]), ("output layer", 4, "out", OUTPUT_COLOR)]
    draw_grouped_violinplot(ax_b, metric_rows, lasso_values, "pca_entropy", b_series, "PCA entropy", "b", legend_cols=1)

    ax_c = fig.add_subplot(bottom[0, 1])
    c_series = [("1-2", None, "1-2", PAIR_COLORS[0]), ("2-3", None, "2-3", PAIR_COLORS[1]), ("1-3", None, "1-3", PAIR_COLORS[2])]
    draw_grouped_violinplot(ax_c, metric_rows, lasso_values, "cka_hidden_pair", c_series, "CKA mat", "c", ylim=(0.65, 1.0), legend_cols=1)

    ax_d = fig.add_subplot(bottom[0, 2])
    d_series = [("1-o", 1, "1-out", OUTPUT_PAIR_COLORS[0]), ("2-o", 2, "2-out", OUTPUT_PAIR_COLORS[1]), ("3-o", 3, "3-out", OUTPUT_PAIR_COLORS[2])]
    draw_grouped_violinplot(ax_d, metric_rows, lasso_values, "cka_output", d_series, "CKA mat", "d", legend_cols=1)

    fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.13)
    paths = export_figure(fig, output_dir, "fig8", formats)
    plt.close(fig)
    return paths


def summarize(metric_rows: list[dict[str, Any]], lasso_values: list[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {"metrics": {}}
    for metric in ["pca_entropy", "cka_hidden_pair", "cka_output"]:
        summary["metrics"][metric] = {}
        for lasso in lasso_values:
            entries = [row for row in metric_rows if row.get("metric") == metric and math.isclose(float(row["lasso"]), float(lasso), abs_tol=1e-9)]
            values = np.asarray([float(row["value"]) for row in entries if np.isfinite(float(row["value"]))], dtype=float)
            summary["metrics"][metric][format_float(lasso)] = {
                "n": int(values.size),
                "median": float(np.nanmedian(values)) if values.size else None,
                "mean": float(np.nanmean(values)) if values.size else None,
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute and draw article Fig. 8 from lasso/gamma sweep run data.")
    parser.add_argument("--summary", type=normal_path, default=SWEEP_SUMMARY)
    parser.add_argument("--output-root", type=normal_path, default=OUTPUT_ROOT)
    parser.add_argument("--article-fig8", type=normal_path, default=ARTICLE_FIG8)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--lasso", default=DEFAULT_LASSO, help="Figure/text l values for Fig. 8b-d, e.g. 0,1,2,3,5,10,15.")
    parser.add_argument("--correlation-lasso", default=DEFAULT_CORRELATION_LASSO, help="Figure/text l values shown in Fig. 8a correlation panels, e.g. 0,1,5,10.")
    parser.add_argument("--fig-l-scale", type=float, default=DEFAULT_FIG_L_SCALE, help="Figure/text l = experiment l * this scale.")
    parser.add_argument("--epoch", default="latest")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional deterministic subsample of x_test for faster representation metrics.")
    parser.add_argument("--correlation-bins", type=int, default=DEFAULT_CORRELATION_BINS)
    parser.add_argument("--correlation-max-runs", type=int, default=DEFAULT_CORRELATION_MAX_RUNS, help="Number of runs per l used for Fig. 8a correlation histograms; use -1 for all selected runs.")
    parser.add_argument("--hidden-mode", choices=["relu", "preactivation"], default=DEFAULT_HIDDEN_MODE, help="Hidden representation used for Fig. 8b-d; Fig. 8a always uses pre-activation responses.")
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--only-plot", action="store_true")
    parser.add_argument("--no-replace-article", action="store_true", help="Do not copy the generated PDF to article/figs/fig8.pdf.")
    args = parser.parse_args()

    if int(args.correlation_bins) <= 0:
        raise ValueError("--correlation-bins must be positive")
    if args.correlation_max_runs is not None and int(args.correlation_max_runs) < 0:
        args.correlation_max_runs = None
    args.output_root.mkdir(parents=True, exist_ok=True)
    formats = parse_formats(args.formats)
    lasso_values = parse_float_list(args.lasso)
    correlation_lasso_values = parse_float_list(args.correlation_lasso)
    args.correlation_lasso_values = correlation_lasso_values
    selected_lasso_values = sorted(set(float(value) for value in (lasso_values + correlation_lasso_values)))
    conditions, missing = selected_conditions(args.summary, float(args.gamma), selected_lasso_values, float(args.fig_l_scale))
    if missing:
        raise FileNotFoundError("Missing requested l values for gamma={:g}: {}".format(float(args.gamma), ", ".join(format_float(x) for x in missing)))

    metric_csv = args.output_root / "figure8_representation_metrics_{}.csv".format(args.hidden_mode)
    hist_csv = args.output_root / "figure8_correlation_hist.csv"
    if args.only_plot:
        metric_rows = read_csv(metric_csv)
        hist_rows = read_csv(hist_csv)
    elif metric_csv.is_file() and hist_csv.is_file() and not args.recompute:
        print("Reading cached Fig. 8 metrics from {}".format(args.output_root))
        metric_rows = read_csv(metric_csv)
        hist_rows = read_csv(hist_csv)
    else:
        metric_rows, hist_rows = compute_all(args, lasso_values, conditions)
        write_csv(metric_csv, metric_rows)
        write_csv(hist_csv, hist_rows)
        print("Wrote {}".format(metric_csv))
        print("Wrote {}".format(hist_csv))

    paths = plot_fig8(metric_rows, hist_rows, lasso_values, correlation_lasso_values, args.output_root, formats)
    pdf_paths = [path for path in paths if path.suffix.lower() == ".pdf"]
    replaced = None
    if pdf_paths and not args.no_replace_article:
        args.article_fig8.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf_paths[0], args.article_fig8)
        replaced = args.article_fig8
        print("Replaced {}".format(args.article_fig8))

    save_json(
        args.output_root / "figure8_summary.json",
        {
            "summary": str(args.summary),
            "output_root": str(args.output_root),
            "article_fig8": str(args.article_fig8),
            "replaced_article_fig8": str(replaced) if replaced else None,
            "gamma": float(args.gamma),
            "lasso": [float(value) for value in lasso_values],
            "correlation_lasso": [float(value) for value in correlation_lasso_values],
            "fig_l_scale": float(args.fig_l_scale),
            "epoch": str(args.epoch),
            "max_runs": args.max_runs,
            "max_samples": args.max_samples,
            "correlation_bins": int(args.correlation_bins),
            "correlation_max_runs": args.correlation_max_runs,
            "hidden_mode": str(args.hidden_mode),
            "metric_csv": str(metric_csv),
            "hist_csv": str(hist_csv),
            "outputs": [str(path) for path in paths],
            **summarize(metric_rows, lasso_values),
        },
    )
    print("Wrote {} figure files under {}".format(len(paths), args.output_root))


if __name__ == "__main__":
    main()