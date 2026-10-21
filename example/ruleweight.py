from __future__ import annotations

import csv
import json
import shutil
import hashlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

import analysis as ana
from analysis import analyze_experiment
from dataset import BinaryPairDataset
from loss import WeightedMSELoss
from main import run_repeats, save_json
from model import TorchModularMLP
from task import BinaryArithmeticTask


# Rule-loss-weighting experiment: three 300-unit hidden layers.
# The new directory separates this protocol from historical 500-unit results.
EXPERIMENT_PREFIX = 'mlp300_binary_ruleweight'
OUTPUT_ROOT = ROOT / 'example/results/fig56_300_protocol_20261020/ruleweight'
SIMULATION_ROOT = OUTPUT_ROOT / 'simulation'
ANALYSIS_ROOT = OUTPUT_ROOT / 'analysis'
FIGURE_ROOT = OUTPUT_ROOT / 'figures'

NUM_SAMPLES = 5000
TRAIN_RATIO = 0.5
OUTPUT_BITS = 7
RULE_INDICES = [0, 1, 2, 3, 4]
CONTROL_CONDITION = None

REPEAT = 30
SEED = 300
EPOCHS = 5000
BATCH_SIZE = 128
LR = 0.05
OPTIMIZER_NAME = 'adam'
WEIGHT_DECAY = 0.0
LOG_EVERY = 100
SNAPSHOT_EVERY = 1000
DEVICE = 'auto'

MLP_HIDDEN_DIM = 300
MLP_NUM_MODULES = 3

UPWEIGHT = 5.0
BASE_RULE_WEIGHT = 1.0
L1 = 5.0
GAMMA = 1.0
REGULARIZATION_EPSILON = 1e-8

RUN_TRAINING = True
RUN_ANALYSIS = True
REUSE_EXISTING_SIMULATION = True
REUSE_EXISTING_ANALYSIS = True

ANALYSIS_METHOD = 'ablation'
ANALYSIS_SUBNET_TARGET = 'weight'
IMPORTANCE_NEURON_THRESHOLD = 1e-4
IMPORTANCE_MIN_SCORE = 1e-9
PERTURBATION_BATCH_SIZE = 64
CORRELATION_SAMPLE_SIZE = 5000
TOKEN_EXTRACTION_METHOD = 'score_quantile'
TOKEN_EXTRACTION_FRACTION = 0.75


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def mean_sem(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float('nan'), float('nan')
    sem = float(np.std(values) / np.sqrt(values.size))
    return float(np.mean(values)), sem


def finite_nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    counts = np.sum(finite, axis=axis)
    totals = np.sum(np.where(finite, values, 0.0), axis=axis)
    return np.divide(totals, counts, out=np.full_like(totals, np.nan, dtype=float), where=counts > 0)


def finite_nanstd(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    means = finite_nanmean(values, axis=axis)
    expanded_means = np.expand_dims(means, axis=axis)
    counts = np.sum(finite, axis=axis)
    squared = np.sum(np.where(finite, (values - expanded_means) ** 2, 0.0), axis=axis)
    variance = np.divide(squared, counts, out=np.full_like(squared, np.nan, dtype=float), where=counts > 0)
    return np.sqrt(variance)


def rule_tag(rule_id: int) -> str:
    return 'rule{}'.format(rule_id)


def display_rule_tag(rule_id: int) -> str:
    return 'rule{}'.format(rule_id + 1)


def condition_tag(upweighted_rule: int | None) -> str:
    if upweighted_rule is None:
        return 'control'
    return rule_tag(upweighted_rule)


def condition_axis_label(upweighted_rule: int | None) -> str:
    if upweighted_rule is None:
        return 'control'
    return display_rule_tag(upweighted_rule)


def rule_weights_for(upweighted_rule: int) -> list[float]:
    return [UPWEIGHT if rule_id == upweighted_rule else BASE_RULE_WEIGHT for rule_id in RULE_INDICES]


def rule_weights_for_condition(upweighted_rule: int | None) -> list[float]:
    if upweighted_rule is None:
        return [BASE_RULE_WEIGHT for _ in RULE_INDICES]
    return rule_weights_for(upweighted_rule)


def experiment_name_for(upweighted_rule: int | None) -> str:
    if upweighted_rule is None:
        name = '{}_control_w{:g}_l{:g}_gamma{:g}_repeat{}'.format(
            EXPERIMENT_PREFIX,
            BASE_RULE_WEIGHT,
            L1,
            GAMMA,
            REPEAT,
        )
    else:
        name = '{}_{}_w{:g}_l{:g}_gamma{:g}_repeat{}'.format(
            EXPERIMENT_PREFIX,
            rule_tag(upweighted_rule),
            UPWEIGHT,
            L1,
            GAMMA,
            REPEAT,
        )
    return name.replace('.', 'p')


def single_rule_combo_columns(combo_labels: list[str]) -> list[int]:
    columns = []
    for rule_id in RULE_INDICES:
        label = str(rule_id)
        if label not in combo_labels:
            raise ValueError('single-rule combo label {!r} not found in {}'.format(label, combo_labels))
        columns.append(combo_labels.index(label))
    return columns


def saved_run_identity(run_dir: Path, upweighted_rule: int | None) -> dict:
    config = load_json(run_dir / 'config.json')
    expected = {'epochs': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
                'optimizer_name': OPTIMIZER_NAME, 'weight_decay': WEIGHT_DECAY}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError('Saved training config mismatch: {} {}'.format(run_dir, key))
    loss = config.get('loss_config', {})
    for key, value in {'l1': L1, 'gamma': GAMMA, 'regularization_epsilon': REGULARIZATION_EPSILON,
                       'rule_weights': rule_weights_for_condition(upweighted_rule)}.items():
        if loss.get(key) != value:
            raise ValueError('Saved loss config mismatch: {} {}'.format(run_dir, key))
    loaded = ana.load_run_npz(run_dir)
    shapes = [tuple(loaded['state'][spec['key']].shape) for spec in ana.weight_block_specs(loaded['state'])]
    expected_shapes = [(MLP_HIDDEN_DIM, 2*OUTPUT_BITS)] + [(MLP_HIDDEN_DIM, MLP_HIDDEN_DIM)]*(MLP_NUM_MODULES-1) + [(len(RULE_INDICES)*OUTPUT_BITS, MLP_HIDDEN_DIM)]
    if shapes != expected_shapes or list(loaded['rule_indices']) != RULE_INDICES:
        raise ValueError('Saved model architecture/task mismatch: {}'.format(run_dir))
    return {'run': run_dir.name, 'seed': config['seed'],
            'input_label_sha256': hashlib.sha256(loaded['x_test'].tobytes()+loaded['y_test'].tobytes()).hexdigest()}


def extract_condition_metrics(analysis_dir: Path, upweighted_rule: int | None) -> dict:
    run_paths = sorted(analysis_dir.glob('run_*/analysis.json'))
    if not run_paths:
        raise FileNotFoundError('No analysis.json files found under {}'.format(analysis_dir))

    pairing = []
    retained_runs = []
    neuron_ratio_runs = []
    weight_ratio_runs = []
    neuron_count_runs = []
    specialization_runs = []
    baseline_acc_runs = []
    combo_labels = None

    for path in run_paths:
        run_dir = analysis_dir.parent.parent / 'simulation' / analysis_dir.name / path.parent.name
        pairing.append(saved_run_identity(run_dir, upweighted_rule))
        result = load_json(path)
        weight_combo = result.get('weight_combo_subnetwork', {})
        if not ana.weight_combo_protocol_matches(weight_combo, mask_method=ANALYSIS_METHOD,
                min_score=IMPORTANCE_MIN_SCORE, extraction_method=TOKEN_EXTRACTION_METHOD,
                fraction=TOKEN_EXTRACTION_FRACTION, output_eps=1e-8,
                max_elements_per_block=None, state=ana.load_run_npz(run_dir)['state']):
            raise ValueError('Incompatible weight-analysis cache: {}'.format(path))
        combo = weight_combo
        if combo_labels is None:
            combo_labels = list(combo['combo_labels'])
        columns = single_rule_combo_columns(combo_labels)
        retained_source = result.get('weight_combo_retained_accuracy', combo.get('network_retained_accuracy', []))
        retained = np.asarray(retained_source, dtype=float)[:, columns]
        weight_all = ana._weight_subnet_retained_weight_ratios([combo])
        weight_ratios = weight_all[0, columns] if weight_all.size else np.full(len(columns), np.nan, dtype=float)
        neuron_ratios = np.full(len(columns), np.nan, dtype=float)
        neuron_counts = np.full(len(columns), np.nan, dtype=float)
        baseline_acc = np.asarray(combo['baseline_accuracy'], dtype=float)

        margins = []
        for rule_pos in range(len(RULE_INDICES)):
            own = retained[rule_pos, rule_pos]
            off_rule = np.delete(retained[:, rule_pos], rule_pos)
            margins.append(own - float(np.nanmean(off_rule)))

        retained_runs.append(retained)
        neuron_ratio_runs.append(neuron_ratios)
        weight_ratio_runs.append(weight_ratios)
        neuron_count_runs.append(neuron_counts)
        specialization_runs.append(np.asarray(margins, dtype=float))
        baseline_acc_runs.append(baseline_acc)

    retained_runs = np.stack(retained_runs, axis=0)
    neuron_ratio_runs = np.stack(neuron_ratio_runs, axis=0)
    weight_ratio_runs = np.stack(weight_ratio_runs, axis=0)
    neuron_count_runs = np.stack(neuron_count_runs, axis=0)
    specialization_runs = np.stack(specialization_runs, axis=0)
    baseline_acc_runs = np.stack(baseline_acc_runs, axis=0)
    num_runs = retained_runs.shape[0]
    up_pos = RULE_INDICES.index(upweighted_rule) if upweighted_rule is not None else None

    if up_pos is None:
        target_retained_mean = float('nan')
        target_retained_sem = float('nan')
        target_neuron_ratio_mean = float('nan')
        target_neuron_ratio_sem = float('nan')
        target_weight_ratio_mean = float('nan')
        target_weight_ratio_sem = float('nan')
        target_specialization_margin_mean = float('nan')
        target_specialization_margin_sem = float('nan')
    else:
        target_retained_mean = float(np.nanmean(retained_runs[:, up_pos, up_pos]))
        target_retained_sem = float(np.nanstd(retained_runs[:, up_pos, up_pos]) / np.sqrt(num_runs))
        target_neuron_ratio_mean = float(finite_nanmean(neuron_ratio_runs[:, up_pos], axis=0))
        target_neuron_ratio_sem = float(finite_nanstd(neuron_ratio_runs[:, up_pos], axis=0) / np.sqrt(num_runs))
        target_weight_ratio_mean = float(np.nanmean(weight_ratio_runs[:, up_pos]))
        target_weight_ratio_sem = float(np.nanstd(weight_ratio_runs[:, up_pos]) / np.sqrt(num_runs))
        target_specialization_margin_mean = float(np.nanmean(specialization_runs[:, up_pos]))
        target_specialization_margin_sem = float(np.nanstd(specialization_runs[:, up_pos]) / np.sqrt(num_runs))

    return {
        'condition': condition_tag(upweighted_rule),
        'upweighted_rule': upweighted_rule,
        'analysis_dir': str(analysis_dir),
        'pairing': pairing,
        'num_runs': len(run_paths),
        'retained_mean': finite_nanmean(retained_runs, axis=0),
        'retained_sem': finite_nanstd(retained_runs, axis=0) / np.sqrt(num_runs),
        'diag_retained_mean': finite_nanmean(np.diagonal(retained_runs, axis1=1, axis2=2), axis=0),
        'diag_retained_sem': finite_nanstd(np.diagonal(retained_runs, axis1=1, axis2=2), axis=0) / np.sqrt(num_runs),
        'neuron_ratio_runs': neuron_ratio_runs,
        'neuron_ratio_mean': finite_nanmean(neuron_ratio_runs, axis=0),
        'neuron_ratio_sem': finite_nanstd(neuron_ratio_runs, axis=0) / np.sqrt(num_runs),
        'weight_ratio_runs': weight_ratio_runs,
        'weight_ratio_mean': finite_nanmean(weight_ratio_runs, axis=0),
        'weight_ratio_sem': finite_nanstd(weight_ratio_runs, axis=0) / np.sqrt(num_runs),
        'neuron_count_mean': finite_nanmean(neuron_count_runs, axis=0),
        'specialization_margin_mean': finite_nanmean(specialization_runs, axis=0),
        'specialization_margin_sem': finite_nanstd(specialization_runs, axis=0) / np.sqrt(num_runs),
        'baseline_acc_runs': baseline_acc_runs,
        'baseline_acc_mean': finite_nanmean(baseline_acc_runs, axis=0),
        'baseline_acc_sem': finite_nanstd(baseline_acc_runs, axis=0) / np.sqrt(num_runs),
        'target_retained_mean': target_retained_mean,
        'target_retained_sem': target_retained_sem,
        'target_neuron_ratio_mean': target_neuron_ratio_mean,
        'target_neuron_ratio_sem': target_neuron_ratio_sem,
        'target_weight_ratio_mean': target_weight_ratio_mean,
        'target_weight_ratio_sem': target_weight_ratio_sem,
        'target_specialization_margin_mean': target_specialization_margin_mean,
        'target_specialization_margin_sem': target_specialization_margin_sem,
    }


def write_summary_tables(condition_metrics: list[dict]) -> dict:
    summary_path = OUTPUT_ROOT / 'ruleweight_summary.json'
    csv_path = OUTPUT_ROOT / 'ruleweight_summary.csv'
    rows = []

    upweighted_metrics = [m for m in condition_metrics if m['upweighted_rule'] is not None]
    control_metric = next((m for m in condition_metrics if m['upweighted_rule'] is None), None)
    empty = np.zeros((0, len(RULE_INDICES)), dtype=float)

    diag_retained = np.stack([m['diag_retained_mean'] for m in upweighted_metrics], axis=0) if upweighted_metrics else empty
    neuron_ratio = np.stack([m['neuron_ratio_mean'] for m in upweighted_metrics], axis=0) if upweighted_metrics else empty
    weight_ratio = np.stack([m['weight_ratio_mean'] for m in upweighted_metrics], axis=0) if upweighted_metrics else empty
    specialization = np.stack([m['specialization_margin_mean'] for m in upweighted_metrics], axis=0) if upweighted_metrics else empty
    whole_network_accuracy = np.stack([m['baseline_acc_mean'] for m in condition_metrics], axis=0) if condition_metrics else empty

    if control_metric is None:
        delta_neuron_ratio = np.full_like(neuron_ratio, np.nan)
        delta_weight_ratio = np.full_like(weight_ratio, np.nan)
        control_neuron_ratio = np.full(len(RULE_INDICES), np.nan, dtype=float)
        control_weight_ratio = np.full(len(RULE_INDICES), np.nan, dtype=float)
        control_whole_accuracy = np.full(len(RULE_INDICES), np.nan, dtype=float)
    else:
        control_neuron_ratio = np.asarray(control_metric['neuron_ratio_mean'], dtype=float)
        control_weight_ratio = np.asarray(control_metric['weight_ratio_mean'], dtype=float)
        control_whole_accuracy = np.asarray(control_metric['baseline_acc_mean'], dtype=float)
        delta_neuron_ratio = neuron_ratio - control_neuron_ratio[None, :]
        delta_weight_ratio = weight_ratio - control_weight_ratio[None, :]

    for metric in condition_metrics:
        up_rule = metric['upweighted_rule']
        is_control = up_rule is None
        for rule_pos, rule_id in enumerate(RULE_INDICES):
            neuron_delta = float(metric['neuron_ratio_mean'][rule_pos] - control_neuron_ratio[rule_pos]) if control_metric is not None else float('nan')
            weight_delta = float(metric['weight_ratio_mean'][rule_pos] - control_weight_ratio[rule_pos]) if control_metric is not None else float('nan')
            whole_delta = float(metric['baseline_acc_mean'][rule_pos] - control_whole_accuracy[rule_pos]) if control_metric is not None else float('nan')
            rows.append({
                'condition': condition_tag(up_rule),
                'upweighted_rule': 'control' if is_control else up_rule,
                'target_rule': rule_id,
                'is_control': is_control,
                'is_upweighted_target': (up_rule == rule_id) if up_rule is not None else False,
                'whole_network_accuracy_mean': float(metric['baseline_acc_mean'][rule_pos]),
                'whole_network_accuracy_sem': float(metric['baseline_acc_sem'][rule_pos]),
                'whole_network_accuracy_delta_vs_control': whole_delta,
                'retained_accuracy_mean': float(metric['diag_retained_mean'][rule_pos]),
                'retained_accuracy_sem': float(metric['diag_retained_sem'][rule_pos]),
                'functional_token_neuron_ratio_mean': float(metric['neuron_ratio_mean'][rule_pos]),
                'functional_token_neuron_ratio_sem': float(metric['neuron_ratio_sem'][rule_pos]),
                'functional_token_neuron_ratio_delta_vs_control': neuron_delta,
                'functional_token_weight_ratio_mean': float(metric['weight_ratio_mean'][rule_pos]),
                'functional_token_weight_ratio_sem': float(metric['weight_ratio_sem'][rule_pos]),
                'functional_token_weight_ratio_delta_vs_control': weight_delta,
                'functional_token_neuron_count_mean': float(metric['neuron_count_mean'][rule_pos]),
                'specialization_margin_mean': float(metric['specialization_margin_mean'][rule_pos]),
                'specialization_margin_sem': float(metric['specialization_margin_sem'][rule_pos]),
            })

    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'config': {
            'rule_indices': RULE_INDICES,
            'upweight': UPWEIGHT,
            'base_rule_weight': BASE_RULE_WEIGHT,
            'control_rule_weights': rule_weights_for_condition(CONTROL_CONDITION),
            'l1': L1,
            'gamma': GAMMA,
            'repeat': REPEAT,
            'seed': SEED,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'analysis_method': ANALYSIS_METHOD,
            'analysis_subnet_target': ANALYSIS_SUBNET_TARGET,
            'token_extraction_method': TOKEN_EXTRACTION_METHOD,
            'token_extraction_fraction': TOKEN_EXTRACTION_FRACTION,
            'reuse_existing_analysis': REUSE_EXISTING_ANALYSIS,
        },
        'rows': rows,
        'matrices': {
            'condition_labels': [condition_axis_label(m['upweighted_rule']) for m in condition_metrics],
            'upweighted_rule_labels': [condition_axis_label(m['upweighted_rule']) for m in upweighted_metrics],
            'whole_network_accuracy_by_condition': whole_network_accuracy,
            'diag_retained_accuracy_by_upweighted_rule': diag_retained,
            'functional_token_neuron_ratio_by_upweighted_rule': neuron_ratio,
            'functional_token_weight_ratio_by_upweighted_rule': weight_ratio,
            'functional_token_neuron_ratio_delta_vs_control': delta_neuron_ratio,
            'functional_token_weight_ratio_delta_vs_control': delta_weight_ratio,
            'specialization_margin_by_upweighted_rule': specialization,
        },
        'control': None if control_metric is None else {
            'condition': condition_tag(control_metric['upweighted_rule']),
            'whole_network_accuracy_mean': control_metric['baseline_acc_mean'],
            'functional_token_neuron_ratio_mean': control_metric['neuron_ratio_mean'],
            'functional_token_weight_ratio_mean': control_metric['weight_ratio_mean'],
            'specialization_margin_mean': control_metric['specialization_margin_mean'],
        },
        'files': {
            'json': str(summary_path),
            'csv': str(csv_path),
        },
    }
    save_json(summary_path, summary)
    return summary


def plot_heatmap(
    matrix: np.ndarray,
    path: Path,
    title: str,
    cbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = 'viridis',
) -> None:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.7))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(RULE_INDICES)))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels([display_rule_tag(r) for r in RULE_INDICES], rotation=38, ha='right', fontsize=13)
    ax.set_yticklabels([display_rule_tag(r) for r in RULE_INDICES[:matrix.shape[0]]], fontsize=13)
    ax.set_xlabel('functional token / retained rule', fontsize=15)
    ax.set_ylabel('upweighted rule', fontsize=15)
    ax.set_title(title, fontsize=16, pad=12)
    cbar = fig.colorbar(im, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel(cbar_label, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def symmetric_color_limit(matrix: np.ndarray) -> float:
    values = np.asarray(matrix, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 1.0
    return max(float(np.nanmax(np.abs(values))), 1e-6)


def violin_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    center = float(np.nanmedian(values))
    epsilon = 0.006
    span = float(np.nanmax(values) - np.nanmin(values)) if values.size > 1 else 0.0
    if span <= epsilon:
        anchors = np.asarray([center - epsilon, center + epsilon], dtype=float)
        values = np.concatenate([values, anchors])
    return values


def plot_whole_network_accuracy_violin(condition_metrics: list[dict]) -> None:
    if not condition_metrics:
        return
    data = []
    positions = []
    rule_ids = []
    num_rules = len(RULE_INDICES)
    inner_gap = 0.12
    group_gap = 1.55
    violin_width = 0.10
    for condition_pos, metric in enumerate(condition_metrics):
        runs = np.asarray(metric['baseline_acc_runs'], dtype=float)
        base_x = condition_pos * group_gap
        for rule_pos, _rule_id in enumerate(RULE_INDICES):
            values = violin_values(runs[:, rule_pos])
            if values.size:
                data.append(values)
                positions.append(base_x + rule_pos * inner_gap)
                rule_ids.append(rule_pos)
    if not data:
        return

    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    fig_width = max(8.0, 1.15 * len(condition_metrics) + 2.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    parts = ax.violinplot(data, positions=positions, widths=violin_width, showmeans=True, showmedians=True, showextrema=False)
    for body, rule_pos in zip(parts['bodies'], rule_ids):
        color = colors[rule_pos % len(colors)]
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.68)
        body.set_linewidth(1.0)
    if 'cmeans' in parts:
        parts['cmeans'].set_color('#222222')
        parts['cmeans'].set_linewidth(1.0)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('#111111')
        parts['cmedians'].set_linewidth(1.35)

    centers = [idx * group_gap + inner_gap * (num_rules - 1) / 2.0 for idx in range(len(condition_metrics))]
    labels = [condition_axis_label(m['upweighted_rule']) for m in condition_metrics]
    finite = np.concatenate([np.asarray(d, dtype=float) for d in data])
    lower = max(0.0, float(np.nanmin(finite)) - 0.035)
    upper = min(1.05, max(1.01, float(np.nanmax(finite)) + 0.02))
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=28, ha='right', fontsize=13)
    ax.set_ylim(lower, upper)
    ax.set_xlabel('upweighted rule condition', fontsize=15)
    ax.set_ylabel('whole-network accuracy', fontsize=15)
    ax.set_title('Whole-network per-rule accuracy', fontsize=16, pad=12)
    ax.tick_params(axis='y', labelsize=13)
    ax.axhline(1.0, color='#666666', linewidth=0.9, linestyle='--', alpha=0.45, zorder=0)
    ax.grid(axis='y', color='#dddddd', linewidth=0.8, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], linewidth=5, alpha=0.75, label=display_rule_tag(rule_id)) for i, rule_id in enumerate(RULE_INDICES)]
    ax.legend(handles=handles, title='evaluated rule', loc='upper left', bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=12, title_fontsize=12, borderaxespad=0.0)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    fig.savefig(FIGURE_ROOT / 'whole_network_relative_accuracy_violin.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def copy_condition_combo_figures(sweep_conditions: list[dict]) -> None:
    for condition in sweep_conditions:
        analysis_dir = Path(condition['analysis_dir'])
        src = analysis_dir / 'figures' / 'weight_retained_combo_accuracy.png'
        if not src.is_file():
            continue
        dst = FIGURE_ROOT / 'weight_retained_combo_accuracy_{}.png'.format(condition['condition'])
        shutil.copy2(src, dst)


def _summary_config_value(summary: dict, key: str):
    if key in summary:
        return summary.get(key)
    config = summary.get('config')
    if isinstance(config, dict):
        return config.get(key)
    return None


def _summary_float_matches(summary: dict, key: str, expected: float, atol: float = 1e-12) -> bool:
    value = _summary_config_value(summary, key)
    try:
        return bool(np.isclose(float(value), float(expected), rtol=0.0, atol=atol))
    except (TypeError, ValueError):
        return False


def analysis_summary_matches_current_config(summary: dict) -> bool:
    expected_strings = {
        'mask_method': ANALYSIS_METHOD,
        'subnet_target': ANALYSIS_SUBNET_TARGET,
        'token_extraction_method': ana.validate_token_extraction_method(TOKEN_EXTRACTION_METHOD),
    }
    for key, expected in expected_strings.items():
        value = _summary_config_value(summary, key)
        if key == 'token_extraction_method':
            try:
                value = ana.validate_token_extraction_method(value)
            except Exception:
                return False
        if str(value).lower().strip() != str(expected).lower().strip():
            return False

    expected_floats = {
        'importance_min_score': IMPORTANCE_MIN_SCORE,
        'importance_neuron_threshold': IMPORTANCE_NEURON_THRESHOLD,
        'importance_score_ratio_threshold_weight': 1e-6,
        'importance_output_eps': 1e-8,
        'token_extraction_fraction': TOKEN_EXTRACTION_FRACTION,
    }
    for key, expected in expected_floats.items():
        if not _summary_float_matches(summary, key, expected):
            return False
    return True


def analysis_ready(analysis_dir: Path) -> bool:
    summary_path = analysis_dir / 'analysis_summary.json'
    figure_path = analysis_dir / 'figures' / 'weight_retained_combo_accuracy.png'
    if not summary_path.is_file() or not figure_path.is_file():
        return False
    try:
        summary = load_json(summary_path)
    except (OSError, json.JSONDecodeError):
        return False
    return analysis_summary_matches_current_config(summary)


def plot_target_bars(condition_metrics: list[dict]) -> None:
    condition_metrics = [m for m in condition_metrics if m['upweighted_rule'] is not None]
    if not condition_metrics:
        return
    x = np.arange(len(condition_metrics))
    labels = [display_rule_tag(m['upweighted_rule']) for m in condition_metrics]
    retained = np.asarray([m['target_retained_mean'] for m in condition_metrics], dtype=float)
    retained_sem = np.asarray([m['target_retained_sem'] for m in condition_metrics], dtype=float)
    neuron_ratio = np.asarray([m['target_weight_ratio_mean'] for m in condition_metrics], dtype=float)
    neuron_ratio_sem = np.asarray([m['target_weight_ratio_sem'] for m in condition_metrics], dtype=float)
    margin = np.asarray([m['target_specialization_margin_mean'] for m in condition_metrics], dtype=float)
    margin_sem = np.asarray([m['target_specialization_margin_sem'] for m in condition_metrics], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    axes[0].bar(x, retained, yerr=retained_sem, capsize=3, color='#0072B2')
    axes[0].set_ylabel('retained accuracy')
    axes[0].set_ylim(0, max(1.05, np.nanmax(retained + retained_sem) * 1.08))
    axes[0].set_title('Upweighted rule retained accuracy')

    axes[1].bar(x, neuron_ratio, yerr=neuron_ratio_sem, capsize=3, color='#009E73')
    axes[1].set_ylabel('functional token weight ratio')
    axes[1].set_ylim(0, max(0.05, np.nanmax(neuron_ratio + neuron_ratio_sem) * 1.18))
    axes[1].set_title('Upweighted rule weight ratio')

    axes[2].bar(x, margin, yerr=margin_sem, capsize=3, color='#D55E00')
    axes[2].axhline(0.0, color='#222222', linewidth=0.8)
    axes[2].set_ylabel('own minus off-rule retained accuracy')
    axes[2].set_title('Specialization margin')

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / 'upweighted_rule_target_summary.png', dpi=300)
    plt.close(fig)


def write_interpretation(summary: dict) -> None:
    rows = summary['rows']
    own_rows = [row for row in rows if row['is_upweighted_target']]
    other_rows = [row for row in rows if not row['is_upweighted_target'] and not row.get('is_control')]

    own_acc = np.asarray([row['retained_accuracy_mean'] for row in own_rows], dtype=float)
    own_ratio = np.asarray([row['functional_token_weight_ratio_mean'] for row in own_rows], dtype=float)
    own_margin = np.asarray([row['specialization_margin_mean'] for row in own_rows], dtype=float)

    deltas = []
    for rule_id in RULE_INDICES:
        own = next((row for row in rows if row['upweighted_rule'] == rule_id and row['target_rule'] == rule_id), None)
        if own is None:
            continue
        proxy = [row for row in other_rows if row['target_rule'] == rule_id]
        proxy_acc, _ = mean_sem(np.asarray([row['retained_accuracy_mean'] for row in proxy], dtype=float))
        proxy_ratio, _ = mean_sem(np.asarray([row['functional_token_weight_ratio_mean'] for row in proxy], dtype=float))
        proxy_margin, _ = mean_sem(np.asarray([row['specialization_margin_mean'] for row in proxy], dtype=float))
        deltas.append({
            'rule': rule_id,
            'retained_accuracy_delta_vs_other_upweights': own['retained_accuracy_mean'] - proxy_acc,
            'weight_ratio_delta_vs_other_upweights': own['functional_token_weight_ratio_mean'] - proxy_ratio,
            'specialization_margin_delta_vs_other_upweights': own['specialization_margin_mean'] - proxy_margin,
        })

    mean_acc_delta = float(np.nanmean([d['retained_accuracy_delta_vs_other_upweights'] for d in deltas]))
    mean_ratio_delta = float(np.nanmean([d['weight_ratio_delta_vs_other_upweights'] for d in deltas]))
    mean_margin_delta = float(np.nanmean([d['specialization_margin_delta_vs_other_upweights'] for d in deltas]))

    if mean_ratio_delta < 0 and mean_acc_delta >= -0.02 and mean_margin_delta > 0:
        verdict = 'The sweep supports the refinement-rather-than-expansion interpretation.'
    elif mean_ratio_delta < 0 and mean_acc_delta >= -0.02:
        verdict = 'The sweep partly supports refinement: emphasized tasks use no more tokens while retaining performance, but specialization-margin evidence is weak or mixed.'
    elif mean_ratio_delta >= 0:
        verdict = 'The sweep does not support the smaller-token refinement claim: emphasized rules use equal or larger functional-token ratios on average.'
    else:
        verdict = 'The sweep is mixed; retained accuracy, token ratio, and specialization changes do not move together cleanly.'

    lines = [
        '# Rule Weight Analysis',
        '',
        verdict,
        '',
        '## Aggregate Checks',
        '',
        '- Mean upweighted-rule retained accuracy: {:.4f}'.format(float(np.nanmean(own_acc))),
        '- Mean upweighted-rule functional token weight ratio: {:.4f}'.format(float(np.nanmean(own_ratio))),
        '- Mean upweighted-rule specialization margin: {:.4f}'.format(float(np.nanmean(own_margin))),
        '- Mean retained-accuracy delta vs the same rule under other upweight conditions: {:+.4f}'.format(mean_acc_delta),
        '- Mean weight-ratio delta vs the same rule under other upweight conditions: {:+.4f}'.format(mean_ratio_delta),
        '- Mean specialization-margin delta vs the same rule under other upweight conditions: {:+.4f}'.format(mean_margin_delta),
        '',
        '## Per-Rule Deltas',
        '',
    ]
    for delta in deltas:
        lines.append(
            '- {}: retained accuracy {:+.4f}, weight ratio {:+.4f}, specialization margin {:+.4f}'.format(
                display_rule_tag(delta['rule']),
                delta['retained_accuracy_delta_vs_other_upweights'],
                delta['weight_ratio_delta_vs_other_upweights'],
                delta['specialization_margin_delta_vs_other_upweights'],
            )
        )

    lines.extend([
        '',
        '## Reading Guide',
        '',
        'Evidence for the claim should look like high retained accuracy for the upweighted rule, a lower or unchanged single-rule functional-token ratio, and a larger own-vs-off-rule specialization margin.',
        'If the ratio increases, the likely explanation is expansion rather than refinement. If retained accuracy drops while the ratio shrinks, the penalty may be too strong or the upweighted task may be competing destructively with other rules.',
    ])
    (OUTPUT_ROOT / 'ruleweight_interpretation.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SIMULATION_ROOT.mkdir(parents=True, exist_ok=True)
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    dataset = BinaryPairDataset(num_samples=NUM_SAMPLES, pattern_size=OUTPUT_BITS, train_ratio=TRAIN_RATIO)
    task = BinaryArithmeticTask(rule_indices=RULE_INDICES)
    sweep_conditions = []

    for weighted_rule in [CONTROL_CONDITION] + RULE_INDICES:
        experiment_name = experiment_name_for(weighted_rule)
        rule_weights = rule_weights_for_condition(weighted_rule)
        output_dir = SIMULATION_ROOT / experiment_name

        if REUSE_EXISTING_SIMULATION and output_dir.exists() and (output_dir / 'summary.json').is_file():
            print('\n=== Reusing existing simulation: {} ==='.format(output_dir))
        elif RUN_TRAINING:
            model = TorchModularMLP(hidden_dim=MLP_HIDDEN_DIM, num_modules=MLP_NUM_MODULES, device=DEVICE)
            loss = WeightedMSELoss(
                rule_weights=rule_weights,
                l1=L1,
                gamma=GAMMA,
                regularization_epsilon=REGULARIZATION_EPSILON,
            )
            print('\n=== Rule weight sweep | condition={} | weights={} | l={} | gamma={} ==='.format(
                condition_axis_label(weighted_rule),
                rule_weights,
                L1,
                GAMMA,
            ))
            output_dir = run_repeats(
                dataset=dataset,
                task=task,
                model=model,
                loss=loss,
                experiment_name=experiment_name,
                output_root=SIMULATION_ROOT,
                repeat=REPEAT,
                seed=SEED,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                log_every=LOG_EVERY,
                optimizer_name=OPTIMIZER_NAME,
                weight_decay=WEIGHT_DECAY,
                snapshot_every=SNAPSHOT_EVERY,
            )
        else:
            raise FileNotFoundError('Simulation not found and RUN_TRAINING=False: {}'.format(output_dir))

        analysis_dir = ANALYSIS_ROOT / output_dir.name
        if RUN_ANALYSIS:
            if REUSE_EXISTING_ANALYSIS and analysis_ready(analysis_dir):
                print('\n=== Reusing existing analysis: {} ==='.format(analysis_dir))
            else:
                analyze_experiment(
                    output_dir,
                    max_weight_elements_per_block=None,
                    mask_method=ANALYSIS_METHOD,
                    importance_mask_mode='score_ratio',
                    importance_mask_mode_neuron='score_ratio',
                    importance_mask_mode_weight='score_ratio',
                    importance_neuron_threshold=IMPORTANCE_NEURON_THRESHOLD,
                    importance_score_ratio_threshold_weight=1e-6,
                    importance_output_eps=1e-8,
                    importance_min_score=IMPORTANCE_MIN_SCORE,
                    perturbation_batch_size=PERTURBATION_BATCH_SIZE,
                    analysis_device=DEVICE,
                    subnet_target=ANALYSIS_SUBNET_TARGET,
                    analysis_dir=analysis_dir,
                    correlation_sample_size=CORRELATION_SAMPLE_SIZE,
                    token_extraction_method=TOKEN_EXTRACTION_METHOD,
                    token_extraction_fraction=TOKEN_EXTRACTION_FRACTION,
                )

        sweep_conditions.append({
            'condition': condition_tag(weighted_rule),
            'upweighted_rule': weighted_rule,
            'rule_weights': rule_weights,
            'output_dir': str(output_dir),
            'analysis_dir': str(analysis_dir),
        })
        save_json(OUTPUT_ROOT / 'ruleweight_conditions.json', {'conditions': sweep_conditions})

    condition_metrics = [
        extract_condition_metrics(Path(condition['analysis_dir']), condition['upweighted_rule'])
        for condition in sweep_conditions
    ]
    summary = write_summary_tables(condition_metrics)
    matrices = summary['matrices']

    plot_heatmap(
        np.asarray(matrices['diag_retained_accuracy_by_upweighted_rule'], dtype=float),
        FIGURE_ROOT / 'single_rule_retained_accuracy_heatmap.png',
        'Single-rule retained accuracy after rule upweighting',
        'retained accuracy',
        vmin=0,
    )
    plot_heatmap(
        np.asarray(matrices['functional_token_weight_ratio_by_upweighted_rule'], dtype=float),
        FIGURE_ROOT / 'functional_token_weight_ratio_heatmap.png',
        'Weight-defined functional token ratio after rule upweighting',
        'retained weight ratio',
        vmin=0,
    )
    plot_heatmap(
        np.asarray(matrices['specialization_margin_by_upweighted_rule'], dtype=float),
        FIGURE_ROOT / 'specialization_margin_heatmap.png',
        'Functional token specialization margin',
        'own - off-rule retained accuracy',
    )

    weight_delta = np.asarray(matrices['functional_token_weight_ratio_delta_vs_control'], dtype=float)
    weight_delta_limit = symmetric_color_limit(weight_delta)
    plot_heatmap(
        weight_delta,
        FIGURE_ROOT / 'functional_token_weight_ratio_delta_vs_control_heatmap.png',
        'Weight-defined functional token ratio delta vs control',
        'delta retained weight ratio',
        vmin=-weight_delta_limit,
        vmax=weight_delta_limit,
        cmap='coolwarm',
    )

    plot_whole_network_accuracy_violin(condition_metrics)
    copy_condition_combo_figures(sweep_conditions)
    plot_target_bars(condition_metrics)
    write_interpretation(summary)

    print('Rule-weight analysis finished: {}'.format(OUTPUT_ROOT))
