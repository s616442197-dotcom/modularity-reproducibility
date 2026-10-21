# Modularity reproducibility

Research code for **Sparsity-driven emergence of functional modularity in multitask neural networks**. This repository packages the current main manuscript and supplement, their scientific implementation, saved run configurations, and compact per-run observations.

**Verification status:** statistics and all six main figures plus Fig. S1 have been rebuilt from the supplied observations; small training/analysis integrations pass. The complete 923-run, 5,000-epoch training protocol has an executable entry point but has **not** been rerun in this packaging task. Numerical agreement of a fresh full training run is therefore not yet demonstrated. See [validation](docs/validation.md).

**Current Fig.5–6/S1 data:** Fig.5 now uses 180 newly trained models with three 300-unit hidden layers (six conditions, seeds 300–329). Fig.6 recomposition and the cost experiment use exhaustive scalar-weight extraction, quantile 0.75 and score floor 1e-9. The earlier 500-unit and neuron-extraction observations are not used by the supported entry point. See [protocol](docs/protocol.md) and [current source manifest](docs/current_source_manifest.json).

## Install

Tested with Python 3.10.20 on Linux/WSL. Create a separate environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate `.venv\Scripts\activate` instead. GPU training requires a compatible NVIDIA driver and PyTorch build. The inspected environment used PyTorch 2.11.0+cu128; CPU execution is supported. Exact versions are in [runtime.json](docs/runtime.json).

For PDF manuscript compilation, install TeX Live or MiKTeX with `pdflatex`, `bibtex`, and packages used by the TeX files (including `xr-hyper`, `natbib`, `amsmath`, `amssymb`, `graphicx`, `booktabs`, `hyperref`, `geometry`, and `fancyhdr`). `arxiv.sty` is included. `--skip-tex` is an explicit option for systems without TeX.

## Reproduce

All commands work from another working directory if `reproduce.py` is addressed by its absolute path. Output directories must be empty; prior results are never overwritten.

```bash
# Recompute statistics and redraw editable vector PDFs from per-run observations.
# Also compile the main manuscript and supplement with the newly drawn figures.
python reproduce.py --from-results --output outputs/from-results

# Same numerical/figure reconstruction without requiring TeX.
python reproduce.py --from-results --skip-tex --output outputs/plots

# Small training + extraction + NIP + geometry + recomposition integrations.
# Uses deliberately reduced models/data and is NOT a paper-scale experiment.
python reproduce.py --smoke --device cpu --output outputs/smoke

# Independent numerical checks against the exported reference observations.
python tests/verify_published.py outputs/checks

# Complete training and analysis from fresh initialization, then figures and TeX.
# This does not read results/published or the fixture checkpoints.
python reproduce.py --all --device cuda --output outputs/full

# Compile the supplied manuscript with the unmodified reference vector figures.
python reproduce.py --compile-reference --output outputs/reference
```

Full training is computationally expensive: 923 saved run configurations, regular checkpoints, exhaustive single-weight ablation where applicable, and the separate 3-repeat cost experiment. Resource use has not been benchmarked for a full rerun. There is no automatic resume: preserve an interrupted output, inspect its records, and use a fresh output directory for a new full run. Fixed seeds do not guarantee bitwise equality across hardware or software versions.

The reconstruction output includes `figures/main_fig1.pdf` through `main_fig6.pdf`, `figures/supp_training_cost.pdf`, `figures/statistics.json`, and (unless skipped) `manuscript/modularity.pdf` and `manuscript/sup.pdf`. Reference figures keep the original layout; portable redraws preserve the scientific panel content/order but use portable fonts and may differ in spacing. PDF/SVG schematics remain vector graphics. PNG previews are rendered from the final merged PDFs, including schematics and panel labels, using the pinned `pypdfium2` dependency.

Every result-based/full run validates its own data directory before rendering. `validation.json` separates execution, data validation and reproduction status. Published-data PASS covers the bundled observations and reference statistics only; it is not a fresh-training claim. Fresh `--all` results remain PARTIAL / reproduction NOT_CHECKED after configuration, shape, mask, pairing and statistics checks: their reference differences are recorded for scientific review, without demanding bitwise equality or automatically certifying the manuscript's conclusions. To recheck a fresh output explicitly, run `python tests/verify_published.py outputs/full/checks_review --data outputs/full/results --mode fresh --run-root outputs/full`. Regression tests are available via `python tests/test_delivery_validation.py`.

## Figure-to-code map

| Current manuscript panel | Inputs | Scientific implementation |
| --- | --- | --- |
| Fig.1a,b; Fig.2e | Model architecture | `scripts/draw_schematics.py`, `model.py` |
| Fig.1c–f | Binary MLP, 30 runs | `analysis.py`, `example/figure2.py` |
| Fig.2a,b,d,f | Pattern/MNIST/MLP Taylor/Transformer observations | `analysis.py`, `example/figure4.py` |
| Fig.2c | Row-normalized Taylor retained-accuracy matrix | `example/figure4.py::taylor_sensitivity_matrix` |
| Fig.3a,b | Single-rule masks, accuracy and selected-weight fractions | `scripts/pipeline.py::nip`, `example/figure5.py` |
| Fig.3c,d | Importance-weighted normalized inverse participation (NIP) | `analysis.py::importance_weighted_specialization_summary` |
| Fig.4a | Taylor NIP at 1,000–5,000 epochs | `scripts/pipeline.py::nip`, `example/modularity_time.py` |
| Fig.4b–e | Correlation, PCA entropy, hidden/output CKA | `example/figure8.py` |
| Fig.5a–e | Six task-weight conditions, 30 runs each; 150 paired trials | `example/ruleweight.py`, `example/figure6.py` |
| Fig.6a–c | Rule-output/input recomposition, 120 runs | `example/figure10.py`, `example/manuscript_figures.py` |
| Fig.6d; Fig.S1 | Multi-task versus five separate models, 3 repeats | `example/composite/multitask_vs_separate_cost.py` |

Historical script numbers differ from the final manuscript numbering. Use `reproduce.py` for the supported portable workflow; the retained original plot modules expose lower-level functions and their standalone defaults refer to the historical directory structure.

## Contents and provenance

- `analysis.py`, `model.py`, `dataset.py`, `task.py`, `loss.py`, `main.py`: original scientific implementation.
- `configs/training_runs.json`: saved per-run seeds, optimizer, loss and schedule settings; tensor dimensions and scalar checkpoint metadata.
- `configs/analysis.json`: the explicitly adopted extraction/NIP protocol.
- `results/published/`: run-level arrays, CSV observations, NIP records and cost records; statistical summaries are recomputed.
- `tests/fixtures/`: one real MLP checkpoint and one real Transformer checkpoint for inspection. These do not replace full training.
- `data/MNIST/`: the existing MNIST training image/label archives required to rebuild paired-digit inputs. See [data attribution](data/README.md).
- `manuscript/`: source manuscript, supplement, bibliography, template and unchanged reference figures.
- `docs/source_manifest.json`: relative source-file names and SHA-256 hashes used during extraction; not a download manifest. The full original experiment directory (approximately 141 GB) remains outside this repository and was not deleted.

The Transformer selection excludes runs `002, 011, 014, 017, 020, 022, 024` with batch size 500; 23 batch-size-128 runs remain. Original files are preserved in the source project. This is an explicit inclusion rule, not a silent failure filter.

The repository is private. No new open-source license is assigned; choose a suitable license before public release. Third-party datasets and the supplied TeX template retain their own terms.
