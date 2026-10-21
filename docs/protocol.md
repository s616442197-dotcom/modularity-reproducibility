# Protocol and evidence

## Parameter provenance

`configs/training_runs.json` is exported from saved `config.json` records. It contains 31 groups and 923 included runs: 113 baseline/robustness runs, 630 regularization-grid runs, and 180 task-weighting runs. The seven excluded Transformer configurations are not included. Saved weight tensor dimensions override directory names.

Dataset reconstruction was compared against `x_test` and `y_test` in one saved run from each of the binary, sparse-pattern, MNIST-pair, Transformer and task-weighting families. All five comparisons were exactly equal. See `dataset_validation.json`. Each uses 5,000 samples, a 50:50 train/test split, and 7-bit targets. The MNIST pair inputs contain two 784-dimensional images; sparse patterns use sparsity 0.7.

The current model constructors retain the source initialization logic (MLP initialization scale 1.0; Transformer 0.6). These are current implementation settings, not independent proof of every historical software/environment setting. The new task is not a reconstruction of an old environment.

## Extraction and scoring

- Adopted per user instruction: zero-score lower bound `importance_min_score = 1e-9`. Latest Fig.3/Fig.4a analysis settings are used for new runs. This adoption must not be misreported as recovery of a missing historical log.
- Quantile parameter `tau = 0.75`, computed separately for each weight matrix and rule. The numeric threshold varies with the score distribution. No legacy fixed cutoff or `panel_a_threshold` controls this path.
- All candidates in the analyzed matrices are evaluated; `max_weight_elements_per_block = None`.
- Each single-weight intervention measures importance. Extracted masks are then applied simultaneously to all included weight matrices, including the output layer. Biases/unmasked parameters remain unchanged. Multi-rule subnetworks use mask unions.
- Retained-weight fraction counts selected mask positions over all analyzed weight positions. It is not the proportion of nonzero weights in the complete network. Ties under `>=` and zero-score filtering can move it away from 25%.
- Taylor analysis uses four random sign probes, seed 0, and squared weight-times-output-gradient scores, normalized by the sample count. It estimates weight-squared times expected squared output sensitivity, not absolute loss-gradient Taylor importance.
- Fig.3c,d use importance-weighted NIP, with output-energy epsilon `1e-24` for this ablation/NIP calculation. Fig.4a uses the Taylor/NIP path with output-energy epsilon `1e-8`. Other weight extraction uses `1e-8`. These epsilons are distinct from the `1e-9` score filter.
- The regularization-grid displayed `l` equals five times its saved loss `l1` coefficient. Other experiment families retain their own saved `l1` without reinterpretation.

`scripts/pipeline.py` uses `configs/analysis.json` for new weight analyses. Bundled observations are exported original results; changing a protocol setting does not magically recompute those observations. Use the full training/analysis path for new data.

## Statistical units

Fig.3a,b bars show mean ± sample SD (`ddof=1`). Violin guide lines connect medians at shared x coordinates. Fig.3 NIP files retain 25 identical legacy threshold entries per run/layer; these collapse to one run/layer measurement, not 25 independent replicates. There are 540 unique condition/run/layer measurements.

Fig.4 representation metrics use 30 runs per condition. Correlation histograms use the recorded first four runs per condition (0,1,5,10), and 100 bins. Fig.5 includes 30 runs per condition and 150 paired trial-level comparisons; its SEM follows the source implementation (`ddof=0`). Fig.6 recomposition includes 30 runs at each of four regularization values.

The separate cost experiment uses seeds 830–832, 600 epochs, 2,000 binary samples, 50% training split, SGD learning rate 0.05, batch size 128, three 300-unit hidden layers, loss coefficients `l1=single_l1=gamma=1`, and zero optimizer weight decay. Extraction evaluates all 194,700 scalar weights, grouped into 3,045 perturbation batches (batch size 64), on 1,000 analysis samples. This gives 194,700,000 scalar-weight/sample evaluations and a batching proxy of 3,045,000. The training and extraction proxies have different scales: their sum is only the predefined heuristic index, not FLOPs, measured speedup, or an equal-accuracy comparison. Historical training seconds and newly measured extraction seconds are not added.

## Current Fig.5 protocol and definitions

The 2026-10-20 rerun replaces all six task-weighting groups with 30 new runs each (seeds 300–329), three 300-unit hidden layers, 5,000 epochs, Adam learning rate 0.05 and batch size 128. Saved configurations and all 180 checkpoint dimensions were checked during this update. The old 500-unit exception is superseded. Fig.6 uses 120 original 300-unit parent models with new weight extraction and two recomposition paths (360 evaluation rows); cost analysis uses three joint and fifteen separate trained models with the auxiliary 600-epoch SGD protocol above.

Rule accuracy is mean bit agreement after thresholding predictions at >0.5. It is not exact output-vector accuracy. NIP uses the numerical active-mass floor epsilon_M=1e-12 (distinct from epsilon_S=1e-9), and is undefined/NaN if no token is active. Fig.5d error bars combine the two marginal SEMs by their root sum of squares (ddof=0), not the SEM of paired differences. Fig.5e uses shared seeds and controls; its pooled p-value is nominal and unadjusted for grouping. Sparse-pattern examples were not deduplicated across splits, MNIST was split by image pairs, and extraction and subnetwork evaluation reuse the held-out set.

## Limits of current validation

Per-run observation replay, dataset reconstruction and small integrations have been tested. A fresh, paper-scale full retraining has not been completed here. Replaying observations does not establish training reproducibility; small-model tests do not establish numerical agreement at full width and 5,000 epochs. No historical parameter is filled in and labeled “confirmed” solely because it was a default.
