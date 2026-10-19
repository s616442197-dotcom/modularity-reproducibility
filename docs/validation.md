# Validation record

Validation date: 2026-10-19. The original source tree, manuscript and reference figures were not edited or deleted during packaging.

## Evidence and limits

| Scope | Status | Evidence |
| --- | --- | --- |
| Saved observations → statistics | PASS | All five robustness families have the expected run counts and finite 5×31 accuracy arrays. Fig.5's three matrices match the original summary exactly (maximum absolute differences 0, 0, 0; acceptance tolerance 1e-12). |
| Run selection and statistical units | PASS | Transformer retains 23 runs; the seven batch-size-500 runs are excluded. NIP's 13,500 legacy records collapse to 540 run/layer observations; duplicates agree within 1e-12. Fig.5 contains 150 paired trials. |
| Dataset reconstruction | PASS | Binary, sparse-pattern, MNIST-pair, Transformer and task-weighting test inputs and targets match one saved run per family exactly. See `dataset_validation.json`. |
| Small training and scientific analysis | PASS | Six cases (binary, pattern, MNIST, Transformer, task-weight control and upweighting) train for 2 epochs on 40 samples. Extraction returns finite 5×31 arrays. Additional NIP, Taylor, representation, recomposition and task-weight aggregation checks pass. |
| Cost integration | PASS | A reduced 1-repeat, 2-epoch, 40-sample experiment trains the multitask model and five separate models and performs neuron extraction. This checks execution, not the paper's numerical cost ratio. |
| Independent-directory execution | PASS | A temporary copy runs observation replay, the six-case smoke suite and numerical checks with Python file-access audit hooks denying both original project trees. This is a Python audit-hook check, **not** a container/OS sandbox or a fresh dependency installation. |
| Figures and TeX | PASS | Six main vector PDFs and Fig.S1 rebuilt; main and supplement compile to 18 and 16 pages with no unresolved cross-references/citations. Rendered pages and figure panels were visually inspected. |
| Manuscript/code consistency | PARTIAL | Fig.5 checkpoints have 500 hidden units, unlike the broad manuscript description of 300. See `protocol.md`. No new complete prose/theory review is claimed by this packaging check. |
| Scientific conclusions from fresh training | PARTIAL | The 923-run full protocol is implemented but was not executed at paper scale. Neither small integrations nor replay of observations establish numerical agreement from fresh full training. |
| Full checkpoint reevaluation | PARTIAL | Two representative checkpoints are supplied, not all 923 model states. The full original results stay local. This delivery does not claim a complete checkpoint reevaluation. |

Machine-readable records are in `validation/`. Commands use the versions in `runtime.json`. TeX compilation was tested with Windows MiKTeX; scientific execution was tested under Linux/WSL. Ten non-fatal supplement warnings concern mathematical tokens in PDF bookmarks; there were no missing figures, undefined references or overfull boxes in the checked final logs.

## Repeat the checks

```bash
python tests/verify_published.py outputs/checks
python reproduce.py --smoke --output outputs/smoke
python reproduce.py --from-results --output outputs/replay
python scripts/isolation_check.py --deny /path/to/original/source --output outputs/isolation
```

For a machine without TeX, add `--skip-tex` to observation replay. The isolation check deliberately omits TeX and reports that limitation. It preserves its temporary copy for inspection. Output paths in these examples must be new/empty.

The final portable redraw uses the same scientific panel content and ordering with DejaVu Sans and fixed canvases. Its geometry is not a byte-for-byte reproduction of the reference PDFs. Reference PDFs in `manuscript/` remain unchanged. The vector schematic merge rejects placements outside the PDF canvas.

## Packaging fixes

- Removed runtime dependencies on source-project absolute paths; exported provenance paths are relative names.
- Fixed an MNIST path that worked on a case-insensitive Windows filesystem but failed in a Linux temporary directory. The bundled archives now resolve through `data/MNIST/raw` on both systems.
- Disabled inherited automatic tight cropping for fixed-canvas figure composition; this prevents loss of the top schematic panels. Panel labels are constrained inside the canvas.
- Alternated main/supplement TeX passes so references to the other document resolve before validation.

The adopted `1e-9` score filter and latest Fig.3/Fig.4a settings are explicitly distinguished from recovered historical records. The bundled observations retain their original numerical values. Running with a newly adopted parameter cannot be asserted to reproduce those values until its analyses have actually been rerun.

No original files were cleaned up. The complete approximately 141-GB source result tree remains outside Git; the repository contains compact run-level observations, configurations, required MNIST inputs and two fixture checkpoints. No new open-source license was assigned.
