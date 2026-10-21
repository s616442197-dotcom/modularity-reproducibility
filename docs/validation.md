# Current validation — 2026-10-21 audit fixes

The current delivery passed result-based rendering, six reduced training/analysis smoke cases, scalar-weight extraction smoke, published-statistic checks and regression tests for input selection, pairing and mask construction. Fig.5 contains 180 width-300 runs, 150 paired observations and pooled r = 0.7835066365; Fig.6b–c contains 360 evaluations. The 18 actual auxiliary-model records independently confirm 600 epochs of SGD, l_train = 1, gamma = 1 and regularization offset = 1e-8.

The renderer now draws black means/red medians, restores the reference Fig.5d axes and associated errors, positions the Fig.5e legend outside, and generates PNG previews only after the final vector PDF is assembled. Formal manuscript figures and the main text were retained unchanged. Sup S8.7 now states the auxiliary regularization coefficients.

The pipeline validates its actual input directory before reporting status. Published checks concern supplied observations only. Fresh-run output can pass structural/configuration checks while remaining PARTIAL / reproduction NOT_CHECKED until its scientific results are reviewed; execution success is never automatically a scientific reproduction claim.

Rendering and regression checks also passed in a copied repository with a Python file-open hook denying original project directories. This is not an OS sandbox or a fresh dependency installation. The existing modularity_env was used. Full fresh training of all 923 runs was not repeated. Main/Sup and the combined submission source were compiled, and the generated ZIP was unpacked and compiled independently. Word-to-TeX text and 195 native equations match; highlighting was checked against ql1019 after accepting revisions. Final page previews were inspected.

See [current_validation.json](current_validation.json) and [current_source_manifest.json](current_source_manifest.json). The delivered revision is identified by the Git commit containing these files; confirm its remote hash when receiving the repository. Earlier validation records remain in [history](history/).
