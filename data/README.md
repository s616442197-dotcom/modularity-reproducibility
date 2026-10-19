# Data

`MNIST/raw/train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` are the MNIST training split already present in the source project. They are included to avoid a network download and to reconstruct the saved paired-digit inputs.

MNIST attribution: Yann LeCun, Corinna Cortes, and Christopher J. C. Burges, *The MNIST database of handwritten digits*. The dataset is an external work, not a newly licensed contribution of this repository. The original download implementation and its source URLs are in `dataset.py`.

Binary arithmetic and sparse-pattern inputs are generated locally from the saved seeds. Per-run measurements in `results/published/` and fixture checkpoints are outputs of the supplied research project.
