from __future__ import annotations

import gzip
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PairData:
    x: np.ndarray
    left_bits: np.ndarray
    right_bits: np.ndarray
    left_int: np.ndarray
    right_int: np.ndarray


@dataclass
class DatasetBundle:
    train: PairData
    test: PairData
    pattern_size: int


def int_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((values[:, None].astype(np.int64) >> shifts[None, :]) & 1).astype(np.float32)


def bits_to_int(bits: np.ndarray) -> np.ndarray:
    width = bits.shape[1]
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    return (bits.astype(np.int64) * weights[None, :]).sum(axis=1)


class BinaryPairDataset:
    def __init__(self, num_samples: int = 1024, pattern_size: int = 5, train_ratio: float = 0.5, unique_pairs: bool = True):
        self.num_samples = num_samples
        self.pattern_size = pattern_size
        self.train_ratio = train_ratio
        self.unique_pairs = unique_pairs

    def build(self, seed: int) -> DatasetBundle:
        rng = np.random.default_rng(seed)
        n_patterns = 1 << self.pattern_size
        max_pairs = n_patterns * n_patterns
        if self.unique_pairs and self.num_samples > max_pairs:
            raise ValueError('num_samples={} is larger than {} unique pairs'.format(self.num_samples, max_pairs))
        if self.unique_pairs:
            pair_code = rng.choice(max_pairs, size=self.num_samples, replace=False)
            left_int = pair_code // n_patterns
            right_int = pair_code % n_patterns
        else:
            left_int = rng.integers(0, n_patterns, size=self.num_samples)
            right_int = rng.integers(0, n_patterns, size=self.num_samples)
        left_bits = int_to_bits(left_int, self.pattern_size)
        right_bits = int_to_bits(right_int, self.pattern_size)
        data = self._make_pair_data(left_bits, right_bits, left_int, right_int)
        return self._split(data, rng)

    def _make_pair_data(self, left_bits: np.ndarray, right_bits: np.ndarray, left_int: np.ndarray, right_int: np.ndarray) -> PairData:
        x = np.concatenate([left_bits, right_bits], axis=1).astype(np.float32)
        return PairData(x=x, left_bits=left_bits.astype(np.float32), right_bits=right_bits.astype(np.float32), left_int=left_int.astype(np.int64), right_int=right_int.astype(np.int64))

    def _split(self, data: PairData, rng: np.random.Generator) -> DatasetBundle:
        order = rng.permutation(len(data.x))
        n_train = int(round(len(order) * self.train_ratio))
        train_idx = order[:n_train]
        test_idx = order[n_train:]
        def take(idx: np.ndarray) -> PairData:
            return PairData(x=data.x[idx], left_bits=data.left_bits[idx], right_bits=data.right_bits[idx], left_int=data.left_int[idx], right_int=data.right_int[idx])
        return DatasetBundle(train=take(train_idx), test=take(test_idx), pattern_size=self.pattern_size)


class SparsePatternDataset(BinaryPairDataset):
    def __init__(self, num_samples: int = 1024, pattern_size: int = 8, train_ratio: float = 0.5, sparsity: float = 0.5):
        super().__init__(num_samples=num_samples, pattern_size=pattern_size, train_ratio=train_ratio, unique_pairs=False)
        self.sparsity = sparsity

    def build(self, seed: int) -> DatasetBundle:
        rng = np.random.default_rng(seed)
        left_bits = (rng.random((self.num_samples, self.pattern_size)) > self.sparsity).astype(np.float32)
        right_bits = (rng.random((self.num_samples, self.pattern_size)) > self.sparsity).astype(np.float32)
        left_int = bits_to_int(left_bits)
        right_int = bits_to_int(right_bits)
        data = self._make_pair_data(left_bits, right_bits, left_int, right_int)
        return self._split(data, rng)


MNIST_URL_BASE = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
MNIST_FILES = {
    'train': ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte'),
    'test': ('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte'),
}


def _open_idx(path: Path):
    return gzip.open(path, 'rb') if path.suffix == '.gz' else open(path, 'rb')


def _read_idx_images(path: Path) -> np.ndarray:
    with _open_idx(path) as f:
        magic, count, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError('Invalid MNIST image file magic {} in {}'.format(magic, path))
        data = np.frombuffer(f.read(rows * cols * count), dtype=np.uint8)
    return data.reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with _open_idx(path) as f:
        magic, count = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError('Invalid MNIST label file magic {} in {}'.format(magic, path))
        data = np.frombuffer(f.read(count), dtype=np.uint8)
    return data.astype(np.int64)


def _mnist_raw_dirs(root: Path) -> list[Path]:
    return [root / 'MNIST' / 'raw', root / 'raw', root]


def _find_mnist_file(root: Path, filename: str) -> Path | None:
    for raw_dir in _mnist_raw_dirs(root):
        for candidate in [raw_dir / filename, raw_dir / '{}.gz'.format(filename)]:
            if candidate.exists():
                return candidate
    return None


def _ensure_mnist_file(root: Path, filename: str, download: bool) -> Path:
    existing = _find_mnist_file(root, filename)
    if existing is not None:
        return existing
    if not download:
        raise FileNotFoundError(
            'Missing MNIST file {} under {}. Put IDX files in MNIST/raw, raw, or root, or set download=True.'.format(filename, root)
        )
    raw_dir = root / 'MNIST' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = raw_dir / '{}.gz'.format(filename)
    url = MNIST_URL_BASE + '{}.gz'.format(filename)
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:
        raise RuntimeError(
            'Could not download {}. Check network access, install torchvision, or download MNIST IDX files manually into {}.'.format(url, raw_dir)
        ) from exc
    return target


def _load_mnist_split(root: Path, split: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
    image_name, label_name = MNIST_FILES[split]
    image_path = _ensure_mnist_file(root, image_name, download)
    label_path = _ensure_mnist_file(root, label_name, download)
    images = _read_idx_images(image_path)
    labels = _read_idx_labels(label_path)
    if len(images) != len(labels):
        raise ValueError('MNIST images and labels have different lengths: {} vs {}'.format(len(images), len(labels)))
    return images, labels


class MNISTPairDataset:
    def __init__(
        self,
        num_samples: int = 4096,
        pattern_size: int = 7,
        train_ratio: float = 0.8,
        root: str | Path = 'data/mnist',
        split: str = 'train',
        download: bool = True,
        normalize: bool = True,
        balanced_pairs: bool = True,
        output_bits: int | None = None,
    ):
        self.num_samples = int(num_samples)
        self.pattern_size = int(output_bits if output_bits is not None else pattern_size)
        self.train_ratio = float(train_ratio)
        self.root = Path(root)
        self.split = split
        self.download = bool(download)
        self.normalize = bool(normalize)
        self.balanced_pairs = bool(balanced_pairs)

    def build(self, seed: int) -> DatasetBundle:
        if self.num_samples <= 0:
            raise ValueError('num_samples must be positive')
        rng = np.random.default_rng(seed)
        images, labels = self._load_source_images()
        flat_images = images.reshape(images.shape[0], -1).astype(np.float32)
        if self.normalize:
            flat_images /= 255.0

        digit_indices = [np.flatnonzero(labels == digit) for digit in range(10)]
        missing = [digit for digit, idx in enumerate(digit_indices) if len(idx) == 0]
        if missing:
            raise ValueError('MNIST split {} is missing digit labels {}'.format(self.split, missing))

        left_digits, right_digits = self._sample_digit_pairs(rng)
        left_idx = self._sample_image_indices(left_digits, digit_indices, rng)
        right_idx = self._sample_image_indices(right_digits, digit_indices, rng)
        left_images = flat_images[left_idx]
        right_images = flat_images[right_idx]
        x = np.concatenate([left_images, right_images], axis=1).astype(np.float32)
        data = PairData(
            x=x,
            left_bits=left_images.astype(np.float32),
            right_bits=right_images.astype(np.float32),
            left_int=left_digits.astype(np.int64),
            right_int=right_digits.astype(np.int64),
        )
        return self._split(data, rng)

    def _load_source_images(self) -> tuple[np.ndarray, np.ndarray]:
        split = self.split.lower().strip()
        if split == 'both':
            train_images, train_labels = _load_mnist_split(self.root, 'train', self.download)
            test_images, test_labels = _load_mnist_split(self.root, 'test', self.download)
            return np.concatenate([train_images, test_images], axis=0), np.concatenate([train_labels, test_labels], axis=0)
        if split not in MNIST_FILES:
            raise ValueError("split must be 'train', 'test', or 'both', got {!r}".format(self.split))
        return _load_mnist_split(self.root, split, self.download)

    def _sample_digit_pairs(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        if self.balanced_pairs:
            repeats = int(np.ceil(self.num_samples / 100))
            pair_codes = np.tile(np.arange(100, dtype=np.int64), repeats)[: self.num_samples]
            rng.shuffle(pair_codes)
            return pair_codes // 10, pair_codes % 10
        return rng.integers(0, 10, size=self.num_samples, dtype=np.int64), rng.integers(0, 10, size=self.num_samples, dtype=np.int64)

    def _sample_image_indices(self, digits: np.ndarray, digit_indices: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
        out = np.empty(len(digits), dtype=np.int64)
        for digit in range(10):
            mask = digits == digit
            count = int(mask.sum())
            if count:
                out[mask] = rng.choice(digit_indices[digit], size=count, replace=True)
        return out

    def _split(self, data: PairData, rng: np.random.Generator) -> DatasetBundle:
        order = rng.permutation(len(data.x))
        n_train = int(round(len(order) * self.train_ratio))
        train_idx = order[:n_train]
        test_idx = order[n_train:]
        def take(idx: np.ndarray) -> PairData:
            return PairData(x=data.x[idx], left_bits=data.left_bits[idx], right_bits=data.right_bits[idx], left_int=data.left_int[idx], right_int=data.right_int[idx])
        return DatasetBundle(train=take(train_idx), test=take(test_idx), pattern_size=self.pattern_size)