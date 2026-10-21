"""Regression checks for input selection, honest status, and union-mask validation."""
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('verify_published', ROOT/'tests/verify_published.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
from scripts.verify_run_records import verify_profile


class DeliveryValidation(unittest.TestCase):
    def test_custom_data_and_fresh_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root/'data'
            shutil.copytree(ROOT/'results/published', data)
            path = data/'cost/extraction_runs.json'
            rows = json.loads(path.read_text())
            for row in rows:
                row['token_retained_relative_accuracy_mean'] -= .01
            path.write_text(json.dumps(rows))
            with redirect_stdout(StringIO()):
                result = checker.main(root/'fresh', data, 'fresh')
            self.assertEqual(result['data_directory'], str(data.resolve()))
            self.assertAlmostEqual(result['cost_retained_relative_accuracy'], .9427236425015575)
            self.assertEqual(result['status'], 'PARTIAL')
            self.assertEqual(result['reproduction_status'], 'NOT_CHECKED')
            with self.assertRaises(AssertionError), redirect_stdout(StringIO()):
                checker.main(root/'published', data, 'published')
            # Missing trials fail validation even in fresh mode.
            metrics_path = data/'ruleweight_runs.json'
            metrics = json.loads(metrics_path.read_text())
            metrics[0]['pairing'] = metrics[0]['pairing'][:-1]
            metrics_path.write_text(json.dumps(metrics))
            with self.assertRaises((AssertionError, ValueError)), redirect_stdout(StringIO()):
                checker.main(root/'invalid', data, 'fresh')

    def test_quantile_zero_filter_and_union_masks(self):
        # Ties at zero must be filtered; multi-rule extraction is a union.
        scores = [[0., 0., 0., 0.], [0., 1., 2., 3.]]
        block = dict(key='output_block_linear1_weight', shape=[1, 4],
                     mask_value_name='ablation_scores', ablation_scores=scores,
                     candidate_flat_indices=[0, 1, 2, 3],
                     important_mask=[[False]*4, [False, False, False, True]],
                     keep_flat_indices=[[], [3], [3]])
        profile = dict(token_extraction_method='score_quantile', token_extraction_fraction=.75,
                       importance_min_score=1e-9, importance_output_eps=1e-8,
                       max_elements_per_block=None, blocks=[block])
        self.assertEqual(verify_profile(profile), 1)
        wrong = copy.deepcopy(profile)
        wrong['blocks'][0]['keep_flat_indices'][2] = []
        with self.assertRaises(AssertionError):
            verify_profile(wrong)
        wrong = copy.deepcopy(profile)
        wrong['blocks'][0]['candidate_flat_indices'] = [0, 1, 2]
        with self.assertRaises(AssertionError):
            verify_profile(wrong)


if __name__ == '__main__':
    unittest.main()
