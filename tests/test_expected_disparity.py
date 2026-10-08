"""Scientific checks for signed, time-weighted, paired RDS estimates.

Run: .venv/bin/python -m unittest discover -s tests -p 'test_expected_disparity.py'
"""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from RDS_analysis.expected_disparity import (
    analyze_paired, cost_to_pmf, load_paired_archive, pmf_moments,
    window_trial_means, window_weights,
)


class ExpectedDisparityTests(unittest.TestCase):
    def test_reflected_distribution_has_opposite_mean_and_equal_variance(self):
        bins = np.array([-10.0, 0.0, 10.0])
        left = np.array([0.1, 0.2, 0.7])
        right = left[::-1]
        mu_left, var_left = pmf_moments(left, bins)
        mu_right, var_right = pmf_moments(right, bins)
        self.assertAlmostEqual(float(mu_left), 6.0)
        self.assertAlmostEqual(float(mu_right), -6.0)
        self.assertAlmostEqual(float(var_left), float(var_right))

    def test_cost_softmax_stability_and_pmf_axis(self):
        costs = np.array([[10001.0, 10000.0], [10000.0, 10001.0]])
        p = cost_to_pmf(costs, axis=0)
        mean, _ = pmf_moments(p, [-2, 2], axis=0)
        np.testing.assert_allclose(mean, [2 * np.tanh(0.5), -2 * np.tanh(0.5)])
        with self.assertRaises(ValueError):
            pmf_moments([-0.1, 1.1], [-1, 1])
        with self.assertRaises(ValueError):
            pmf_moments([0.2, 0.2], [-1, 1])

    def test_30hz_full_window_is_frame_mean(self):
        times = np.arange(30) / 30
        weights = window_weights(times, (0, 1), time_end=1)
        np.testing.assert_allclose(weights, np.full(30, 1 / 30))

    def test_partial_frame_duration_is_not_sample_count(self):
        times = np.arange(3) / 30
        weights = window_weights(times, (1 / 60, 1 / 12), time_end=0.1)
        np.testing.assert_allclose(weights, [0.25, 0.5, 0.25])
        irregular = window_weights([0, 0.1, 0.9], (0, 1), time_end=1)
        np.testing.assert_allclose(irregular, [0.1, 0.8, 0.1])

    def test_single_static_frame_is_held_for_its_declared_duration(self):
        z = np.array([[[7.0, -7.0]]])
        mean = window_trial_means(z, [0], (0, 2), time_end=2)
        np.testing.assert_allclose(mean, [[7, -7]])

    def test_linear_boundary_interpolation_integrates_a_ramp(self):
        weights = window_weights([0, 0.2, 1], (0.1, 0.7), method="linear")
        self.assertAlmostEqual(float(weights.sum()), 1)
        self.assertAlmostEqual(float(weights @ np.array([0, 0.2, 1])), 0.4)

    def test_sample_windows_are_explicit_and_half_open(self):
        weights = window_weights([0, 1, 2], (0, 2), method="samples")
        np.testing.assert_allclose(weights, [0.5, 0.5, 0])
        with self.assertRaises(ValueError):
            window_weights([0, 1], (3, 4), method="samples")

    def test_time_coverage_and_order_cannot_be_guessed(self):
        for times, bounds, kwargs in (
            ([0, 1], (0, 2), {}),
            ([0, 1], (0, 3), {"time_end": 2}),
            ([0, 0], (0, 1), {"time_end": 1}),
            ([0, 1], (-1, 1), {"method": "linear"}),
        ):
            with self.subTest(times=times, bounds=bounds, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    window_weights(times, bounds, **kwargs)

    def test_paired_bootstrap_preserves_known_eye_residual(self):
        # Widely different trials with exactly paired eye sum=0.5. Independent
        # eye resampling would invent variability in this known constant.
        left = np.array([1.0, 20.0, -8.0, 12.0])[:, None]
        right = -left + 0.5
        z = np.stack((np.repeat(left, 30, axis=1), np.repeat(right, 30, axis=1)), axis=-1)
        result, _ = analyze_paired(
            z, np.arange(30) / 30, ["crds"] * 4, [0.3] * 4, [10] * 4,
            (0, 1), time_end=1, time_unit="seconds", n_bootstrap=300,
        )
        group = result["groups"][0]
        self.assertEqual(group["n_trials"], 4)  # not 120 frames
        np.testing.assert_allclose(group["eye_sum"]["ci_px"], [0.5, 0.5], atol=1e-12)
        self.assertAlmostEqual(group["eye_sum"]["mean_px"], 0.5)
        self.assertGreater(group["left"]["standard_error_px"], 0)

    def test_targets_are_not_pooled_and_one_trial_has_no_interval(self):
        result, _ = analyze_paired(
            np.array([[[10, -10]], [[-10, 10]]]), [0], ["crds", "crds"],
            [0.3, 0.3], [10, -10], (0, 1), time_end=1, time_unit="seconds",
            n_bootstrap=10,
        )
        self.assertEqual(len(result["groups"]), 2)
        for group in result["groups"]:
            self.assertEqual(group["aligned_gain"], 1)
            self.assertIsNone(group["left"]["ci_px"])

    def test_legacy_files_and_missing_eye_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "legacy.npy", np.zeros((3, 4, 5)))
            with self.assertRaisesRegex(ValueError, "legacy"):
                load_paired_archive(path / "legacy.npy")
            np.savez(path / "incomplete.npz", z=np.zeros((1, 1, 2)))
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_paired_archive(path / "incomplete.npz")


class PairedInferenceTests(unittest.TestCase):
    def test_actual_small_architectures_accept_physical_eye_adapter(self):
        import torch
        from config.config_gcnet_lr import ConfigGCNet as LRConfig
        from config.config_gcnet_gn_lr import ConfigGCNet as GNConfig
        from GC_Net_LR.modules.gcnet import build_gcnet as build_lr
        from GC_NetGN_LR.modules.gcnet import build_gcnet as build_gn
        from RDS_analysis.paired_disparity import predict_paired_roi

        original_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for config_type, builder in ((LRConfig, build_lr), (GNConfig, build_gn)):
                config = config_type(max_disp=32, img_height=32, img_width=64,
                                     base_channels=8, n_resBlocks=1,
                                     compile_mode=None, load_state=False)
                model = builder(config)
                images = torch.randn(1, 3, 32, 64)
                result = predict_paired_roi(model, images, images.flip(-1), (8, 24, 16, 48))
                self.assertEqual(result.shape, (1, 2))
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue(((result >= -16) & (result <= 15)).all())
        finally:
            torch.set_num_threads(original_threads)

    def test_physical_images_unchanged_and_both_outputs_measured(self):
        import torch
        from RDS_analysis.paired_disparity import predict_paired_roi

        class EyeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []

            def forward(self, inputs):
                self.calls.append((inputs.left, inputs.right, inputs.ref.clone()))
                # Deliberately asymmetric: right must be measured as -1,
                # rather than manufactured by negating left's +2 prediction.
                value = torch.where(inputs.ref == 1, 2.0, -1.0)
                return value[:, None, None].expand(len(inputs.ref), 4, 6)

        model = EyeModel().train()
        left, right = torch.ones(2, 3, 4, 6), torch.zeros(2, 3, 4, 6)
        result = predict_paired_roi(model, left, right, (1, 3, 1, 4), roi_right=(1, 3, 2, 5))
        np.testing.assert_allclose(result, [[2, -1], [2, -1]])
        self.assertTrue(model.training)
        for call_left, call_right, _ in model.calls:
            self.assertIs(call_left, left)
            self.assertIs(call_right, right)
        self.assertEqual(model.calls[0][2].tolist(), [1, 1])
        self.assertEqual(model.calls[1][2].tolist(), [-1, -1])

    def test_invalid_roi_and_nonfinite_model_output_are_rejected(self):
        import torch
        from RDS_analysis.paired_disparity import predict_paired_roi

        class BadModel(torch.nn.Module):
            def forward(self, inputs):
                return torch.full((1, 4, 6), float("nan"))

        model = BadModel().train()
        inputs = torch.zeros(1, 3, 4, 6)
        with self.assertRaisesRegex(ValueError, "ROI"):
            predict_paired_roi(model, inputs, inputs, (0, 4, 0, 7))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            predict_paired_roi(model, inputs, inputs, (0, 4, 0, 6))
        self.assertTrue(model.training)


class CollectionTests(unittest.TestCase):
    def test_collection_uses_real_rds_generator_and_records_30hz(self):
        """Exercise checkpoint I/O/generation/ROI/archive with a light model fixture."""
        from argparse import Namespace
        from contextlib import redirect_stdout
        import io
        from unittest.mock import patch
        import torch
        from scripts.analysis.run_expected_disparity import collect

        class FixtureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.amplitude = torch.nn.Parameter(torch.tensor(4.0))

            def forward(self, inputs):
                n, _, height, width = inputs.left.shape
                return (inputs.ref * self.amplitude)[:, None, None].expand(n, height, width)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            checkpoint = path / "fixture.pth"
            torch.save({"state_dict": FixtureModel().state_dict()}, checkpoint)
            args = Namespace(
                checkpoint=checkpoint, model="GC_Net_LR", interaction="sum_diff",
                seed=27, batch_size=1, device="cpu", dot_density=[0.02],
                disparities=[10], trials=2, frames=2, fps=30, amp="float32",
                roi=None, output=path / "paired.npz",
            )
            with patch("GC_Net_LR.modules.gcnet.build_gcnet", return_value=FixtureModel()):
                with redirect_stdout(io.StringIO()):
                    collect(args)
            data = load_paired_archive(args.output)
            self.assertEqual(data["z"].shape, (6, 2, 2))
            np.testing.assert_allclose(data["times"], [0, 1 / 30])
            self.assertAlmostEqual(data["time_end"], 2 / 30)
            np.testing.assert_allclose(data["z"], np.broadcast_to([4, -4], (6, 2, 2)))
            self.assertEqual(set(data["rds_type"]), {"ards", "hmrds", "crds"})
            np.testing.assert_array_equal(data["target_disparity"], np.full(6, 10))

    def test_real_crds_displacement_matches_declared_geometric_sign(self):
        from contextlib import redirect_stdout
        import io
        from joblib import parallel_config
        from RDS.DataHandler_RDS import RDS_Handler

        state = np.random.get_state()
        try:
            np.random.seed(91)
            with parallel_config(backend="sequential"), redirect_stdout(io.StringIO()):
                left, right, _ = RDS_Handler.generate_rds(1, 0.1, [-10], 1, True, False)
        finally:
            np.random.set_state(state)
        patch = left[0, 84:172, 160:352, 0]
        candidates = np.arange(-16, 17)
        scores = [(patch * right[0, 84:172, 160 - d:352 - d, 0]).mean() for d in candidates]
        self.assertEqual(int(candidates[np.argmax(scores)]), 10)


if __name__ == "__main__":
    unittest.main()
