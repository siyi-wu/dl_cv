from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch import nn

from miniddpm.diffusion import GaussianDiffusion, cosine_beta_schedule, linear_beta_schedule
from miniddpm.evaluation import MNISTClassifier, classification_statistics, frechet_feature_distance
from miniddpm.model import EnhancedUNet, MiniUNet, build_model


class ZeroNoiseModel(nn.Module):
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class DiffusionTests(unittest.TestCase):
    def test_schedules_are_valid(self):
        for schedule in (linear_beta_schedule(1000), cosine_beta_schedule(1000)):
            self.assertEqual(tuple(schedule.shape), (1000,))
            self.assertTrue(torch.all(schedule > 0))
            self.assertTrue(torch.all(schedule < 1))

    def test_q_sample_matches_closed_form(self):
        diffusion = GaussianDiffusion(timesteps=20)
        x_start = torch.ones(2, 1, 4, 4)
        noise = torch.full_like(x_start, 0.25)
        timesteps = torch.tensor([0, 19])
        actual, returned_noise = diffusion.q_sample(x_start, timesteps, noise)
        expected = (
            diffusion.alpha_bars[timesteps].sqrt()[:, None, None, None] * x_start
            + (1 - diffusion.alpha_bars[timesteps]).sqrt()[:, None, None, None] * noise
        )
        self.assertTrue(torch.equal(returned_noise, noise))
        self.assertTrue(torch.allclose(actual, expected))

    def test_respaced_samplers_return_finite_images(self):
        diffusion = GaussianDiffusion(timesteps=10, schedule="cosine")
        model = ZeroNoiseModel()
        initial = torch.randn(2, 1, 8, 8)
        for sampler in ("ancestral", "ddim"):
            result = diffusion.sample(
                model,
                tuple(initial.shape),
                steps=5,
                sampler=sampler,
                initial_noise=initial,
            )
            self.assertEqual(tuple(result.shape), tuple(initial.shape))
            self.assertTrue(torch.isfinite(result).all())
            self.assertGreaterEqual(float(result.min()), -1.0)
            self.assertLessEqual(float(result.max()), 1.0)


class ModelTests(unittest.TestCase):
    def test_shape_and_parameter_budget(self):
        model = MiniUNet()
        inputs = torch.randn(2, 1, 28, 28)
        outputs = model(inputs, torch.tensor([0, 999]))
        self.assertEqual(tuple(outputs.shape), tuple(inputs.shape))
        self.assertLessEqual(model.parameter_count, 5_000_000)

    def test_enhanced_shape_budget_and_factory(self):
        model = EnhancedUNet()
        inputs = torch.randn(2, 1, 28, 28)
        outputs = model(inputs, torch.tensor([0, 999]))
        self.assertEqual(tuple(outputs.shape), tuple(inputs.shape))
        self.assertGreater(model.parameter_count, MiniUNet().parameter_count)
        self.assertLessEqual(model.parameter_count, 5_000_000)
        self.assertIsInstance(build_model("enhanced"), EnhancedUNet)


class EvaluationTests(unittest.TestCase):
    def test_classifier_shapes_and_coverage_statistics(self):
        classifier = MNISTClassifier(feature_dim=32)
        images = torch.randn(4, 1, 28, 28)
        features = classifier.forward_features(images)
        logits = classifier(images)
        self.assertEqual(tuple(features.shape), (4, 32))
        self.assertEqual(tuple(logits.shape), (4, 10))
        statistics = classification_statistics(torch.eye(10) * 10)
        self.assertEqual(statistics["class_coverage"], 10)
        self.assertAlmostEqual(statistics["normalized_class_entropy"], 1.0, places=5)

    def test_frechet_distance_is_zero_for_identical_features(self):
        features = torch.randn(64, 16)
        self.assertAlmostEqual(frechet_feature_distance(features, features), 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
