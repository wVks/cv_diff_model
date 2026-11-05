import numpy as np
import pytest
import torch
from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric


def test_fid_identical_features_is_zero_and_translation_has_known_distance():
    features = torch.from_numpy(np.random.default_rng(7).normal(size=(64, 4)))
    original = fid_features_to_statistics(features)
    same = fid_statistics_to_metric(original, original, verbose=False)["frechet_inception_distance"]
    shifted = fid_features_to_statistics(features + torch.tensor([1., 2., 0., 0.]))
    distance = fid_statistics_to_metric(original, shifted, verbose=False)["frechet_inception_distance"]
    assert same == pytest.approx(0., abs=1e-8)
    assert distance == pytest.approx(5., abs=1e-8)
