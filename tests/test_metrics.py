import pytest
from faceattend.metrics import classification_metrics


def test_known_metrics():
    metrics = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["accuracy"] == .75
    assert metrics["macro_f1"] == pytest.approx((2/3 + .8) / 7)
    assert metrics["confusion_matrix"][0][:2] == [1, 1]
