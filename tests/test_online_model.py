import random

import pytest

from trading_agent.model.online_model import LabelQueue, OnlineDirectionModel


def test_online_model_learns_separable_pattern():
    random.seed(0)
    model = OnlineDirectionModel(seed=7)
    for _ in range(400):
        x = {"a": random.gauss(0, 1), "b": random.gauss(0, 1)}
        y = 1 if (x["a"] - 0.5 * x["b"]) > 0 else 0
        model.learn(x, y)

    assert model.n_samples == 400
    assert model.rolling_score() > 0.7  # ROC AUC znatelne lepsi nez nahoda (0.5)


def test_online_model_save_load_roundtrip(tmp_path):
    random.seed(1)
    model = OnlineDirectionModel(seed=3)
    for _ in range(100):
        x = {"a": random.gauss(0, 1)}
        model.learn(x, 1 if x["a"] > 0 else 0)

    path = tmp_path / "model.pkl"
    model.save(str(path))
    reloaded = OnlineDirectionModel.load_or_create(str(path))

    assert reloaded.n_samples == model.n_samples
    probe = {"a": 1.5}
    assert reloaded.predict_proba(probe) == pytest.approx(model.predict_proba(probe))


def test_online_model_load_or_create_missing_file_returns_fresh(tmp_path):
    model = OnlineDirectionModel.load_or_create(str(tmp_path / "missing.pkl"))
    assert model.n_samples == 0


def test_label_queue_resolves_after_horizon():
    queue = LabelQueue(horizon=3)
    queue.push("AAPL", {"f": 1}, reference_price=100.0)

    assert queue.advance_and_resolve("AAPL", 101.0) == []
    assert queue.advance_and_resolve("AAPL", 102.0) == []
    resolved = queue.advance_and_resolve("AAPL", 105.0)
    assert resolved == [({"f": 1}, 1)]
    assert queue.pending_count("AAPL") == 0


def test_label_queue_label_is_direction_of_price_move():
    queue = LabelQueue(horizon=1)
    queue.push("AAPL", {"f": 1}, reference_price=100.0)
    resolved = queue.advance_and_resolve("AAPL", 90.0)
    assert resolved[0][1] == 0  # cena klesla -> label 0
