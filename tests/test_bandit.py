import random

import pytest

from trading_agent.strategy.bandit import EpsilonGreedyBandit


def test_bandit_converges_to_best_arm():
    random.seed(0)
    bandit = EpsilonGreedyBandit(["a", "b", "c"], epsilon_start=0.5, epsilon_min=0.05, epsilon_decay=0.98, seed=42)
    true_rewards = {"a": 0.001, "b": 0.02, "c": -0.01}

    selections = {"a": 0, "b": 0, "c": 0}
    for _ in range(2000):
        arm = bandit.select_arm()
        selections[arm] += 1
        bandit.update(arm, random.gauss(true_rewards[arm], 0.01))

    assert max(bandit.values, key=bandit.values.get) == "b"
    assert selections["b"] > selections["a"]
    assert selections["b"] > selections["c"]
    assert bandit.epsilon == pytest.approx(bandit.epsilon_min, abs=1e-6)


def test_bandit_persistence_roundtrip(tmp_path):
    bandit = EpsilonGreedyBandit(["x", "y"], seed=1)
    bandit.update("x", 0.05)
    bandit.update("x", 0.03)
    bandit.update("y", -0.02)

    path = tmp_path / "bandit.json"
    bandit.save(str(path))

    loaded = EpsilonGreedyBandit.load_or_create(str(path), ["x", "y"])
    assert loaded.values == pytest.approx(bandit.values)
    assert loaded.counts == bandit.counts


def test_bandit_load_or_create_handles_missing_file(tmp_path):
    path = tmp_path / "does_not_exist.json"
    bandit = EpsilonGreedyBandit.load_or_create(str(path), ["a", "b"])
    assert bandit.values == {"a": 0.0, "b": 0.0}


def test_bandit_load_or_create_adds_new_arms(tmp_path):
    original = EpsilonGreedyBandit(["a", "b"], seed=1)
    original.update("a", 0.1)
    path = tmp_path / "bandit.json"
    original.save(str(path))

    loaded = EpsilonGreedyBandit.load_or_create(str(path), ["a", "b", "c_new"])
    assert "c_new" in loaded.arms
    assert loaded.values["c_new"] == 0.0
