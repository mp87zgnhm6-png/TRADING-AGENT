"""Epsilon-greedy multi-armed bandit pro vyber mezi obchodnimi strategiemi.

Toto je "meta" uroven samouceni agenta: ML model se uci CO predikovat,
bandit se uci KTEREMU zdroji signalu v soucasnem rezimu trhu vice duverovat.
Kazda paze = jedna strategie; odmena = realizovane PnL% z obchodu, ktery
podle jejiho signalu agent otevrel a uzavrel. Stav je perzistovan na disk,
takze se uceni prenasi mezi restarty agenta.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from trading_agent.logging_setup import get_logger

logger = get_logger("bandit")


class EpsilonGreedyBandit:
    def __init__(
        self,
        arms: list[str],
        epsilon_start: float = 0.3,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        seed: Optional[int] = None,
    ):
        self.arms = list(arms)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.counts: dict[str, int] = {a: 0 for a in self.arms}
        self.values: dict[str, float] = {a: 0.0 for a in self.arms}
        self._rng = random.Random(seed)

    def select_arm(self) -> str:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if self._rng.random() < self.epsilon:
            return self._rng.choice(self.arms)
        return max(self.arms, key=lambda a: self.values[a])

    def update(self, arm: str, reward: float) -> None:
        if arm not in self.values:
            self.arms.append(arm)
            self.values[arm] = 0.0
            self.counts[arm] = 0
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def stats(self) -> dict:
        return {"epsilon": round(self.epsilon, 4), "counts": dict(self.counts), "values": dict(self.values)}

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "arms": self.arms,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "counts": self.counts,
            "values": self.values,
        }
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        Path(tmp).replace(path)

    @classmethod
    def load_or_create(cls, path: str, arms: list[str], seed: Optional[int] = None) -> "EpsilonGreedyBandit":
        instance = cls(arms, seed=seed)
        p = Path(path)
        if not p.exists():
            logger.info("Zadny ulozeny bandit na %s, zacinam s rovnomernymi vahami.", path)
            return instance
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            instance.arms = list(payload["arms"])
            instance.epsilon = payload["epsilon"]
            instance.epsilon_min = payload.get("epsilon_min", instance.epsilon_min)
            instance.epsilon_decay = payload.get("epsilon_decay", instance.epsilon_decay)
            instance.counts = {a: payload["counts"].get(a, 0) for a in instance.arms}
            instance.values = {a: payload["values"].get(a, 0.0) for a in instance.arms}
            for a in arms:
                if a not in instance.arms:
                    instance.arms.append(a)
                    instance.counts[a] = 0
                    instance.values[a] = 0.0
            logger.info("Bandit nacten z %s: %s", path, instance.stats())
        except Exception as exc:
            logger.warning("Nepodarilo se nacist bandit z %s (%s), zacinam znovu.", path, exc)
        return instance
