"""Spoji ML model a pravidlove strategie do jednoho obchodniho signalu.

Bandit vybira, cim se agent v danem okamziku ridi (`strategy_used`); pokud se
alespon dve ze tri strategii shodnou na smeru, jistota (confidence) vybrane
strategie se mirne navysi (nezavisle potvrzeni signalu).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from trading_agent.model.online_model import OnlineDirectionModel
from trading_agent.strategy import rules
from trading_agent.strategy.bandit import EpsilonGreedyBandit

ARM_ML_MODEL = "ml_model"
ARM_TREND_FOLLOWING = "trend_following"
ARM_MEAN_REVERSION = "mean_reversion"
ALL_ARMS = [ARM_ML_MODEL, ARM_TREND_FOLLOWING, ARM_MEAN_REVERSION]

AGREEMENT_BONUS = 0.1


@dataclass
class TradeSignal:
    symbol: str
    direction: int  # -1 sell/short, 0 hold, 1 buy
    confidence: float
    strategy_used: str
    components: dict = field(default_factory=dict)


def _ml_signal(model: OnlineDirectionModel, features: dict) -> tuple[int, float]:
    proba_up = model.predict_proba(features)
    direction = 1 if proba_up > 0.5 else (-1 if proba_up < 0.5 else 0)
    confidence = abs(proba_up - 0.5) * 2.0
    return direction, confidence


class SignalEngine:
    def __init__(self, model: OnlineDirectionModel, bandit: EpsilonGreedyBandit):
        self.model = model
        self.bandit = bandit

    def generate(self, symbol: str, features: dict) -> TradeSignal:
        components = {
            ARM_ML_MODEL: _ml_signal(self.model, features),
            ARM_TREND_FOLLOWING: rules.trend_following_signal(features),
            ARM_MEAN_REVERSION: rules.mean_reversion_signal(features),
        }

        chosen_arm = self.bandit.select_arm()
        direction, confidence = components[chosen_arm]

        agreeing = [d for d, _ in components.values() if d == direction and d != 0]
        if direction != 0 and len(agreeing) >= 2:
            confidence = min(1.0, confidence + AGREEMENT_BONUS)

        return TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            strategy_used=chosen_arm,
            components=components,
        )
