"""Online (inkrementalni) samoucici se model smeru ceny.

Toto je jadro "self-learningu" agenta: misto jednorazoveho natrenovani a
nasazeni pouzivame `river` - kazda nova svicka, jakmile je znamy jeji skutecny
vysledek, model okamzite doucuje (`learn_one`). `HoeffdingAdaptiveTreeClassifier`
navic uvnitr bezi ADWIN detekci concept driftu, takze se model prubezne
prizpusobuje meninimu se chovani trhu bez explicitniho periodickeho retrainu.
"""

from __future__ import annotations

import pickle
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from river import metrics, preprocessing, tree

from trading_agent.logging_setup import get_logger

logger = get_logger("model")


def _build_pipeline(seed: int = 42):
    return preprocessing.StandardScaler() | tree.HoeffdingAdaptiveTreeClassifier(
        grace_period=50,
        delta=1e-5,
        seed=seed,
    )


class OnlineDirectionModel:
    """Predikuje pravdepodobnost, ze cena za `horizon` svicek bude vyssi nez ted."""

    def __init__(self, seed: int = 42):
        self._lock = threading.Lock()
        self.pipeline = _build_pipeline(seed)
        self.metric = metrics.ROCAUC()
        self.n_samples = 0

    def predict_proba(self, features: dict) -> float:
        with self._lock:
            proba = self.pipeline.predict_proba_one(features)
        return float(proba.get(1, 0.5))

    def learn(self, features: dict, label: int) -> None:
        """'Test-then-train': nejdriv se vyhodnoti rolling metrika na dosud nevidenem
        vzorku, teprve pak se na nem model doucí - poctivy odhad online vykonu bez
        nutnosti drzet samostatnou validacni sadu."""
        with self._lock:
            proba = self.pipeline.predict_proba_one(features)
            self.metric.update(label, proba.get(1, 0.0))
            self.pipeline.learn_one(features, label)
            self.n_samples += 1

    def rolling_score(self) -> float:
        try:
            return float(self.metric.get())
        except Exception:
            return 0.5

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with self._lock:
            with open(tmp_path, "wb") as f:
                pickle.dump(
                    {"pipeline": self.pipeline, "metric": self.metric, "n_samples": self.n_samples},
                    f,
                )
        Path(tmp_path).replace(path)

    @classmethod
    def load_or_create(cls, path: str, seed: int = 42) -> "OnlineDirectionModel":
        instance = cls(seed=seed)
        p = Path(path)
        if not p.exists():
            logger.info("Zadny ulozeny model na %s, zacinam s novym modelem.", path)
            return instance
        try:
            with open(p, "rb") as f:
                state = pickle.load(f)
            instance.pipeline = state["pipeline"]
            instance.metric = state["metric"]
            instance.n_samples = state.get("n_samples", 0)
            logger.info(
                "Model nacten z %s (n_samples=%d, rolling_auc=%.4f)",
                path,
                instance.n_samples,
                instance.rolling_score(),
            )
        except Exception as exc:  # poskozeny/nekompatibilni checkpoint nesmi shodit agenta
            logger.warning("Nepodarilo se nacist model z %s (%s), zacinam s novym.", path, exc)
        return instance


@dataclass
class PendingSample:
    features: dict
    reference_price: float
    bar_index: int


class LabelQueue:
    """Fronta vzorku cekajicich na 'label', dokud neuplyne predikcni horizont.

    Drzena jen v pameti - pri restartu agenta se par rozpracovanych vzorku
    (max `horizon` svicek na symbol, typicky nekolik minut dat) jednoduse
    zahodi; neni to skoda, protoze historicke svicky se pri startu stejne
    znovu naseji a noveho pendingu rychle pribude.
    """

    def __init__(self, horizon: int):
        self.horizon = max(1, horizon)
        self._queues: dict[str, deque[PendingSample]] = {}
        self._bar_index: dict[str, int] = {}

    def push(self, symbol: str, features: dict, reference_price: float) -> None:
        idx = self._bar_index.get(symbol, 0)
        self._queues.setdefault(symbol, deque()).append(PendingSample(features, reference_price, idx))

    def advance_and_resolve(self, symbol: str, new_close: float) -> list[tuple[dict, int]]:
        """Zavola se na kazdou novou svicku symbolu. Vrati seznam (features, label)
        pro vsechny vzorky, kterym prave uplynul horizont."""
        idx = self._bar_index.get(symbol, 0) + 1
        self._bar_index[symbol] = idx
        q = self._queues.setdefault(symbol, deque())
        resolved: list[tuple[dict, int]] = []
        while q and idx - q[0].bar_index >= self.horizon:
            sample = q.popleft()
            label = 1 if new_close > sample.reference_price else 0
            resolved.append((sample.features, label))
        return resolved

    def pending_count(self, symbol: str) -> int:
        return len(self._queues.get(symbol, ()))
