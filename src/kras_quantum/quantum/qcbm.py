"""
QCBM - Quantum Circuit Born Machine.

Makale referansi: "Quantum generative model: QCBM model"

    "The QCBM is a variational quantum algorithm that uses the foundational
     principles of quantum mechanics, particularly the Born rule, to generate
     complex and diverse data samples. ... As per the Born rule, given a
     measurement basis, ... the probability of observing a specific outcome
     |x> is expressed as |<x|psi(theta)>|^2."

    "In the context of QCBM training, the exact NLL functions as the primary
     loss function ... NLL(theta) = -sum_{x in D} log p_theta(x) ...
     In our project, we used COBYLA for our optimizer."

    "The initial probability of the samples, P(X(i)), was computed on the basis
     of the rewards returned by the Chemistry42 model. These reward-based
     probabilities were then passed through a softmax function ... The resulting
     values served as the 'true' probabilities of the samples and were used as
     the target values during the model's training process."

Yani egitim dongusu (Algorithm 2):
    hedef dagilim  <- softmax(odul)      [seyrek: sadece orneklenen bitstring'ler]
    kayip          <- exact NLL
    optimizasyon   <- COBYLA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from kras_quantum.config import QCBMConfig
from kras_quantum.quantum.ansatz import AnsatzSpec, build_ansatz
from kras_quantum.quantum.backend import AerBackend, QuantumBackend

logger = logging.getLogger(__name__)

__all__ = ["QCBM", "TargetDistribution", "softmax", "TrainingHistory"]

# log(0) = -inf'i engellemek icin olasilik tabani
_EPS = 1e-12


def softmax(rewards: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Odul vektorunu olasilik dagilimina cevirir.

    Makale: "P(x) = softmax(R(x))"

    Sayisal kararlilik icin en buyuk deger cikarilir (log-sum-exp hilesi).
    """
    r = np.asarray(rewards, dtype=float)
    if r.size == 0:
        return r
    if temperature <= 0:
        raise ValueError(f"temperature > 0 olmali, verilen: {temperature}")
    z = (r - r.max()) / temperature
    e = np.exp(z)
    return e / e.sum()


@dataclass(frozen=True)
class TargetDistribution:
    """QCBM'in ogrenmeye calistigi seyrek hedef dagilim.

    16 qubit icin olasi sonuc sayisi 2^16 = 65536'dir; ancak her epoch'ta
    yalnizca orneklenen birkac bin bitstring odullendirilir. Bu yuzden hedefi
    seyrek tutuyoruz: (indeks, olasilik) ciftleri.

    Attributes
    ----------
    indices : np.ndarray[int]
        Bitstring'lerin tamsayi karsiliklari.
    probabilities : np.ndarray[float]
        Toplami 1 olan hedef olasiliklar (softmax(odul) ciktisi).
    """

    indices: np.ndarray
    probabilities: np.ndarray

    def __post_init__(self) -> None:
        if self.indices.shape != self.probabilities.shape:
            raise ValueError(
                f"indices ve probabilities ayni boyutta olmali: "
                f"{self.indices.shape} vs {self.probabilities.shape}"
            )
        if self.indices.size == 0:
            raise ValueError("Hedef dagilim bos olamaz")
        total = float(self.probabilities.sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"Hedef olasiliklarin toplami 1 olmali, bulunan: {total}")

    @classmethod
    def from_rewards(
        cls,
        bitstrings: np.ndarray,
        rewards: np.ndarray,
        temperature: float = 1.0,
    ) -> "TargetDistribution":
        """Odullerden hedef dagilim kurar (Makale: P(x) = softmax(R(x))).

        Ayni bitstring birden fazla kez orneklenmisse odulleri ortalanir.
        """
        bits = np.asarray(bitstrings, dtype=np.int64)
        rew = np.asarray(rewards, dtype=float)
        if bits.shape != rew.shape:
            raise ValueError(f"bitstrings ve rewards ayni boyutta olmali: {bits.shape} vs {rew.shape}")

        uniq, inverse = np.unique(bits, return_inverse=True)
        summed = np.bincount(inverse, weights=rew, minlength=uniq.size)
        counts = np.bincount(inverse, minlength=uniq.size)
        mean_rewards = summed / counts

        return cls(indices=uniq, probabilities=softmax(mean_rewards, temperature))

    def __len__(self) -> int:
        return int(self.indices.size)


@dataclass
class TrainingHistory:
    """Bir `fit` cagrisinin kaydi."""

    loss: list[float] = field(default_factory=list)
    n_evaluations: int = 0
    converged: bool = False
    message: str = ""

    @property
    def initial_loss(self) -> float | None:
        return self.loss[0] if self.loss else None

    @property
    def final_loss(self) -> float | None:
        return self.loss[-1] if self.loss else None

    @property
    def improvement(self) -> float | None:
        """Kayiptaki toplam azalma (pozitif = iyilesme)."""
        if len(self.loss) < 2:
            return None
        return self.loss[0] - self.loss[-1]


class QCBM:
    """Quantum Circuit Born Machine.

    Kuantum devresi bir olasilik dagilimi kodlar; Born kurali ile
    p_theta(x) = |<x|psi(theta)>|^2. Parametreler theta, uretilen dagilimi
    hedef dagilima yaklastiracak sekilde klasik bir optimize edici (COBYLA)
    ile ayarlanir.

    Hibrit hatta QCBM bir "prior" gorevi gorur: urettigi ornekler LSTM'e
    beslenir (Makale, Denklem 9 ve Extended Data Fig. 3a).

    Example
    -------
    >>> qcbm = QCBM(QCBMConfig(n_qubits=4, n_layers=1, n_epochs=1))
    >>> samples = qcbm.sample(shots=128)          # egitilmemis prior'dan ornek
    >>> target = TargetDistribution.from_rewards(samples, rewards)
    >>> history = qcbm.fit(target)
    """

    def __init__(
        self,
        config: QCBMConfig | None = None,
        backend: QuantumBackend | None = None,
        ansatz_mode: str = "fig5",
    ) -> None:
        self.config = config or QCBMConfig()
        self.spec = AnsatzSpec(
            n_qubits=self.config.n_qubits,
            n_layers=self.config.n_layers,
            mode=ansatz_mode,
            entanglement=self.config.entanglement,
        )
        self.circuit, self.theta = build_ansatz(self.spec)
        self.backend = backend or AerBackend(
            n_qubits=self.config.n_qubits, seed=self.config.seed
        )

        self._rng = np.random.default_rng(self.config.seed)
        # Baslangic parametreleri rastgele (Makale: "the model is initially
        # configured with randomly assigned parameters |psi(theta)>").
        self.params = self._rng.uniform(0, 2 * np.pi, size=self.n_parameters)
        self.history: list[TrainingHistory] = []

    # ------------------------------------------------------------------
    # Ozellikler
    # ------------------------------------------------------------------
    @property
    def n_qubits(self) -> int:
        return self.spec.n_qubits

    @property
    def n_parameters(self) -> int:
        return self.spec.n_parameters

    @property
    def n_states(self) -> int:
        """Olasi olcum sonucu sayisi, 2**n_qubits."""
        return 2**self.n_qubits

    # ------------------------------------------------------------------
    # Ileri gecis
    # ------------------------------------------------------------------
    def probabilities(self, params: np.ndarray | None = None) -> np.ndarray:
        """Born kurali dagilimi p_theta(x), uzunluk 2**n_qubits."""
        p = self.params if params is None else params
        return self.backend.probabilities(self.circuit, self.theta, p)

    def sample(self, shots: int | None = None, params: np.ndarray | None = None) -> np.ndarray:
        """Prior'dan ornek ceker.

        Returns
        -------
        np.ndarray[int], uzunluk `shots`
            Her eleman bir bitstring'in tamsayi karsiligi. Sonuclar, olcum
            sayimlari acilarak dondurulur (sirasi karistirilmis halde).
        """
        shots = shots or self.config.shots
        counts = self.backend.sample(self.circuit, self.theta, self.params if params is None else params, shots)

        samples = np.repeat(
            np.fromiter(counts.keys(), dtype=np.int64, count=len(counts)),
            np.fromiter(counts.values(), dtype=np.int64, count=len(counts)),
        )
        self._rng.shuffle(samples)
        return samples

    # ------------------------------------------------------------------
    # Kayip
    # ------------------------------------------------------------------
    def exact_nll(self, target: TargetDistribution, params: np.ndarray | None = None) -> float:
        """Exact negative log-likelihood (Makale'nin birincil kayip fonksiyonu).

            NLL(theta) = - sum_{x in D} P_hedef(x) * log p_theta(x)

        Makale formulu agirliksiz yazar (`-sum log p_theta(x)`), ancak hemen
        ardindan softmax(odul) degerlerini "hedef degerler" olarak tanimlar.
        Odul-agirlikli form, bu iki ifadeyi tutarli kilar: yuksek odullu
        bitstring'ler kayba daha cok katkida bulunur. Agirliksiz varyant icin
        `exact_nll_unweighted` kullanin.
        """
        probs = self.probabilities(params)
        model_p = np.clip(probs[target.indices], _EPS, None)
        return float(-np.sum(target.probabilities * np.log(model_p)))

    def exact_nll_unweighted(
        self, target: TargetDistribution, params: np.ndarray | None = None
    ) -> float:
        """Makale metnindeki birebir form: -sum_{x in D} log p_theta(x)."""
        probs = self.probabilities(params)
        model_p = np.clip(probs[target.indices], _EPS, None)
        return float(-np.sum(np.log(model_p)))

    def kl_divergence(self, target: TargetDistribution, params: np.ndarray | None = None) -> float:
        """KL(hedef || model) - yakinsamayi izlemek icin tani metrigi."""
        probs = self.probabilities(params)
        model_p = np.clip(probs[target.indices], _EPS, None)
        tgt = np.clip(target.probabilities, _EPS, None)
        return float(np.sum(tgt * np.log(tgt / model_p)))

    # ------------------------------------------------------------------
    # Egitim (Makale, Algorithm 2)
    # ------------------------------------------------------------------
    def fit(
        self,
        target: TargetDistribution,
        max_iterations: int | None = None,
        weighted: bool = True,
        callback=None,
    ) -> TrainingHistory:
        """QCBM parametrelerini hedef dagilima gore COBYLA ile optimize eder.

        Makale, Algorithm 2:
            1: Initialize QCBM
            3: while not converged:
            4:   Compute exact NLL loss function
            6:   Adjust parameters using an optimizer
            11:  Train QCBM model with adjusted probabilities

        COBYLA turev gerektirmez; bu, parametre-kaydirma (parameter-shift)
        ile gradyan hesaplamanin donanimda pahali oldugu NISQ ortaminda
        makul bir secimdir.
        """
        max_iterations = max_iterations or self.config.max_iterations

        # COBYLA, ilk simpleksi kurabilmek icin en az (parametre sayisi + 2)
        # degerlendirme ister; daha azi verilirse scipy uyari basip degeri
        # kendisi yukseltir. Devre buyudukce (16 qubit -> 200+ parametre)
        # yapilandirmadaki sabit deger yetersiz kalir, bu yuzden alt siniri
        # burada uyguluyoruz.
        min_iterations = self.n_parameters + 2
        if max_iterations < min_iterations:
            logger.debug(
                "COBYLA iterasyon sayisi %d -> %d yukseltildi (%d parametre)",
                max_iterations,
                min_iterations,
                self.n_parameters,
            )
            max_iterations = min_iterations

        loss_fn = self.exact_nll if weighted else self.exact_nll_unweighted
        history = TrainingHistory()

        def objective(p: np.ndarray) -> float:
            value = loss_fn(target, p)
            history.loss.append(value)
            history.n_evaluations += 1
            if callback is not None:
                callback(history.n_evaluations, value, p)
            return value

        result = minimize(
            objective,
            x0=self.params,
            method=self.config.optimizer,  # Makale: COBYLA
            options={"maxiter": max_iterations, "disp": False},
        )

        self.params = np.asarray(result.x, dtype=float)
        history.converged = bool(result.success)
        history.message = str(result.message)
        self.history.append(history)

        logger.debug(
            "QCBM fit: %d degerlendirme, kayip %.4f -> %.4f (%s)",
            history.n_evaluations,
            history.initial_loss if history.initial_loss is not None else float("nan"),
            history.final_loss if history.final_loss is not None else float("nan"),
            history.message,
        )
        return history

    def reset_parameters(self, seed: int | None = None) -> None:
        """Parametreleri yeniden rastgele baslatir."""
        rng = np.random.default_rng(seed if seed is not None else self.config.seed)
        self.params = rng.uniform(0, 2 * np.pi, size=self.n_parameters)
        self.history.clear()

    def describe(self) -> str:
        """Kosu kaydina yazilacak insan-okur ozet."""
        return (
            f"QCBM(qubits={self.n_qubits}, layers={self.spec.n_layers}, "
            f"ansatz={self.spec.mode!r}, entanglement={self.spec.entanglement!r}, "
            f"parametre={self.n_parameters}, durum_uzayi=2^{self.n_qubits}={self.n_states}, "
            f"backend={self.backend.name!r}, optimize_edici={self.config.optimizer})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.describe()}>"
