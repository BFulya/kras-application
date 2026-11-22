"""
Hibrit kuantum-klasik uretici hat (Makale, Algorithm 1).

Makale, Algorithm 1 - "quantum-assisted drug discovery using an LSTM framework":

     1: Initialize: LSTM, QCBM, filter (Chemistry42)
     2: Generate initial samples X_i from QCBM
     3: while not converged do
     4:     Train LSTM with X_i
     5:     LSTM generates a new compound from the current samples X_i
     6:     Validate the new compound with the filter
     7:     if new compound is valid then
     8:         Compute rewards for the new compound
     9:         Compute probabilities P(X_i) for each new compound
    10:         Train QCBM with X_i and P(X_i)
    11:         Generate new X_i from QCBM
    12:     endif
    13: endwhile

Dongunun fikri
--------------
QCBM, molekulleri DOGRUDAN uretmez; LSTM'i kosullandiran prior ornekleri
uretir. Geri besleme su zincirle kapanir:

    prior bitstring  ->  LSTM  ->  molekul  ->  odul R(x)
                                                   |
    QCBM <- P(x)=softmax(R(x)) hedef dagilimi <-----+

Yani QCBM, "iyi molekullere yol acan prior'lari" ogrenir. Bu, 16 qubit'lik
bir devrenin cok daha buyuk bir kimyasal uzayi yonlendirmesini saglar -
makalenin "qubit limitations" sorununa getirdigi cozum budur:

    "we propose a hybrid quantum-classical model that addresses qubit
     limitations and combines quantum and classical approaches"

Uc kol (Makale, Fig. 1d)
------------------------
    classical    -> prior yok (vanilla LSTM)       -> 3 bilesik secildi
    quantum_sim  -> QCBM, kuantum simulatorunde    -> 4 bilesik secildi
    quantum_hw   -> QCBM, gercek kuantum donaniminda -> 8 bilesik secildi
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from kras_quantum.classical.lstm import PriorConditionedLSTM, bitstrings_to_prior
from kras_quantum.classical.trainer import train_lstm
from kras_quantum.config import Config
from kras_quantum.data.stoned import selfies_to_smiles
from kras_quantum.data.tokenizer import SelfiesTokenizer
from kras_quantum.quantum.backend import QuantumBackend, get_backend
from kras_quantum.quantum.qcbm import QCBM, TargetDistribution
from kras_quantum.scoring.reward import RewardFunction

logger = logging.getLogger(__name__)

__all__ = ["HybridGenerator", "EpochRecord", "GenerationResult", "ARMS"]

# Fig. 1d'deki uc uretim kolu
ARMS = ("classical", "quantum_sim", "quantum_hw")


@dataclass
class EpochRecord:
    """Tek bir egitim epoch'unun kaydi."""

    epoch: int
    reward_stage: str              # "local_filter" | "chemistry42_proxy"
    n_generated: int               # LSTM'in urettigi ham molekul sayisi
    n_valid: int                   # filtreyi gecen molekul sayisi
    mean_reward: float
    max_reward: float
    lstm_loss: float
    qcbm_loss_before: float | None = None
    qcbm_loss_after: float | None = None
    duration_s: float = 0.0

    @property
    def validity_rate(self) -> float:
        """Makalenin "success rate" metrigi (Extended Data Fig. 1)."""
        return self.n_valid / self.n_generated if self.n_generated else 0.0

    def __str__(self) -> str:  # pragma: no cover
        qcbm = ""
        if self.qcbm_loss_before is not None and self.qcbm_loss_after is not None:
            qcbm = f" | QCBM NLL {self.qcbm_loss_before:.3f}->{self.qcbm_loss_after:.3f}"
        return (
            f"epoch {self.epoch:3d} [{self.reward_stage:17s}] "
            f"basari %{self.validity_rate*100:5.1f} "
            f"({self.n_valid}/{self.n_generated}) | "
            f"odul ort={self.mean_reward:.3f} maks={self.max_reward:.3f} | "
            f"LSTM {self.lstm_loss:.3f}{qcbm} | {self.duration_s:.1f}s"
        )


@dataclass
class GenerationResult:
    """Uretim sonucu: molekuller ve onlari ureten prior'lar."""

    smiles: list[str] = field(default_factory=list)
    selfies: list[str] = field(default_factory=list)
    bitstrings: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    rewards: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def __len__(self) -> int:
        return len(self.smiles)


class HybridGenerator:
    """QCBM + LSTM hibrit uretici (Makale, Algorithm 1).

    Parameters
    ----------
    arm
        "classical"   -> prior yok (makalenin vanilla LSTM kontrolu)
        "quantum_sim" -> QCBM, Aer simulatorunde
        "quantum_hw"  -> QCBM, gercek IBM donaniminda
    tokenizer
        SELFIES sozlugu (egitim setinden kurulmus olmali).
    reward_function
        R(x). Verilmezse varsayilan (vekil PLI) kullanilir.

    Example
    -------
    >>> gen = HybridGenerator(arm="quantum_sim", tokenizer=tok)
    >>> gen.fit(training_sequences, n_epochs=30)
    >>> sonuc = gen.generate(1000)
    """

    def __init__(
        self,
        arm: str,
        tokenizer: SelfiesTokenizer,
        config: Config | None = None,
        reward_function: RewardFunction | None = None,
        backend: QuantumBackend | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"arm {ARMS} icinden biri olmali, verilen: {arm!r}")

        self.arm = arm
        self.tokenizer = tokenizer
        self.config = config or Config()
        self.reward = reward_function or RewardFunction()
        self.device = torch.device(device)

        self.is_quantum = arm != "classical"
        prior_dim = self.config.qcbm.n_qubits if self.is_quantum else 0

        # --- kuantum bilesen (Algorithm 1, satir 1) ---
        self.qcbm: QCBM | None = None
        if self.is_quantum:
            if backend is None:
                kind = "hw" if arm == "quantum_hw" else "sim"
                backend = get_backend(
                    kind, n_qubits=self.config.qcbm.n_qubits, seed=self.config.qcbm.seed
                )
            self.qcbm = QCBM(self.config.qcbm, backend=backend)

        # --- klasik bilesen (Algorithm 1, satir 1) ---
        self.lstm = PriorConditionedLSTM(
            vocab_size=tokenizer.vocab_size,
            prior_dim=prior_dim,
            config=self.config.lstm,
            pad_id=tokenizer.pad_id,
        ).to(self.device)

        self.history: list[EpochRecord] = []
        self._rng = np.random.default_rng(self.config.lstm.seed)

    # ------------------------------------------------------------------
    # Prior uretimi
    # ------------------------------------------------------------------
    def _sample_priors(self, n: int) -> np.ndarray:
        """QCBM'den n prior bitstring ceker (Algorithm 1, satir 2 ve 11).

        Klasik kolda prior yoktur; bos dizi doner.
        """
        if not self.is_quantum or self.qcbm is None:
            return np.array([], dtype=np.int64)

        samples = self.qcbm.sample(shots=max(n, self.config.qcbm.shots))

        # Istenenden az ornek geldiyse tekrarla, fazlaysa kirp
        if len(samples) < n:
            samples = np.resize(samples, n)
        return samples[:n]

    # ------------------------------------------------------------------
    # Egitim dongusu (Algorithm 1)
    # ------------------------------------------------------------------
    def fit(
        self,
        training_sequences: np.ndarray,
        n_epochs: int | None = None,
        n_generate_per_epoch: int = 256,
        lstm_epochs_per_cycle: int = 2,
        verbose: bool = True,
    ) -> list[EpochRecord]:
        """Hibrit dongunun tamamini calistirir.

        Parameters
        ----------
        training_sequences : (n_molekul, seq_len)
            Tokenlastirilmis egitim seti (STONED-SELFIES + sanal tarama).
        n_epochs
            Dis dongu adimi. Makale QCBM'i 30 epoch egitir.
        n_generate_per_epoch
            Her epoch'ta LSTM'den cekilip odullendirilecek molekul sayisi.
            Bu, QCBM'in hedef dagiliminin cozunurlugunu belirler.
        lstm_epochs_per_cycle
            Her dis dongu adiminda LSTM'in kac epoch egitilecegi.
            Makale LSTM'i toplam 40, QCBM'i 30 epoch egitir; oran ~1.33.
        """
        n_epochs = n_epochs if n_epochs is not None else self.config.qcbm.n_epochs
        n_training = len(training_sequences)

        if verbose:
            logger.info("Hibrit egitim basliyor - kol: %s", self.arm)
            logger.info("  %s", self.lstm.describe())
            if self.qcbm is not None:
                logger.info("  %s", self.qcbm.describe())

        # Algorithm 1, satir 2: QCBM'den ilk ornekler
        prior_bitstrings = self._sample_priors(n_training)

        for epoch in range(n_epochs):
            start = time.time()
            self.reward.set_epoch(epoch)

            # --- satir 4: LSTM'i mevcut prior'larla egit ---
            priors = (
                bitstrings_to_prior(prior_bitstrings, self.config.qcbm.n_qubits)
                if self.is_quantum
                else None
            )
            lstm_history = train_lstm(
                self.lstm,
                training_sequences,
                priors,
                n_epochs=lstm_epochs_per_cycle,
                device=self.device,
                verbose=False,
            )

            # --- satir 5: LSTM yeni bilesikler uretsin ---
            generated = self._generate_with_rewards(n_generate_per_epoch)

            # --- satir 6-9: dogrula, odullendir, P(X_i) hesapla ---
            n_valid = int(np.sum(generated.rewards > 0))
            mean_reward = float(generated.rewards.mean()) if len(generated) else 0.0
            max_reward = float(generated.rewards.max()) if len(generated) else 0.0

            record = EpochRecord(
                epoch=epoch,
                reward_stage=self.reward.stage_name,
                n_generated=len(generated),
                n_valid=n_valid,
                mean_reward=mean_reward,
                max_reward=max_reward,
                lstm_loss=lstm_history.final_loss or float("nan"),
            )

            # --- satir 10-11: QCBM'i egit ve yeni prior'lar uret ---
            if self.is_quantum and self.qcbm is not None and n_valid > 0:
                target = TargetDistribution.from_rewards(
                    generated.bitstrings, generated.rewards
                )
                record.qcbm_loss_before = self.qcbm.exact_nll(target)
                self.qcbm.fit(target)
                record.qcbm_loss_after = self.qcbm.exact_nll(target)

                prior_bitstrings = self._sample_priors(n_training)
            elif self.is_quantum and n_valid == 0:
                # Hicbir gecerli molekul yoksa QCBM'i egitmek anlamsiz
                # (Algorithm 1, satir 7: "if new compound is valid then").
                # Prior'lari yeniliyoruz ki model takilip kalmasin.
                logger.debug("Epoch %d: gecerli molekul yok, QCBM egitimi atlandi", epoch)
                prior_bitstrings = self._sample_priors(n_training)

            record.duration_s = time.time() - start
            self.history.append(record)

            if verbose:
                logger.info("%s", record)

        return self.history

    # ------------------------------------------------------------------
    # Uretim
    # ------------------------------------------------------------------
    def _generate_with_rewards(self, n: int) -> GenerationResult:
        """n molekul uretir ve odullendirir; prior izini korur.

        Prior izi kritiktir: QCBM'in hedef dagilimini kurabilmek icin her
        molekulun HANGI bitstring'den dogdugu bilinmelidir.
        """
        bitstrings = self._sample_priors(n)

        prior_tensor = (
            bitstrings_to_prior(bitstrings, self.config.qcbm.n_qubits)
            if self.is_quantum
            else None
        )

        selfies_list = self.lstm.sample(
            n_samples=n,
            tokenizer=self.tokenizer,
            prior=prior_tensor,
            max_length=self.config.lstm.max_sequence_length,
            device=self.device,
        )

        smiles_list = [selfies_to_smiles(s) if s else None for s in selfies_list]
        # Cozulemeyen molekuller bos dize olur; odul fonksiyonu bunlara 0 verir
        clean_smiles = [s if s is not None else "" for s in smiles_list]

        rewards = self.reward(clean_smiles)

        return GenerationResult(
            smiles=clean_smiles,
            selfies=selfies_list,
            bitstrings=bitstrings if self.is_quantum else np.zeros(n, dtype=np.int64),
            rewards=rewards,
        )

    def generate(self, n: int, batch_size: int = 512) -> GenerationResult:
        """Egitilmis modelden n molekul uretir.

        Makale, nihai secimde her koldan 1M bilesik ornekler (Fig. 1d).
        Buyuk n degerleri icin uretim parcalara bolunur.
        """
        all_smiles: list[str] = []
        all_selfies: list[str] = []
        all_bitstrings: list[np.ndarray] = []
        all_rewards: list[np.ndarray] = []

        remaining = n
        while remaining > 0:
            size = min(batch_size, remaining)
            batch = self._generate_with_rewards(size)

            all_smiles.extend(batch.smiles)
            all_selfies.extend(batch.selfies)
            all_bitstrings.append(batch.bitstrings)
            all_rewards.append(batch.rewards)
            remaining -= size

        return GenerationResult(
            smiles=all_smiles,
            selfies=all_selfies,
            bitstrings=np.concatenate(all_bitstrings) if all_bitstrings else np.array([]),
            rewards=np.concatenate(all_rewards) if all_rewards else np.array([]),
        )

    # ------------------------------------------------------------------
    def describe(self) -> str:
        parcalar = [f"HybridGenerator[kol={self.arm}]", f"  {self.lstm.describe()}"]
        if self.qcbm is not None:
            parcalar.append(f"  {self.qcbm.describe()}")
        else:
            parcalar.append("  kuantum prior yok (klasik kontrol kolu)")
        parcalar.append(f"  odul: {self.reward!r}")
        return "\n".join(parcalar)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<HybridGenerator kol={self.arm!r} kuantum={self.is_quantum}>"
