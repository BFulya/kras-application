"""
Birlesik odul fonksiyonu - Chemistry42'nin acik kaynak vekili.

Makale referansi: "Experimental validation methods"

    "Our selection process relied on a tiered system of the following criteria:
       - Structural and compositional parameter filters...
       - Property evaluation for molecular weight, lipophilicity...
       - Medicinal chemistry filters to exclude problematic structural motifs.
       - A synthetic accessibility assessment based on the ReRSA model.
       - Three-dimensional pharmacophoric analysis in reference to an X-ray
         cocrystal structure (PDB 7EW9).
       - PLI scoring to estimate binding efficiency.
       - An overall reward calculation integrating the above scores"

    Sentez adayligi ("For synthesis candidacy"):
       - Clearance of all Chemistry42 filters.
       - An aggregate reward value exceeding 0.7.
       - A PLI score indicative of strong KRAS binding (< -8 kcal/mol).
       - A pharmacophore match score above 0.7.
       - Favorable synthetic accessibility with a ReRSA score under 5.

Odulun QCBM egitimindeki rolu (Makale, Fig. 1b ve Algorithm 1-2):

    P(x) = softmax(R(x))

Yani bu modulun urettigi R(x), dogrudan QCBM'in hedef dagilimini belirler.
Odul yanlissa kuantum model yanlis bolgeye yonelir - bu yuzden bilesenler
ayri ayri test edilmis ve PLI vekili gercek docking'e karsi kalibre
edilmistir (scripts/02_calibrate_surrogate.py).

Makale ayrica odul fonksiyonunun uyarlanabilir oldugunu belirtir:
    "the reward function can be tailored to a user's specific criteria,
     including accounting for off-target effects by evaluating multiple
     docking scenarios."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from kras_quantum.config import RewardConfig
from kras_quantum.scoring.filters import FilterResult, evaluate_filters
from kras_quantum.scoring.pharmacophore import pharmacophore_match_score
from kras_quantum.scoring.pli import PLIScorer, SurrogatePLI
from kras_quantum.scoring.synthetic import sa_to_reward, synthetic_accessibility

logger = logging.getLogger(__name__)

__all__ = ["RewardBreakdown", "RewardFunction", "is_synthesis_candidate"]

# PLI skorunun [0,1] odul bilesenine dogrusal haritalanmasinda kullanilan
# referans: bu degerde (kcal/mol) bilesen 1.0 olur.
PLI_REWARD_FLOOR = -12.0


@dataclass
class RewardBreakdown:
    """Tek bir molekulun odul dokumu.

    Bilesenleri ayri ayri saklamak, modelin NEDEN belirli molekullere
    yoneldigini incelemeyi mumkun kilar.
    """

    smiles: str
    total: float

    # Bilesenler, hepsi [0, 1]
    filter_score: float = 0.0
    pli_score_normalized: float = 0.0
    pharmacophore_score: float = 0.0
    synthetic_score: float = 0.0

    # Ham degerler (makale esikleriyle dogrudan kiyaslanabilir)
    pli_raw: float = 0.0           # kcal/mol
    sa_raw: float = 10.0           # 1-10
    filter_result: FilterResult | None = None

    @property
    def passed_filters(self) -> bool:
        return self.filter_result is not None and self.filter_result.passed

    def as_dict(self) -> dict[str, float | str | bool]:
        return {
            "smiles": self.smiles,
            "total_reward": self.total,
            "pli_kcal_mol": self.pli_raw,
            "pharmacophore": self.pharmacophore_score,
            "sa_score": self.sa_raw,
            "passed_filters": self.passed_filters,
        }

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"odul={self.total:.3f} | PLI={self.pli_raw:+.2f} kcal/mol | "
            f"farmakofor={self.pharmacophore_score:.2f} | SA={self.sa_raw:.2f} | "
            f"filtre={'gecti' if self.passed_filters else 'kaldi'}"
        )


def _normalize_pli(pli_raw: float) -> float:
    """PLI skorunu (kcal/mol) [0, 1] odul bilesenine cevirir.

    Dogrusal harita: 0 kcal/mol -> 0.0,  PLI_REWARD_FLOOR (-12) -> 1.0.
    Makalenin sentez esigi olan -8 kcal/mol bu haritada 0.67'ye karsilik
    gelir; toplam odul esigi 0.7 ile tutarli bir buyukluk.
    """
    if pli_raw >= 0:
        return 0.0
    return float(min(pli_raw / PLI_REWARD_FLOOR, 1.0))


class RewardFunction:
    """Makalenin R(x) odul fonksiyonu.

    QCBM egitiminde P(x) = softmax(R(x)) olarak kullanilir.

    Makale, egitim sirasinda iki asamali bir odul kullanir:

        "The reward value for each sample was computed at every step using
         the local filter until epoch 20, after which we selected Chemistry42."

    Bu davranis `epoch` parametresiyle saglanir: erken epoch'larda ucuz ve
    hosgorulu "local filter" odulu, sonra tam (ve daha kati) bilesik odul.
    Gerekcesi acik - egitimin basinda model neredeyse hic gecerli molekul
    uretmez; hemen kati kriter uygulamak odul sinyalini duzlestirir
    (neredeyse tum ornekler 0 alir) ve ogrenme durur.

    Parameters
    ----------
    pli_scorer
        PLI skorlayici. Varsayilan hizli vekildir; nihai siralamada
        VinaDocking verilmelidir.
    config
        Agirliklar ve esikler.

    Example
    -------
    >>> reward = RewardFunction()
    >>> reward.set_epoch(0)              # local filter asamasi
    >>> values = reward(["CCO", "c1ccccc1"])
    >>> reward.set_epoch(25)             # tam odul asamasi
    """

    def __init__(
        self,
        pli_scorer: PLIScorer | None = None,
        config: RewardConfig | None = None,
    ) -> None:
        self.config = config or RewardConfig()
        self.pli_scorer = pli_scorer or SurrogatePLI()
        self.epoch = 0
        self._cache: dict[str, RewardBreakdown] = {}

    # ------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Egitim epoch'unu bildirir (odul asamasini belirler)."""
        self.epoch = int(epoch)

    @property
    def uses_local_filter_only(self) -> bool:
        """Makale: epoch 20'ye kadar yalnizca local filter."""
        return self.epoch < self.config.local_filter_until_epoch

    @property
    def stage_name(self) -> str:
        return "local_filter" if self.uses_local_filter_only else "chemistry42_proxy"

    # ------------------------------------------------------------------
    def score(self, smiles: str, use_cache: bool = True) -> RewardBreakdown:
        """Tek molekulun odul dokumunu hesaplar."""
        cache_key = f"{self.stage_name}:{smiles}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        breakdown = self._compute(smiles)

        if use_cache:
            self._cache[cache_key] = breakdown
        return breakdown

    def _compute(self, smiles: str) -> RewardBreakdown:
        if not smiles:
            return RewardBreakdown(smiles=smiles, total=0.0)

        stringent = not self.uses_local_filter_only
        filter_result = evaluate_filters(smiles, stringent=stringent)

        # Filtreyi gecemeyen molekul sifir odul alir: makale, filtre
        # temizligini sentez adayligi icin ON KOSUL sayar.
        if not filter_result.passed:
            return RewardBreakdown(
                smiles=smiles,
                total=0.0,
                filter_score=0.0,
                filter_result=filter_result,
            )

        # --- erken asama: yalnizca local filter ---
        # Makale, epoch 20'ye kadar odulu local filtreden alir. Bu asamada
        # pahali bilesenleri (docking, farmakofor) hesaplamiyoruz; hem
        # makaleye sadik hem de cok daha hizli.
        if self.uses_local_filter_only:
            return RewardBreakdown(
                smiles=smiles,
                total=1.0,
                filter_score=1.0,
                filter_result=filter_result,
            )

        # --- tam asama: Chemistry42 vekili ---
        pli_raw = self.pli_scorer.score(smiles)
        pli_normalized = _normalize_pli(pli_raw)

        pharmacophore = pharmacophore_match_score(smiles)

        sa_raw = synthetic_accessibility(smiles)
        synthetic = sa_to_reward(sa_raw, threshold=self.config.max_rersa_score)

        weights = self.config.weights
        total = (
            weights["filters"] * 1.0  # filtre gecildi
            + weights["pli"] * pli_normalized
            + weights["pharmacophore"] * pharmacophore
            + weights["synthetic_access"] * synthetic
        )

        return RewardBreakdown(
            smiles=smiles,
            total=float(total),
            filter_score=1.0,
            pli_score_normalized=pli_normalized,
            pharmacophore_score=pharmacophore,
            synthetic_score=synthetic,
            pli_raw=pli_raw,
            sa_raw=sa_raw,
            filter_result=filter_result,
        )

    # ------------------------------------------------------------------
    def __call__(self, smiles_list: list[str]) -> np.ndarray:
        """Toplu odul: QCBM egitimine beslenecek R(x) vektoru."""
        return np.array([self.score(s).total for s in smiles_list], dtype=float)

    def score_batch(self, smiles_list: list[str]) -> list[RewardBreakdown]:
        return [self.score(s) for s in smiles_list]

    def clear_cache(self) -> None:
        self._cache.clear()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<RewardFunction asama={self.stage_name!r} epoch={self.epoch} "
            f"pli={self.pli_scorer.name!r}>"
        )


# ---------------------------------------------------------------------------
# Sentez adayligi
# ---------------------------------------------------------------------------
@dataclass
class CandidateVerdict:
    """Sentez adayligi karari ve gerekcesi."""

    is_candidate: bool
    reasons: list[str] = field(default_factory=list)
    breakdown: RewardBreakdown | None = None

    def __bool__(self) -> bool:
        return self.is_candidate

    def __str__(self) -> str:  # pragma: no cover
        if self.is_candidate:
            return "SENTEZ ADAYI"
        return "ADAY DEGIL: " + "; ".join(self.reasons)


def is_synthesis_candidate(
    breakdown: RewardBreakdown, config: RewardConfig | None = None
) -> CandidateVerdict:
    """Makalenin sentez adayligi kriterlerini uygular.

    Makale, "For synthesis candidacy, we applied more demanding conditions":
      - Tum filtreler gecilmeli
      - Toplam odul > 0.7
      - PLI < -8 kcal/mol
      - Farmakofor eslesmesi > 0.7
      - ReRSA (vekil: SA skoru) < 5
    """
    config = config or RewardConfig()
    reasons: list[str] = []

    if not breakdown.passed_filters:
        failures = (
            ", ".join(breakdown.filter_result.failures)
            if breakdown.filter_result
            else "bilinmiyor"
        )
        reasons.append(f"filtreler gecilemedi ({failures})")

    if breakdown.total <= config.min_total_reward:
        reasons.append(
            f"toplam odul {breakdown.total:.3f} <= {config.min_total_reward}"
        )

    if breakdown.pli_raw >= config.max_pli_score:
        reasons.append(
            f"PLI {breakdown.pli_raw:.2f} >= {config.max_pli_score} kcal/mol"
        )

    if breakdown.pharmacophore_score <= config.min_pharmacophore_match:
        reasons.append(
            f"farmakofor {breakdown.pharmacophore_score:.3f} <= "
            f"{config.min_pharmacophore_match}"
        )

    if breakdown.sa_raw >= config.max_rersa_score:
        reasons.append(f"SA skoru {breakdown.sa_raw:.2f} >= {config.max_rersa_score}")

    return CandidateVerdict(
        is_candidate=not reasons, reasons=reasons, breakdown=breakdown
    )
