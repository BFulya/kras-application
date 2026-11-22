"""
Bilesik secim akisi (Makale, Fig. 1d).

Makale, Fig. 1d | "Identification of inhibitor candidates":

    1M bilesik (her koldan) ->
      [Molecular docking via Chemistry42] ->
      [Filtering based on pharmacophores] ->
      [Sorting based on PLI score] ->
      [Structural novelty] ->
      [Synthetic complexity] ->
    15 bilesik (3 klasik + 8 kuantum-HW + 4 kuantum-SIM)

Makale, "Experimental validation methods" bolumunde son adimi soyle anlatir:

    "After screening, molecules were clustered and ranked within clusters by
     chemical similarity, allowing for expert analysis to further prioritize
     on the basis of novelty and structural intricacy, culminating in a
     selection of 100-150 molecules for potential synthesis."

Kumeleme adimi onemlidir: en yuksek skorlu 15 molekul cogunlukla birbirinin
cok yakin analogu olur. Kumeleme, sentez listesinin kimyasal cesitlilik
tasimasini saglar - tek bir kimyasal seriye bahse girmek yerine birkac
farkli kemotipi denemek.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from kras_quantum.config import CampaignConfig, RewardConfig
from kras_quantum.data.stoned import StonedConfig, canonical_smiles, tanimoto_similarity
from kras_quantum.scoring.pharmacophore import pharmacophore_match_score
from kras_quantum.scoring.reward import RewardBreakdown, RewardFunction
from kras_quantum.scoring.synthetic import synthetic_accessibility

logger = logging.getLogger(__name__)

__all__ = ["Candidate", "SelectionReport", "select_candidates", "cluster_by_similarity"]


@dataclass
class Candidate:
    """Secim surecinden gecen tek bir bilesik."""

    smiles: str
    arm: str                       # hangi kol uretti
    reward: RewardBreakdown
    novelty: float = 1.0           # egitim setine en yuksek benzerligin tersi
    cluster_id: int = -1

    @property
    def pli(self) -> float:
        return self.reward.pli_raw

    @property
    def total_reward(self) -> float:
        return self.reward.total

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"[{self.arm:11s}] odul={self.total_reward:.3f} "
            f"PLI={self.pli:+.2f} yenilik={self.novelty:.2f} "
            f"kume={self.cluster_id:2d} | {self.smiles[:60]}"
        )


@dataclass
class SelectionReport:
    """Secim akisinin her adimindaki hayatta kalan sayilari (Fig. 1d)."""

    n_input: int = 0
    n_after_validity: int = 0
    n_after_filters: int = 0
    n_after_pharmacophore: int = 0
    n_after_pli: int = 0
    n_after_novelty: int = 0
    n_after_synthetic: int = 0
    n_clusters: int = 0
    n_selected: int = 0
    per_arm: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover
        return "\n".join(
            [
                "Secim akisi (Makale, Fig. 1d):",
                f"  girdi bilesik            : {self.n_input:>8,}",
                f"  gecerli molekul          : {self.n_after_validity:>8,}",
                f"  filtreleri gecen         : {self.n_after_filters:>8,}",
                f"  farmakofor esigini gecen : {self.n_after_pharmacophore:>8,}",
                f"  PLI esigini gecen        : {self.n_after_pli:>8,}",
                f"  yapisal olarak yeni      : {self.n_after_novelty:>8,}",
                f"  sentezlenebilir          : {self.n_after_synthetic:>8,}",
                f"  kume sayisi              : {self.n_clusters:>8,}",
                f"  SECILEN                  : {self.n_selected:>8,}",
                f"  kol dagilimi             : {self.per_arm}",
            ]
        )


# ---------------------------------------------------------------------------
# Yapisal yenilik
# ---------------------------------------------------------------------------
def compute_novelty(
    smiles: str,
    reference_smiles: list[str],
    config: StonedConfig | None = None,
) -> float:
    """Egitim setine gore yapisal yenilik, [0, 1].

    Makale, Fig. 1d'de "Structural novelty" adimini icerir ve sonucta
    "potential chemotypes that expand the chemical space explored in KRAS
    inhibitor design" elde edildigini belirtir.

    yenilik = 1 - (referans kumeye en yuksek Tanimoto benzerligi)

    1.0'a yakin  -> tamamen yeni kemotip
    0.0'a yakin  -> egitim setindeki bir molekulun kopyasi
    """
    config = config or StonedConfig()
    if not reference_smiles:
        return 1.0

    max_similarity = max(
        tanimoto_similarity(smiles, reference, config) for reference in reference_smiles
    )
    return float(1.0 - max_similarity)


# ---------------------------------------------------------------------------
# Kumeleme
# ---------------------------------------------------------------------------
def cluster_by_similarity(
    smiles_list: list[str],
    similarity_threshold: float = 0.6,
    config: StonedConfig | None = None,
) -> list[int]:
    """Molekulleri kimyasal benzerlige gore kumeler (Taylor-Butina benzeri).

    Makale: "molecules were clustered and ranked within clusters by chemical
    similarity".

    Basit, belirlenimci bir acgozlu (greedy) kumeleme: liste sirasiyla
    gezilir; molekul mevcut bir kume merkezine `similarity_threshold`
    uzerinde benziyorsa o kumeye katilir, aksi halde yeni kume acar.

    Girdi odule gore sirali verilirse kume merkezleri en iyi molekuller
    olur - bu da sentez listesinin hem kaliteli hem cesitli olmasini saglar.

    Returns
    -------
    Her molekul icin kume kimligi (girdi ile ayni sirada).
    """
    config = config or StonedConfig()
    cluster_ids: list[int] = []
    centroids: list[str] = []

    for smiles in smiles_list:
        assigned = -1
        for cluster_id, centroid in enumerate(centroids):
            if tanimoto_similarity(smiles, centroid, config) >= similarity_threshold:
                assigned = cluster_id
                break

        if assigned < 0:
            assigned = len(centroids)
            centroids.append(smiles)

        cluster_ids.append(assigned)

    return cluster_ids


def _select_diverse(candidates: list[Candidate], n: int) -> list[Candidate]:
    """Kumeler arasinda gezerek cesitli bir alt kume secer.

    Her turda her kumeden en iyi (henuz secilmemis) molekul alinir; boylece
    secim tek bir kimyasal seriye yiglmaz. Yeterli kume yoksa en iyi
    skorlulardan tamamlanir.
    """
    if n <= 0 or not candidates:
        return []

    by_cluster: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        by_cluster.setdefault(candidate.cluster_id, []).append(candidate)

    # Her kume kendi icinde odule gore sirali
    for cluster in by_cluster.values():
        cluster.sort(key=lambda c: c.total_reward, reverse=True)

    # Kumeleri en iyi uyelerinin odulune gore sirala
    ordered_clusters = sorted(
        by_cluster.values(), key=lambda c: c[0].total_reward, reverse=True
    )

    selected: list[Candidate] = []
    round_index = 0
    while len(selected) < n:
        added_this_round = False
        for cluster in ordered_clusters:
            if len(selected) >= n:
                break
            if round_index < len(cluster):
                selected.append(cluster[round_index])
                added_this_round = True
        if not added_this_round:
            break  # tum kumeler tukendi
        round_index += 1

    return selected[:n]


# ---------------------------------------------------------------------------
# Ana secim akisi
# ---------------------------------------------------------------------------
def select_candidates(
    molecules_by_arm: dict[str, list[str]],
    reward_function: RewardFunction,
    training_smiles: list[str] | None = None,
    campaign: CampaignConfig | None = None,
    reward_config: RewardConfig | None = None,
    min_novelty: float = 0.4,
    cluster_threshold: float = 0.6,
    verbose: bool = True,
) -> tuple[list[Candidate], SelectionReport]:
    """Fig. 1d secim akisini uygular.

    Parameters
    ----------
    molecules_by_arm
        {"classical": [...], "quantum_hw": [...], "quantum_sim": [...]}
    reward_function
        Tam asamada (epoch >= 20) calisiyor olmali; aksi halde PLI ve
        farmakofor bilesenleri hesaplanmaz.
    training_smiles
        Yapisal yenilik olcumu icin referans kume (egitim seti).
    min_novelty
        Egitim setine fazla benzeyen molekuller elenir. Makale sayisal bir
        esik vermez; "structural novelty" adimini nicelestirmek icin
        kullaniyoruz.

    Returns
    -------
    (secilen_adaylar, rapor)
        Kol basina secim sayilari makaledeki dagilimi izler:
        klasik 3, kuantum-HW 8, kuantum-SIM 4 (toplam 15).
    """
    campaign = campaign or CampaignConfig()
    reward_config = reward_config or RewardConfig()
    report = SelectionReport()

    if reward_function.uses_local_filter_only:
        logger.warning(
            "Odul fonksiyonu hala 'local_filter' asamasinda; secim icin "
            "tam asama gerekli. reward_function.set_epoch(%d) cagriliyor.",
            reward_config.local_filter_until_epoch,
        )
        reward_function.set_epoch(reward_config.local_filter_until_epoch)

    # Kol basina hedef secim sayisi (Makale, Fig. 1d)
    quota = {
        "classical": campaign.n_selected_classical,
        "quantum_hw": campaign.n_selected_quantum_hw,
        "quantum_sim": campaign.n_selected_quantum_sim,
    }

    survivors: list[Candidate] = []

    for arm, smiles_list in molecules_by_arm.items():
        report.n_input += len(smiles_list)

        # --- gecerlilik ve tekillestirme ---
        seen: set[str] = set()
        unique: list[str] = []
        for smiles in smiles_list:
            if not smiles:
                continue
            canonical = canonical_smiles(smiles)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            unique.append(canonical)
        report.n_after_validity += len(unique)

        for smiles in unique:
            # --- filtreler + odul (Fig. 1d: docking ve filtreleme) ---
            breakdown = reward_function.score(smiles)
            if not breakdown.passed_filters:
                continue
            report.n_after_filters += 1

            # --- farmakofor esigi ---
            if breakdown.pharmacophore_score <= reward_config.min_pharmacophore_match:
                continue
            report.n_after_pharmacophore += 1

            # --- PLI esigi (< -8 kcal/mol) ---
            if breakdown.pli_raw >= reward_config.max_pli_score:
                continue
            report.n_after_pli += 1

            # --- yapisal yenilik ---
            novelty = (
                compute_novelty(smiles, training_smiles) if training_smiles else 1.0
            )
            if novelty < min_novelty:
                continue
            report.n_after_novelty += 1

            # --- sentetik karmasiklik (ReRSA vekili < 5) ---
            if synthetic_accessibility(smiles) >= reward_config.max_rersa_score:
                continue
            report.n_after_synthetic += 1

            survivors.append(
                Candidate(smiles=smiles, arm=arm, reward=breakdown, novelty=novelty)
            )

    if not survivors:
        logger.warning("Hicbir bilesik secim kriterlerini gecemedi.")
        return [], report

    # --- kumeleme: once odule gore sirala ki kume merkezleri en iyiler olsun ---
    survivors.sort(key=lambda c: c.total_reward, reverse=True)
    cluster_ids = cluster_by_similarity(
        [c.smiles for c in survivors], similarity_threshold=cluster_threshold
    )
    for candidate, cluster_id in zip(survivors, cluster_ids):
        candidate.cluster_id = cluster_id
    report.n_clusters = len(set(cluster_ids))

    # --- kol basina kota ile cesitli secim ---
    selected: list[Candidate] = []
    for arm, n_wanted in quota.items():
        arm_candidates = [c for c in survivors if c.arm == arm]
        arm_selected = _select_diverse(arm_candidates, n_wanted)
        selected.extend(arm_selected)
        report.per_arm[arm] = len(arm_selected)

        if len(arm_selected) < n_wanted:
            logger.warning(
                "%s kolundan %d bilesik istendi, %d bulundu",
                arm,
                n_wanted,
                len(arm_selected),
            )

    selected.sort(key=lambda c: c.total_reward, reverse=True)
    report.n_selected = len(selected)

    if verbose:
        logger.info("\n%s", report)

    return selected, report
