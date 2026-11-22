"""
Skorlama ve odul katmani - Chemistry42'nin acik kaynak vekili.

Makale, uretilen molekulleri Chemistry42 ile degerlendirir (Fig. 1b sag kol,
Fig. 1d). Chemistry42 tescillidir; bu paket ayni islevleri acik araclarla
yeniden kurar:

  filters       -> yapisal, fizikokimyasal ve med-chem filtreleri
  pli           -> protein-ligand etkilesim skoru (gercek AutoDock Vina
                   docking veya hizli vekil)
  pharmacophore -> Extended Data Fig. 4 farmakofor modeli (TH-Z816 / PDB 7EW9)
  synthetic     -> sentetik erisilebilirlik (ReRSA vekili: SA Score)
  reward        -> bunlari birlestiren R(x); QCBM'e P(x)=softmax(R(x)) olarak girer
"""

from __future__ import annotations

from kras_quantum.scoring.filters import (
    FilterResult,
    FilterThresholds,
    compute_properties,
    evaluate_filters,
    find_unwanted_motifs,
    passes_chemistry42_filter,
    passes_local_filter,
)
from kras_quantum.scoring.pharmacophore import (
    PharmacophoreFeatures,
    extract_features,
    pharmacophore_match_score,
    pharmacophore_similarity,
)
from kras_quantum.scoring.pli import (
    DockingBox,
    PLIScorer,
    SurrogatePLI,
    VinaDocking,
    get_pli_scorer,
    load_receptor_config,
)
from kras_quantum.scoring.reward import (
    CandidateVerdict,
    RewardBreakdown,
    RewardFunction,
    is_synthesis_candidate,
)
from kras_quantum.scoring.synthetic import (
    sa_to_reward,
    synthetic_accessibility,
    synthetic_accessibility_batch,
)

__all__ = [
    # filtreler
    "FilterResult",
    "FilterThresholds",
    "evaluate_filters",
    "compute_properties",
    "find_unwanted_motifs",
    "passes_local_filter",
    "passes_chemistry42_filter",
    # PLI
    "PLIScorer",
    "SurrogatePLI",
    "VinaDocking",
    "DockingBox",
    "get_pli_scorer",
    "load_receptor_config",
    # farmakofor
    "PharmacophoreFeatures",
    "extract_features",
    "pharmacophore_match_score",
    "pharmacophore_similarity",
    # sentetik erisilebilirlik
    "synthetic_accessibility",
    "synthetic_accessibility_batch",
    "sa_to_reward",
    # odul
    "RewardFunction",
    "RewardBreakdown",
    "CandidateVerdict",
    "is_synthesis_candidate",
]
