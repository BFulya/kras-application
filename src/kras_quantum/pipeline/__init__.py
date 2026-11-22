"""
Hibrit hat: QCBM + LSTM dongusu ve bilesik secimi.

hybrid    -> Makale, Algorithm 1 (kuantum-klasik geri besleme dongusu)
selection -> Makale, Fig. 1d (1M bilesikten 15 sentez adayina)
"""

from __future__ import annotations

from kras_quantum.pipeline.hybrid import (
    ARMS,
    EpochRecord,
    GenerationResult,
    HybridGenerator,
)
from kras_quantum.pipeline.selection import (
    Candidate,
    SelectionReport,
    cluster_by_similarity,
    compute_novelty,
    select_candidates,
)

__all__ = [
    "HybridGenerator",
    "EpochRecord",
    "GenerationResult",
    "ARMS",
    "Candidate",
    "SelectionReport",
    "select_candidates",
    "cluster_by_similarity",
    "compute_novelty",
]
