"""
kras_quantum
============

Nature Biotechnology 43, 1954-1959 (2025) makalesindeki hibrit
kuantum-klasik uretici molekul tasarim hattinin acik kaynak uygulamasi.

  Vakili M.G., Gorgulla C., Snider J., Nigam A. et al.
  "Quantum-computing-enhanced algorithm unveils potential KRAS inhibitors"
  doi:10.1038/s41587-024-02526-3

Hattin uc asamasi (Makale, Fig. 1):
  1. Egitim verisi uretimi   -> kras_quantum.data
  2. Yeni molekul uretimi    -> kras_quantum.quantum (QCBM) + kras_quantum.classical (LSTM)
  3. Secim ve dogrulama      -> kras_quantum.scoring + kras_quantum.pipeline
"""

from __future__ import annotations

from kras_quantum.config import DEFAULT_CONFIG, Config

__version__ = "0.1.0"
__paper_doi__ = "10.1038/s41587-024-02526-3"

__all__ = ["Config", "DEFAULT_CONFIG", "__version__", "__paper_doi__"]
