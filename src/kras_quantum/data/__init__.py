"""
Egitim verisi uretimi (Makale, Fig. 1a).

Uc kaynak birlestirilir:
  1. Cekirdek inhibitorler  -> seed_library  (makalede ~650 molekul)
  2. STONED-SELFIES analoglari -> stoned     (makalede 850.000 molekul)
  3. VirtualFlow sanal tarama hitleri        (makalede 250.000 molekul)

Uretici modelin girdisi SELFIES token dizileridir -> tokenizer.
"""

from __future__ import annotations

from kras_quantum.data.seed_library import (
    DEFAULT_SEED_PATH,
    KNOWN_KRAS_INHIBITOR_NAMES,
    SeedCompound,
    load_seed_library,
    write_seed_library,
)
from kras_quantum.data.stoned import (
    StonedConfig,
    canonical_smiles,
    draw_mutation_count,
    generate_analogs,
    mutate_selfies,
    randomize_smiles,
    selfies_to_smiles,
    smiles_to_selfies,
    tanimoto_similarity,
)
from kras_quantum.data.tokenizer import SelfiesTokenizer, SpecialTokens

__all__ = [
    # cekirdek kutuphane
    "SeedCompound",
    "load_seed_library",
    "write_seed_library",
    "DEFAULT_SEED_PATH",
    "KNOWN_KRAS_INHIBITOR_NAMES",
    # STONED-SELFIES
    "StonedConfig",
    "canonical_smiles",
    "smiles_to_selfies",
    "selfies_to_smiles",
    "randomize_smiles",
    "mutate_selfies",
    "draw_mutation_count",
    "generate_analogs",
    "tanimoto_similarity",
    # tokenizer
    "SelfiesTokenizer",
    "SpecialTokens",
]
