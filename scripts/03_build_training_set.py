"""
Egitim setini olusturur (Makale, Fig. 1a - "Training data generation").

Makale uc kaynagi birlestirir:

  1. ~650 deneysel KRAS inhibitoru (literaturden)
  2. 850.000 STONED-SELFIES analogu (cekirdeklerden mutasyonla)
  3. 250.000 VirtualFlow 2.0 sanal tarama hiti (Enamine REAL, 100M taranan)
                                                       = ~1.1M molekul

Bu depodaki farklar
-------------------
  (1) Cekirdek set PubChem'den gelen ~19 bilesiktir (makaledeki 650'lik
      Insilico ic derlemesi yayinlanmamistir).
  (2) STONED-SELFIES asamasi aynen uygulanir, ancak uretilen analog sayisi
      --n-analogs ile olceklenir (varsayilan 20.000; makaledeki 850.000
      tek makinede saatler surer).
  (3) VirtualFlow asamasi ATLANIR. 100 milyon molekulluk Enamine REAL
      taramasi bulut olcegi bir istir (makale AWS kullanmistir) ve tek
      makinede yeniden uretilemez. Bu nedenle egitim seti makaleye gore
      yapisal olarak daha dar bir kimyasal alani kapsar.

Kullanim:
    python scripts/03_build_training_set.py
    python scripts/03_build_training_set.py --n-analogs 50000 --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kras_quantum.config import DATA_CACHE_DIR, DataConfig  # noqa: E402
from kras_quantum.data import (  # noqa: E402
    SelfiesTokenizer,
    StonedConfig,
    generate_analogs,
    load_seed_library,
    smiles_to_selfies,
)
from kras_quantum.scoring.filters import passes_local_filter  # noqa: E402


def build_analog_pool(
    seed_smiles: list[str],
    n_total: int,
    rng: np.random.Generator,
    apply_filter: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Cekirdeklerden analog havuzu uretir (Makale, Fig. 1a "Hit expansion").

    Makale, uretilen analoglari sentezlenebilirlik acisindan filtreler:
    "The resulting derivatives were filtered on the basis of synthesizability".
    """
    config = StonedConfig()
    pool: set[str] = set()
    start = time.time()

    # Cekirdekler uzerinde tur tur gez: sentezlenebilirlik filtresi analoglarin
    # bir kismini eledigi icin tek tur cogunlukla hedefe ulasmaz. Her turda
    # kalan acigi cekirdeklere bolusturuyoruz.
    max_rounds = 12
    for round_index in range(max_rounds):
        if len(pool) >= n_total:
            break

        remaining = n_total - len(pool)
        per_seed = max(remaining // len(seed_smiles) + 1, 1)

        pool_before_round = len(pool)

        for index, smiles in enumerate(seed_smiles):
            if len(pool) >= n_total:
                break

            try:
                analogs = generate_analogs(
                    smiles, n_analogs=per_seed, config=config, rng=rng
                )
            except ValueError:
                continue

            if apply_filter:
                analogs = [a for a in analogs if passes_local_filter(a)]

            pool.update(analogs)

            if verbose:
                elapsed = time.time() - start
                print(
                    f"  tur {round_index + 1}, cekirdek {index + 1}/{len(seed_smiles)}: "
                    f"havuz {len(pool):,}/{n_total:,} ({elapsed:.0f}s)",
                    end="\r",
                )

        gained = len(pool) - pool_before_round
        if verbose:
            print(
                f"  tur {round_index + 1} bitti: havuz {len(pool):,}/{n_total:,} "
                f"(+{gained:,}, {time.time() - start:.0f}s)" + " " * 20
            )

        # Yeni molekul gelmiyorsa kimyasal uzay tukenmistir; zorlamayalim
        if gained == 0:
            if verbose:
                print(
                    "  Yeni benzersiz analog uretilemiyor; havuz doygunluga ulasti."
                )
            break

    return sorted(pool)[:n_total]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-analogs",
        type=int,
        default=20_000,
        help="Uretilecek STONED-SELFIES analog sayisi (makale: 850.000)",
    )
    parser.add_argument("--max-length", type=int, default=72, help="SELFIES token ust siniri")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--no-filter", action="store_true", help="Sentezlenebilirlik filtresini atla")
    parser.add_argument("--out-dir", type=Path, default=DATA_CACHE_DIR)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data_config = DataConfig()

    print("=" * 70)
    print("EGITIM SETI OLUSTURMA (Makale, Fig. 1a)")
    print("=" * 70)

    # --- 1) cekirdek inhibitorler ---
    print("\n[1/4] Cekirdek inhibitorler yukleniyor...")
    compounds = load_seed_library()
    seed_smiles = [c.smiles for c in compounds]
    print(f"      {len(seed_smiles)} bilesik (makalede ~{data_config.n_seed_inhibitors})")

    # --- 2) STONED-SELFIES analoglari ---
    print(f"\n[2/4] STONED-SELFIES ile {args.n_analogs:,} analog uretiliyor...")
    print(f"      (makalede {data_config.n_stoned_analogs:,})")
    analogs = build_analog_pool(
        seed_smiles, args.n_analogs, rng, apply_filter=not args.no_filter
    )
    print(f"      {len(analogs):,} benzersiz analog uretildi")

    # --- 3) VirtualFlow asamasi (atlandi) ---
    print("\n[3/4] VirtualFlow sanal tarama: ATLANDI")
    print(f"      (makalede {data_config.n_virtual_screened:,} molekul tarandi,")
    print(f"       en iyi {data_config.n_virtual_hits:,} eklendi - bulut olcegi is)")

    # --- 4) tokenlastirma ---
    print("\n[4/4] SELFIES'e cevriliyor ve tokenlastiriliyor...")
    all_smiles = seed_smiles + analogs

    selfies_list = []
    for smiles in all_smiles:
        selfies_str = smiles_to_selfies(smiles)
        if selfies_str:
            selfies_list.append(selfies_str)

    print(f"      {len(selfies_list):,} molekul SELFIES'e cevrildi")

    tokenizer = SelfiesTokenizer.from_corpus(selfies_list)
    sequences = tokenizer.encode_batch(selfies_list, max_length=args.max_length)
    print(f"      sozluk boyutu: {tokenizer.vocab_size}")
    print(f"      dizi matrisi : {sequences.shape}")

    # --- kaydet ---
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "training_sequences.npy", sequences)
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "training_smiles.txt").write_text(
        "\n".join(all_smiles), encoding="utf-8"
    )

    metadata = {
        "n_seed_compounds": len(seed_smiles),
        "n_analogs": len(analogs),
        "n_total_molecules": len(all_smiles),
        "n_tokenized": len(selfies_list),
        "vocab_size": tokenizer.vocab_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "synthesizability_filter": not args.no_filter,
        "paper_reference": {
            "n_seed_compounds": data_config.n_seed_inhibitors,
            "n_stoned_analogs": data_config.n_stoned_analogs,
            "n_virtual_hits": data_config.n_virtual_hits,
            "n_total": data_config.total_training_molecules,
        },
        "note": (
            "VirtualFlow sanal tarama asamasi atlandi (bulut olcegi). "
            "Cekirdek set PubChem'den; makaledeki 650 molekulluk Insilico "
            "ic derlemesi yayinlanmamistir."
        ),
    }
    (out_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"\nKaydedildi: {out_dir}")
    print(f"  training_sequences.npy  {sequences.shape}")
    print(f"  tokenizer.json          {tokenizer.vocab_size} token")
    print(f"  training_smiles.txt     {len(all_smiles):,} molekul")
    print("\nSonraki adim: python scripts/04_run_campaign.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
