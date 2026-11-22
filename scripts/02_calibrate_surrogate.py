"""
Vekil PLI skorlayiciyi gercek Vina docking'ine karsi kalibre eder.

Neden gerekli
-------------
Egitim dongusu, her epoch'ta binlerce molekule odul atar; gercek docking
bu olcekte uygulanamaz (molekul basina ~10 s). Bu yuzden hizli bir vekil
skorlayici kullaniyoruz (kras_quantum.scoring.pli.SurrogatePLI).

Vekil kotu kalibre edilmisse, QCBM yanlis bir odul sinyaliyle egitilir -
yani model "iyi" sandigi ama aslinda kotu baglanan molekullere yonelir.
Bu betik, vekilin gercek docking skorlariyla ne kadar ortustugunu olcer.

Onemli olan MUTLAK deger degil SIRALAMA'dir: odul hesabinda softmax
kullanildigi icin molekuller arasi goreli sira belirleyicidir. Bu yuzden
birincil olcut Spearman sira korelasyonudur.

Kullanim:
    python scripts/02_calibrate_surrogate.py --n-molecules 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kras_quantum.config import PROJECT_ROOT  # noqa: E402
from kras_quantum.data import generate_analogs, load_seed_library  # noqa: E402
from kras_quantum.scoring.filters import passes_local_filter  # noqa: E402
from kras_quantum.scoring.pli import SurrogatePLI, VinaDocking  # noqa: E402


def build_molecule_set(n_molecules: int, seed: int = 0) -> list[tuple[str, str]]:
    """Kalibrasyon kumesi: gercek inhibitorler + onlardan turetilmis analoglar.

    Analoglari da dahil ediyoruz cunku vekil, egitim sirasinda tam olarak
    bu tur molekulleri (uretici modelin ciktilarini) skorlayacak - yalnizca
    optimize edilmis ilaclarda degil, genis kalite yelpazesinde dogru
    siralama yapmasi gerekir.
    """
    rng = np.random.default_rng(seed)
    compounds = load_seed_library()

    molecules: list[tuple[str, str]] = [(c.name, c.smiles) for c in compounds]

    # Cekirdeklerden analoglar uret (kalite yelpazesini genislet)
    n_analogs_needed = max(n_molecules - len(molecules), 0)
    if n_analogs_needed:
        per_seed = max(n_analogs_needed // len(compounds) + 1, 1)
        for compound in compounds:
            if len(molecules) >= n_molecules:
                break
            try:
                analogs = generate_analogs(compound.smiles, n_analogs=per_seed, rng=rng)
            except ValueError:
                continue
            for i, analog in enumerate(analogs):
                if len(molecules) >= n_molecules:
                    break
                # Yalnizca temel filtreyi gecenler - cop molekuller siralamayi
                # yapay olarak kolaylastirir
                if passes_local_filter(analog):
                    molecules.append((f"{compound.name}_analog{i}", analog))

    return molecules[:n_molecules]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-molecules", type=int, default=40)
    parser.add_argument("--exhaustiveness", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "surrogate_calibration.json",
    )
    args = parser.parse_args()

    print(f"Kalibrasyon kumesi hazirlaniyor ({args.n_molecules} molekul)...")
    molecules = build_molecule_set(args.n_molecules, args.seed)
    print(f"  {len(molecules)} molekul hazir\n")

    docking = VinaDocking(exhaustiveness=args.exhaustiveness)
    surrogate = SurrogatePLI()

    names: list[str] = []
    smiles_used: list[str] = []
    vina_scores: list[float] = []
    surrogate_scores: list[float] = []

    print(f"{'molekul':<28} {'Vina':>8} {'vekil':>8}")
    print("-" * 48)

    start = time.time()
    for name, smiles in molecules:
        vina_score = docking.score(smiles)
        if vina_score == 0.0:  # docking basarisiz - kalibrasyona katma
            print(f"{name:<28} {'BASARISIZ':>8}")
            continue

        surrogate_score = surrogate.score(smiles)
        names.append(name)
        smiles_used.append(smiles)
        vina_scores.append(vina_score)
        surrogate_scores.append(surrogate_score)
        print(f"{name:<28} {vina_score:8.2f} {surrogate_score:8.2f}")

    elapsed = time.time() - start

    if len(vina_scores) < 5:
        print("\nYeterli basarili docking yok; kalibrasyon yapilamadi.")
        return 1

    vina = np.array(vina_scores)
    surrogate_array = np.array(surrogate_scores)

    # --- korelasyonlar ---
    from scipy.stats import pearsonr, spearmanr

    spearman, spearman_p = spearmanr(vina, surrogate_array)
    pearson, pearson_p = pearsonr(vina, surrogate_array)

    # --- dogrusal yeniden olcekleme: vekili Vina olcegine tasi ---
    slope, intercept = np.polyfit(surrogate_array, vina, 1)
    rescaled = slope * surrogate_array + intercept
    mae = float(np.mean(np.abs(rescaled - vina)))

    print("\n" + "=" * 48)
    print(f"Molekul sayisi      : {len(vina)}")
    print(f"Toplam sure         : {elapsed:.0f} s ({elapsed/len(vina):.1f} s/molekul)")
    print(f"Spearman (sira)     : {spearman:+.3f}  (p={spearman_p:.2g})   <- birincil olcut")
    print(f"Pearson  (dogrusal) : {pearson:+.3f}  (p={pearson_p:.2g})")
    print(f"Yeniden olcekleme   : vina ~ {slope:.3f} * vekil + {intercept:.3f}")
    print(f"Olceklenmis MAE     : {mae:.2f} kcal/mol")
    print(f"Vina araligi        : [{vina.min():.2f}, {vina.max():.2f}]")
    print(f"Vekil araligi       : [{surrogate_array.min():.2f}, {surrogate_array.max():.2f}]")

    if spearman > 0.6:
        verdict = "IYI - vekil siralamasi gercek docking ile buyuk olcude ortusuyor"
    elif spearman > 0.35:
        verdict = "ORTA - vekil kaba siralama icin kullanilabilir, nihai secim Vina ile yapilmali"
    else:
        verdict = "ZAYIF - vekil odul sinyali olarak guvenilir degil"
    print(f"\nDegerlendirme: {verdict}")

    payload = {
        "n_molecules": len(vina),
        "exhaustiveness": args.exhaustiveness,
        "spearman": float(spearman),
        "spearman_p": float(spearman_p),
        "pearson": float(pearson),
        "rescale_slope": float(slope),
        "rescale_intercept": float(intercept),
        "rescaled_mae": mae,
        "verdict": verdict,
        "molecules": [
            {"name": n, "smiles": sm, "vina": float(v), "surrogate": float(s)}
            for n, sm, v, s in zip(names, smiles_used, vina_scores, surrogate_scores)
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Sonuclar yazildi: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
