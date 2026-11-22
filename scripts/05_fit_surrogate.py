"""
Vekil PLI modelini gercek docking verisinden ogrenir.

Neden
-----
Egitim dongusu her epoch'ta yuzlerce molekule odul atar; gercek docking
molekul basina ~10-18 saniye surer ve bu olcekte uygulanamaz. Bu yuzden
hizli bir vekil skorlayici gerekir.

Ilk uygulamada vekil ELLE tasarlanmisti (ligand verimliligi + fizikokimyasal
terimler). Gercek docking'e karsi olculdugunde ise yaramadigi gorulde:
ilk kalibrasyon setinde Spearman -0.396 (TERS korele), daha buyuk ikinci
sette +0.032 (korelasyon yok). Elle yazilmis formul, dar Switch-II cebinin
davranisini yakalayamiyor.

Bu betik bunun yerine, gercek Vina skorlarindan basit ve seffaf bir model
OGRENIR: standartlastirilmis moleküler tanimlayicilar uzerinde ridge
regresyonu (kapali formda, ek bagimlilik olmadan). Performans k-kat capraz
dogrulama ile DURUSTCE raporlanir - egitim verisindeki uyum degil, gorulmemis
molekullerdeki siralama basarisi.

Onemli olan mutlak deger degil SIRALAMA'dir: odulde softmax kullanildigi
icin molekuller arasi goreli sira belirleyicidir. Birincil olcut bu yuzden
capraz dogrulanmis Spearman korelasyonudur.

Kullanim:
    python scripts/02_calibrate_surrogate.py --n-molecules 150 --out results/calibration_set.json
    python scripts/05_fit_surrogate.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kras_quantum.config import PROJECT_ROOT  # noqa: E402
from kras_quantum.scoring.filters import compute_properties  # noqa: E402
from kras_quantum.scoring.pli import SURROGATE_MODEL_PATH, SURROGATE_FEATURES  # noqa: E402


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standartlastirilmis ridge regresyonu (kapali form).

        w = (Z'Z + lam*I)^-1 Z'y      (kesisim cezalandirilmaz)

    Returns
    -------
    (agirliklar, ortalama, standart_sapma)
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0

    Z = (X - mean) / std
    Z1 = np.hstack([np.ones((len(Z), 1)), Z])

    penalty = np.eye(Z1.shape[1])
    penalty[0, 0] = 0.0  # kesisim terimi duzenlilestirilmez

    weights = np.linalg.solve(Z1.T @ Z1 + lam * penalty, Z1.T @ y)
    return weights, mean, std


def ridge_predict(X: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    Z = (X - mean) / std
    return np.hstack([np.ones((len(Z), 1)), Z]) @ weights


def cross_validate(
    X: np.ndarray, y: np.ndarray, lam: float, n_folds: int = 5, seed: int = 0
) -> tuple[float, float]:
    """k-kat capraz dogrulama -> (Spearman, MAE)."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    folds = np.array_split(indices, n_folds)

    predictions = np.zeros_like(y)
    for fold in folds:
        train = np.setdiff1d(indices, fold)
        weights, mean, std = ridge_fit(X[train], y[train], lam)
        predictions[fold] = ridge_predict(X[fold], weights, mean, std)

    rho, _ = spearmanr(predictions, y)
    mae = float(np.mean(np.abs(predictions - y)))
    return float(rho), mae


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=PROJECT_ROOT / "results" / "calibration_set.json",
        help="02_calibrate_surrogate.py ciktisi",
    )
    parser.add_argument("--out", type=Path, default=SURROGATE_MODEL_PATH)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.calibration.exists():
        print(f"Kalibrasyon verisi bulunamadi: {args.calibration}")
        print("Once toplayin:")
        print("    python scripts/02_calibrate_surrogate.py --n-molecules 150 \\")
        print("           --out results/calibration_set.json")
        return 1

    payload = json.loads(args.calibration.read_text(encoding="utf-8"))
    molecules = payload["molecules"]

    if not molecules or "smiles" not in molecules[0]:
        print(
            "Kalibrasyon dosyasinda SMILES yok. 02_calibrate_surrogate.py'yi "
            "guncel surumle yeniden calistirin."
        )
        return 1

    # --- tanimlayici matrisi ---
    rows, targets, skipped = [], [], 0
    for entry in molecules:
        properties = compute_properties(entry["smiles"])
        if properties is None:
            skipped += 1
            continue
        rows.append([properties[key] for key in SURROGATE_FEATURES])
        targets.append(entry["vina"])

    X = np.array(rows, dtype=float)
    y = np.array(targets, dtype=float)

    print(f"Kalibrasyon verisi : {len(y)} molekul ({skipped} atlandi)")
    print(f"Vina araligi       : [{y.min():.2f}, {y.max():.2f}]  ort={y.mean():.2f} std={y.std():.2f}")
    print(f"Tanimlayici sayisi : {X.shape[1]}")

    if len(y) < 25:
        print("\nUYARI: 25'ten az molekul; model guvenilir olmayabilir.")

    # --- duzenlilestirme katsayisini CV ile sec ---
    print(f"\n{args.n_folds}-kat capraz dogrulama:")
    print(f"{'lambda':>10}{'Spearman':>12}{'MAE':>9}")

    candidates = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    scores = []
    for lam in candidates:
        rho, mae = cross_validate(X, y, lam, args.n_folds, args.seed)
        scores.append((rho, mae, lam))
        print(f"{lam:>10.1f}{rho:>+12.3f}{mae:>9.2f}")

    best_rho, best_mae, best_lam = max(scores, key=lambda s: s[0])

    # --- nihai model tum veriyle ---
    weights, mean, std = ridge_fit(X, y, best_lam)

    print(f"\nSecilen lambda     : {best_lam}")
    print(f"CV Spearman        : {best_rho:+.3f}   <- gorulmemis molekullerde siralama")
    print(f"CV MAE             : {best_mae:.2f} kcal/mol")

    print("\nTanimlayici agirliklari (standartlastirilmis; negatif = daha iyi baglanma):")
    for name, weight in sorted(
        zip(SURROGATE_FEATURES, weights[1:]), key=lambda t: -abs(t[1])
    ):
        print(f"  {name:<20}{weight:+.3f}")

    if best_rho > 0.6:
        verdict = "IYI - vekil siralamasi gercek docking ile buyuk olcude ortusuyor"
    elif best_rho > 0.35:
        verdict = (
            "ORTA - vekil kaba siralama icin kullanilabilir; "
            "nihai secim daima gercek docking ile yapilmali"
        )
    else:
        verdict = "ZAYIF - vekil odul sinyali olarak guvenilir degil"
    print(f"\nDegerlendirme: {verdict}")

    # --- kaydet ---
    model = {
        "model_type": "ridge_regression",
        "features": list(SURROGATE_FEATURES),
        "intercept": float(weights[0]),
        "weights": [float(w) for w in weights[1:]],
        "feature_mean": [float(m) for m in mean],
        "feature_std": [float(s) for s in std],
        "lambda": float(best_lam),
        "cv_spearman": best_rho,
        "cv_mae": best_mae,
        "n_training_molecules": int(len(y)),
        "n_folds": args.n_folds,
        "target_range": [float(y.min()), float(y.max())],
        # Uygulanabilirlik alani (applicability domain).
        # Model yalnizca egitim verisine benzeyen molekullerde anlamlidir.
        # Ornegin benzen (6 agir atom) bu araligin cok disindadir ve model
        # ona gercek disi derecede iyi bir skor verir (yuksek aromatik
        # fraksiyon katsayisi nedeniyle), oysa gercek docking -4.64 verir.
        # Alan disindaki molekuller icin skorlayici FAILED_SCORE doner.
        "applicability_domain": {
            "heavy_atoms_min": float(X[:, SURROGATE_FEATURES.index("heavy_atoms")].min()),
            "heavy_atoms_max": float(X[:, SURROGATE_FEATURES.index("heavy_atoms")].max()),
            "tolerance": 0.25,
        },
        "verdict": verdict,
        "note": (
            "Gercek AutoDock Vina skorlarindan ogrenildi (PDB 7EW9, Switch-II "
            "cebi). Bu model BU resepture ozgudur; baska bir hedef icin "
            "scripts/02 ve 05 yeniden calistirilmalidir. Mutlak degerler "
            "yaklasiktir; amac goreli siralamadir."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"\nModel kaydedildi: {args.out}")
    print("SurrogatePLI bu modeli otomatik yukleyecek.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
