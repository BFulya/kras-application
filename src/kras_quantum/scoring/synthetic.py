"""
Sentetik erisilebilirlik skorlamasi.

Makale referansi: "Experimental validation methods"

    "A synthetic accessibility assessment based on the ReRSA model."
    Sentez adayligi esigi: "Favorable synthetic accessibility with a
    ReRSA score under 5."

ReRSA (Retrosynthesis-related Synthetic Accessibility), Insilico Medicine'in
patentli modelidir (makale kaynak 58: WO 2021/229454 A1) ve acik degildir.

Vekil olarak Ertl & Schuffenhauer'in SA Score'unu kullaniyoruz
(J. Cheminform. 1:8, 2009; RDKit ile birlikte dagitilir). Her iki skor da
1-10 araliginda ve ayni yonde calisir (dusuk = kolay sentezlenir), bu yuzden
makaledeki "< 5" esigi dogrudan uygulanabilir.

Fark: ReRSA retrosentetik yol analizine dayanir; SA Score ise parcacik
(fragment) katkilari ve yapisal karmasiklik cezalarina dayanir. Sonuclar
korele ancak ozdes degildir.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

__all__ = [
    "synthetic_accessibility",
    "synthetic_accessibility_batch",
    "sa_to_reward",
    "SA_MIN",
    "SA_MAX",
]

# Ertl-Schuffenhauer SA Score araligi
SA_MIN = 1.0   # onemsiz derecede kolay
SA_MAX = 10.0  # cok zor / erisilemez

# Molekul cozulemezse en kotu skor
SA_FAILED = SA_MAX


@lru_cache(maxsize=1)
def _sascorer():
    """RDKit Contrib icindeki sascorer modulunu yukler.

    sascorer, RDKit'in ana paketinde degil Contrib dizinindedir; bu yuzden
    import yoluna elle eklenmesi gerekir.
    """
    import os
    import sys

    from rdkit.Chem import RDConfig

    sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if sa_path not in sys.path:
        sys.path.append(sa_path)

    import sascorer  # type: ignore[import-not-found]

    return sascorer


def synthetic_accessibility(smiles: str) -> float:
    """Molekulun sentetik erisilebilirlik skoru (1 = kolay, 10 = zor).

    Makale'nin ReRSA skorunun vekilidir; sentez adayligi esigi: < 5.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return SA_FAILED

    try:
        return float(_sascorer().calculateScore(mol))
    except Exception as exc:  # pragma: no cover - beklenmeyen RDKit hatasi
        logger.debug("SA skoru hesaplanamadi (%s): %s", smiles[:40], exc)
        return SA_FAILED


def synthetic_accessibility_batch(smiles_list: list[str]) -> list[float]:
    return [synthetic_accessibility(s) for s in smiles_list]


def sa_to_reward(sa_score: float, threshold: float = 5.0) -> float:
    """SA skorunu [0, 1] araliginda bir odul bilesenine cevirir.

    Esik (makalede 5.0) ve altindaki degerler yuksek odul alir; esigin
    uzerinde odul hizla duser.

        sa=1  -> 1.00   (onemsiz sentez)
        sa=5  -> 0.50   (makale esigi, tam orta)
        sa=10 -> 0.00   (erisilemez)
    """
    sa = float(min(max(sa_score, SA_MIN), SA_MAX))

    if sa <= threshold:
        # [SA_MIN, threshold] -> [1.0, 0.5]
        span = threshold - SA_MIN
        if span <= 0:
            return 1.0
        return 1.0 - 0.5 * (sa - SA_MIN) / span

    # (threshold, SA_MAX] -> (0.5, 0.0]
    span = SA_MAX - threshold
    if span <= 0:
        return 0.5
    return 0.5 * (1.0 - (sa - threshold) / span)
