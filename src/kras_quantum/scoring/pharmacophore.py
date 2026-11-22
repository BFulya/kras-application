"""
Farmakofor eslesme skorlamasi.

Makale referansi: Extended Data Fig. 4 | "Pharmacophore Model Depiction"

    "Pharmacophore Model Depiction for KRAS Inhibitor TH-Z816 Based on the
     Co-crystallized Ligand Structure Analyzed with Chemistry42 (PDB: 7EW9).
     ... A blue sphere represents a critical ring system that supports
     structural integrity, a green sphere highlights a hydrophobic moiety
     essential for binding affinity, and a cyan sphere indicates a hydrogen
     bond donor that contributes to interaction specificity with the KRAS
     protein. ... the pharmacophore interactions within the KRAS Switch-II
     binding pocket."

Sentez adayligi esigi ("Experimental validation methods"):
    "A pharmacophore match score above 0.7."

Yani makalenin farmakofor modeli UC ozellikten olusur:
    1. Kritik halka sistemi   (yapisal bütünlük)
    2. Hidrofobik grup        (baglanma afinitesi)
    3. Hidrojen bagi donoru   (etkilesim ozgullugu)

Bu modul iki tamamlayici sinyali birlestirir:

  a) Ozellik varligi - makalenin tanimladigi uc ozelligin adayda bulunup
     bulunmadigi (RDKit kimyasal ozellik tanimlari ile).
  b) Farmakofor parmak izi benzerligi - adayin, referans ligand TH-Z816 ile
     Gobbi 2B farmakofor parmak izi uzerinden Tanimoto benzerligi. Bu,
     ozelliklerin yalnizca VARLIGINI degil, aralarindaki topolojik
     DUZENINI de yakalar; farmakoforun ozu budur.

Chemistry42'nin 3B hizalama tabanli skoru tescillidir ve birebir
yeniden uygulanamaz. Buradaki skor bagimsiz bir vekildir; esik degeri
(0.7) makaleden alinmistir ancak olcekler ozdes degildir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

__all__ = [
    "PharmacophoreFeatures",
    "REFERENCE_LIGAND_NAME",
    "extract_features",
    "pharmacophore_match_score",
    "pharmacophore_similarity",
]

# Makalenin farmakofor referansi (Extended Data Fig. 4, PDB 7EW9)
REFERENCE_LIGAND_NAME = "TH-Z816"


@dataclass(frozen=True)
class PharmacophoreFeatures:
    """Makalenin uc farmakofor ozelliginin sayimlari."""

    aromatic_rings: int       # mavi kure: kritik halka sistemi
    hydrophobic_groups: int   # yesil kure: hidrofobik grup
    hbond_donors: int         # cyan kure: hidrojen bagi donoru

    # Tani amacli ek bilgi
    hbond_acceptors: int = 0
    total_rings: int = 0

    @property
    def has_all_required(self) -> bool:
        """Uc temel ozelligin hepsi mevcut mu?"""
        return (
            self.aromatic_rings >= 1
            and self.hydrophobic_groups >= 1
            and self.hbond_donors >= 1
        )


def _rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


@lru_cache(maxsize=1)
def _feature_factory():
    """RDKit'in yerlesik kimyasal ozellik tanimlarini yukler."""
    import os

    from rdkit import RDConfig
    from rdkit.Chem import ChemicalFeatures

    fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    return ChemicalFeatures.BuildFeatureFactory(fdef)


def extract_features(smiles: str) -> PharmacophoreFeatures | None:
    """Molekulun farmakofor ozelliklerini cikarir."""
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        features = _feature_factory().GetFeaturesForMol(mol)
    except Exception as exc:  # pragma: no cover
        logger.debug("Ozellik cikarimi basarisiz (%s): %s", smiles[:40], exc)
        return None

    counts: dict[str, int] = {}
    for feature in features:
        family = feature.GetFamily()
        counts[family] = counts.get(family, 0) + 1

    ring_info = mol.GetRingInfo()
    aromatic_rings = sum(
        1
        for ring in ring_info.AtomRings()
        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
    )

    return PharmacophoreFeatures(
        aromatic_rings=aromatic_rings,
        hydrophobic_groups=counts.get("Hydrophobe", 0) + counts.get("LumpedHydrophobe", 0),
        hbond_donors=counts.get("Donor", 0),
        hbond_acceptors=counts.get("Acceptor", 0),
        total_rings=ring_info.NumRings(),
    )


# ---------------------------------------------------------------------------
# Farmakofor parmak izi benzerligi
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _pharm2d_fingerprint(smiles: str):
    """Gobbi 2B farmakofor parmak izi.

    Atom ciftleri arasindaki farmakoforik ozellik-mesafe oruntulerini kodlar;
    ozelliklerin yalnizca varligini degil DUZENINI de yakalar.
    """
    from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
    except Exception as exc:  # pragma: no cover
        logger.debug("Farmakofor parmak izi uretilemedi (%s): %s", smiles[:40], exc)
        return None


def pharmacophore_similarity(smiles: str, reference_smiles: str) -> float:
    """Iki molekulun farmakofor parmak izi Tanimoto benzerligi (0-1)."""
    from rdkit import DataStructs

    fp_query = _pharm2d_fingerprint(smiles)
    fp_reference = _pharm2d_fingerprint(reference_smiles)

    if fp_query is None or fp_reference is None:
        return 0.0

    return float(DataStructs.TanimotoSimilarity(fp_query, fp_reference))


# ---------------------------------------------------------------------------
# Birlesik eslesme skoru
# ---------------------------------------------------------------------------
def _feature_presence_score(features: PharmacophoreFeatures) -> float:
    """Makalenin uc ozelligi icin [0, 1] skorlama.

    Her ozellik esit agirlikli (1/3). Ozellik varsa tam puan; birden fazla
    aromatik halka (TH-Z816'da oldugu gibi cok halkali sistem) tam puan alir,
    tek halka kismi puan.
    """
    # Mavi kure: kritik halka sistemi
    if features.aromatic_rings >= 2:
        ring_score = 1.0
    elif features.aromatic_rings == 1:
        ring_score = 0.6
    else:
        ring_score = 0.0

    # Yesil kure: hidrofobik grup
    if features.hydrophobic_groups >= 2:
        hydrophobic_score = 1.0
    elif features.hydrophobic_groups == 1:
        hydrophobic_score = 0.7
    else:
        hydrophobic_score = 0.0

    # Cyan kure: hidrojen bagi donoru
    # Makale "ozgulluk" vurgusu yapar; cok sayida donor secicilige zarar
    # verdigi icin 1-2 donor en yuksek puani alir.
    if features.hbond_donors in (1, 2):
        donor_score = 1.0
    elif features.hbond_donors >= 3:
        donor_score = 0.6
    else:
        donor_score = 0.0

    return (ring_score + hydrophobic_score + donor_score) / 3.0


def pharmacophore_match_score(
    smiles: str,
    reference_smiles: str | None = None,
    presence_weight: float = 0.5,
) -> float:
    """Farmakofor eslesme skoru, [0, 1].

    Makale sentez adayligi icin bu skorun 0.7'nin uzerinde olmasini sart kosar.

    Parameters
    ----------
    reference_smiles
        Referans ligand (varsayilan: TH-Z816, cekirdek kutuphaneden yuklenir).
        Yuklenemezse skor yalnizca ozellik varligina dayanir.
    presence_weight
        Ozellik varligi ile parmak izi benzerligi arasindaki agirlik dengesi.
        0.5 -> esit katki.
    """
    features = extract_features(smiles)
    if features is None:
        return 0.0

    presence = _feature_presence_score(features)

    if reference_smiles is None:
        reference_smiles = _default_reference_smiles()

    if reference_smiles is None:
        # Referans yok: yalnizca ozellik varligina dayan
        return presence

    similarity = pharmacophore_similarity(smiles, reference_smiles)
    return float(presence_weight * presence + (1.0 - presence_weight) * similarity)


@lru_cache(maxsize=1)
def _default_reference_smiles() -> str | None:
    """Cekirdek kutuphaneden TH-Z816'yi yukler (Extended Data Fig. 4 referansi)."""
    try:
        from kras_quantum.data.seed_library import load_seed_library

        for compound in load_seed_library():
            if compound.name.upper() == REFERENCE_LIGAND_NAME.upper():
                return compound.smiles
    except (FileNotFoundError, ValueError) as exc:
        logger.warning(
            "Farmakofor referansi (%s) yuklenemedi: %s. "
            "Skor yalnizca ozellik varligina dayanacak.",
            REFERENCE_LIGAND_NAME,
            exc,
        )
    return None
