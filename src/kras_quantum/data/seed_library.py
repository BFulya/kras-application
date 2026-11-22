"""
Cekirdek (seed) KRAS inhibitor kutuphanesi.

Makale referansi: "Data acquisition and preprocessing" ve Fig. 1a

    "Our preliminary dataset, sourced from Insilico Medicine, included
     approximately 650 data points. These were selectively collated from
     existing literature, specifically targeting the KRAS-G12D mutant."

ONEMLI - veri kaynagi farki
---------------------------
Makaledeki 650 molekulluk cekirdek set Insilico Medicine'in ic
derlemesidir ve yayinla birlikte dagitilmamistir. Bu depoda o seti
UYDURMUYORUZ. Bunun yerine, literaturde adi gecen KRAS inhibitorlerinin
yapilarini PubChem'den isimle cekiyoruz (bkz. scripts/00_fetch_seed_library.py).

Sonuc, makaledeki 650 molekulun yerine gecmez; buyukluk mertebesi olarak
cok daha kucuktur (~15-25 bilesik). Hat yine de calisir, cunku STONED-SELFIES
asamasi cekirdek basina binlerce analog uretir - ancak sonuclarin makaleyle
birebir ayni olmasi BEKLENMEMELIDIR. README'deki "Makaleden sapmalar"
bolumune bakin.

Kendi cekirdek setiniz varsa (ChEMBL disa aktarimi, ic derleme, vb.)
`load_seed_library(path=...)` ile dogrudan yukleyebilirsiniz; CSV'de
`name` ve `smiles` sutunlari aranir.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

from kras_quantum.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "SeedCompound",
    "DEFAULT_SEED_PATH",
    "KNOWN_KRAS_INHIBITOR_NAMES",
    "load_seed_library",
    "write_seed_library",
]


DEFAULT_SEED_PATH = PROJECT_ROOT / "data" / "seed_kras_inhibitors.csv"


# Literaturde adi gecen KRAS (ve yakin yolak) inhibitorleri.
# Bunlar PubChem'de isimle aranir; yapilar oradan gelir, elle yazilmaz.
# Kaynaklar: makale kaynak 21 (MRTX1133), 22, 59 (pan-RAS), 60-63
# (KRAS inhibitor derlemeleri).
KNOWN_KRAS_INHIBITOR_NAMES: tuple[str, ...] = (
    # --- KRAS G12C, klinik asamada ---
    "sotorasib",        # AMG 510
    "adagrasib",        # MRTX849
    "divarasib",        # GDC-6036
    "opnurasib",        # JDQ443
    "glecirasib",       # JAB-21822
    "garsorasib",       # D-1553
    "olomorasib",       # LY3537982
    "fulzerasib",       # GFH925 / IBI351
    # --- KRAS G12C, arac bilesikler ---
    "ARS-1620",
    "ARS-853",
    # --- KRAS G12D ---
    "MRTX1133",         # Makale kaynak 21; secicilik icin D12 ile etkilesir
    "TH-Z816",          # Makale, Extended Data Fig. 4 - farmakofor referansi (PDB 7EW9)
    "TH-Z835",
    # --- pan-KRAS / pan-RAS ---
    "BI-2865",
    "BI-2493",
    # --- SOS1 aracili RAS yolak inhibitorleri ---
    "BI-1701963",
    "BAY-293",
    # --- Erken donem RAS baglayicilar ---
    "Kobe0065",
    "Kobe2602",
)


@dataclass(frozen=True)
class SeedCompound:
    """Cekirdek kutuphanedeki tek bir bilesik."""

    name: str
    smiles: str
    source: str = ""          # or. "PubChem CID 137278711"
    inchikey: str = ""
    note: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.name}: {self.smiles}"


def load_seed_library(
    path: str | Path | None = None, validate: bool = True
) -> list[SeedCompound]:
    """Cekirdek kutuphaneyi CSV'den yukler.

    Parameters
    ----------
    path
        CSV yolu. Verilmezse `data/seed_kras_inhibitors.csv` kullanilir.
    validate
        True ise her SMILES RDKit ile cozulur; cozulemeyenler atlanir
        ve uyari loglanir.

    Raises
    ------
    FileNotFoundError
        Dosya yoksa - kullaniciya once fetch scriptini calistirmasi soylenir.
    """
    path = Path(path) if path is not None else DEFAULT_SEED_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Cekirdek kutuphane bulunamadi: {path}\n"
            f"Once yapilari indirin:\n"
            f"    python scripts/00_fetch_seed_library.py\n"
            f"veya kendi CSV'nizi ('name','smiles' sutunlariyla) bu yola koyun."
        )

    compounds: list[SeedCompound] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"name", "smiles"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} dosyasinda eksik sutun(lar): {sorted(missing)}")

        for row in reader:
            smiles = (row.get("smiles") or "").strip()
            name = (row.get("name") or "").strip()
            if not smiles:
                continue
            compounds.append(
                SeedCompound(
                    name=name,
                    smiles=smiles,
                    source=(row.get("source") or "").strip(),
                    inchikey=(row.get("inchikey") or "").strip(),
                    note=(row.get("note") or "").strip(),
                )
            )

    if validate:
        compounds = _validate(compounds)

    if not compounds:
        raise ValueError(f"{path} icinde gecerli bilesik yok")

    logger.info("Cekirdek kutuphane yuklendi: %d bilesik (%s)", len(compounds), path)
    return compounds


def _validate(compounds: list[SeedCompound]) -> list[SeedCompound]:
    """SMILES'leri RDKit ile cozerek gecersizleri eler."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    valid: list[SeedCompound] = []
    for compound in compounds:
        if Chem.MolFromSmiles(compound.smiles) is None:
            logger.warning(
                "Gecersiz SMILES atlandi - %s: %s", compound.name, compound.smiles
            )
            continue
        valid.append(compound)
    return valid


def write_seed_library(
    compounds: list[SeedCompound], path: str | Path | None = None
) -> Path:
    """Cekirdek kutuphaneyi CSV olarak yazar."""
    path = Path(path) if path is not None else DEFAULT_SEED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["name", "smiles", "inchikey", "source", "note"]
        )
        writer.writeheader()
        for compound in compounds:
            writer.writerow(
                {
                    "name": compound.name,
                    "smiles": compound.smiles,
                    "inchikey": compound.inchikey,
                    "source": compound.source,
                    "note": compound.note,
                }
            )

    logger.info("Cekirdek kutuphane yazildi: %d bilesik -> %s", len(compounds), path)
    return path
