"""
Cekirdek KRAS inhibitor kutuphanesini PubChem'den indirir.

Makale, 650 molekulluk cekirdek seti Insilico Medicine'in ic derlemesinden
alir; bu set yayinla dagitilmamistir. Burada, literaturde adi gecen KRAS
inhibitorlerinin yapilarini PubChem'in acik REST arayuzunden isimle
cekiyoruz - boylece depoda elle yazilmis (ve sessizce yanlis olabilecek)
hicbir SMILES bulunmaz.

Kullanim:
    python scripts/00_fetch_seed_library.py
    python scripts/00_fetch_seed_library.py --out data/benim_setim.csv

Yalnizca Python standart kutuphanesini kullanir; RDKit gerektirmez, bu
yuzden konteyner disinda da calisir.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Depo kokunu import yoluna ekle
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kras_quantum.data.seed_library import (  # noqa: E402
    DEFAULT_SEED_PATH,
    KNOWN_KRAS_INHIBITOR_NAMES,
    SeedCompound,
    write_seed_library,
)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
# PubChem, saniyede 5 istegi asmamayi ister.
REQUEST_DELAY_S = 0.25
TIMEOUT_S = 30


def fetch_compound(name: str) -> SeedCompound | None:
    """PubChem'den isme gore yapi ceker; bulunamazsa None."""
    url = (
        f"{PUBCHEM_BASE}/compound/name/{urllib.parse.quote(name)}"
        f"/property/SMILES,InChIKey,MolecularFormula/JSON"
    )

    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            print(f"  [BULUNAMADI] {name}")
        else:
            print(f"  [HTTP {exc.code}]   {name}")
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  [AG HATASI]   {name}: {exc}")
        return None
    except json.JSONDecodeError:
        print(f"  [BOZUK YANIT] {name}")
        return None

    try:
        record = payload["PropertyTable"]["Properties"][0]
    except (KeyError, IndexError):
        print(f"  [BOS KAYIT]   {name}")
        return None

    smiles = record.get("SMILES") or record.get("ConnectivitySMILES")
    if not smiles:
        print(f"  [SMILES YOK]  {name}")
        return None

    cid = record.get("CID", "")
    print(f"  [TAMAM]       {name:<14} CID {cid}  {record.get('MolecularFormula', '')}")

    return SeedCompound(
        name=name,
        smiles=smiles,
        inchikey=record.get("InChIKey", ""),
        source=f"PubChem CID {cid}",
        note="Makale, Fig. 1a: literaturden derlenen deneysel KRAS inhibitorleri",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_SEED_PATH,
        help=f"Cikti CSV yolu (varsayilan: {DEFAULT_SEED_PATH})",
    )
    args = parser.parse_args()

    print(f"PubChem'den {len(KNOWN_KRAS_INHIBITOR_NAMES)} bilesik araniyor...\n")

    compounds: list[SeedCompound] = []
    for name in KNOWN_KRAS_INHIBITOR_NAMES:
        compound = fetch_compound(name)
        if compound is not None:
            compounds.append(compound)
        time.sleep(REQUEST_DELAY_S)

    if not compounds:
        print("\nHicbir bilesik indirilemedi. Ag baglantisini kontrol edin.")
        return 1

    path = write_seed_library(compounds, args.out)

    bulunamayan = len(KNOWN_KRAS_INHIBITOR_NAMES) - len(compounds)
    print(
        f"\n{len(compounds)}/{len(KNOWN_KRAS_INHIBITOR_NAMES)} bilesik yazildi -> {path}"
    )
    if bulunamayan:
        print(f"{bulunamayan} bilesik PubChem'de isimle bulunamadi (yukarida isaretli).")
    print(
        "\nNOT: Bu set, makaledeki 650 molekulluk Insilico ic derlemesinin "
        "yerine gecmez;\nSTONED-SELFIES asamasi icin cekirdek gorevi gorur."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
