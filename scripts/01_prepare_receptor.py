"""
Docking icin KRAS reseptorunu hazirlar.

Makale referansi
----------------
Iki farkli yapi kullanilir:

  * PDB 5US4 -> sanal tarama ("Virtual screening process"):
      "The protein structure used in the screen was PDB 5US4, which was
       prepared before the virtual screen with Schrodinger's protein
       preparation wizard (addition of hydrogens and protonation state
       prediction). The size of the docking box was 14 x 14 x 20 A^3."

  * PDB 7EW9 -> farmakofor / PLI referansi (Extended Data Fig. 4):
      "Three-dimensional pharmacophoric analysis in reference to an X-ray
       cocrystal structure (PDB 7EW9)" - TH-Z816'nin KRAS-G12D'ye bagli
       ko-kristal yapisi, Switch-II cebi.

Varsayilan olarak 7EW9 kullaniyoruz: makalenin PLI skorlamasini ve
farmakofor modelini dayandirdigi yapi budur ve ko-kristal ligandi
(HET kodu 05C = TH-Z816) docking kutusunu kesin olarak konumlandirir.

Schrodinger tescillidir; hidrojen ekleme ve protonasyon OpenBabel ile
yapilir (pH 7.4). Bu, makalenin hazirlik adimiyla ayni amaca hizmet eder
ancak birebir ayni arac degildir.

Kullanim:
    python scripts/01_prepare_receptor.py
    python scripts/01_prepare_receptor.py --pdb-id 5US4 --chain A
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kras_quantum.config import PROJECT_ROOT  # noqa: E402

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
RECEPTOR_DIR = PROJECT_ROOT / "data" / "receptor"

# Reseptorde TUTULACAK heteroatom gruplari.
# KRAS islevsel olarak nukleotid baglidir; GDP ve koordine Mg2+ proteinin
# katlanmasinin parcasidir ve Switch bolgelerinin konumunu belirler.
# Makale NMR deneylerini de GDP yuklu kosulda yapmistir.
KEEP_HETATMS = {"GDP", "GTP", "GNP", "MG"}

# Cikarilacaklar: su molekulleri ve ko-kristal inhibitor
# (inhibitorun yerine kendi molekullerimizi yerlestirecegiz)
DROP_HETATMS = {"HOH", "WAT", "SO4", "PO4", "CL", "NA", "EDO", "GOL", "ACT"}

# Makalenin farmakofor referansi 7EW9'daki ko-kristal ligand
COCRYSTAL_LIGANDS = {
    "7EW9": "05C",   # TH-Z816 (Extended Data Fig. 4)
}


def download_pdb(pdb_id: str, out_path: Path) -> Path:
    """RCSB'den PDB dosyasini indirir (zaten varsa yeniden indirmez)."""
    if out_path.exists():
        print(f"  PDB onbellekte: {out_path}")
        return out_path

    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    print(f"  Indiriliyor: {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url, timeout=60) as response:
        out_path.write_bytes(response.read())

    print(f"  Kaydedildi: {out_path} ({out_path.stat().st_size // 1024} KB)")
    return out_path


def split_structure(
    pdb_path: Path, chain: str, ligand_code: str | None
) -> tuple[list[str], list[str]]:
    """PDB'yi reseptor ve ko-kristal ligand satirlarina ayirir.

    Returns
    -------
    (reseptor_satirlari, ligand_satirlari)
    """
    receptor_lines: list[str] = []
    ligand_lines: list[str] = []

    for line in pdb_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        record = line[:6].strip()

        if record not in ("ATOM", "HETATM"):
            continue

        line_chain = line[21]
        residue = line[17:20].strip()
        # Alternatif konformasyonlardan yalnizca ilkini al
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue

        if line_chain != chain:
            continue

        if record == "ATOM":
            receptor_lines.append(line)
            continue

        # HETATM
        if ligand_code and residue == ligand_code:
            ligand_lines.append(line)
        elif residue in KEEP_HETATMS:
            receptor_lines.append(line)
        # DROP_HETATMS ve bilinmeyen heteroatomlar atlanir

    return receptor_lines, ligand_lines


def compute_box_center(ligand_lines: list[str]) -> tuple[float, float, float]:
    """Ko-kristal ligandin agirlik merkezi -> docking kutusu merkezi.

    Bu, baglanma cebini tahmin etmek yerine DENEYSEL olarak konumlandirir:
    kutu, kristal yapida inhibitorun gercekten bulundugu yere oturur
    (7EW9'da Switch-II cebi).
    """
    if not ligand_lines:
        raise ValueError(
            "Ko-kristal ligand bulunamadi; kutu merkezi hesaplanamiyor. "
            "--center-x/y/z ile elle verin."
        )

    coords = [
        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        for line in ligand_lines
    ]
    n = len(coords)
    return (
        sum(c[0] for c in coords) / n,
        sum(c[1] for c in coords) / n,
        sum(c[2] for c in coords) / n,
    )


def write_pdbqt(pdb_lines: list[str], pdb_out: Path, pdbqt_out: Path, ph: float = 7.4) -> Path:
    """Reseptoru PDBQT formatina cevirir (OpenBabel, rigid).

    Makale, Schrodinger'in protein hazirlama sihirbazini kullanir
    (hidrojen ekleme + protonasyon durumu tahmini). Burada ayni islemi
    OpenBabel ile yapiyoruz.
    """
    from openbabel import pybel

    pdb_out.parent.mkdir(parents=True, exist_ok=True)
    pdb_out.write_text("\n".join(pdb_lines) + "\nEND\n", encoding="utf-8")

    molecule = next(pybel.readfile("pdb", str(pdb_out)))
    molecule.OBMol.CorrectForPH(ph)   # protonasyon durumu
    molecule.addh()                   # hidrojen ekleme
    molecule.OBMol.SetAutomaticPartialCharge(True)

    # -xr: rigid reseptor (donebilir bag yazma)
    molecule.write("pdbqt", str(pdbqt_out), overwrite=True, opt={"r": None})
    return pdbqt_out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb-id",
        default="7EW9",
        help="RCSB PDB kodu (varsayilan: 7EW9 - makalenin farmakofor referansi)",
    )
    parser.add_argument("--chain", default="A", help="Kullanilacak zincir (varsayilan: A)")
    parser.add_argument(
        "--ligand-code",
        default=None,
        help="Ko-kristal ligand HET kodu; verilmezse bilinen esleme kullanilir",
    )
    parser.add_argument(
        "--box-padding",
        type=float,
        default=5.0,
        help="Ligandin sinirlayici kutusuna eklenecek bosluk (A, varsayilan: 5)",
    )
    parser.add_argument("--ph", type=float, default=7.4, help="Protonasyon pH'i")
    args = parser.parse_args()

    pdb_id = args.pdb_id.upper()
    ligand_code = args.ligand_code or COCRYSTAL_LIGANDS.get(pdb_id)

    print(f"KRAS reseptoru hazirlaniyor: PDB {pdb_id}, zincir {args.chain}")
    if ligand_code:
        print(f"  Ko-kristal ligand: {ligand_code}")

    raw_pdb = download_pdb(pdb_id, RECEPTOR_DIR / f"{pdb_id}.pdb")
    receptor_lines, ligand_lines = split_structure(raw_pdb, args.chain, ligand_code)

    if not receptor_lines:
        print(f"HATA: {args.chain} zincirinde protein atomu bulunamadi.")
        return 1

    print(f"  Reseptor atomu: {len(receptor_lines)}  |  ligand atomu: {len(ligand_lines)}")

    # --- docking kutusu ---
    if ligand_lines:
        center = compute_box_center(ligand_lines)
        coords = [
            (float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in ligand_lines
        ]
        size = tuple(
            max(
                max(c[i] for c in coords) - min(c[i] for c in coords) + 2 * args.box_padding,
                14.0,  # makaledeki kutu kenarlarindan kucuk olmasin
            )
            for i in range(3)
        )
    else:
        print("  UYARI: ko-kristal ligand yok; kutu protein merkezine konuluyor.")
        coords = [
            (float(l[30:38]), float(l[38:46]), float(l[46:54])) for l in receptor_lines
        ]
        n = len(coords)
        center = tuple(sum(c[i] for c in coords) / n for i in range(3))
        size = (14.0, 14.0, 20.0)  # makaledeki varsayilan kutu

    print(f"  Kutu merkezi: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    print(f"  Kutu boyutu : ({size[0]:.1f}, {size[1]:.1f}, {size[2]:.1f}) A")

    # --- PDBQT ---
    clean_pdb = RECEPTOR_DIR / f"{pdb_id}_{args.chain}_receptor.pdb"
    pdbqt = RECEPTOR_DIR / f"{pdb_id}_{args.chain}_receptor.pdbqt"
    print("  PDBQT yaziliyor (hidrojen ekleme + protonasyon)...")
    write_pdbqt(receptor_lines, clean_pdb, pdbqt, ph=args.ph)
    print(f"  Reseptor PDBQT: {pdbqt} ({pdbqt.stat().st_size // 1024} KB)")

    if ligand_lines:
        ref_ligand = RECEPTOR_DIR / f"{pdb_id}_{ligand_code}_reference.pdb"
        ref_ligand.write_text("\n".join(ligand_lines) + "\nEND\n", encoding="utf-8")
        print(f"  Referans ligand: {ref_ligand}")

    # --- yapilandirma ---
    config = {
        "pdb_id": pdb_id,
        "chain": args.chain,
        "cocrystal_ligand": ligand_code,
        "receptor_pdbqt": str(pdbqt.relative_to(PROJECT_ROOT)),
        "center": {"x": center[0], "y": center[1], "z": center[2]},
        "size": {"x": size[0], "y": size[1], "z": size[2]},
        "ph": args.ph,
        "note": (
            "Kutu merkezi, ko-kristal ligandin agirlik merkezinden hesaplandi "
            "(deneysel baglanma bolgesi). Makale, sanal taramada PDB 5US4 ve "
            "14x14x20 A kutu kullanmistir; 7EW9 ise makalenin farmakofor ve "
            "PLI referans yapisidir (Extended Data Fig. 4)."
        ),
    }
    config_path = RECEPTOR_DIR / "receptor_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\nYapilandirma yazildi: {config_path}")
    print("Reseptor docking icin hazir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
