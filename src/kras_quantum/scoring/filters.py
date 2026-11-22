"""
Yapisal ve med-chem filtreleri.

Makale referansi
----------------
Iki ayri filtre kumesi kullanilir ve karsilastirilir (Extended Data Fig. 1a,b):

  * "Local Filters"      -> Tartarus kiyaslama platformundan (makale kaynak 26)
  * "Chemistry42 Filters" -> daha katı, tescilli kume

Makale, Tartarus filtrelerini soyle tanimlar:

    "these molecules were required to pass a comprehensive set of filters
     designed to eliminate reactive, unsynthesizable or unstable groups"

Chemistry42 tarafinda ise ("Experimental validation methods"):

    "Structural and compositional parameter filters, such as hydrogen bond
     donor count and aromatic atom fraction."
    "Property evaluation for molecular weight, lipophilicity and other
     physicochemical traits."
    "Medicinal chemistry filters to exclude problematic structural motifs."

Chemistry42 tescillidir; buradaki kume ACIK KAYNAK BIR VEKILDIR. Esikler
makalede acikca verilen yerlerde makaleden, verilmeyen yerlerde yerlesik
med-chem uygulamasindan (Lipinski, Veber, PAINS) alinmistir. Bu bir
yeniden-uygulama degil, benzer amacli bagimsiz bir kumedir.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

__all__ = [
    "FilterResult",
    "FilterThresholds",
    "UNWANTED_SMARTS",
    "passes_local_filter",
    "passes_chemistry42_filter",
    "evaluate_filters",
]


# ---------------------------------------------------------------------------
# Istenmeyen yapisal motifler
# ---------------------------------------------------------------------------
# Reaktif, kararsiz veya sentezlenemez gruplar. Tartarus'un "local filter"
# fikrini izler: uretici modellerin sikca urettigi, ilac benzeri olmayan
# motifleri eler.
UNWANTED_SMARTS: dict[str, str] = {
    # --- reaktif elektrofiller ---
    "acil_halojenur": "[CX3](=O)[F,Cl,Br,I]",
    "aldehit": "[CX3H1](=O)[#6]",
    "izosiyanat": "[NX2]=[CX2]=[OX1]",
    "izotiyosiyanat": "[NX2]=[CX2]=[SX1]",
    "epoksit": "[OX2r3]1[#6r3][#6r3]1",
    "aziridin": "[NX3r3]1[#6r3][#6r3]1",
    "alkil_halojenur": "[CX4][Cl,Br,I]",
    "sulfonil_halojenur": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "anhidrit": "[CX3](=O)[OX2][CX3](=O)",
    # --- kararsiz / sorunlu gruplar ---
    "peroksit": "[OX2][OX2]",
    "azo": "[NX2]=[NX2]",
    "azid": "[NX2]=[NX2+]=[NX1-]",
    "diazo": "[CX3]=[NX2+]=[NX1-]",
    "nitrozo": "[NX2]=[OX1]",
    "tiyol": "[SX2H]",
    "hidrazin": "[NX3][NX3]",
    "fosfor_halojenur": "[PX3,PX4][F,Cl,Br,I]",
    # --- heteroatom zincirleri (kararsiz) ---
    "uc_heteroatom_zinciri": "[NX3,OX2,SX2][NX3,OX2,SX2][NX3,OX2,SX2]",
    # --- toksikofor / PAINS benzeri ---
    "kinon": "[#6]1(=O)[#6][#6][#6](=O)[#6][#6]1",
    "nitro_aromatik": "[a][NX3](=O)=O",
    "michael_akseptoru_aldehit": "[CX3]=[CX3][CX3H1]=[OX1]",
}


@dataclass(frozen=True)
class FilterThresholds:
    """Fizikokimyasal esikler.

    Makale, "Property evaluation for molecular weight, lipophilicity and other
    physicochemical traits" der ancak sayisal esik vermez. Asagidakiler
    yerlesik ilac-benzerligi kurallarindan alinmistir:
      - Lipinski "Rule of Five" (MW<=500, logP<=5, HBD<=5, HBA<=10)
      - Veber (donebilir bag<=10, TPSA<=140)

    KRAS inhibitorleri bilinerek buyuk molekullerdir (MRTX1133: 601 Da,
    sotorasib: 561 Da), bu yuzden MW ust siniri klasik 500 yerine 650'ye
    cekilmistir - aksi halde makalenin kendi cekirdek bilesikleri elenirdi.
    """

    min_heavy_atoms: int = 10
    max_heavy_atoms: int = 70

    min_molecular_weight: float = 150.0
    max_molecular_weight: float = 650.0   # KRAS kimyasal alanina uyarlandi

    min_logp: float = -3.0
    max_logp: float = 6.5

    max_hbd: int = 5                      # Makale: "hydrogen bond donor count"
    max_hba: int = 12
    max_rotatable_bonds: int = 12
    max_tpsa: float = 160.0

    min_aromatic_fraction: float = 0.0    # Makale: "aromatic atom fraction"
    max_aromatic_fraction: float = 0.85

    # Halka sayisi RDKit'te SSSR olarak sayilir; kaynasmis (fused) sistemler
    # yuksek deger verir. Esik, cekirdek kutuphanedeki GERCEK inhibitorlere
    # gore kalibre edildi: MRTX1133 (makale kaynak 21, G12D secicilik
    # referansi) 8 halka icerir ve 7'lik esikle eleniyordu.
    max_rings: int = 9
    max_ring_size: int = 8
    min_sp3_fraction: float = 0.0

    # Radikal elektronlar: uretici modeller SELFIES uzerinden [CH], [C] gibi
    # karben/radikal atomlar uretebilir. Bunlar kimyasal olarak kararsiz ve
    # sentezlenemezdir; makalenin "eliminate reactive, unsynthesizable or
    # unstable groups" olcutune dogrudan girerler. Her iki filtre kumesinde
    # de sifir tolerans uygulanir.
    max_radical_electrons: int = 0

    # Yalnizca "stringent" (Chemistry42 vekili) kumede uygulanir
    max_formal_charge: int = 2
    max_unwanted_motifs: int = 0


@dataclass
class FilterResult:
    """Tek bir molekulun filtre sonucu."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    properties: dict[str, float] = field(default_factory=dict)

    @property
    def n_failures(self) -> int:
        return len(self.failures)

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:  # pragma: no cover
        if self.passed:
            return "GECTI"
        return "KALDI: " + ", ".join(self.failures)


# ---------------------------------------------------------------------------
# RDKit yardimcilari
# ---------------------------------------------------------------------------
def _rdkit():
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem


@lru_cache(maxsize=1)
def _compiled_smarts() -> list[tuple[str, object]]:
    """SMARTS desenlerini bir kez derler (her molekulde yeniden derlemek pahali)."""
    Chem = _rdkit()
    compiled = []
    for name, smarts in UNWANTED_SMARTS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            logger.warning("SMARTS derlenemedi, atlaniyor: %s = %s", name, smarts)
            continue
        compiled.append((name, pattern))
    return compiled


def compute_properties(smiles: str) -> dict[str, float] | None:
    """Molekulun fizikokimyasal tanimlayicilarini hesaplar.

    Returns
    -------
    dict veya None (molekul cozulemezse)
    """
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(ring) for ring in ring_info.AtomRings()]

    return {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "heavy_atoms": float(heavy_atoms),
        "aromatic_fraction": (aromatic_atoms / heavy_atoms) if heavy_atoms else 0.0,
        "sp3_fraction": rdMolDescriptors.CalcFractionCSP3(mol),
        "n_rings": float(ring_info.NumRings()),
        "max_ring_size": float(max(ring_sizes)) if ring_sizes else 0.0,
        "formal_charge": float(abs(Chem.GetFormalCharge(mol))),
        "n_radical_electrons": float(
            sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
        ),
        "n_unwanted_motifs": float(count_unwanted_motifs(smiles)),
    }


def count_unwanted_motifs(smiles: str) -> int:
    """Molekuldeki istenmeyen yapisal motif sayisi."""
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return len(UNWANTED_SMARTS)  # cozulemeyen molekul: en kotu durum

    return sum(1 for _, pattern in _compiled_smarts() if mol.HasSubstructMatch(pattern))


def find_unwanted_motifs(smiles: str) -> list[str]:
    """Hangi istenmeyen motiflerin bulundugunu dondurur (tani amacli)."""
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["cozulemeyen_molekul"]

    return [name for name, pattern in _compiled_smarts() if mol.HasSubstructMatch(pattern)]


# ---------------------------------------------------------------------------
# Filtre degerlendirmesi
# ---------------------------------------------------------------------------
def evaluate_filters(
    smiles: str,
    thresholds: FilterThresholds | None = None,
    stringent: bool = False,
) -> FilterResult:
    """Molekulu filtre kumesinden gecirir.

    Parameters
    ----------
    stringent
        False -> "local filter" (Tartarus benzeri): temel gecerlilik,
                 boyut ve reaktif grup elemesi.
        True  -> "Chemistry42 filter" vekili: ek olarak tum fizikokimyasal
                 esikler ve yuk siniri.

    Makale, iki kumenin katiliginin farkli oldugunu Extended Data Fig. 1a,b'de
    gosterir: local filtrede basari orani ~%52-64 iken Chemistry42 filtresinde
    ~%8-10'a duser.
    """
    thresholds = thresholds or FilterThresholds()
    properties = compute_properties(smiles)

    if properties is None:
        return FilterResult(passed=False, failures=["cozulemeyen_molekul"])

    failures: list[str] = []

    def check(condition: bool, label: str) -> None:
        if not condition:
            failures.append(label)

    # --- her iki kumede de uygulanan temel kontroller ---
    check(
        thresholds.min_heavy_atoms <= properties["heavy_atoms"] <= thresholds.max_heavy_atoms,
        f"agir_atom_sayisi({properties['heavy_atoms']:.0f})",
    )
    check(
        properties["n_unwanted_motifs"] <= thresholds.max_unwanted_motifs,
        f"istenmeyen_motif({properties['n_unwanted_motifs']:.0f})",
    )
    check(
        properties["n_radical_electrons"] <= thresholds.max_radical_electrons,
        f"radikal_elektron({properties['n_radical_electrons']:.0f})",
    )
    check(
        properties["max_ring_size"] <= thresholds.max_ring_size,
        f"halka_boyutu({properties['max_ring_size']:.0f})",
    )
    check(properties["n_rings"] <= thresholds.max_rings, f"halka_sayisi({properties['n_rings']:.0f})")

    # --- yalnizca kati (Chemistry42 vekili) kumede ---
    if stringent:
        check(
            thresholds.min_molecular_weight
            <= properties["molecular_weight"]
            <= thresholds.max_molecular_weight,
            f"molekul_agirligi({properties['molecular_weight']:.1f})",
        )
        check(
            thresholds.min_logp <= properties["logp"] <= thresholds.max_logp,
            f"logp({properties['logp']:.2f})",
        )
        check(properties["hbd"] <= thresholds.max_hbd, f"hbd({properties['hbd']:.0f})")
        check(properties["hba"] <= thresholds.max_hba, f"hba({properties['hba']:.0f})")
        check(properties["tpsa"] <= thresholds.max_tpsa, f"tpsa({properties['tpsa']:.1f})")
        check(
            properties["rotatable_bonds"] <= thresholds.max_rotatable_bonds,
            f"donebilir_bag({properties['rotatable_bonds']:.0f})",
        )
        check(
            thresholds.min_aromatic_fraction
            <= properties["aromatic_fraction"]
            <= thresholds.max_aromatic_fraction,
            f"aromatik_fraksiyon({properties['aromatic_fraction']:.2f})",
        )
        check(
            properties["sp3_fraction"] >= thresholds.min_sp3_fraction,
            f"sp3_fraksiyon({properties['sp3_fraction']:.2f})",
        )
        check(
            properties["formal_charge"] <= thresholds.max_formal_charge,
            f"formal_yuk({properties['formal_charge']:.0f})",
        )

    return FilterResult(passed=not failures, failures=failures, properties=properties)


def passes_local_filter(smiles: str, thresholds: FilterThresholds | None = None) -> bool:
    """Tartarus "local filter" vekili (Extended Data Fig. 1a).

    Makale, QCBM'i epoch 20'ye kadar bu filtreyle odullendirir:
    "The reward value for each sample was computed at every step using the
     local filter until epoch 20, after which we selected Chemistry42."
    """
    return evaluate_filters(smiles, thresholds, stringent=False).passed


def passes_chemistry42_filter(smiles: str, thresholds: FilterThresholds | None = None) -> bool:
    """Chemistry42 filtre kumesi vekili (Extended Data Fig. 1b) - daha kati."""
    return evaluate_filters(smiles, thresholds, stringent=True).passed
