"""
PLI (protein-ligand etkilesim) skorlamasi.

Makale referansi
----------------
    "We used Chemistry42 to screen these samples for pharmacological viability
     and ranked them on the basis of their docking scores (protein-ligand
     interaction (PLI) score)."

    Sentez adayligi esigi: "A PLI score indicative of strong KRAS binding
     (< -8 kcal/mol)."

    Sanal taramada: "The docking program used was QuickVina 2, with
     exhaustiveness set to 1 in both stages of the screen."

Iki skorlayici sunulur
----------------------
VinaDocking
    GERCEK molekuler docking (AutoDock Vina). QuickVina 2, Vina'nin
    hizlandirilmis bir catalidir; ayni skorlama fonksiyonunu kullanir,
    dolayisiyla uretilen degerler makaledeki kcal/mol esikleriyle
    dogrudan kiyaslanabilir.
    Maliyet: molekul basina ~0.5-5 saniye. 1M molekul icin uygun degildir.

SurrogatePLI
    Hizli vekil skorlayici. Docking YAPMAZ; ligand verimliligi ve
    fizikokimyasal tamamlayicilik uzerinden bir tahmin uretir.
    Egitim dongusunde (her epoch binlerce molekul) kullanilir.

Bu iki kademeli yapi, makalenin kendi akisini yansitir: ucuz bir odul
sinyaliyle egitim, pahali ve guvenilir bir skorla nihai siralama.

ONEMLI: SurrogatePLI'nin urettigi degerler kcal/mol birimindeymis gibi
olceklenmistir ancak GERCEK DOCKING DEGILDIR. Nihai bilesik secimi daima
VinaDocking ile yapilmalidir.
"""

from __future__ import annotations

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from kras_quantum.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "PLIScorer",
    "SurrogatePLI",
    "VinaDocking",
    "DockingBox",
    "load_receptor_config",
    "get_pli_scorer",
]

DEFAULT_RECEPTOR_CONFIG = PROJECT_ROOT / "data" / "receptor" / "receptor_config.json"

# Docking basarisiz oldugunda dondurulecek deger: kotu ama sonlu.
# Sonsuz kullanmiyoruz cunku odul hesabinda NaN yayilmasina yol acar.
FAILED_SCORE = 0.0


@dataclass(frozen=True)
class DockingBox:
    """Docking arama kutusu (merkez ve kenar uzunluklari, Angstrom)."""

    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float

    @property
    def center(self) -> list[float]:
        return [self.center_x, self.center_y, self.center_z]

    @property
    def size(self) -> list[float]:
        return [self.size_x, self.size_y, self.size_z]


def load_receptor_config(path: str | Path | None = None) -> tuple[Path, DockingBox]:
    """scripts/01_prepare_receptor.py tarafindan yazilan yapilandirmayi okur."""
    path = Path(path) if path is not None else DEFAULT_RECEPTOR_CONFIG

    if not path.exists():
        raise FileNotFoundError(
            f"Reseptor yapilandirmasi bulunamadi: {path}\n"
            f"Once reseptoru hazirlayin:\n"
            f"    python scripts/01_prepare_receptor.py"
        )

    config = json.loads(path.read_text(encoding="utf-8"))
    receptor = PROJECT_ROOT / config["receptor_pdbqt"]

    if not receptor.exists():
        raise FileNotFoundError(
            f"Reseptor PDBQT bulunamadi: {receptor}\n"
            f"scripts/01_prepare_receptor.py yeniden calistirilmali."
        )

    box = DockingBox(
        center_x=config["center"]["x"],
        center_y=config["center"]["y"],
        center_z=config["center"]["z"],
        size_x=config["size"]["x"],
        size_y=config["size"]["y"],
        size_z=config["size"]["z"],
    )
    return receptor, box


# ---------------------------------------------------------------------------
# Ortak arayuz
# ---------------------------------------------------------------------------
class PLIScorer(ABC):
    """PLI skorlayici arayuzu.

    Sozlesme: daha NEGATIF skor = daha iyi baglanma (kcal/mol gelenegi).
    Basarisizlik durumunda FAILED_SCORE (0.0) donulur.
    """

    name: str = "abstract"
    is_real_docking: bool = False

    @abstractmethod
    def score(self, smiles: str) -> float:
        """Tek molekul icin PLI skoru (kcal/mol)."""

    def score_batch(self, smiles_list: list[str]) -> list[float]:
        """Varsayilan: tek tek skorla. Alt siniflar toplu hale getirebilir."""
        return [self.score(s) for s in smiles_list]

    def __repr__(self) -> str:  # pragma: no cover
        tur = "gercek docking" if self.is_real_docking else "vekil"
        return f"<{type(self).__name__} name={self.name!r} tur={tur}>"


# ---------------------------------------------------------------------------
# Hizli vekil skorlayici
# ---------------------------------------------------------------------------
class SurrogatePLI(PLIScorer):
    """Docking YAPMAYAN hizli tahmin edici.

    Gerekce
    -------
    Makalenin egitim dongusu her epoch'ta binlerce molekule odul atar.
    Gercek docking bu olcekte uygulanamaz (molekul basina saniyeler).
    Makale bu islevi Chemistry42'nin PLI skoruna yaptirir; Chemistry42
    tescilli oldugu ve yayinlanmis bir denklemi bulunmadigi icin burada
    acik, belgelenmis bir vekil kullaniyoruz.

    Model
    -----
    Temel: ligand verimliligi (LE). Deneysel olarak, iyi optimize edilmis
    ilac benzeri ligandlarda LE ~ 0.3 kcal/mol/agir atom civarindadir
    (Hopkins et al.). Buradan:

        taban_skor = -LE_ref * agir_atom_sayisi

    Uzerine, baglanma cebiyle tamamlayiciligi yakalayan carpanlar eklenir:
      + aromatik halkalar   -> Switch-II cebi hidrofobiktir, istiflenme katkisi
      + hidrofobik uyum     -> logP'nin optimum araliga yakinligi
      + H-bag kapasitesi    -> yonlu etkilesimler (asiri polarlik cezalandirilir)
      - donebilir bag cezasi-> baglanmada kaybedilen konformasyonel entropi

    SINIR: Bu bir enerji hesabi DEGILDIR. Molekullerin goreli siralamasi
    icin tasarlanmistir; mutlak degerleri deneysel afiniteye karsilik gelmez.
    Nihai secim daima VinaDocking ile yapilmalidir.
    """

    name = "surrogate"
    is_real_docking = False

    # Referans ligand verimliligi (kcal/mol / agir atom)
    LE_REFERENCE = 0.32
    # Switch-II cebi icin optimum lipofilisite araligi
    OPTIMAL_LOGP = 3.5
    LOGP_TOLERANCE = 2.0

    def score(self, smiles: str) -> float:
        from kras_quantum.scoring.filters import compute_properties

        properties = compute_properties(smiles)
        if properties is None:
            return FAILED_SCORE

        heavy_atoms = properties["heavy_atoms"]
        if heavy_atoms < 1:
            return FAILED_SCORE

        # --- taban: ligand verimliligi ---
        # Buyuk molekuller daha cok temas kurar, ancak katki dogrusal degildir;
        # karekok sonmesi, asiri buyuk molekullerin skoru sismesini engeller.
        base = -self.LE_REFERENCE * heavy_atoms * (1.0 - 0.004 * max(heavy_atoms - 25, 0))

        # --- aromatik istiflenme (Switch-II cebi hidrofobik) ---
        aromatic_bonus = -0.45 * properties["aromatic_fraction"] * math.sqrt(heavy_atoms)

        # --- lipofilik tamamlayicilik ---
        logp_deviation = abs(properties["logp"] - self.OPTIMAL_LOGP) / self.LOGP_TOLERANCE
        lipophilic_term = 1.2 * min(logp_deviation**2, 4.0)  # pozitif = ceza

        # --- yonlu H-baglari (asiri polarlik cezalandirilir) ---
        hbond_capacity = min(properties["hbd"], 3) + min(properties["hba"], 6)
        hbond_bonus = -0.25 * hbond_capacity
        polarity_penalty = 0.015 * max(properties["tpsa"] - 120.0, 0.0)

        # --- konformasyonel entropi kaybi ---
        rotatable_penalty = 0.20 * properties["rotatable_bonds"]

        score = (
            base
            + aromatic_bonus
            + hbond_bonus
            + lipophilic_term
            + polarity_penalty
            + rotatable_penalty
        )

        # Fiziksel olarak makul araliga kelepcele
        return float(max(min(score, 0.0), -16.0))


# ---------------------------------------------------------------------------
# Gercek docking
# ---------------------------------------------------------------------------
class VinaDocking(PLIScorer):
    """AutoDock Vina ile gercek molekuler docking.

    Makale QuickVina 2 kullanir (kaynak 48); QuickVina 2, AutoDock Vina'nin
    arama heuristigini hizlandiran bir catalidir ve AYNI skorlama
    fonksiyonunu kullanir. Dolayisiyla uretilen kcal/mol degerleri
    makaledeki "< -8 kcal/mol" esigiyle dogrudan kiyaslanabilir.

    Parameters
    ----------
    exhaustiveness
        Arama yogunlugu. Makale sanal taramada 1 kullanir (100M molekul
        tarandigi icin); nihai siralama icin 8-16 onerilir.
    n_poses
        Uretilecek poz sayisi; en iyi pozun skoru dondurulur.
    """

    name = "vina"
    is_real_docking = True

    def __init__(
        self,
        receptor_pdbqt: str | Path | None = None,
        box: DockingBox | None = None,
        exhaustiveness: int = 8,
        n_poses: int = 9,
        seed: int = 1234,
        n_cpu: int = 0,  # 0 = tum cekirdekler
        embed_seed: int = 42,
    ) -> None:
        if receptor_pdbqt is None or box is None:
            loaded_receptor, loaded_box = load_receptor_config()
            receptor_pdbqt = receptor_pdbqt or loaded_receptor
            box = box or loaded_box

        self.receptor_pdbqt = Path(receptor_pdbqt)
        self.box = box
        self.exhaustiveness = exhaustiveness
        self.n_poses = n_poses
        self.seed = seed
        self.n_cpu = n_cpu
        self.embed_seed = embed_seed
        self._vina = None

    # -- kurulum -------------------------------------------------------
    @property
    def vina(self):
        """Vina nesnesini tembel kurar (reseptor ve kutu bir kez yuklenir)."""
        if self._vina is None:
            from vina import Vina

            engine = Vina(sf_name="vina", cpu=self.n_cpu, seed=self.seed, verbosity=0)
            engine.set_receptor(str(self.receptor_pdbqt))
            engine.compute_vina_maps(center=self.box.center, box_size=self.box.size)
            self._vina = engine
            logger.info(
                "Vina hazir: reseptor=%s, merkez=%s, kutu=%s",
                self.receptor_pdbqt.name,
                [round(c, 2) for c in self.box.center],
                [round(s, 1) for s in self.box.size],
            )
        return self._vina

    # -- ligand hazirlama ----------------------------------------------
    def prepare_ligand_pdbqt(self, smiles: str) -> str | None:
        """SMILES -> 3B konformer -> PDBQT dizesi.

        Adimlar:
          1. Hidrojen ekle (docking icin acik hidrojen gerekli)
          2. ETKDG ile 3B konformer uret
          3. MMFF94 ile geometriyi optimize et
          4. Meeko ile PDBQT'ye cevir (donebilir baglari isaretler)
        """
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem

        RDLogger.DisableLog("rdApp.*")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        params = AllChem.ETKDGv3()
        params.randomSeed = self.embed_seed
        if AllChem.EmbedMolecule(mol, params) != 0:
            # Rastgele koordinatlarla yeniden dene (zor halka sistemleri)
            params.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params) != 0:
                logger.debug("3B konformer uretilemedi: %s", smiles[:60])
                return None

        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=400)
        except Exception:
            pass  # optimizasyon basarisiz olsa da ham konformer kullanilabilir

        try:
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            setups = MoleculePreparation().prepare(mol)
            if not setups:
                return None
            pdbqt, success, error = PDBQTWriterLegacy.write_string(setups[0])
            if not success:
                logger.debug("Meeko PDBQT basarisiz (%s): %s", smiles[:40], error)
                return None
            return pdbqt
        except Exception as exc:
            logger.debug("Ligand hazirlama hatasi (%s): %s", smiles[:40], exc)
            return None

    # -- skorlama ------------------------------------------------------
    def score(self, smiles: str) -> float:
        """Molekulu reseptore yerlestirir ve en iyi pozun skorunu dondurur."""
        pdbqt = self.prepare_ligand_pdbqt(smiles)
        if pdbqt is None:
            return FAILED_SCORE

        try:
            engine = self.vina
            engine.set_ligand_from_string(pdbqt)
            engine.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)
            energies = engine.energies(n_poses=1)
            if len(energies) == 0:
                return FAILED_SCORE
            return float(energies[0][0])  # ilk sutun: toplam baglanma enerjisi
        except Exception as exc:
            logger.debug("Docking basarisiz (%s): %s", smiles[:40], exc)
            return FAILED_SCORE

    def dock_to_file(self, smiles: str, out_path: str | Path) -> float:
        """Docking yapar ve pozlari PDBQT dosyasina yazar (gorsellestirme icin)."""
        pdbqt = self.prepare_ligand_pdbqt(smiles)
        if pdbqt is None:
            return FAILED_SCORE

        engine = self.vina
        engine.set_ligand_from_string(pdbqt)
        engine.dock(exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        engine.write_poses(str(out_path), n_poses=self.n_poses, overwrite=True)

        return float(engine.energies(n_poses=1)[0][0])


# ---------------------------------------------------------------------------
# Fabrika
# ---------------------------------------------------------------------------
def get_pli_scorer(kind: str = "surrogate", **kwargs) -> PLIScorer:
    """PLI skorlayici olusturur.

    kind
      "surrogate" -> hizli vekil (egitim dongusu icin)
      "vina"      -> gercek docking (nihai siralama icin)
      "auto"      -> reseptor hazirsa vina, degilse vekil
    """
    kind = kind.lower()

    if kind == "auto":
        try:
            load_receptor_config()
            kind = "vina"
        except FileNotFoundError:
            logger.info("Reseptor hazir degil; vekil PLI skorlayiciya duseldu.")
            kind = "surrogate"

    if kind == "surrogate":
        return SurrogatePLI()
    if kind == "vina":
        return VinaDocking(**kwargs)
    raise ValueError(f"Bilinmeyen PLI skorlayici: {kind!r} ('surrogate'|'vina'|'auto')")
