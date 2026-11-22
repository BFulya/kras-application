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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kras_quantum.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

__all__ = [
    "PLIScorer",
    "SurrogatePLI",
    "VinaDocking",
    "DockingBox",
    "load_receptor_config",
    "get_pli_scorer",
    "SURROGATE_MODEL_PATH",
    "SURROGATE_FEATURES",
]

DEFAULT_RECEPTOR_CONFIG = PROJECT_ROOT / "data" / "receptor" / "receptor_config.json"

# Vekil modelin dosya yolu ve kullandigi tanimlayicilar.
# Model, scripts/05_fit_surrogate.py tarafindan gercek docking verisinden
# ogrenilir (bkz. SurrogatePLI aciklamasi).
SURROGATE_MODEL_PATH = PROJECT_ROOT / "data" / "surrogate_pli_model.json"

SURROGATE_FEATURES: tuple[str, ...] = (
    "heavy_atoms",
    "molecular_weight",
    "logp",
    "hba",
    "hbd",
    "tpsa",
    "rotatable_bonds",
    "aromatic_fraction",
    "sp3_fraction",
    "n_rings",
    "max_ring_size",
)

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
    """Docking YAPMAYAN hizli tahmin edici (ogrenilmis model).

    Gerekce
    -------
    Makalenin egitim dongusu her epoch'ta yuzlerce molekule odul atar.
    Gercek docking bu olcekte uygulanamaz (molekul basina ~10-18 saniye).
    Makale bu islevi Chemistry42'nin PLI skoruna yaptirir; Chemistry42
    tescilli ve yayinlanmis bir denklemi olmadigi icin burada acik,
    belgelenmis bir vekil kullaniyoruz.

    Neden elle yazilmis formul degil?
    ---------------------------------
    Ilk uygulama ligand verimliligi ve fizikokimyasal terimlerden ELLE
    kuruldu. Gercek docking'e karsi olculdugunde ise yaramadigi gorulde:

        ilk kalibrasyon seti (34 molekul) : Spearman -0.396  (TERS korele!)
        daha buyuk set      (65 molekul) : Spearman +0.032  (korelasyon yok)

    Teshis: formul molekul boyutunu odullendiriyordu, oysa dar Switch-II
    cebinde buyuk molekuller daha KOTU skor aliyor. Sezgiye dayali bir
    fizikokimyasal formul bu cebin davranisini yakalayamiyor.

    Bu yuzden vekil artik GERCEK DOCKING VERISINDEN OGRENILIR:
    standartlastirilmis moleküler tanimlayicilar uzerinde ridge regresyonu
    (scripts/05_fit_surrogate.py). Performans capraz dogrulama ile olculur
    ve model dosyasinda saklanir.

    SINIRLAR
    --------
    * Bu bir enerji hesabi DEGILDIR; amac goreli siralamadir.
    * Model, uzerinde egitildigi resepture (PDB 7EW9, Switch-II cebi)
      ozgudur. Baska bir hedef icin scripts/02 ve 05 yeniden calistirilmali.
    * Nihai bilesik secimi DAIMA VinaDocking ile yapilmalidir.
    """

    name = "surrogate"
    is_real_docking = False

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else SURROGATE_MODEL_PATH
        self._model: dict | None = None
        self._warned = False

    # ------------------------------------------------------------------
    @property
    def model(self) -> dict | None:
        """Ogrenilmis modeli tembel yukler."""
        if self._model is None and self.model_path.exists():
            self._model = json.loads(self.model_path.read_text(encoding="utf-8"))
            logger.info(
                "Vekil PLI modeli yuklendi: CV Spearman=%+.3f, %d molekulde egitildi",
                self._model.get("cv_spearman", float("nan")),
                self._model.get("n_training_molecules", 0),
            )
        return self._model

    @property
    def is_calibrated(self) -> bool:
        return self.model is not None

    @property
    def cv_spearman(self) -> float | None:
        """Modelin capraz dogrulanmis sira korelasyonu (varsa)."""
        model = self.model
        return model.get("cv_spearman") if model else None

    # ------------------------------------------------------------------
    def score(self, smiles: str) -> float:
        from kras_quantum.scoring.filters import compute_properties

        model = self.model
        if model is None:
            if not self._warned:
                logger.warning(
                    "Vekil PLI modeli bulunamadi (%s). PLI bileseni 0 donecek, "
                    "yani odulde etkisiz kalacak. Modeli uretmek icin:\n"
                    "    python scripts/02_calibrate_surrogate.py --n-molecules 150 "
                    "--out results/calibration_set.json\n"
                    "    python scripts/05_fit_surrogate.py",
                    self.model_path,
                )
                self._warned = True
            return FAILED_SCORE

        properties = compute_properties(smiles)
        if properties is None:
            return FAILED_SCORE

        # --- uygulanabilirlik alani (applicability domain) ---
        # Ogrenilmis model yalnizca egitim verisine benzeyen molekullerde
        # anlamlidir. Alan disinda ekstrapolasyon yapar ve yaniltici degerler
        # uretir: ornegin benzen (6 agir atom, aromatik fraksiyon 1.0) model
        # tarafindan cok iyi skorlanir, oysa gercek docking -4.64 verir.
        # Bu yuzden alan disindaki molekulleri skorlamayi reddediyoruz.
        domain = model.get("applicability_domain")
        if domain is not None:
            heavy_atoms = properties["heavy_atoms"]
            tolerance = domain.get("tolerance", 0.25)
            low = domain["heavy_atoms_min"] * (1.0 - tolerance)
            high = domain["heavy_atoms_max"] * (1.0 + tolerance)

            if not (low <= heavy_atoms <= high):
                logger.debug(
                    "Molekul uygulanabilirlik alani disinda (%d agir atom, "
                    "gecerli aralik %.0f-%.0f): %s",
                    heavy_atoms,
                    low,
                    high,
                    smiles[:50],
                )
                return FAILED_SCORE

        features = np.array(
            [properties[name] for name in model["features"]], dtype=float
        )
        mean = np.array(model["feature_mean"], dtype=float)
        std = np.array(model["feature_std"], dtype=float)
        weights = np.array(model["weights"], dtype=float)

        standardized = (features - mean) / np.where(std == 0, 1.0, std)
        score = float(model["intercept"] + standardized @ weights)

        # Egitim araliginin cok disina tasan tahminleri kelepcele: ridge
        # modeli aykiri tanimlayicilarda fiziksel olmayan degerler uretebilir.
        low, high = model.get("target_range", [-16.0, 0.0])
        margin = 2.0
        return float(np.clip(score, low - margin, min(high + margin, 0.0)))


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
