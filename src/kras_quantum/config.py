"""
Merkezi konfigurasyon.

Buradaki sayilarin tamami makaleden gelir:
  Vakili, Gorgulla, Snider, Nigam et al.,
  "Quantum-computing-enhanced algorithm unveils potential KRAS inhibitors",
  Nature Biotechnology 43, 1954-1959 (2025).
  doi:10.1038/s41587-024-02526-3

Her alanin yanindaki yorum, degerin makalede gectigi yeri gosterir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Proje yollari
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"


# --------------------------------------------------------------------------
# 1) Egitim verisi uretimi  (Makale: Fig. 1a, "Data acquisition and preprocessing")
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Fig. 1a: 650 + 850K + 250K = ~1.1M molekul."""

    # Literaturden derlenen, deneysel olarak dogrulanmis KRAS inhibitorleri.
    n_seed_inhibitors: int = 650           # Makale: "approximately 650 known KRAS inhibitors"

    # STONED-SELFIES ile uretilen analoglar.
    n_stoned_analogs: int = 850_000        # Makale: "added 850,000 molecules"
    stoned_max_mutations: int = 500        # Makale: "mutations ... up to 500 times"

    # Benzerlik kabuklari (Fig. 1a: "Similarity > 0.6" ve "Similarity > 0.4")
    similarity_inner_shell: float = 0.6
    similarity_outer_shell: float = 0.4

    # VirtualFlow 2.0 sanal tarama (Enamine REAL).
    n_virtual_screened: int = 100_000_000  # Makale: "screen 100 million molecules"
    n_virtual_hits: int = 250_000          # Makale: "top 250,000 with the best docking scores"

    # Docking kutusu (Makale, "Virtual screening process")
    docking_receptor_pdb: str = "5US4"     # Sanal tarama icin kullanilan yapi
    docking_box_size: tuple[float, float, float] = (14.0, 14.0, 20.0)  # Angstrom
    docking_exhaustiveness: int = 1        # QuickVina 2, her iki asamada da 1

    # Farmakofor referans yapisi (Makale, Extended Data Fig. 4)
    pharmacophore_pdb: str = "7EW9"        # TH-Z816 ko-kristal yapisi

    @property
    def total_training_molecules(self) -> int:
        """Makale: "a single dataset containing 1.1 million data points"."""
        return self.n_seed_inhibitors + self.n_stoned_analogs + self.n_virtual_hits


# --------------------------------------------------------------------------
# 2) Kuantum bilesen: QCBM  (Makale: "Quantum generative model: QCBM model")
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class QCBMConfig:
    """QCBM = Quantum Circuit Born Machine.

    Makale: "Our QCBM model was built with 16 qubits and four layers and we had
    96 parameters to optimize in total." + "We used linear topology for our project."

    NOT (kod/makale tutarsizligi): Extended Data Fig. 5'teki ansatz semasi
    (Rx - Rz - Rxx(linear) - Rz - Rx) 4 qubit icin tam 19 parametre verir
    (4 + 4 + 3 + 4 + 4). Ayni semayi 16 qubit'e olceklemek 96'dan farkli bir
    sayi verir. Bu yuzden ansatz'i konfigure edilebilir biraktik ve
    `QCBMAnsatz.num_parameters` degerini calistirma aninda raporluyoruz.
    Ayrintili tartisma icin README'deki "Makaleden sapmalar" bolumune bakin.
    """

    n_qubits: int = 16                     # Makale: 16-qubit IBM islemcisi (Guadalupe, Falcon r4P)
    n_layers: int = 4                      # Makale: "four layers"
    entanglement: str = "linear"           # Makale: "We used linear topology"

    optimizer: str = "COBYLA"              # Makale: "we used COBYLA for our optimizer"
    max_iterations: int = 60               # COBYLA'nin epoch basina adim sayisi
    n_epochs: int = 30                     # Makale: "QCBM model ... spanning 30 epochs"

    shots: int = 4096                      # Her epoch'ta donanimdan alinan olcum sayisi
    seed: int = 1234


# --------------------------------------------------------------------------
# 3) Klasik bilesen: LSTM  (Makale: "Classical model: LSTM model")
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LSTMConfig:
    """SELFIES token dizileri uzerinde calisan uretici LSTM.

    Kuantum prior'u LSTM hucresine giris seviyesinde eklenir:
        x'_t = X(i) ++ x_t   (concat)   VEYA   x'_t = X(i) + x_t   (add)
    Makale, Denklem (9) ve Extended Data Fig. 3a.
    """

    embedding_dim: int = 64                # Optuna ile taranan hiperparametre
    hidden_dim: int = 256                  # Optuna ile taranan hiperparametre
    n_layers: int = 2                      # Optuna ile taranan hiperparametre
    dropout: float = 0.2                   # Makale: "regularization techniques such as dropout"

    prior_merge: str = "concat"            # "concat" | "add"  (Makale, Denklem 9)

    batch_size: int = 256
    learning_rate: float = 1e-3            # Adam (Makale: "trained using the Adam optimizer")
    n_epochs: int = 40                     # Makale: "LSTM model was trained over a total of 40 epochs"
    max_sequence_length: int = 72          # SELFIES token cinsinden ust sinir

    sampling_temperature: float = 1.0      # Makale: "we tuned the sampling temperature"
    seed: int = 1234


# --------------------------------------------------------------------------
# 4) Odul / filtre  (Makale: "Experimental validation methods")
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardConfig:
    """Chemistry42 tescilli bir platformdur; burada acik kaynak bir vekil
    (surrogate) skorlayici kullaniyoruz. Esikler makaledeki sentez
    kriterlerinden birebir alinmistir.
    """

    # Makale: "The reward value ... was computed at every step using the local
    # filter until epoch 20, after which we selected Chemistry42."
    local_filter_until_epoch: int = 20

    # Sentez adayligi icin gereken esikler (Makale, "For synthesis candidacy"):
    min_total_reward: float = 0.7          # "aggregate reward value exceeding 0.7"
    max_pli_score: float = -8.0            # "PLI score ... (< -8 kcal/mol)"
    min_pharmacophore_match: float = 0.7   # "pharmacophore match score above 0.7"
    max_rersa_score: float = 5.0           # "ReRSA score under 5"

    # Odul bilesenlerinin agirliklari (makalede acikca verilmemistir;
    # tiered kriter listesini yansitacak sekilde secildi).
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "filters": 0.30,          # yapisal/med-chem filtreleri
            "pli": 0.30,              # protein-ligand etkilesim skoru
            "pharmacophore": 0.20,    # 3B farmakofor eslesmesi
            "synthetic_access": 0.20, # ReRSA vekili (SA score)
        }
    )


# --------------------------------------------------------------------------
# 5) Uretim ve secim  (Makale: Fig. 1d)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CampaignConfig:
    """Fig. 1d: uc modelin her birinden 1M bilesik -> 15 sentez adayi."""

    n_samples_per_model: int = 1_000_000   # Makale: "1M compounds" (her kol icin)
    n_selected_total: int = 15             # Makale: "15 promising compounds were selected"

    # Fig. 1d'deki kol basina secim dagilimi
    n_selected_classical: int = 3          # "Selected 3 compounds"  (Classical / vanilla LSTM)
    n_selected_quantum_hw: int = 8         # "Selected 8 compounds"  (Quantum HW)
    n_selected_quantum_sim: int = 4        # "Selected 4 compounds"  (Quantum SIM)

    # Benchmark (Extended Data Fig. 1)
    benchmark_repeats: int = 5             # "we repeated the experiments five times"
    benchmark_samples_fig_a: int = 1_000   # "sampling n=1,000 compounds"
    benchmark_samples_fig_c: int = 5_000   # "sampling n=5,000 compounds"
    qubit_sweep: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16)  # Extended Data Fig. 1c


# --------------------------------------------------------------------------
# Tek giris noktasi
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    qcbm: QCBMConfig = field(default_factory=QCBMConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    campaign: CampaignConfig = field(default_factory=CampaignConfig)


DEFAULT_CONFIG = Config()


__all__ = [
    "PROJECT_ROOT",
    "RESULTS_DIR",
    "DATA_CACHE_DIR",
    "DataConfig",
    "QCBMConfig",
    "LSTMConfig",
    "RewardConfig",
    "CampaignConfig",
    "Config",
    "DEFAULT_CONFIG",
]
