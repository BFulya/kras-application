"""
Hibrit hat testleri (Makale, Algorithm 1 ve Fig. 1d).

Testler kucuk yapilandirmalarla calisir (4-6 qubit, kisa dongu); makaledeki
16 qubit / 30 epoch kosusu tam deney betikleriyle yapilir.
"""

from __future__ import annotations

import numpy as np
import pytest

from kras_quantum.config import (
    CampaignConfig,
    Config,
    LSTMConfig,
    QCBMConfig,
    RewardConfig,
)
from kras_quantum.data import load_seed_library, smiles_to_selfies
from kras_quantum.data.tokenizer import SelfiesTokenizer
from kras_quantum.pipeline import (
    ARMS,
    HybridGenerator,
    cluster_by_similarity,
    compute_novelty,
    select_candidates,
)
from kras_quantum.scoring import RewardFunction

SMALL_CONFIG = Config(
    qcbm=QCBMConfig(n_qubits=4, n_layers=1, shots=256, seed=5, max_iterations=15),
    lstm=LSTMConfig(
        embedding_dim=16,
        hidden_dim=32,
        n_layers=1,
        dropout=0.0,
        batch_size=16,
        max_sequence_length=24,
    ),
)


@pytest.fixture(scope="module")
def training_data():
    """Kucuk ama gercek bir egitim seti: cekirdek inhibitorler."""
    compounds = load_seed_library()
    selfies_list = [smiles_to_selfies(c.smiles) for c in compounds]
    selfies_list = [s for s in selfies_list if s]

    tokenizer = SelfiesTokenizer.from_corpus(selfies_list)
    sequences = tokenizer.encode_batch(selfies_list, max_length=24)
    return tokenizer, sequences, [c.smiles for c in compounds]


# ---------------------------------------------------------------------------
# Kurulum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", ARMS[:2])  # classical, quantum_sim (hw gercek cihaz ister)
def test_kollar_kurulabilir(training_data, arm):
    tokenizer, _, _ = training_data
    gen = HybridGenerator(arm=arm, tokenizer=tokenizer, config=SMALL_CONFIG)

    assert gen.arm == arm
    assert (gen.qcbm is not None) == (arm != "classical")


def test_klasik_kolda_kuantum_bilesen_yok(training_data):
    """Makale'nin "vanilla LSTM" kontrolu."""
    tokenizer, _, _ = training_data
    gen = HybridGenerator(arm="classical", tokenizer=tokenizer, config=SMALL_CONFIG)

    assert gen.qcbm is None
    assert not gen.is_quantum
    assert gen.lstm.prior_dim == 0


def test_kuantum_kolunda_qcbm_dogru_boyutta(training_data):
    tokenizer, _, _ = training_data
    gen = HybridGenerator(arm="quantum_sim", tokenizer=tokenizer, config=SMALL_CONFIG)

    assert gen.qcbm is not None
    assert gen.qcbm.n_qubits == 4
    assert gen.lstm.prior_dim == 4


def test_gecersiz_kol_reddedilir(training_data):
    tokenizer, _, _ = training_data
    with pytest.raises(ValueError, match="arm"):
        HybridGenerator(arm="kuantum", tokenizer=tokenizer, config=SMALL_CONFIG)


# ---------------------------------------------------------------------------
# Uretim
# ---------------------------------------------------------------------------
def test_klasik_kol_molekul_uretir(training_data):
    tokenizer, _, _ = training_data
    gen = HybridGenerator(arm="classical", tokenizer=tokenizer, config=SMALL_CONFIG)

    sonuc = gen.generate(12)

    assert len(sonuc) == 12
    assert len(sonuc.rewards) == 12
    assert np.all(sonuc.rewards >= 0)


def test_kuantum_kol_prior_izini_korur(training_data):
    """QCBM hedef dagilimi icin her molekulun prior'i bilinmeli."""
    tokenizer, _, _ = training_data
    gen = HybridGenerator(arm="quantum_sim", tokenizer=tokenizer, config=SMALL_CONFIG)

    sonuc = gen.generate(10)

    assert len(sonuc.bitstrings) == 10
    assert sonuc.bitstrings.min() >= 0
    assert sonuc.bitstrings.max() < 2**4  # 4 qubit


def test_uretim_parcali_calisir(training_data):
    tokenizer, _, _ = training_data
    gen = HybridGenerator(arm="classical", tokenizer=tokenizer, config=SMALL_CONFIG)

    sonuc = gen.generate(20, batch_size=7)  # parca boyutu bolunmuyor

    assert len(sonuc) == 20


# ---------------------------------------------------------------------------
# Egitim dongusu (Algorithm 1)
# ---------------------------------------------------------------------------
def test_klasik_dongu_calisir(training_data):
    tokenizer, sequences, _ = training_data
    gen = HybridGenerator(arm="classical", tokenizer=tokenizer, config=SMALL_CONFIG)

    history = gen.fit(
        sequences, n_epochs=3, n_generate_per_epoch=16, lstm_epochs_per_cycle=1,
        verbose=False,
    )

    assert len(history) == 3
    assert all(r.n_generated == 16 for r in history)


def test_kuantum_dongu_calisir_ve_qcbm_egitilir(training_data):
    """Algorithm 1'in tamami: QCBM -> LSTM -> odul -> QCBM."""
    tokenizer, sequences, _ = training_data
    gen = HybridGenerator(arm="quantum_sim", tokenizer=tokenizer, config=SMALL_CONFIG)

    history = gen.fit(
        sequences, n_epochs=3, n_generate_per_epoch=16, lstm_epochs_per_cycle=1,
        verbose=False,
    )

    assert len(history) == 3
    # En az bir epoch'ta QCBM egitilmis olmali (gecerli molekul uretildiyse)
    egitilen = [r for r in history if r.qcbm_loss_before is not None]
    assert egitilen, "QCBM hic egitilmedi"

    for record in egitilen:
        assert record.qcbm_loss_after <= record.qcbm_loss_before + 1e-6


def test_odul_asamasi_epoch_20de_degisir(training_data):
    """Makale: "using the local filter until epoch 20, after which Chemistry42"."""
    tokenizer, sequences, _ = training_data
    gen = HybridGenerator(arm="classical", tokenizer=tokenizer, config=SMALL_CONFIG)

    # Epoch 19 -> local filter, epoch 20 -> tam odul
    gen.reward.set_epoch(19)
    assert gen.reward.stage_name == "local_filter"
    gen.reward.set_epoch(20)
    assert gen.reward.stage_name == "chemistry42_proxy"


def test_epoch_kaydi_basari_orani_hesaplar(training_data):
    """Makalenin "success rate" metrigi (Extended Data Fig. 1)."""
    tokenizer, sequences, _ = training_data
    gen = HybridGenerator(arm="classical", tokenizer=tokenizer, config=SMALL_CONFIG)

    history = gen.fit(
        sequences, n_epochs=2, n_generate_per_epoch=16, lstm_epochs_per_cycle=1,
        verbose=False,
    )

    for record in history:
        assert 0.0 <= record.validity_rate <= 1.0
        assert record.n_valid <= record.n_generated


# ---------------------------------------------------------------------------
# Yapisal yenilik
# ---------------------------------------------------------------------------
def test_ayni_molekul_sifir_yenilik_alir():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    assert compute_novelty(aspirin, [aspirin]) == pytest.approx(0.0, abs=1e-6)


def test_ilgisiz_molekul_yuksek_yenilik_alir():
    yenilik = compute_novelty("CCO", ["CC(=O)Oc1ccccc1C(=O)O"])
    assert yenilik > 0.5


def test_bos_referans_kumesi_tam_yenilik_verir():
    assert compute_novelty("CCO", []) == 1.0


# ---------------------------------------------------------------------------
# Kumeleme
# ---------------------------------------------------------------------------
def test_ayni_molekuller_ayni_kumeye_duser():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    kumeler = cluster_by_similarity([aspirin, aspirin, aspirin])
    assert len(set(kumeler)) == 1


def test_farkli_molekuller_ayri_kumelere_duser():
    kumeler = cluster_by_similarity(["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"])
    assert len(set(kumeler)) >= 2


def test_kumeleme_her_molekule_kimlik_atar():
    molekuller = [c.smiles for c in load_seed_library()[:8]]
    kumeler = cluster_by_similarity(molekuller)

    assert len(kumeler) == len(molekuller)
    assert all(k >= 0 for k in kumeler)


def test_dusuk_esik_daha_az_kume_uretir():
    molekuller = [c.smiles for c in load_seed_library()]
    gevsek = len(set(cluster_by_similarity(molekuller, similarity_threshold=0.3)))
    siki = len(set(cluster_by_similarity(molekuller, similarity_threshold=0.9)))
    assert gevsek <= siki


# ---------------------------------------------------------------------------
# Secim akisi (Fig. 1d)
# ---------------------------------------------------------------------------
def test_secim_kol_kotalarina_uyar(training_data):
    """Makale, Fig. 1d: klasik 3, kuantum-HW 8, kuantum-SIM 4 = 15."""
    _, _, training_smiles = training_data

    # Gercek inhibitorleri aday havuzu olarak kullan: kriterleri gecmeleri
    # beklenir, boylece kota mantigi test edilebilir.
    compounds = [c.smiles for c in load_seed_library()]
    havuz = {
        "classical": compounds,
        "quantum_hw": compounds,
        "quantum_sim": compounds,
    }

    odul = RewardFunction()
    odul.set_epoch(25)

    secilen, rapor = select_candidates(
        havuz,
        odul,
        training_smiles=None,     # yenilik filtresini devre disi birak
        min_novelty=0.0,
        verbose=False,
    )

    campaign = CampaignConfig()
    assert rapor.per_arm.get("classical", 0) <= campaign.n_selected_classical
    assert rapor.per_arm.get("quantum_hw", 0) <= campaign.n_selected_quantum_hw
    assert rapor.per_arm.get("quantum_sim", 0) <= campaign.n_selected_quantum_sim
    assert len(secilen) <= campaign.n_selected_total


def test_secim_raporu_huni_sayilarini_izler():
    compounds = [c.smiles for c in load_seed_library()]
    odul = RewardFunction()
    odul.set_epoch(25)

    _, rapor = select_candidates(
        {"classical": compounds}, odul, min_novelty=0.0, verbose=False
    )

    # Huni daralmali: her adim bir oncekinden kucuk veya esit
    assert rapor.n_after_validity <= rapor.n_input
    assert rapor.n_after_filters <= rapor.n_after_validity
    assert rapor.n_after_pharmacophore <= rapor.n_after_filters
    assert rapor.n_after_pli <= rapor.n_after_pharmacophore
    assert rapor.n_after_synthetic <= rapor.n_after_novelty


def test_secim_cop_molekulleri_eler():
    odul = RewardFunction()
    odul.set_epoch(25)

    secilen, rapor = select_candidates(
        {"classical": ["CCO", "C", "CC", "!!!gecersiz", ""]},
        odul,
        min_novelty=0.0,
        verbose=False,
    )

    assert secilen == []
    assert rapor.n_after_filters == 0


def test_secim_yinelenen_molekulleri_tekillestirir():
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    odul = RewardFunction()
    odul.set_epoch(25)

    _, rapor = select_candidates(
        {"classical": [aspirin] * 10}, odul, min_novelty=0.0, verbose=False
    )

    assert rapor.n_after_validity == 1


def test_yenilik_esigi_egitim_setindekileri_eler():
    """Egitim setinin kopyalari "yeni kemotip" sayilmaz."""
    compounds = [c.smiles for c in load_seed_library()]
    odul = RewardFunction()
    odul.set_epoch(25)

    _, rapor = select_candidates(
        {"classical": compounds},
        odul,
        training_smiles=compounds,   # havuz = egitim seti
        min_novelty=0.4,
        verbose=False,
    )

    assert rapor.n_after_novelty == 0, "Egitim setinin kopyalari yeni sayildi"
