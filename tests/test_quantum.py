"""QCBM ve ansatz testleri.

Testler kucuk qubit sayilariyla calisir (hizli olsun diye); makaledeki
16 qubit yapilandirmasi yalnizca parametre sayimi duzeyinde dogrulanir.
"""

from __future__ import annotations

import numpy as np
import pytest

from kras_quantum.config import QCBMConfig
from kras_quantum.quantum import (
    QCBM,
    AnsatzSpec,
    TargetDistribution,
    build_ansatz,
    count_parameters,
    softmax,
)


# ---------------------------------------------------------------------------
# Ansatz
# ---------------------------------------------------------------------------
def test_fig5_ansatz_4qubit_3layer_sema_ile_ayni_parametre_sayisi():
    """Extended Data Fig. 5, 4 qubit icin theta_0..theta_18 = 19 parametre gosterir.

    Sema tek bir [Rx, Rz, Rxx, Rz, Rx] dizilimidir; bizim gosterimimizde
    bu, 1 "katman" + kapanis rotasyonlarina karsilik gelir.
    """
    spec = AnsatzSpec(n_qubits=4, n_layers=1, mode="fig5", entanglement="linear")
    # 1 * (4 Rx + 4 Rz + 3 Rxx) + (4 Rz + 4 Rx) = 11 + 8 = 19
    assert spec.n_parameters == 19

    circuit, theta = build_ansatz(spec)
    assert len(theta) == 19
    assert circuit.num_qubits == 4


def test_hardware_efficient_mod_makaledeki_96_parametreyi_verir():
    """Makale metni: "16 qubits ... 96 parameters to optimize in total".

    Bu sayi ancak parametresiz entangler kullanan varyantla elde edilir:
    2 * 16 qubit * 3 katman = 96.
    """
    spec = AnsatzSpec(n_qubits=16, n_layers=3, mode="hardware_efficient")
    assert spec.n_parameters == 96


def test_lineer_topoloji_dogru_sayida_dolanikilik_kapisi_kurar():
    """Makale: "We used linear topology for our project." -> n-1 cift."""
    spec = AnsatzSpec(n_qubits=6, n_layers=2, mode="fig5", entanglement="linear")
    circuit, _ = build_ansatz(spec)

    n_rxx = sum(1 for inst in circuit.data if inst.operation.name == "rxx")
    assert n_rxx == 2 * (6 - 1)  # katman basina n-1


def test_circular_topoloji_bir_fazla_cift_ekler():
    linear = AnsatzSpec(n_qubits=6, n_layers=2, entanglement="linear").n_parameters
    circular = AnsatzSpec(n_qubits=6, n_layers=2, entanglement="circular").n_parameters
    assert circular - linear == 2  # katman basina 1 ek cift, 2 katman


def test_count_parameters_kurulan_devreyle_tutarli():
    for n_qubits in (2, 3, 5, 8):
        for n_layers in (1, 2, 4):
            for mode in ("fig5", "hardware_efficient"):
                spec = AnsatzSpec(n_qubits=n_qubits, n_layers=n_layers, mode=mode)
                _, theta = build_ansatz(spec)
                assert len(theta) == count_parameters(spec), (n_qubits, n_layers, mode)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_qubits": 1, "n_layers": 1},            # cok az qubit
        {"n_qubits": 4, "n_layers": 0},            # gecersiz katman
        {"n_qubits": 4, "n_layers": 1, "mode": "yok"},
        {"n_qubits": 4, "n_layers": 1, "entanglement": "full"},
    ],
)
def test_gecersiz_ansatz_tanimi_reddedilir(kwargs):
    with pytest.raises(ValueError):
        AnsatzSpec(**kwargs)


# ---------------------------------------------------------------------------
# softmax / hedef dagilim
# ---------------------------------------------------------------------------
def test_softmax_olasilik_dagilimi_verir():
    """Makale: P(x) = softmax(R(x))."""
    probs = softmax(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs > 0)
    assert probs[2] > probs[1] > probs[0]  # yuksek odul -> yuksek olasilik


def test_softmax_buyuk_degerlerde_tasmaz():
    probs = softmax(np.array([1000.0, 1001.0]))
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(np.isfinite(probs))


def test_hedef_dagilim_odullerden_kurulur():
    bitstrings = np.array([0, 1, 2, 3])
    rewards = np.array([0.1, 0.9, 0.5, 0.2])
    target = TargetDistribution.from_rewards(bitstrings, rewards)

    assert len(target) == 4
    assert np.isclose(target.probabilities.sum(), 1.0)
    # En yuksek odullu bitstring (1) en yuksek hedef olasiligi almali
    assert target.indices[np.argmax(target.probabilities)] == 1


def test_hedef_dagilim_tekrarlanan_bitstringleri_birlestirir():
    bitstrings = np.array([5, 5, 7])
    rewards = np.array([0.4, 0.6, 1.0])
    target = TargetDistribution.from_rewards(bitstrings, rewards)

    assert len(target) == 2
    assert set(target.indices.tolist()) == {5, 7}


def test_normalize_edilmemis_hedef_reddedilir():
    with pytest.raises(ValueError, match="toplami 1"):
        TargetDistribution(indices=np.array([0, 1]), probabilities=np.array([0.3, 0.3]))


# ---------------------------------------------------------------------------
# QCBM
# ---------------------------------------------------------------------------
@pytest.fixture
def small_qcbm() -> QCBM:
    return QCBM(QCBMConfig(n_qubits=4, n_layers=1, shots=512, seed=7, max_iterations=25))


def test_born_kurali_gecerli_olasilik_dagilimi_verir(small_qcbm):
    """p(x) = |<x|psi(theta)>|^2 -> toplami 1, hepsi negatif olmayan."""
    probs = small_qcbm.probabilities()

    assert probs.shape == (2**4,)
    assert np.isclose(probs.sum(), 1.0)
    assert np.all(probs >= 0)


def test_ornekleme_istenen_sayida_gecerli_bitstring_dondurur(small_qcbm):
    samples = small_qcbm.sample(shots=256)

    assert samples.shape == (256,)
    assert samples.min() >= 0
    assert samples.max() < small_qcbm.n_states


def test_egitim_nll_kaybini_azaltir(small_qcbm):
    """Algorithm 2: COBYLA, exact NLL'i dusurmeli."""
    rng = np.random.default_rng(0)
    bitstrings = rng.integers(0, small_qcbm.n_states, size=64)
    rewards = rng.random(64)
    target = TargetDistribution.from_rewards(bitstrings, rewards)

    before = small_qcbm.exact_nll(target)
    history = small_qcbm.fit(target)
    after = small_qcbm.exact_nll(target)

    assert history.n_evaluations > 0
    assert after <= before + 1e-9, f"kayip artti: {before} -> {after}"
    assert history.final_loss is not None


def test_egitim_tek_tepe_noktasina_olasilik_yigar():
    """Hedef tek bir bitstring ise, QCBM o duruma olasilik kaydirmali."""
    qcbm = QCBM(QCBMConfig(n_qubits=3, n_layers=2, seed=11, max_iterations=250))
    hedef_durum = 5
    target = TargetDistribution(
        indices=np.array([hedef_durum]), probabilities=np.array([1.0])
    )

    onceki = qcbm.probabilities()[hedef_durum]
    qcbm.fit(target)
    sonraki = qcbm.probabilities()[hedef_durum]

    assert sonraki > onceki
    assert sonraki > 0.5, f"hedef duruma yeterince olasilik yiglmadi: {sonraki:.3f}"


def test_agirliksiz_nll_makale_formuluyle_ayni():
    """-sum_{x in D} log p_theta(x) elle hesaplananla ayni olmali."""
    qcbm = QCBM(QCBMConfig(n_qubits=3, n_layers=1, seed=3))
    target = TargetDistribution(
        indices=np.array([0, 3]), probabilities=np.array([0.5, 0.5])
    )

    probs = qcbm.probabilities()
    beklenen = -np.sum(np.log(np.clip(probs[target.indices], 1e-12, None)))

    assert np.isclose(qcbm.exact_nll_unweighted(target), beklenen)


def test_kl_diverjansi_negatif_olmaz(small_qcbm):
    rng = np.random.default_rng(1)
    bitstrings = rng.integers(0, small_qcbm.n_states, size=16)
    target = TargetDistribution.from_rewards(bitstrings, rng.random(16))

    assert small_qcbm.kl_divergence(target) >= -1e-9


def test_parametre_sifirlama_gecmisi_temizler(small_qcbm):
    rng = np.random.default_rng(2)
    target = TargetDistribution.from_rewards(
        rng.integers(0, small_qcbm.n_states, size=8), rng.random(8)
    )
    small_qcbm.fit(target)
    assert small_qcbm.history

    small_qcbm.reset_parameters()
    assert not small_qcbm.history
    assert small_qcbm.params.shape == (small_qcbm.n_parameters,)


def test_makale_yapilandirmasi_16_qubit_kurulur():
    """Varsayilan config makaledeki 16 qubit / 4 katman olmali."""
    cfg = QCBMConfig()
    assert cfg.n_qubits == 16
    assert cfg.n_layers == 4
    assert cfg.entanglement == "linear"
    assert cfg.optimizer == "COBYLA"

    qcbm = QCBM(cfg)
    assert qcbm.n_states == 65536
    assert "16" in qcbm.describe()
