"""Kiyaslama testleri (Makale, Extended Data Fig. 1)."""

from __future__ import annotations

import numpy as np
import pytest

from kras_quantum.benchmark import (
    BenchmarkResult,
    PriorComparison,
    fit_qubit_trend,
    run_prior_benchmark,
)
from kras_quantum.config import Config, LSTMConfig, QCBMConfig
from kras_quantum.data import load_seed_library, smiles_to_selfies
from kras_quantum.data.tokenizer import SelfiesTokenizer

SMALL_CONFIG = Config(
    qcbm=QCBMConfig(n_qubits=4, n_layers=1, shots=128, seed=3, max_iterations=10),
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
    selfies_list = [smiles_to_selfies(c.smiles) for c in load_seed_library()]
    selfies_list = [s for s in selfies_list if s]
    tokenizer = SelfiesTokenizer.from_corpus(selfies_list)
    return tokenizer, tokenizer.encode_batch(selfies_list, max_length=24)


# ---------------------------------------------------------------------------
# Sonuc nesnesi
# ---------------------------------------------------------------------------
def test_ortalama_ve_standart_sapma_hesaplanir():
    """Makale protokolu: "Mean +/- Std" raporlanir."""
    sonuc = BenchmarkResult(
        label="test",
        local_filter_rates=[60.0, 62.0, 64.0],
        chemistry42_rates=[9.0, 10.0, 11.0],
    )

    assert sonuc.local_mean == pytest.approx(62.0)
    assert sonuc.local_std == pytest.approx(np.std([60.0, 62.0, 64.0]))
    assert sonuc.chemistry42_mean == pytest.approx(10.0)


def test_bos_sonuc_sifir_dondurur():
    sonuc = BenchmarkResult(label="bos")
    assert sonuc.local_mean == 0.0
    assert sonuc.chemistry42_mean == 0.0


def test_sonuc_sozluge_cevrilir():
    sonuc = BenchmarkResult(label="QCBM (SIM)", local_filter_rates=[63.56], n_qubits=16)
    d = sonuc.as_dict()

    assert d["label"] == "QCBM (SIM)"
    assert d["n_qubits"] == 16
    assert d["n_repeats"] == 1


# ---------------------------------------------------------------------------
# Kuantum avantaji hesabi
# ---------------------------------------------------------------------------
def test_kuantum_avantaji_klasik_taban_cizgiye_gore_hesaplanir():
    """Makale, QCBM-LSTM icin %21.5 iyilesme bildirir.

    Burada hesabin dogrulugunu makalenin kendi sayilariyla denetliyoruz
    (Extended Data Fig. 1a): QCBM(SIM) 63.56, Klasik LSTM 52.30
    -> (63.56 - 52.30) / 52.30 = %21.5
    """
    karsilastirma = PriorComparison(
        results=[
            BenchmarkResult(label="Klasik LSTM", local_filter_rates=[52.30]),
            BenchmarkResult(label="QCBM (SIM)", local_filter_rates=[63.56]),
        ]
    )

    avantaj = karsilastirma.quantum_advantage()

    assert "QCBM (SIM)" in avantaj
    assert avantaj["QCBM (SIM)"] == pytest.approx(21.5, abs=0.1)


def test_klasik_kol_yoksa_avantaj_hesaplanmaz():
    karsilastirma = PriorComparison(
        results=[BenchmarkResult(label="QCBM (SIM)", local_filter_rates=[60.0])]
    )
    assert karsilastirma.quantum_advantage() == {}


def test_karsilastirma_tablosu_uretilir():
    karsilastirma = PriorComparison(
        results=[
            BenchmarkResult(label="Klasik LSTM", local_filter_rates=[52.3]),
            BenchmarkResult(label="QCBM (SIM)", local_filter_rates=[63.6]),
        ]
    )
    tablo = karsilastirma.as_table()

    assert "Klasik LSTM" in tablo
    assert "QCBM (SIM)" in tablo


# ---------------------------------------------------------------------------
# Qubit egilimi
# ---------------------------------------------------------------------------
def test_qubit_egilimi_dogrusal_iliskiyi_yakalar():
    """Makale: "success rate correlated roughly linearly with the number
    of qubits" (Extended Data Fig. 1c)."""
    sonuclar = [
        BenchmarkResult(label=f"{q} qubit", local_filter_rates=[50.0 + 0.5 * q], n_qubits=q)
        for q in (2, 4, 8, 16)
    ]

    egilim = fit_qubit_trend(sonuclar)

    assert egilim["slope"] == pytest.approx(0.5, abs=1e-6)
    assert egilim["r_squared"] == pytest.approx(1.0, abs=1e-6)


def test_yetersiz_nokta_egilim_vermez():
    sonuclar = [BenchmarkResult(label="4 qubit", local_filter_rates=[55.0], n_qubits=4)]
    assert fit_qubit_trend(sonuclar) == {}


# ---------------------------------------------------------------------------
# Uctan uca kiyaslama (kucuk olcek)
# ---------------------------------------------------------------------------
def test_kiyaslama_her_kol_icin_sonuc_uretir(training_data):
    tokenizer, sequences = training_data

    karsilastirma = run_prior_benchmark(
        tokenizer,
        sequences,
        arms=("classical", "quantum_sim"),
        config=SMALL_CONFIG,
        n_train_epochs=1,
        n_samples=40,
        n_repeats=2,
        verbose=False,
    )

    assert len(karsilastirma.results) == 2
    for sonuc in karsilastirma.results:
        assert len(sonuc.local_filter_rates) == 2   # n_repeats
        assert 0.0 <= sonuc.local_mean <= 100.0
        assert 0.0 <= sonuc.chemistry42_mean <= 100.0


def test_kati_filtre_orani_gevsek_filtreden_yuksek_olamaz(training_data):
    """Chemistry42 filtresi local filtreden kati (Extended Data Fig. 1a,b)."""
    tokenizer, sequences = training_data

    karsilastirma = run_prior_benchmark(
        tokenizer,
        sequences,
        arms=("classical",),
        config=SMALL_CONFIG,
        n_train_epochs=1,
        n_samples=40,
        n_repeats=2,
        verbose=False,
    )

    sonuc = karsilastirma.results[0]
    assert sonuc.chemistry42_mean <= sonuc.local_mean + 1e-9
