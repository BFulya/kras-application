"""Prior-kosullu LSTM testleri (Makale, Denklem 9-15)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kras_quantum.classical import (
    PriorConditionedLSTM,
    bitstrings_to_prior,
    train_lstm,
)
from kras_quantum.config import LSTMConfig
from kras_quantum.data.stoned import selfies_to_smiles, smiles_to_selfies
from kras_quantum.data.tokenizer import SelfiesTokenizer

SMALL_CONFIG = LSTMConfig(
    embedding_dim=16, hidden_dim=32, n_layers=1, dropout=0.0, batch_size=8, n_epochs=3
)


@pytest.fixture
def tokenizer() -> SelfiesTokenizer:
    molekuller = ["CCO", "CCC", "c1ccccc1", "CC(=O)O", "CCN", "CCCl"]
    return SelfiesTokenizer.from_corpus([smiles_to_selfies(s) for s in molekuller])


@pytest.fixture
def sequences(tokenizer: SelfiesTokenizer) -> np.ndarray:
    molekuller = ["CCO", "CCC", "c1ccccc1", "CC(=O)O", "CCN", "CCCl"] * 4
    return tokenizer.encode_batch(
        [smiles_to_selfies(s) for s in molekuller], max_length=24
    )


# ---------------------------------------------------------------------------
# Prior kodlama
# ---------------------------------------------------------------------------
def test_bitstring_prior_vektorune_acilir():
    """QCBM tamsayi ciktisi -> {0,1}^n vektoru."""
    prior = bitstrings_to_prior(np.array([5]), n_qubits=4)  # 5 = 0b0101

    assert prior.shape == (1, 4)
    assert prior[0].tolist() == [0.0, 1.0, 0.0, 1.0]


def test_prior_toplu_cevrilir():
    prior = bitstrings_to_prior(np.array([0, 15, 8]), n_qubits=4)

    assert prior.shape == (3, 4)
    assert prior[0].tolist() == [0, 0, 0, 0]
    assert prior[1].tolist() == [1, 1, 1, 1]
    assert prior[2].tolist() == [1, 0, 0, 0]


def test_16_qubit_prior_makale_genisliginde():
    """Makale: 16 qubit'lik IBM islemcisi."""
    rng = np.random.default_rng(0)
    prior = bitstrings_to_prior(rng.integers(0, 2**16, size=32), n_qubits=16)

    assert prior.shape == (32, 16)
    assert set(prior.unique().tolist()) <= {0.0, 1.0}


def test_aralik_disi_bitstring_reddedilir():
    with pytest.raises(ValueError, match="araliginda"):
        bitstrings_to_prior(np.array([16]), n_qubits=4)  # 16 >= 2^4


# ---------------------------------------------------------------------------
# Model yapisi
# ---------------------------------------------------------------------------
def test_concat_birlestirme_giris_boyutunu_buyutur(tokenizer):
    """Denklem (9): x'_t = X(i) ++ x_t."""
    config = LSTMConfig(embedding_dim=16, hidden_dim=32, n_layers=1, prior_merge="concat")
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=16, config=config)

    assert model.lstm.input_size == 16 + 16
    assert model.prior_projection is None


def test_add_birlestirme_prior_projeksiyonu_kullanir(tokenizer):
    """Denklem (9): x'_t = X(i) + x_t - boyutlar esitlenmeli."""
    config = LSTMConfig(embedding_dim=16, hidden_dim=32, n_layers=1, prior_merge="add")
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=16, config=config)

    assert model.lstm.input_size == 16
    assert model.prior_projection is not None
    assert model.prior_projection.in_features == 16
    assert model.prior_projection.out_features == 16


def test_klasik_model_prior_kullanmaz(tokenizer):
    """Makale'nin "vanilla LSTM" kontrol kolu (Extended Data Fig. 1)."""
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=0, config=SMALL_CONFIG)

    assert not model.is_quantum_conditioned
    assert model.lstm.input_size == SMALL_CONFIG.embedding_dim
    assert "klasik" in model.describe()


def test_gecersiz_birlestirme_yontemi_reddedilir(tokenizer):
    config = LSTMConfig(prior_merge="carp")
    with pytest.raises(ValueError, match="concat"):
        PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=16, config=config)


def test_pad_tokeni_gomulmede_sifirlanir(tokenizer):
    """padding_idx=0 -> dolgu gomulmesi egitilmez ve sifir kalir."""
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=0, config=SMALL_CONFIG)
    assert torch.all(model.embedding.weight[tokenizer.pad_id] == 0)


# ---------------------------------------------------------------------------
# Ileri gecis
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("merge", ["concat", "add"])
def test_ileri_gecis_dogru_sekilde_logit_verir(tokenizer, merge):
    config = LSTMConfig(embedding_dim=16, hidden_dim=32, n_layers=1, prior_merge=merge)
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=8, config=config)

    tokens = torch.randint(0, tokenizer.vocab_size, (4, 12))
    prior = bitstrings_to_prior(np.array([1, 2, 3, 4]), n_qubits=8)

    logits, hidden = model(tokens, prior)

    assert logits.shape == (4, 12, tokenizer.vocab_size)
    assert hidden[0].shape == (1, 4, 32)


def test_farkli_prior_farkli_cikti_verir(tokenizer):
    """Kuantum prior gercekten uretimi kosullandirmali (yoksa anlamsiz olur)."""
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=8, config=SMALL_CONFIG)
    model.eval()

    tokens = torch.randint(0, tokenizer.vocab_size, (2, 10))
    prior_a = bitstrings_to_prior(np.array([0, 0]), n_qubits=8)
    prior_b = bitstrings_to_prior(np.array([255, 255]), n_qubits=8)

    with torch.no_grad():
        logits_a, _ = model(tokens, prior_a)
        logits_b, _ = model(tokens, prior_b)

    assert not torch.allclose(logits_a, logits_b), "Prior ciktiyi etkilemiyor"


def test_yanlis_prior_boyutu_reddedilir(tokenizer):
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=16, config=SMALL_CONFIG)
    tokens = torch.randint(0, tokenizer.vocab_size, (2, 10))

    with pytest.raises(ValueError, match="Prior boyutu"):
        model(tokens, bitstrings_to_prior(np.array([1, 2]), n_qubits=8))


def test_yanlis_prior_yigin_boyutu_reddedilir(tokenizer):
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=8, config=SMALL_CONFIG)
    tokens = torch.randint(0, tokenizer.vocab_size, (4, 10))

    with pytest.raises(ValueError, match="yigin boyutu"):
        model(tokens, bitstrings_to_prior(np.array([1, 2]), n_qubits=8))


# ---------------------------------------------------------------------------
# Kayip ve egitim
# ---------------------------------------------------------------------------
def test_kayip_skaler_ve_pozitif(tokenizer, sequences):
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=0, config=SMALL_CONFIG)
    loss = model.loss(torch.as_tensor(sequences))

    assert loss.ndim == 0
    assert float(loss.detach()) > 0


def test_dolgu_tokenlari_kayba_katilmaz(tokenizer):
    """ignore_index=pad -> dolgu uzunlugu kaybi degistirmemeli."""
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=0, config=SMALL_CONFIG)
    model.eval()

    selfies_str = smiles_to_selfies("CCO")
    kisa = torch.as_tensor(tokenizer.encode_batch([selfies_str], max_length=12))
    uzun = torch.as_tensor(tokenizer.encode_batch([selfies_str], max_length=24))

    with torch.no_grad():
        assert float(model.loss(kisa)) == pytest.approx(float(model.loss(uzun)), abs=1e-5)


def test_egitim_kaybi_azalir(tokenizer, sequences):
    """Makale: Adam + NLL kaybi."""
    torch.manual_seed(0)
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=0, config=SMALL_CONFIG)

    history = train_lstm(model, sequences, n_epochs=25, learning_rate=0.01, verbose=False)

    assert len(history.train_loss) == 25
    assert history.final_loss < history.train_loss[0], "Kayip dusmedi"


def test_kuantum_kosullu_egitim_calisir(tokenizer, sequences):
    """QCBM prior'lari ile uctan uca egitim."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=16, config=SMALL_CONFIG)
    priors = bitstrings_to_prior(
        rng.integers(0, 2**16, size=len(sequences)), n_qubits=16
    )

    history = train_lstm(model, sequences, priors, n_epochs=20, learning_rate=0.01, verbose=False)

    assert history.final_loss < history.train_loss[0]


def test_prior_gerektiren_model_priorsuz_egitilemez(tokenizer, sequences):
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=16, config=SMALL_CONFIG)

    with pytest.raises(ValueError, match="priors"):
        train_lstm(model, sequences, priors=None, verbose=False)


def test_uyumsuz_prior_sayisi_reddedilir(tokenizer, sequences):
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=16, config=SMALL_CONFIG)
    priors = bitstrings_to_prior(np.array([1, 2, 3]), n_qubits=16)  # cok az

    with pytest.raises(ValueError, match="ayni sayida satira"):
        train_lstm(model, sequences, priors, verbose=False)


# ---------------------------------------------------------------------------
# Ornekleme
# ---------------------------------------------------------------------------
def test_ornekleme_istenen_sayida_dize_verir(tokenizer):
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=0, config=SMALL_CONFIG)
    ornekler = model.sample(n_samples=5, tokenizer=tokenizer, max_length=20)

    assert len(ornekler) == 5
    assert all(isinstance(s, str) for s in ornekler)


def test_uretilen_selfies_her_zaman_gecerli_molekul(tokenizer, sequences):
    """SELFIES'in saglamlik garantisi: uretim gecersiz molekul veremez.

    Makale bu gosterimi tam olarak bu nedenle secer.
    """
    torch.manual_seed(1)
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=0, config=SMALL_CONFIG)
    train_lstm(model, sequences, n_epochs=10, learning_rate=0.01, verbose=False)

    ornekler = model.sample(n_samples=25, tokenizer=tokenizer, max_length=24)

    for s in ornekler:
        if not s:       # bos dize (hemen EOS) kabul edilebilir
            continue
        assert selfies_to_smiles(s) is not None, f"Gecersiz SELFIES uretildi: {s}"


def test_kuantum_kosullu_ornekleme(tokenizer):
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=16, config=SMALL_CONFIG)
    prior = bitstrings_to_prior(np.array([42, 1337, 7]), n_qubits=16)

    ornekler = model.sample(n_samples=3, tokenizer=tokenizer, prior=prior, max_length=20)
    assert len(ornekler) == 3


def test_kuantum_model_priorsuz_orneklenemez(tokenizer):
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=16, config=SMALL_CONFIG)

    with pytest.raises(ValueError, match="prior"):
        model.sample(n_samples=3, tokenizer=tokenizer, max_length=20)


def test_dusuk_sicaklik_cesitliligi_azaltir(tokenizer, sequences):
    """Makale: sicaklik "belirlenimcilik ve rastgelelik arasindaki dengeyi" ayarlar."""
    torch.manual_seed(2)
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=0, config=SMALL_CONFIG)
    train_lstm(model, sequences, n_epochs=15, learning_rate=0.01, verbose=False)

    torch.manual_seed(3)
    soguk = model.sample(n_samples=40, tokenizer=tokenizer, max_length=24, temperature=0.1)
    torch.manual_seed(3)
    sicak = model.sample(n_samples=40, tokenizer=tokenizer, max_length=24, temperature=2.0)

    assert len(set(soguk)) <= len(set(sicak))


def test_gecersiz_sicaklik_reddedilir(tokenizer):
    model = PriorConditionedLSTM(tokenizer.vocab_size, prior_dim=0, config=SMALL_CONFIG)

    with pytest.raises(ValueError, match="temperature"):
        model.sample(n_samples=2, tokenizer=tokenizer, temperature=0.0)
