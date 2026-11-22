"""Veri katmani testleri: cekirdek kutuphane, STONED-SELFIES, tokenizer."""

from __future__ import annotations

import numpy as np
import pytest

from kras_quantum.data.seed_library import (
    KNOWN_KRAS_INHIBITOR_NAMES,
    SeedCompound,
    load_seed_library,
    write_seed_library,
)
from kras_quantum.data.stoned import (
    StonedConfig,
    canonical_smiles,
    generate_analogs,
    mutate_selfies,
    randomize_smiles,
    selfies_to_smiles,
    smiles_to_selfies,
    tanimoto_similarity,
)
from kras_quantum.data.tokenizer import SelfiesTokenizer

# Testlerde kullanilan kucuk, bilinen molekuller
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
BENZENE = "c1ccccc1"
TOLUENE = "Cc1ccccc1"
ETHANOL = "CCO"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# SMILES <-> SELFIES temelleri
# ---------------------------------------------------------------------------
def test_kanonik_smiles_ayni_molekulu_ayni_dizeye_indirger():
    """Farkli yazimlar ayni kanonik forma inmeli."""
    assert canonical_smiles("C1=CC=CC=C1") == canonical_smiles("c1ccccc1")


def test_gecersiz_smiles_none_dondurur():
    assert canonical_smiles("bu bir molekul degil") is None


def test_selfies_gidis_donus_molekulu_korur():
    """SMILES -> SELFIES -> SMILES ayni molekulu vermeli."""
    for smiles in (ASPIRIN, BENZENE, ETHANOL):
        selfies_str = smiles_to_selfies(smiles)
        assert selfies_str, f"SELFIES'e cevrilemedi: {smiles}"
        assert selfies_to_smiles(selfies_str) == canonical_smiles(smiles)


def test_randomize_smiles_yazimi_degistirir_molekulu_degistirmez():
    """Makale: "we first randomized the string using RDKit"."""
    rng = np.random.default_rng(0)
    hedef = canonical_smiles(ASPIRIN)

    yazimlar = set()
    for _ in range(20):
        randomized = randomize_smiles(ASPIRIN, rng)
        assert randomized is not None
        # Yazim degisti ama molekul ayni kaldi
        assert canonical_smiles(randomized) == hedef
        yazimlar.add(randomized)

    assert len(yazimlar) > 1, "Randomizasyon hic farkli yazim uretmedi"


# ---------------------------------------------------------------------------
# Mutasyon
# ---------------------------------------------------------------------------
def test_mutasyon_her_zaman_gecerli_molekul_verir(rng):
    """SELFIES'in saglamlik garantisi: mutasyon gecersiz molekul uretemez.

    STONED algoritmasinin tum dayanagi budur (Makale kaynak 24).
    """
    seed_selfies = smiles_to_selfies(ASPIRIN)

    for _ in range(100):
        mutated = mutate_selfies(seed_selfies, rng, n_mutations=int(rng.integers(1, 20)))
        smiles = selfies_to_smiles(mutated)
        assert smiles is not None, f"Mutasyon gecersiz molekul uretti: {mutated}"


def test_mutasyon_diziyi_degistirir(rng):
    seed_selfies = smiles_to_selfies(ASPIRIN)
    degisen = sum(
        mutate_selfies(seed_selfies, rng, n_mutations=5) != seed_selfies
        for _ in range(20)
    )
    assert degisen >= 18, "Mutasyonlarin cogu diziyi degistirmedi"


def test_mutasyon_diziyi_bosaltmaz(rng):
    """Cok sayida silme bile en az bir token birakmali."""
    seed_selfies = smiles_to_selfies(ETHANOL)
    for _ in range(50):
        mutated = mutate_selfies(seed_selfies, rng, n_mutations=50)
        assert mutated, "Mutasyon bos dize uretti"
        assert selfies_to_smiles(mutated) is not None


# ---------------------------------------------------------------------------
# Benzerlik
# ---------------------------------------------------------------------------
def test_molekulun_kendisiyle_benzerligi_bir():
    assert tanimoto_similarity(ASPIRIN, ASPIRIN) == pytest.approx(1.0)


def test_benzer_molekuller_benzemeyenlerden_yuksek_skor_alir():
    benzen_toluen = tanimoto_similarity(BENZENE, TOLUENE)
    benzen_aspirin = tanimoto_similarity(BENZENE, ASPIRIN)
    assert benzen_toluen > benzen_aspirin


def test_gecersiz_girdi_sifir_benzerlik_verir():
    assert tanimoto_similarity(ASPIRIN, "gecersiz!!") == 0.0


# ---------------------------------------------------------------------------
# Analog uretimi (hit expansion, Fig. 1a)
# ---------------------------------------------------------------------------
def test_analoglar_benzersiz_ve_cekirdekten_farkli():
    analogs = generate_analogs(ASPIRIN, n_analogs=25, rng=np.random.default_rng(1))

    assert len(analogs) == 25
    assert len(set(analogs)) == 25, "Yinelenen analog uretildi"
    assert canonical_smiles(ASPIRIN) not in analogs


def test_uretilen_analoglarin_hepsi_gecerli():
    analogs = generate_analogs(ASPIRIN, n_analogs=30, rng=np.random.default_rng(2))
    assert all(canonical_smiles(s) is not None for s in analogs)


def test_benzerlik_esigi_uygulanir():
    """Fig. 1a kabuklari: "Similarity > 0.6" / "Similarity > 0.4"."""
    config = StonedConfig()
    esik = 0.4

    analogs = generate_analogs(
        ASPIRIN,
        n_analogs=10,
        rng=np.random.default_rng(3),
        min_similarity=esik,
        max_attempts_factor=200,
    )

    for analog in analogs:
        benzerlik = tanimoto_similarity(ASPIRIN, analog, config)
        assert benzerlik >= esik, f"Esigin altinda analog gecti: {benzerlik:.3f}"


def test_dar_kabuk_genis_kabuktan_daha_benzer_analoglar_verir():
    """Yuksek esik -> cekirdege daha yakin analoglar."""
    ortalama = {}
    for esik in (0.4, 0.6):
        analogs = generate_analogs(
            ASPIRIN,
            n_analogs=8,
            rng=np.random.default_rng(4),
            min_similarity=esik,
            max_attempts_factor=300,
        )
        ortalama[esik] = np.mean(
            [tanimoto_similarity(ASPIRIN, a) for a in analogs]
        )

    assert ortalama[0.6] > ortalama[0.4]


def test_cozulemeyen_cekirdek_hata_verir():
    with pytest.raises(ValueError, match="cozulemedi"):
        generate_analogs("gecersiz molekul", n_analogs=5)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def test_tokenizer_korpustan_sozluk_kurar():
    korpus = [smiles_to_selfies(s) for s in (ASPIRIN, BENZENE, ETHANOL)]
    tok = SelfiesTokenizer.from_corpus(korpus)

    assert tok.vocab_size > 4  # 4 ozel token + gercek token'lar
    assert tok.pad_id == 0     # nn.Embedding(padding_idx=0) icin sart


def test_ozel_tokenlar_sabit_indekslerde():
    tok = SelfiesTokenizer.from_corpus([smiles_to_selfies(ETHANOL)])
    assert (tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id) == (0, 1, 2, 3)


def test_kodlama_cozme_gidis_donusu():
    selfies_str = smiles_to_selfies(ASPIRIN)
    tok = SelfiesTokenizer.from_corpus([selfies_str])

    ids = tok.encode(selfies_str, max_length=64)
    assert tok.decode(ids) == selfies_str


def test_kodlama_sabit_uzunluk_ve_ozel_tokenlar_uretir():
    selfies_str = smiles_to_selfies(ETHANOL)
    tok = SelfiesTokenizer.from_corpus([selfies_str])

    ids = tok.encode(selfies_str, max_length=16)
    assert ids.shape == (16,)
    assert ids[0] == tok.bos_id
    assert tok.eos_id in ids.tolist()
    assert ids[-1] == tok.pad_id  # kisa dizi pad'lendi


def test_uzun_dizi_kirpilir_ama_uzunluk_korunur():
    selfies_str = smiles_to_selfies(ASPIRIN)
    tok = SelfiesTokenizer.from_corpus([selfies_str])

    ids = tok.encode(selfies_str, max_length=6)
    assert ids.shape == (6,)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id  # kirpilsa da kapatildi


def test_bilinmeyen_token_unk_olur():
    tok = SelfiesTokenizer.from_corpus([smiles_to_selfies(ETHANOL)])
    ids = tok.encode("[Br]", max_length=8)  # korpusta gecmeyen token
    assert tok.unk_id in ids.tolist()


def test_toplu_kodlama_dogru_sekil_verir():
    korpus = [smiles_to_selfies(s) for s in (ASPIRIN, BENZENE, ETHANOL)]
    tok = SelfiesTokenizer.from_corpus(korpus)

    batch = tok.encode_batch(korpus, max_length=48)
    assert batch.shape == (3, 48)
    assert tok.decode_batch(batch) == korpus


def test_saglam_alfabe_tokenizeri_genis_sozluk_verir():
    tok = SelfiesTokenizer.from_robust_alphabet()
    assert tok.vocab_size > 50


def test_tokenizer_kaydedilip_yuklenebilir(tmp_path):
    korpus = [smiles_to_selfies(s) for s in (ASPIRIN, ETHANOL)]
    tok = SelfiesTokenizer.from_corpus(korpus)

    path = tok.save(tmp_path / "vocab.json")
    yeniden = SelfiesTokenizer.load(path)

    assert yeniden.vocab == tok.vocab
    assert yeniden.vocab_size == tok.vocab_size
    assert yeniden.decode(yeniden.encode(korpus[0], 64)) == korpus[0]


# ---------------------------------------------------------------------------
# Cekirdek kutuphane
# ---------------------------------------------------------------------------
def test_cekirdek_kutuphane_yuklenir_ve_tum_yapilari_gecerli():
    """data/seed_kras_inhibitors.csv PubChem'den indirilmis gercek yapilar."""
    compounds = load_seed_library()

    assert len(compounds) >= 15, "Cekirdek kutuphane beklenenden kucuk"
    for compound in compounds:
        assert compound.name
        assert canonical_smiles(compound.smiles) is not None, compound.name
        assert compound.source.startswith("PubChem"), compound.name


def test_cekirdek_kutuphane_bilinen_inhibitorleri_icerir():
    """Makalede ve KRAS literaturunde adi gecen anahtar bilesikler."""
    isimler = {c.name.lower() for c in load_seed_library()}

    # MRTX1133: makale kaynak 21, G12D secicilik referansi
    # TH-Z816 : makale Extended Data Fig. 4, farmakofor referansi (PDB 7EW9)
    for beklenen in ("mrtx1133", "th-z816", "sotorasib", "adagrasib"):
        assert beklenen in isimler, f"Cekirdek sette eksik: {beklenen}"


def test_cekirdek_bilesikler_birbirinden_farkli():
    compounds = load_seed_library()
    kanonik = [canonical_smiles(c.smiles) for c in compounds]
    assert len(set(kanonik)) == len(kanonik), "Cekirdek sette yinelenen yapi var"


def test_eksik_dosya_yol_gosteren_hata_verir(tmp_path):
    with pytest.raises(FileNotFoundError, match="00_fetch_seed_library"):
        load_seed_library(tmp_path / "yok.csv")


def test_gecersiz_yapilar_yuklemede_elenir(tmp_path):
    path = write_seed_library(
        [
            SeedCompound(name="gecerli", smiles=ETHANOL, source="PubChem CID 702"),
            SeedCompound(name="bozuk", smiles="!!gecersiz!!", source="PubChem CID 0"),
        ],
        tmp_path / "karisik.csv",
    )

    compounds = load_seed_library(path)
    assert [c.name for c in compounds] == ["gecerli"]


def test_cekirdek_isim_listesi_yinelenme_icermez():
    isimler = [n.lower() for n in KNOWN_KRAS_INHIBITOR_NAMES]
    assert len(set(isimler)) == len(isimler)


def test_cekirdekten_analog_uretimi_calisir():
    """Uctan uca: gercek bir KRAS inhibitorunden analog uretimi."""
    compounds = load_seed_library()
    mrtx1133 = next(c for c in compounds if c.name == "MRTX1133")

    analogs = generate_analogs(
        mrtx1133.smiles, n_analogs=5, rng=np.random.default_rng(9)
    )

    assert len(analogs) == 5
    assert all(canonical_smiles(a) is not None for a in analogs)
