"""Skorlama katmani testleri: filtreler, PLI, farmakofor, SA, odul."""

from __future__ import annotations

import numpy as np
import pytest

from kras_quantum.config import RewardConfig
from kras_quantum.data import load_seed_library
from kras_quantum.scoring import (
    RewardFunction,
    SurrogatePLI,
    compute_properties,
    evaluate_filters,
    extract_features,
    find_unwanted_motifs,
    is_synthesis_candidate,
    pharmacophore_match_score,
    pharmacophore_similarity,
    passes_chemistry42_filter,
    passes_local_filter,
    sa_to_reward,
    synthetic_accessibility,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
BENZENE = "c1ccccc1"
ACIL_KLORUR = "CC(=O)Cl"          # reaktif - elenmeli
EPOKSIT = "C1CO1"                 # reaktif - elenmeli
DEV_MOLEKUL = "C" * 100           # asiri buyuk - elenmeli


@pytest.fixture(scope="module")
def seed_compounds():
    return load_seed_library()


# ---------------------------------------------------------------------------
# Filtreler
# ---------------------------------------------------------------------------
def test_gercek_kras_inhibitorlerinin_cogu_filtreleri_gecer(seed_compounds):
    """Kritik tutarlilik kontrolu.

    Makalenin kendi referans bilesikleri filtreden gecemiyorsa filtre
    yanlis kalibre edilmis demektir. Bu test, esikleri gercek veriye
    baglar.
    """
    gecen = sum(1 for c in seed_compounds if passes_chemistry42_filter(c.smiles))
    oran = gecen / len(seed_compounds)

    assert oran >= 0.8, (
        f"Gercek KRAS inhibitorlerinin yalnizca %{oran*100:.0f}'i filtreyi geciyor; "
        f"esikler fazla dar olabilir"
    )


def test_mrtx1133_filtreleri_gecer(seed_compounds):
    """MRTX1133 makalenin G12D secicilik referansidir (kaynak 21).

    Kaynasmis halka sistemi nedeniyle 8 SSSR halkasi icerir; halka sayisi
    esigi bu bilesige gore kalibre edildi.
    """
    mrtx = next(c for c in seed_compounds if c.name == "MRTX1133")
    sonuc = evaluate_filters(mrtx.smiles, stringent=True)

    assert sonuc.passed, f"MRTX1133 elendi: {sonuc.failures}"


def test_th_z816_filtreleri_gecer(seed_compounds):
    """TH-Z816, makalenin farmakofor referansidir (Extended Data Fig. 4)."""
    ligand = next(c for c in seed_compounds if c.name == "TH-Z816")
    assert passes_chemistry42_filter(ligand.smiles)


def test_reaktif_gruplar_elenir():
    assert not passes_local_filter(ACIL_KLORUR)
    assert not passes_local_filter(EPOKSIT)


def test_radikal_iceren_molekuller_elenir():
    """Uretici model SELFIES uzerinden karben/radikal atomlar uretebilir.

    Bu molekuller kimyasal olarak kararsiz ve sentezlenemezdir; makalenin
    "eliminate reactive, unsynthesizable or unstable groups" olcutune
    dogrudan girerler.

    Asagidaki yapi, gercek bir kampanya kosusunda secim hunisinin SONUNA
    kadar gelmis bir adaydir - filtre bunu kaciriyordu.
    """
    radikal_aday = "[CH]N=CCC(=O)CNCc1cccc2c1Oc1ccc-2c(F)c1C#C"

    sonuc = evaluate_filters(radikal_aday, stringent=False)
    assert not sonuc.passed
    assert any("radikal" in f for f in sonuc.failures)

    # Radikalsiz esdegeri etkilenmemeli
    assert compute_properties(ASPIRIN)["n_radical_electrons"] == 0


def test_istenmeyen_motif_dogru_tanimlanir():
    assert "acil_halojenur" in find_unwanted_motifs(ACIL_KLORUR)
    assert "epoksit" in find_unwanted_motifs(EPOKSIT)
    assert find_unwanted_motifs(ASPIRIN) == []


def test_cok_kucuk_molekul_elenir():
    assert not passes_local_filter(ETHANOL)   # 3 agir atom, min 10


def test_cok_buyuk_molekul_elenir():
    assert not passes_local_filter(DEV_MOLEKUL)


def test_cozulemeyen_molekul_elenir():
    sonuc = evaluate_filters("bu molekul degil")
    assert not sonuc.passed
    assert "cozulemeyen_molekul" in sonuc.failures


def test_kati_filtre_gevsek_filtreden_daha_secicidir(seed_compounds):
    """Makale, Extended Data Fig. 1a,b: Chemistry42 filtresi local'den kati.

    Local filtrede basari orani ~%52-64, Chemistry42'de ~%8-10'a duser.
    """
    rng = np.random.default_rng(0)
    from kras_quantum.data import generate_analogs

    molekuller = []
    for compound in seed_compounds[:4]:
        molekuller.extend(generate_analogs(compound.smiles, n_analogs=15, rng=rng))

    local_gecen = sum(1 for s in molekuller if passes_local_filter(s))
    kati_gecen = sum(1 for s in molekuller if passes_chemistry42_filter(s))

    assert kati_gecen <= local_gecen


def test_ozellikler_hesaplanir():
    ozellikler = compute_properties(ASPIRIN)

    assert ozellikler is not None
    assert ozellikler["molecular_weight"] == pytest.approx(180.16, abs=0.1)
    assert ozellikler["heavy_atoms"] == 13
    assert 0 <= ozellikler["aromatic_fraction"] <= 1


def test_cozulemeyen_molekul_ozellik_vermez():
    assert compute_properties("!!!") is None


# ---------------------------------------------------------------------------
# Sentetik erisilebilirlik (ReRSA vekili)
# ---------------------------------------------------------------------------
def test_sa_skoru_gecerli_aralikta():
    for smiles in (ASPIRIN, BENZENE, ETHANOL):
        sa = synthetic_accessibility(smiles)
        assert 1.0 <= sa <= 10.0


def test_basit_molekul_karmasiktan_kolay_sentezlenir(seed_compounds):
    basit = synthetic_accessibility(BENZENE)
    karmasik = synthetic_accessibility(
        next(c for c in seed_compounds if c.name == "MRTX1133").smiles
    )
    assert basit < karmasik


def test_gercek_ilaclar_makul_sa_skoru_alir(seed_compounds):
    """Onaylanmis/klinik bilesikler sentezlenebilir olmali."""
    for isim in ("sotorasib", "adagrasib", "ARS-1620"):
        bilesik = next(c for c in seed_compounds if c.name == isim)
        sa = synthetic_accessibility(bilesik.smiles)
        assert sa < 6.5, f"{isim} icin SA skoru beklenenden yuksek: {sa:.2f}"


def test_cozulemeyen_molekul_en_kotu_sa_alir():
    assert synthetic_accessibility("!!!") == 10.0


def test_sa_odul_donusumu_makale_esigine_oturur():
    """Makale esigi: ReRSA < 5."""
    assert sa_to_reward(1.0) == pytest.approx(1.0)
    assert sa_to_reward(5.0) == pytest.approx(0.5)   # esik tam ortada
    assert sa_to_reward(10.0) == pytest.approx(0.0)
    assert sa_to_reward(3.0) > sa_to_reward(7.0)


# ---------------------------------------------------------------------------
# Farmakofor (Extended Data Fig. 4)
# ---------------------------------------------------------------------------
def test_farmakofor_ozellikleri_cikarilir(seed_compounds):
    ligand = next(c for c in seed_compounds if c.name == "TH-Z816")
    ozellikler = extract_features(ligand.smiles)

    assert ozellikler is not None
    # Makalenin uc ozelligi: halka sistemi, hidrofobik grup, H-bagi donoru
    assert ozellikler.aromatic_rings >= 1
    assert ozellikler.hydrophobic_groups >= 1
    assert ozellikler.has_all_required


def test_referans_ligand_kendisiyle_tam_benzer(seed_compounds):
    ligand = next(c for c in seed_compounds if c.name == "TH-Z816")
    assert pharmacophore_similarity(ligand.smiles, ligand.smiles) == pytest.approx(1.0)


def test_referans_ligand_yuksek_eslesme_skoru_alir(seed_compounds):
    """TH-Z816, farmakofor modelinin TURETILDIGI molekuldur.

    Makale sentez esigi 0.7; referans ligandin kendisi bunu rahatca
    gecmelidir, aksi halde skorlama kalibrasyonu bozuktur.
    """
    ligand = next(c for c in seed_compounds if c.name == "TH-Z816")
    skor = pharmacophore_match_score(ligand.smiles)

    assert skor > 0.7, f"Referans ligand esigi gecemedi: {skor:.3f}"


def test_gercek_inhibitorlerin_cogu_farmakofor_esigini_gecer(seed_compounds):
    """Kritik kalibrasyon kontrolu (makale esigi: 0.7).

    Ilk uygulamada ozellik-varligi agirligi 0.5'ti ve gercek KRAS
    inhibitorlerinin yalnizca 3/19'u esigi geciyordu - yani makalenin kendi
    referans ilaclari "farmakofor uyumsuz" sayiliyordu. Agirlik 0.75'e
    cekildi; bu test o kalibrasyonu sabitler.
    """
    skorlar = np.array([pharmacophore_match_score(c.smiles) for c in seed_compounds])
    gecen_orani = float((skorlar > 0.7).mean())

    assert gecen_orani >= 0.7, (
        f"Gercek inhibitorlerin yalnizca %{gecen_orani*100:.0f}'i farmakofor "
        f"esigini geciyor (ortalama skor {skorlar.mean():.3f})"
    )


def test_ilgisiz_molekul_dusuk_eslesme_alir():
    assert pharmacophore_match_score(ETHANOL) < 0.5


def test_kras_inhibitorleri_ilgisiz_molekullerden_yuksek_skor_alir(seed_compounds):
    inhibitor_skorlari = [
        pharmacophore_match_score(c.smiles) for c in seed_compounds[:8]
    ]
    kontrol_skorlari = [
        pharmacophore_match_score(s) for s in (ETHANOL, BENZENE, "CCCCCC")
    ]

    assert np.mean(inhibitor_skorlari) > np.mean(kontrol_skorlari)


def test_cozulemeyen_molekul_sifir_eslesme():
    assert pharmacophore_match_score("!!!") == 0.0


# ---------------------------------------------------------------------------
# Vekil PLI
# ---------------------------------------------------------------------------
def test_vekil_pli_negatif_skor_verir(seed_compounds):
    skorlayici = SurrogatePLI()
    for bilesik in seed_compounds[:5]:
        assert skorlayici.score(bilesik.smiles) < 0


def test_vekil_pli_modeli_kalibre_edilmis():
    """Vekil artik elle yazilmis bir formul degil, ogrenilmis bir modeldir.

    Elle yazilmis surum gercek docking'e karsi ise yaramiyordu
    (Spearman -0.396 ve +0.032). Model, scripts/05_fit_surrogate.py ile
    gercek Vina skorlarindan ogrenilir ve data/ altinda saklanir.
    """
    skorlayici = SurrogatePLI()

    assert skorlayici.is_calibrated, (
        "Vekil PLI modeli bulunamadi; scripts/05_fit_surrogate.py calistirilmali"
    )
    assert skorlayici.cv_spearman is not None
    assert skorlayici.cv_spearman > 0.3, (
        f"Model capraz dogrulamada zayif: Spearman={skorlayici.cv_spearman:+.3f}"
    )


def test_vekil_pli_gercek_inhibitorleri_siralar(seed_compounds):
    """Vekil, gercek inhibitorlere kucuk/ilgisiz molekullerden daha iyi
    (daha negatif) skor vermeli."""
    skorlayici = SurrogatePLI()

    inhibitorler = np.mean([skorlayici.score(c.smiles) for c in seed_compounds])
    kontroller = np.mean([skorlayici.score(s) for s in (ETHANOL, BENZENE, "CCCCCC")])

    assert inhibitorler < kontroller


def test_vekil_model_yoksa_etkisiz_kalir(tmp_path):
    """Model dosyasi yoksa skorlayici sessizce yanlis deger uretmemeli."""
    skorlayici = SurrogatePLI(model_path=tmp_path / "yok.json")

    assert not skorlayici.is_calibrated
    assert skorlayici.score(ASPIRIN) == 0.0


def test_vekil_pli_makul_aralikta(seed_compounds):
    """Fiziksel olarak anlamsiz degerler uretilmemeli."""
    skorlayici = SurrogatePLI()
    for bilesik in seed_compounds:
        skor = skorlayici.score(bilesik.smiles)
        assert -16.0 <= skor <= 0.0


def test_vekil_pli_cozulemeyen_molekulde_basarisiz_doner():
    assert SurrogatePLI().score("!!!") == 0.0


def test_vekil_pli_gercek_docking_olmadigini_bildirir():
    """Dokumantasyon sozlesmesi: vekil, gercek docking gibi gorunmemeli."""
    assert SurrogatePLI().is_real_docking is False


# ---------------------------------------------------------------------------
# Odul fonksiyonu
# ---------------------------------------------------------------------------
def test_erken_epoch_yalnizca_local_filter_kullanir():
    """Makale: "using the local filter until epoch 20"."""
    odul = RewardFunction()
    odul.set_epoch(0)

    assert odul.uses_local_filter_only
    assert odul.stage_name == "local_filter"

    odul.set_epoch(20)
    assert not odul.uses_local_filter_only
    assert odul.stage_name == "chemistry42_proxy"


def test_erken_asamada_gecen_molekul_tam_odul_alir(seed_compounds):
    odul = RewardFunction()
    odul.set_epoch(0)

    dokum = odul.score(seed_compounds[0].smiles)
    assert dokum.total == 1.0


def test_filtreyi_gecemeyen_molekul_sifir_odul_alir():
    odul = RewardFunction()
    odul.set_epoch(25)

    assert odul.score(ACIL_KLORUR).total == 0.0
    assert odul.score("!!!").total == 0.0


def test_tam_asama_odulu_bilesenlere_ayrilir(seed_compounds):
    odul = RewardFunction()
    odul.set_epoch(25)

    dokum = odul.score(seed_compounds[0].smiles)

    assert 0.0 <= dokum.total <= 1.0
    assert dokum.pli_raw < 0
    assert 0.0 <= dokum.pharmacophore_score <= 1.0
    assert 1.0 <= dokum.sa_raw <= 10.0
    assert dokum.passed_filters


def test_gercek_inhibitorler_rastgele_molekullerden_yuksek_odul_alir(seed_compounds):
    """Odul sinyali anlamli mi? QCBM'in ogrenecegi sey budur."""
    odul = RewardFunction()
    odul.set_epoch(25)

    inhibitor_odulleri = [odul.score(c.smiles).total for c in seed_compounds[:8]]
    kontrol_odulleri = [odul.score(s).total for s in (ETHANOL, BENZENE, "CCCCCCCC")]

    assert np.mean(inhibitor_odulleri) > np.mean(kontrol_odulleri)


def test_toplu_odul_dizi_dondurur(seed_compounds):
    odul = RewardFunction()
    odul.set_epoch(25)

    degerler = odul([c.smiles for c in seed_compounds[:5]])

    assert isinstance(degerler, np.ndarray)
    assert degerler.shape == (5,)
    assert np.all((degerler >= 0) & (degerler <= 1))


def test_odul_onbellegi_calisir(seed_compounds):
    odul = RewardFunction()
    odul.set_epoch(25)
    smiles = seed_compounds[0].smiles

    ilk = odul.score(smiles)
    ikinci = odul.score(smiles)

    assert ilk is ikinci  # onbellekten ayni nesne

    odul.clear_cache()
    assert odul.score(smiles) is not ilk


def test_asama_degisimi_onbellegi_karistirmaz(seed_compounds):
    """Ayni molekul, farkli asamada farkli odul almali."""
    odul = RewardFunction()
    smiles = seed_compounds[0].smiles

    odul.set_epoch(0)
    erken = odul.score(smiles).total
    odul.set_epoch(25)
    gec = odul.score(smiles).total

    assert erken == 1.0
    assert gec != erken  # tam asamada bilesenler devreye girer


# ---------------------------------------------------------------------------
# Sentez adayligi (makale esikleri)
# ---------------------------------------------------------------------------
def test_zayif_molekul_aday_degil():
    odul = RewardFunction()
    odul.set_epoch(25)

    karar = is_synthesis_candidate(odul.score(ETHANOL))

    assert not karar.is_candidate
    assert karar.reasons


def test_aday_karari_tum_kriterleri_denetler():
    """Makale kriterleri: filtre + odul>0.7 + PLI<-8 + farmakofor>0.7 + SA<5."""
    from kras_quantum.scoring.reward import RewardBreakdown

    config = RewardConfig()

    # Tum kriterleri saglayan yapay dokum
    iyi = RewardBreakdown(
        smiles="x",
        total=0.85,
        pli_raw=-9.5,
        pharmacophore_score=0.80,
        sa_raw=3.2,
        filter_result=evaluate_filters(ASPIRIN),
    )
    assert is_synthesis_candidate(iyi, config).is_candidate

    # Yalnizca PLI esigi ihlal edilirse reddedilmeli
    zayif_pli = RewardBreakdown(
        smiles="x",
        total=0.85,
        pli_raw=-6.0,          # -8'den zayif
        pharmacophore_score=0.80,
        sa_raw=3.2,
        filter_result=evaluate_filters(ASPIRIN),
    )
    karar = is_synthesis_candidate(zayif_pli, config)
    assert not karar.is_candidate
    assert any("PLI" in r for r in karar.reasons)
