"""
STONED-SELFIES ile hit genisletme.

Makale referansi: "STONED-SELFIES" (Methods) ve Fig. 1a

    "We used the STONED-SELFIES algorithm (https://github.com/aspuru-guzik-group/
     stoned-selfies) to mine our initial set of 650 molecules. For a given
     molecule in SMILES format, we first randomized the string using RDKit.
     These randomized strings were then converted into SELFIES. Each SELFIES
     string underwent mutations (in the form of character deletions,
     replacements and additions) up to 500 times. Subsequently, the
     synthesizability and stability of the mutated strings were assessed using
     Chemistry42. We generated 850,000 molecules..."

STONED = Superfast Traversal, Optimization, Novelty, Exploration and Discovery
(Nigam, Pollice, Krenn, Gomes & Aspuru-Guzik, Chem. Sci. 12, 7079-7090, 2021;
makale kaynak 24).

Yontemin kilit noktasi SELFIES gosteriminin %100 saglam olmasidir: rastgele
bir SELFIES dizisi HER ZAMAN gecerli bir molekule cozulur. SMILES'te rastgele
bir karakter degisikligi cogunlukla gecersiz bir dize uretirken, SELFIES'te
mutasyon guvenlidir - genetik/rastgele arama bu yuzden dogrudan uygulanabilir.

Fig. 1a ayrica iki benzerlik kabugu gosterir: cekirdek molekule Tanimoto
benzerligi > 0.6 olan yakin analoglar ve > 0.4 olan daha cesitli analoglar
("Increasing structural diversity").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "draw_mutation_count",
    "MutationKind",
    "StonedConfig",
    "randomize_smiles",
    "mutate_selfies",
    "generate_analogs",
    "tanimoto_similarity",
]


# ---------------------------------------------------------------------------
# rdkit / selfies tembel yukleme
# ---------------------------------------------------------------------------
def _rdkit():
    """RDKit'i gec yukler ve gurultulu loglarini kapatir."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")  # gecersiz ara urunler icin uyari basmasin
    return Chem


def _selfies():
    import selfies as sf

    return sf


MutationKind = ("replace", "add", "delete")


@dataclass(frozen=True)
class StonedConfig:
    """STONED-SELFIES ayarlari."""

    # Makale: "mutations ... up to 500 times" -> ust sinir
    max_mutations: int = 500
    # Mutasyon sayisi dagiliminin sekli (bkz. `draw_mutation_count`).
    # Kucuk degerler agirlikli: cogu analog cekirdege yakin, azi uzak.
    # Fig. 1a'daki "increasing structural diversity" konisi tam olarak budur.
    mutation_decay: float = 0.3
    # Fig. 1a benzerlik kabuklari
    similarity_inner: float = 0.6
    similarity_outer: float = 0.4
    # Morgan parmak izi ayarlari (Tanimoto icin)
    fingerprint_radius: int = 2
    fingerprint_bits: int = 2048
    seed: int = 1234


# ---------------------------------------------------------------------------
# SMILES islemleri
# ---------------------------------------------------------------------------
def canonical_smiles(smiles: str) -> str | None:
    """Kanonik SMILES dondurur; molekul cozulemezse None."""
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def randomize_smiles(smiles: str, rng: np.random.Generator) -> str | None:
    """Ayni molekulun farkli bir SMILES yazimini uretir.

    Makale: "we first randomized the string using RDKit".

    Neden? Ayni molekul farkli atom siralamalariyla farkli SELFIES dizilerine
    cevrilir; mutasyonlar da dizi uzerinde calistigi icin randomizasyon,
    ayni cekirdekten farkli yonlere mutasyon yapilmasini saglar - arama
    uzayinin cesitliligini artirir.
    """
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_order = list(range(mol.GetNumAtoms()))
    rng.shuffle(atom_order)
    renumbered = Chem.RenumberAtoms(mol, [int(i) for i in atom_order])
    return Chem.MolToSmiles(renumbered, canonical=False)


def smiles_to_selfies(smiles: str) -> str | None:
    sf = _selfies()
    try:
        return sf.encoder(smiles)
    except Exception:  # selfies, cesitli dahili istisnalar atabilir
        return None


def selfies_to_smiles(selfies_str: str) -> str | None:
    sf = _selfies()
    try:
        smiles = sf.decoder(selfies_str)
    except Exception:
        return None
    if not smiles:
        return None
    return canonical_smiles(smiles)


# ---------------------------------------------------------------------------
# Mutasyon
# ---------------------------------------------------------------------------
def draw_mutation_count(
    rng: np.random.Generator, config: StonedConfig, scale: float = 1.0
) -> int:
    """Bir analog icin uygulanacak mutasyon sayisini secer.

    Makale mutasyon sayisi icin yalnizca bir UST SINIR verir ("up to 500
    times"), dagilimi belirtmez. Duz (uniform 1..500) secim pratikte ise
    yaramaz: ortalama ~250 mutasyon cekirdegi tamamen yok eder ve Fig. 1a'daki
    yakin kabuk ("Similarity > 0.6") hicbir zaman dolmaz.

    Bunun yerine geometrik dagilim kullaniyoruz: cogu analog cekirdege birkac
    mutasyon uzaklikta, azinligi cok daha uzakta. Bu, Fig. 1a'daki genisleyen
    koni ("Increasing structural diversity") ile tutarlidir ve STONED'un
    yerel kimyasal uzay taramasi fikrine uyar.

    Parameters
    ----------
    scale
        Dagilimi genisletme carpani (>1 daha uzak analoglar). Benzerlik
        esigi saglanamadiginda arama kademeli genisletilirken kullanilir.
    """
    n = int(rng.geometric(config.mutation_decay))
    return int(np.clip(round(n * scale), 1, config.max_mutations))


def mutate_selfies(
    selfies_str: str,
    rng: np.random.Generator,
    alphabet: list[str] | None = None,
    n_mutations: int = 1,
) -> str:
    """SELFIES dizisine rastgele nokta mutasyonlari uygular.

    Makale: "mutations (in the form of character deletions, replacements and
    additions)".

    SELFIES gosterimi saglamdir: alfabeden olusan her dizi gecerli bir
    molekule cozulur. Ancak bu garanti "BOS OLMAYAN molekul" anlamina gelmez -
    ornegin yalniz basina "[=Branch1]" gibi bir dizi bos bir molekule cozulur.
    Bu yuzden mutasyon sonucu bos molekule cozulurse mutasyon geri alinir ve
    yeniden denenir.
    """
    sf = _selfies()
    tokens = list(sf.split_selfies(selfies_str))
    if not tokens:
        return selfies_str

    if alphabet is None:
        alphabet = sorted(sf.get_semantic_robust_alphabet())

    for _ in range(n_mutations):
        if not tokens:
            tokens = [str(rng.choice(alphabet))]
            continue

        onceki = list(tokens)  # bos molekule duserse geri donmek icin

        kind = MutationKind[int(rng.integers(0, len(MutationKind)))]
        position = int(rng.integers(0, len(tokens)))

        if kind == "replace":
            tokens[position] = str(rng.choice(alphabet))
        elif kind == "add":
            tokens.insert(position, str(rng.choice(alphabet)))
        elif len(tokens) > 1:  # "delete" - diziyi tamamen bosaltma
            tokens.pop(position)

        # Bos molekule cozulen diziyi kabul etme
        if selfies_to_smiles("".join(tokens)) is None:
            tokens = onceki

    return "".join(tokens)


# ---------------------------------------------------------------------------
# Benzerlik
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _morgan_generator(radius: int, n_bits: int):
    """Morgan parmak izi ureticisi - kurulumu pahali, yeniden kullanilir."""
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def _fingerprint(smiles: str, config: StonedConfig):
    Chem = _rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = _morgan_generator(config.fingerprint_radius, config.fingerprint_bits)
    return generator.GetFingerprint(mol)


def tanimoto_similarity(
    smiles_a: str, smiles_b: str, config: StonedConfig | None = None
) -> float:
    """Iki molekul arasindaki Morgan/Tanimoto benzerligi (0-1).

    Fig. 1a'daki "Similarity > 0.6" ve "Similarity > 0.4" kabuklari bu
    metrige dayanir.
    """
    from rdkit import DataStructs

    config = config or StonedConfig()
    fp_a = _fingerprint(smiles_a, config)
    fp_b = _fingerprint(smiles_b, config)
    if fp_a is None or fp_b is None:
        return 0.0
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


# ---------------------------------------------------------------------------
# Ana uretim fonksiyonu
# ---------------------------------------------------------------------------
def generate_analogs(
    seed_smiles: str,
    n_analogs: int,
    config: StonedConfig | None = None,
    rng: np.random.Generator | None = None,
    min_similarity: float | None = None,
    max_attempts_factor: int = 20,
) -> list[str]:
    """Tek bir cekirdek molekulden yapisal analoglar uretir.

    Makale, Fig. 1a: cekirdek inhibitorlerden "hit expansion" ile artan
    yapisal cesitlilikte analog bulutlari olusturulur.

    Parameters
    ----------
    seed_smiles
        Cekirdek molekul (deneysel olarak dogrulanmis bir KRAS inhibitoru).
    n_analogs
        Istenen benzersiz analog sayisi.
    min_similarity
        Verilirse, cekirdege Tanimoto benzerligi bu esigin altinda kalan
        analoglar elenir (Fig. 1a kabuklari icin 0.6 veya 0.4).
    max_attempts_factor
        Sonsuz donguye karsi guvenlik: en fazla n_analogs * bu kadar deneme.

    Returns
    -------
    Benzersiz, kanonik SMILES listesi (cekirdegin kendisi haric).
    """
    config = config or StonedConfig()
    rng = rng if rng is not None else np.random.default_rng(config.seed)

    seed_canonical = canonical_smiles(seed_smiles)
    if seed_canonical is None:
        raise ValueError(f"Cekirdek SMILES cozulemedi: {seed_smiles!r}")

    sf = _selfies()
    alphabet = sorted(sf.get_semantic_robust_alphabet())

    analogs: set[str] = set()
    attempts = 0
    max_attempts = max(n_analogs * max_attempts_factor, 100)

    # Benzerlik esigi varsa mutasyon siddetini uyarlamali kis: art arda
    # basarisiz denemeler, cekirdekten fazla uzaklasildigini gosterir.
    scale = 1.0
    consecutive_failures = 0

    while len(analogs) < n_analogs and attempts < max_attempts:
        attempts += 1

        # 1) SMILES'i randomize et (farkli bir SELFIES "yolu" elde etmek icin)
        randomized = randomize_smiles(seed_canonical, rng)
        if randomized is None:
            continue

        # 2) SELFIES'e cevir
        seed_selfies = smiles_to_selfies(randomized)
        if seed_selfies is None:
            continue

        # 3) Mutasyon uygula (makale: en fazla 500 mutasyon)
        n_mut = draw_mutation_count(rng, config, scale)
        mutated = mutate_selfies(seed_selfies, rng, alphabet=alphabet, n_mutations=n_mut)

        # 4) SMILES'e geri cevir
        candidate = selfies_to_smiles(mutated)
        if candidate is None or candidate == seed_canonical or candidate in analogs:
            continue

        # 5) Benzerlik kabugu filtresi (Fig. 1a)
        if min_similarity is not None:
            if tanimoto_similarity(seed_canonical, candidate, config) < min_similarity:
                consecutive_failures += 1
                # 50 basarisiz denemede bir, mutasyonlari kademeli olarak
                # hafiflet (cekirdege daha yakin analoglar ara)
                if consecutive_failures % 50 == 0:
                    scale = max(scale * 0.5, 0.05)
                continue

        consecutive_failures = 0
        analogs.add(candidate)

    if len(analogs) < n_analogs:
        logger.warning(
            "Cekirdek %s icin %d analog istendi, %d uretildi (%d deneme). "
            "Benzerlik esigi cok yuksek olabilir.",
            seed_canonical[:40],
            n_analogs,
            len(analogs),
            attempts,
        )

    return sorted(analogs)
