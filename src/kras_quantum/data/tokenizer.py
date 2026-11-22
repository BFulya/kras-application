"""
SELFIES token sozlugu ve dizi kodlama.

Makale referansi: "The LSTM network processed sequential data of chemical
structures" (Fig. 1b) ve Extended Data Fig. 3a: molekuler bilgi SELFIES
kodlamasinda tutulur, kuantum verisiyle birlestirilip LSTM hucresine girer.

Neden SELFIES (SMILES degil)?
-----------------------------
SELFIES (Krenn et al., makale kaynaklari 39, 65, 66) %100 saglam bir
gosterimdir: gecerli alfabeden olusan HER dizi gecerli bir molekule cozulur.
Uretici model bu sayede "gecersiz molekul" uretemez; SMILES'te uretimin
buyuk kismi parantez/valans hatalarina gider. STONED mutasyonlari da ayni
ozellige dayanir (bkz. kras_quantum.data.stoned).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SelfiesTokenizer", "SpecialTokens"]


@dataclass(frozen=True)
class SpecialTokens:
    """Dizi modeli icin gerekli ozel token'lar."""

    pad: str = "[nop]"    # dolgu; SELFIES'in kendi "islem yok" token'i
    bos: str = "[BOS]"    # dizi basi
    eos: str = "[EOS]"    # dizi sonu
    unk: str = "[UNK]"    # sozlukte olmayan token

    def as_tuple(self) -> tuple[str, str, str, str]:
        # Sira onemli: pad daima 0 indeksinde olmali (nn.Embedding padding_idx)
        return (self.pad, self.bos, self.eos, self.unk)


class SelfiesTokenizer:
    """SELFIES dizeleri <-> tamsayi dizileri.

    Example
    -------
    >>> tok = SelfiesTokenizer.from_corpus(["[C][C][O]", "[C][N]"])
    >>> ids = tok.encode("[C][O]", max_length=8)
    >>> tok.decode(ids)
    '[C][O]'
    """

    def __init__(
        self,
        tokens: list[str],
        special: SpecialTokens | None = None,
    ) -> None:
        self.special = special or SpecialTokens()

        # Ozel token'lar her zaman basta ve sabit sirada
        vocab: list[str] = list(self.special.as_tuple())
        for token in tokens:
            if token not in vocab:
                vocab.append(token)

        self.vocab: list[str] = vocab
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(vocab)}

    # ------------------------------------------------------------------
    # Kurucular
    # ------------------------------------------------------------------
    @classmethod
    def from_corpus(
        cls, selfies_list: list[str], min_frequency: int = 1
    ) -> "SelfiesTokenizer":
        """Egitim kumesinde gecen token'lardan sozluk kurar."""
        import selfies as sf

        counter: Counter[str] = Counter()
        for s in selfies_list:
            if not s:
                continue
            try:
                counter.update(sf.split_selfies(s))
            except Exception:  # bozuk dize - atla
                logger.debug("Token'lara ayrilamadi: %s", s[:60])
                continue

        tokens = sorted(t for t, n in counter.items() if n >= min_frequency)
        logger.info(
            "Sozluk kuruldu: %d token (%d benzersiz aday, min_frequency=%d)",
            len(tokens) + 4,
            len(counter),
            min_frequency,
        )
        return cls(tokens)

    @classmethod
    def from_robust_alphabet(cls) -> "SelfiesTokenizer":
        """SELFIES'in semantik olarak saglam tam alfabesini kullanir.

        Korpus kucuk oldugunda, egitim setinde gorulmeyen token'lari da
        kapsamak icin yararlidir.
        """
        import selfies as sf

        return cls(sorted(sf.get_semantic_robust_alphabet()))

    # ------------------------------------------------------------------
    # Ozellikler
    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[self.special.pad]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[self.special.bos]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[self.special.eos]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[self.special.unk]

    # ------------------------------------------------------------------
    # Kodlama / cozme
    # ------------------------------------------------------------------
    def encode(
        self, selfies_str: str, max_length: int, add_special: bool = True
    ) -> np.ndarray:
        """SELFIES dizesini sabit uzunlukta tamsayi dizisine cevirir.

        Dizi [BOS] ... [EOS] ile sarilir ve `max_length`'e pad'lenir;
        uzunsa kirpilir (ve [EOS] ile kapatilir).
        """
        import selfies as sf

        try:
            tokens = list(sf.split_selfies(selfies_str))
        except Exception:
            tokens = []

        ids = [self.token_to_id.get(t, self.unk_id) for t in tokens]

        if add_special:
            # [BOS] ve [EOS] icin iki yer ayir
            ids = ids[: max_length - 2]
            ids = [self.bos_id] + ids + [self.eos_id]
        else:
            ids = ids[:max_length]

        if len(ids) < max_length:
            ids.extend([self.pad_id] * (max_length - len(ids)))

        return np.asarray(ids, dtype=np.int64)

    def encode_batch(self, selfies_list: list[str], max_length: int) -> np.ndarray:
        """Coklu kodlama: (n_molekul, max_length) dizisi."""
        return np.stack([self.encode(s, max_length) for s in selfies_list])

    def decode(self, ids, strip_special: bool = True) -> str:
        """Tamsayi dizisini SELFIES dizesine geri cevirir."""
        parts: list[str] = []
        specials = set(self.special.as_tuple())

        for i in np.asarray(ids).ravel().tolist():
            token = self.vocab[int(i)] if 0 <= int(i) < self.vocab_size else self.special.unk
            if token == self.special.eos:
                break
            if strip_special and token in specials:
                continue
            parts.append(token)

        return "".join(parts)

    def decode_batch(self, batch, strip_special: bool = True) -> list[str]:
        return [self.decode(row, strip_special) for row in np.asarray(batch)]

    # ------------------------------------------------------------------
    # Kalicilik
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vocab": self.vocab,
            "special": {
                "pad": self.special.pad,
                "bos": self.special.bos,
                "eos": self.special.eos,
                "unk": self.special.unk,
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "SelfiesTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        special = SpecialTokens(**payload["special"])
        # Ozel token'lar kurucu tarafindan yeniden eklenir; yinelenmesinler
        reserved = set(special.as_tuple())
        tokens = [t for t in payload["vocab"] if t not in reserved]
        return cls(tokens, special)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:  # pragma: no cover
        return f"<SelfiesTokenizer vocab_size={self.vocab_size}>"
