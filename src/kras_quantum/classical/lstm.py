"""
Prior-kosullu LSTM uretici modeli.

Makale referansi: "Classical model: LSTM model" ve Extended Data Fig. 3a

    "The prior samples were combined with input samples x'_t = X(i) ++ x_t in
     the LSTM cell. This combination constituted two methods: adding and
     concatenating samples."

    Denklem (9):  x'_t = X(i) ++ x_t   (concat)   VEYA   x'_t = X(i) + x_t  (add)
    Denklem (10)-(15): standart LSTM kapilari, x_t yerine x'_t ile.

Mimarinin ozu
-------------
QCBM, her molekul icin bir "prior ornegi" uretir: 16 qubit'lik bir
bitstring. Bu bitstring, SELFIES token gomulmelerine HER ZAMAN ADIMINDA
eklenir/eklemlenir. Yani kuantum devresi, LSTM'in uretim surecini
kosullandiran bir gizli degisken gorevi gorur.

Makale bu tasarimin gerekcesini soyle acikliyor: kuantum devresi
superpozisyon ve dolanikilik sayesinde karmasik, yuksek boyutlu olasilik
dagilimlarini klasik modellerden daha verimli temsil edebilir; boylece
LSTM'in ornekledigi baslangic dagilimi daha zengin olur.

Kayip: NLL (Makale, Denklem 7-8), optimize edici: Adam (Makale kaynak 70).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kras_quantum.config import LSTMConfig
from kras_quantum.data.tokenizer import SelfiesTokenizer

logger = logging.getLogger(__name__)

__all__ = ["PriorConditionedLSTM", "bitstrings_to_prior", "LSTMTrainingHistory"]


def bitstrings_to_prior(
    bitstrings: np.ndarray | torch.Tensor, n_qubits: int
) -> torch.Tensor:
    """QCBM olcum sonuclarini (tamsayi) ikili prior vektorlerine cevirir.

    QCBM, bitstring'leri tamsayi olarak dondurur (or. 16 qubit icin 0..65535).
    LSTM'e besleyebilmek icin bunlari {0,1}^n_qubits vektorlerine aciyoruz.

    Parameters
    ----------
    bitstrings : (batch,) tamsayi dizisi
    n_qubits   : bit genisligi

    Returns
    -------
    (batch, n_qubits) float tensor; bit sirasi en anlamlidan en az anlamliya.
    """
    if isinstance(bitstrings, torch.Tensor):
        values = bitstrings.to(torch.int64)
    else:
        values = torch.as_tensor(np.asarray(bitstrings), dtype=torch.int64)

    if values.ndim == 0:
        values = values.unsqueeze(0)

    max_value = 1 << n_qubits
    if torch.any(values < 0) or torch.any(values >= max_value):
        raise ValueError(
            f"Bitstring degerleri [0, {max_value}) araliginda olmali; "
            f"bulunan aralik [{int(values.min())}, {int(values.max())}]"
        )

    shifts = torch.arange(n_qubits - 1, -1, -1, dtype=torch.int64, device=values.device)
    bits = (values.unsqueeze(-1) >> shifts) & 1
    return bits.to(torch.float32)


@dataclass
class LSTMTrainingHistory:
    """Egitim kaydi."""

    train_loss: list[float]
    epochs: int

    @property
    def final_loss(self) -> float | None:
        return self.train_loss[-1] if self.train_loss else None

    @property
    def improvement(self) -> float | None:
        if len(self.train_loss) < 2:
            return None
        return self.train_loss[0] - self.train_loss[-1]


class PriorConditionedLSTM(nn.Module):
    """SELFIES uzerinde calisan, kuantum prior'u ile kosullandirilmis LSTM.

    `prior_dim=0` verilirse model tamamen klasiklesir - makaledeki
    "vanilla LSTM" / "Classical" kolu (Extended Data Fig. 1, Fig. 1d).
    Bu, kuantum prior'un katkisini olcmek icin gereken kontrol grubudur.

    Parameters
    ----------
    vocab_size : SELFIES sozluk boyutu
    prior_dim  : kuantum prior bit genisligi (16 = makale); 0 -> klasik model
    config     : hiperparametreler

    Example
    -------
    >>> model = PriorConditionedLSTM(vocab_size=64, prior_dim=16)
    >>> prior = bitstrings_to_prior(np.array([42, 7]), n_qubits=16)
    >>> tokens = model.sample(n_samples=2, prior=prior, max_length=30, ...)
    """

    def __init__(
        self,
        vocab_size: int,
        prior_dim: int = 16,
        config: LSTMConfig | None = None,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.config = config or LSTMConfig()
        self.vocab_size = vocab_size
        self.prior_dim = prior_dim
        self.pad_id = pad_id
        self.merge = self.config.prior_merge

        if self.merge not in ("concat", "add"):
            raise ValueError(
                f"prior_merge 'concat' veya 'add' olmali, verilen: {self.merge!r}"
            )

        embedding_dim = self.config.embedding_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)

        # Prior'un LSTM girisine katilma bicimi (Makale, Denklem 9)
        if prior_dim > 0:
            if self.merge == "concat":
                # x'_t = X(i) ++ x_t  -> giris boyutu buyur
                lstm_input_dim = embedding_dim + prior_dim
                self.prior_projection = None
            else:
                # x'_t = X(i) + x_t  -> toplama icin boyutlar esitlenmeli;
                # prior, ogrenilebilir bir dogrusal katmanla gomme uzayina tasinir
                lstm_input_dim = embedding_dim
                self.prior_projection = nn.Linear(prior_dim, embedding_dim)
        else:
            lstm_input_dim = embedding_dim
            self.prior_projection = None

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.n_layers,
            dropout=self.config.dropout if self.config.n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.output = nn.Linear(self.config.hidden_dim, vocab_size)

    # ------------------------------------------------------------------
    # Ileri gecis
    # ------------------------------------------------------------------
    def _merge_prior(self, embedded: torch.Tensor, prior: torch.Tensor | None) -> torch.Tensor:
        """Makale, Denklem (9): prior'i token gomulmesiyle birlestirir.

        Prior molekul basina sabittir; her zaman adimina yayilir (broadcast).
        """
        if self.prior_dim == 0 or prior is None:
            return embedded

        batch, seq_len, _ = embedded.shape

        if prior.shape[0] != batch:
            raise ValueError(
                f"Prior yigin boyutu girisle uyusmuyor: {prior.shape[0]} vs {batch}"
            )
        if prior.shape[-1] != self.prior_dim:
            raise ValueError(
                f"Prior boyutu {self.prior_dim} olmali, verilen: {prior.shape[-1]}"
            )

        prior = prior.to(embedded.dtype).to(embedded.device)
        # (batch, prior_dim) -> (batch, seq_len, prior_dim)
        expanded = prior.unsqueeze(1).expand(batch, seq_len, self.prior_dim)

        if self.merge == "concat":
            return torch.cat([embedded, expanded], dim=-1)
        return embedded + self.prior_projection(expanded)

    def forward(
        self,
        tokens: torch.Tensor,
        prior: torch.Tensor | None = None,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        tokens : (batch, seq_len) tamsayi token indeksleri
        prior  : (batch, prior_dim) kuantum prior; klasik modelde None
        hidden : onceki gizli durum (ornekleme sirasinda adim adim uretim icin)

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        hidden : guncellenmis (h, c)
        """
        embedded = self.embedding(tokens)                 # (B, T, E)
        merged = self._merge_prior(embedded, prior)       # Denklem (9)
        output, hidden = self.lstm(merged, hidden)        # Denklem (10)-(15)
        return self.output(self.dropout(output)), hidden

    # ------------------------------------------------------------------
    # Kayip (Makale, Denklem 7-8)
    # ------------------------------------------------------------------
    def loss(
        self,
        tokens: torch.Tensor,
        prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Ogretmen-zorlamali NLL kaybi.

        Makale, Denklem (7): L(y, y_hat) = -log(y_hat_y)
        Makale, Denklem (8): yigin kaybi bireysel kayiplarin ortalamasi.

        Dizi kaydirilir: model, her adimda BIR SONRAKI token'i tahmin eder.
        Dolgu (pad) token'lari kayba katilmaz.
        """
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        logits, _ = self.forward(inputs, prior)

        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_id,  # dolgu kayba katkida bulunmasin
        )

    # ------------------------------------------------------------------
    # Ornekleme
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        tokenizer: SelfiesTokenizer,
        prior: torch.Tensor | None = None,
        max_length: int | None = None,
        temperature: float | None = None,
        device: torch.device | str = "cpu",
    ) -> list[str]:
        """Modelden SELFIES dizeleri ureterek ornek ceker.

        Makale: "we tuned the sampling temperature, which defines the balance
        between determinism and stochasticity in the model".

        temperature < 1 -> daha belirlenimci (yuksek olasilikli token'lar)
        temperature > 1 -> daha cesitli (daha duz dagilim)
        """
        self.eval()
        device = torch.device(device)
        self.to(device)

        max_length = max_length or self.config.max_sequence_length
        temperature = (
            self.config.sampling_temperature if temperature is None else temperature
        )
        if temperature <= 0:
            raise ValueError(f"temperature > 0 olmali, verilen: {temperature}")

        if self.prior_dim > 0:
            if prior is None:
                raise ValueError(
                    "Bu model kuantum prior ile kosullandirilmis; "
                    "prior=None yalnizca klasik modelde (prior_dim=0) gecerli"
                )
            if prior.shape[0] != n_samples:
                raise ValueError(
                    f"Prior satir sayisi n_samples ile esit olmali: "
                    f"{prior.shape[0]} vs {n_samples}"
                )
            prior = prior.to(device)

        # [BOS] ile basla
        tokens = torch.full(
            (n_samples, 1), tokenizer.bos_id, dtype=torch.int64, device=device
        )
        generated = [tokens]
        hidden = None
        finished = torch.zeros(n_samples, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits, hidden = self.forward(tokens, prior, hidden)
            next_logits = logits[:, -1, :] / temperature

            probs = F.softmax(next_logits, dim=-1)
            tokens = torch.multinomial(probs, num_samples=1)

            # Bitmis diziler yalnizca pad uretsin
            tokens = torch.where(
                finished.unsqueeze(-1),
                torch.full_like(tokens, tokenizer.pad_id),
                tokens,
            )
            generated.append(tokens)

            finished = finished | (tokens.squeeze(-1) == tokenizer.eos_id)
            if bool(finished.all()):
                break

        sequences = torch.cat(generated, dim=1).cpu().numpy()
        return tokenizer.decode_batch(sequences)

    # ------------------------------------------------------------------
    # Yardimcilar
    # ------------------------------------------------------------------
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def is_quantum_conditioned(self) -> bool:
        return self.prior_dim > 0

    def describe(self) -> str:
        kol = (
            f"kuantum-kosullu (prior_dim={self.prior_dim}, birlestirme={self.merge!r})"
            if self.is_quantum_conditioned
            else "klasik (vanilla LSTM - prior yok)"
        )
        return (
            f"PriorConditionedLSTM[{kol}] "
            f"sozluk={self.vocab_size}, gomme={self.config.embedding_dim}, "
            f"gizli={self.config.hidden_dim}, katman={self.config.n_layers}, "
            f"parametre={self.n_parameters:,}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.describe()}>"
