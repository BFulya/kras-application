"""
LSTM egitim dongusu.

Makale referansi: "Classical model: LSTM model"

    "Training an LSTM involves optimizing the network's weights and biases to
     minimize a specific loss function. This is typically accomplished using
     gradient-based optimization algorithms such as stochastic gradient descent
     or Adam. The backpropagation through time algorithm was used to compute
     the gradients relative to the loss function... The networks were trained
     using the Adam optimizer with the NLL loss function; to mitigate
     overfitting, regularization techniques such as dropout were implemented."

Makale, LSTM'i 40 epoch egitir (bkz. "Benchmark setup").
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from kras_quantum.classical.lstm import LSTMTrainingHistory, PriorConditionedLSTM

logger = logging.getLogger(__name__)

__all__ = ["train_lstm"]


def train_lstm(
    model: PriorConditionedLSTM,
    sequences: np.ndarray | torch.Tensor,
    priors: np.ndarray | torch.Tensor | None = None,
    n_epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    device: torch.device | str = "cpu",
    grad_clip: float = 5.0,
    verbose: bool = True,
) -> LSTMTrainingHistory:
    """Uretici LSTM'i ogretmen zorlamasiyla egitir.

    Parameters
    ----------
    model
        Egitilecek model. `model.prior_dim > 0` ise `priors` zorunludur.
    sequences : (n_molekul, seq_len)
        Tokenlastirilmis SELFIES dizileri.
    priors : (n_molekul, prior_dim), opsiyonel
        Her molekule eslik eden kuantum prior vektoru. Klasik kolda None.
    grad_clip
        Gradyan normu kirpma esigi. RNN'lerde patlayan gradyan sik gorulur;
        makale bunu acikca belirtmez, ancak BPTT ile egitimde standart
        uygulamadir ve egitimi kararli kilar.

    Returns
    -------
    LSTMTrainingHistory
        Epoch basina ortalama kayip.
    """
    config = model.config
    n_epochs = n_epochs if n_epochs is not None else config.n_epochs
    batch_size = batch_size if batch_size is not None else config.batch_size
    learning_rate = learning_rate if learning_rate is not None else config.learning_rate

    device = torch.device(device)
    model.to(device)

    seq_tensor = torch.as_tensor(np.asarray(sequences), dtype=torch.int64)
    if seq_tensor.ndim != 2:
        raise ValueError(f"sequences (n, seq_len) seklinde olmali, verilen: {seq_tensor.shape}")
    if seq_tensor.shape[1] < 2:
        raise ValueError("Diziler en az 2 token uzunlugunda olmali (giris + hedef)")

    if model.prior_dim > 0:
        if priors is None:
            raise ValueError(
                "Model kuantum prior ile kosullandirilmis; `priors` verilmeli. "
                "Klasik kontrol kolu icin modeli prior_dim=0 ile kurun."
            )
        prior_tensor = torch.as_tensor(np.asarray(priors), dtype=torch.float32)
        if prior_tensor.shape[0] != seq_tensor.shape[0]:
            raise ValueError(
                f"priors ve sequences ayni sayida satira sahip olmali: "
                f"{prior_tensor.shape[0]} vs {seq_tensor.shape[0]}"
            )
        dataset = TensorDataset(seq_tensor, prior_tensor)
    else:
        if priors is not None:
            logger.warning(
                "Model klasik (prior_dim=0); verilen `priors` yok sayiliyor."
            )
        dataset = TensorDataset(seq_tensor)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    # Makale: Adam optimize edici
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[float] = []
    model.train()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            tokens = batch[0].to(device)
            prior = batch[1].to(device) if len(batch) > 1 else None

            optimizer.zero_grad()
            loss = model.loss(tokens, prior)  # NLL, Denklem (7)-(8)
            loss.backward()                   # zamanda geri yayilim (BPTT)

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        mean_loss = epoch_loss / max(n_batches, 1)
        history.append(mean_loss)

        if verbose and (epoch % max(n_epochs // 10, 1) == 0 or epoch == n_epochs - 1):
            logger.info("LSTM epoch %d/%d - NLL kaybi: %.4f", epoch + 1, n_epochs, mean_loss)

    return LSTMTrainingHistory(train_loss=history, epochs=n_epochs)
