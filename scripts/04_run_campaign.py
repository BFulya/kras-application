"""
KRAS inhibitor tasarim kampanyasi (Makale, Fig. 1b-d).

Uc kolu egitir, her birinden molekul ureter ve Fig. 1d secim hunisini
uygulayarak sentez adaylarini belirler:

    classical    -> prior yok (vanilla LSTM)          -> 3 bilesik
    quantum_sim  -> QCBM, Aer simulatorunde           -> 4 bilesik
    quantum_hw   -> QCBM, gercek IBM donaniminda      -> 8 bilesik
                                                  toplam 15 bilesik

Olcek farki
-----------
Makale her koldan 1.000.000 bilesik ornekler ve QCBM'i 30 epoch egitir.
Varsayilanlar burada cok daha kucuktur (tek makinede calissin diye);
makale olcegi icin --n-samples ve --n-epochs ile buyutun.

Gercek kuantum donanimi
-----------------------
--arms quantum_hw, IBM Quantum hesabi gerektirir:
    pip install qiskit-ibm-runtime
    python -c "from qiskit_ibm_runtime import QiskitRuntimeService; \\
               QiskitRuntimeService.save_account(channel='ibm_quantum', token='...')"
Her epoch donanimda is kuyruguna girer; makaledeki 30 epoch'luk kosu
saatler surebilir.

Kullanim:
    python scripts/04_run_campaign.py
    python scripts/04_run_campaign.py --arms classical quantum_sim --n-epochs 10
    python scripts/04_run_campaign.py --pli vina   # nihai siralamada gercek docking
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kras_quantum.config import (  # noqa: E402
    DATA_CACHE_DIR,
    RESULTS_DIR,
    CampaignConfig,
    Config,
    LSTMConfig,
    QCBMConfig,
)
from kras_quantum.data import SelfiesTokenizer  # noqa: E402
from kras_quantum.pipeline import HybridGenerator, select_candidates  # noqa: E402
from kras_quantum.scoring import RewardFunction, get_pli_scorer  # noqa: E402


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_training_data(cache_dir: Path) -> tuple[SelfiesTokenizer, np.ndarray, list[str]]:
    """03_build_training_set.py ciktisini yukler."""
    sequences_path = cache_dir / "training_sequences.npy"
    tokenizer_path = cache_dir / "tokenizer.json"
    smiles_path = cache_dir / "training_smiles.txt"

    if not sequences_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Egitim verisi bulunamadi: {cache_dir}\n"
            f"Once olusturun:\n"
            f"    python scripts/03_build_training_set.py"
        )

    sequences = np.load(sequences_path)
    tokenizer = SelfiesTokenizer.load(tokenizer_path)
    training_smiles = (
        smiles_path.read_text(encoding="utf-8").splitlines()
        if smiles_path.exists()
        else []
    )
    return tokenizer, sequences, training_smiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["classical", "quantum_sim"],
        choices=["classical", "quantum_sim", "quantum_hw"],
        help="Calistirilacak kollar (quantum_hw gercek IBM cihazi ister)",
    )
    parser.add_argument("--n-epochs", type=int, default=10, help="Hibrit dongu adedi (makale: 30)")
    parser.add_argument("--n-samples", type=int, default=2000, help="Kol basina uretim (makale: 1.000.000)")
    parser.add_argument("--n-qubits", type=int, default=16, help="QCBM qubit sayisi (makale: 16)")
    parser.add_argument("--n-generate-per-epoch", type=int, default=128)
    parser.add_argument(
        "--pli",
        default="surrogate",
        choices=["surrogate", "vina", "auto"],
        help="PLI skorlayici. Egitimde 'surrogate' zorunlu (hiz); "
        "nihai siralamada 'vina' onerilir",
    )
    parser.add_argument("--min-novelty", type=float, default=0.4)
    parser.add_argument("--cache-dir", type=Path, default=DATA_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    setup_logging(not args.quiet)

    print("=" * 72)
    print("KRAS INHIBITOR TASARIM KAMPANYASI (Makale, Fig. 1b-d)")
    print("=" * 72)

    # --- egitim verisi ---
    tokenizer, sequences, training_smiles = load_training_data(args.cache_dir)
    print(f"\nEgitim seti : {sequences.shape[0]:,} molekul, sozluk {tokenizer.vocab_size}")
    print(f"Kollar      : {', '.join(args.arms)}")
    print(f"Epoch       : {args.n_epochs} (makale: 30)")
    print(f"Uretim/kol  : {args.n_samples:,} (makale: 1.000.000)")
    print(f"QCBM        : {args.n_qubits} qubit (makale: 16)")

    config = Config(
        qcbm=QCBMConfig(n_qubits=args.n_qubits, seed=args.seed),
        lstm=LSTMConfig(seed=args.seed),
    )
    campaign = CampaignConfig()

    # Egitim odulunde daima hizli vekil kullanilir; gercek docking
    # molekul basina ~10 saniye surer ve dongude uygulanamaz.
    training_reward = RewardFunction(pli_scorer=get_pli_scorer("surrogate"))

    molecules_by_arm: dict[str, list[str]] = {}
    histories: dict[str, list[dict]] = {}

    for arm in args.arms:
        print(f"\n{'-' * 72}\nKOL: {arm}\n{'-' * 72}")
        start = time.time()

        generator = HybridGenerator(
            arm=arm,
            tokenizer=tokenizer,
            config=config,
            reward_function=training_reward,
        )
        print(generator.describe())

        # --- egitim (Algorithm 1) ---
        history = generator.fit(
            sequences,
            n_epochs=args.n_epochs,
            n_generate_per_epoch=args.n_generate_per_epoch,
            verbose=not args.quiet,
        )
        histories[arm] = [
            {
                "epoch": r.epoch,
                "stage": r.reward_stage,
                "validity_rate": r.validity_rate,
                "mean_reward": r.mean_reward,
                "max_reward": r.max_reward,
                "lstm_loss": r.lstm_loss,
                "qcbm_loss_before": r.qcbm_loss_before,
                "qcbm_loss_after": r.qcbm_loss_after,
                "duration_s": r.duration_s,
            }
            for r in history
        ]

        # --- uretim ---
        print(f"\n  {args.n_samples:,} bilesik uretiliyor...")
        result = generator.generate(args.n_samples)
        molecules_by_arm[arm] = result.smiles

        gecerli = sum(1 for s in result.smiles if s)
        benzersiz = len({s for s in result.smiles if s})
        print(
            f"  uretildi: {len(result.smiles):,} | gecerli: {gecerli:,} "
            f"| benzersiz: {benzersiz:,} | sure: {time.time() - start:.0f}s"
        )

    # --- secim (Fig. 1d) ---
    print(f"\n{'=' * 72}\nSECIM AKISI (Makale, Fig. 1d)\n{'=' * 72}")

    selection_reward = RewardFunction(pli_scorer=get_pli_scorer(args.pli))
    # Secimde odul daima TAM asamada olmali (PLI + farmakofor + SA bilesenleri
    # hesaplansin diye); bunun icin local-filter asamasinin bittigi epoch'u
    # geciyoruz.
    selection_reward.set_epoch(selection_reward.config.local_filter_until_epoch)
    if args.pli == "vina":
        print("  Nihai siralamada GERCEK docking kullaniliyor (yavas)...")

    selected, report = select_candidates(
        molecules_by_arm,
        selection_reward,
        training_smiles=training_smiles or None,
        campaign=campaign,
        min_novelty=args.min_novelty,
        verbose=False,
    )

    print(report)

    if selected:
        print(f"\n{'=' * 72}\nSECILEN SENTEZ ADAYLARI\n{'=' * 72}")
        for index, candidate in enumerate(selected, 1):
            print(f"{index:2d}. {candidate}")
    else:
        print(
            "\nHicbir bilesik tum kriterleri gecemedi.\n"
            "Daha uzun egitim (--n-epochs), daha fazla uretim (--n-samples) "
            "veya daha gevsek yenilik esigi (--min-novelty) deneyin."
        )

    # --- kaydet ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "arms": args.arms,
            "n_epochs": args.n_epochs,
            "n_samples": args.n_samples,
            "n_qubits": args.n_qubits,
            "pli_scorer": args.pli,
            "min_novelty": args.min_novelty,
            "seed": args.seed,
        },
        "paper_reference": {
            "n_epochs": QCBMConfig().n_epochs,
            "n_samples_per_model": campaign.n_samples_per_model,
            "n_selected_total": campaign.n_selected_total,
            "selection_split": {
                "classical": campaign.n_selected_classical,
                "quantum_hw": campaign.n_selected_quantum_hw,
                "quantum_sim": campaign.n_selected_quantum_sim,
            },
        },
        "training_history": histories,
        "selection_report": {
            "n_input": report.n_input,
            "n_after_validity": report.n_after_validity,
            "n_after_filters": report.n_after_filters,
            "n_after_pharmacophore": report.n_after_pharmacophore,
            "n_after_pli": report.n_after_pli,
            "n_after_novelty": report.n_after_novelty,
            "n_after_synthetic": report.n_after_synthetic,
            "n_clusters": report.n_clusters,
            "n_selected": report.n_selected,
            "per_arm": report.per_arm,
        },
        "candidates": [
            {
                "rank": index,
                "smiles": c.smiles,
                "arm": c.arm,
                "total_reward": c.total_reward,
                "pli_kcal_mol": c.pli,
                "pharmacophore": c.reward.pharmacophore_score,
                "sa_score": c.reward.sa_raw,
                "novelty": c.novelty,
                "cluster_id": c.cluster_id,
            }
            for index, c in enumerate(selected, 1)
        ],
    }

    out_path = args.out_dir / "campaign_results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSonuclar kaydedildi: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
