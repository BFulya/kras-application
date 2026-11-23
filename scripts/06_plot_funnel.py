"""
Secim hunisini gorsellestirir (Makale, Fig. 1d).

Kampanya ciktisindaki (results/campaign_results.json) huni sayilarini
paylasima uygun tek bir PNG olarak cizer.

Neden logaritmik eksen?
    Huni 6.000'den bir avuc bilesige iner. Dogrusal eksende son adimlar
    gorunmez olur; log eksen her adimi okunur kilar. Eksen etiketinde bu
    acikca belirtilir, cunku log eksen dikkatsiz okunursa azalmayi
    oldugundan kucuk gosterir.

Kullanim:
    python scripts/06_plot_funnel.py
    python scripts/06_plot_funnel.py --out results/funnel.png --dark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kras_quantum.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402

# dataviz referans paleti (references/palette.md)
LIGHT = {
    "surface": "#fcfcfb",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "series": "#2a78d6",
    "grid": "#d9d8d4",
}
DARK = {
    "surface": "#1a1a19",
    "text_primary": "#f0efec",
    "text_secondary": "#c3c2b7",
    "series": "#3987e5",
    "grid": "#3a3a37",
}

# Huni adimlari: (JSON anahtari, etiket)
STEPS = [
    ("n_input", "Üretilen bileşik"),
    ("n_after_validity", "Geçerli molekül"),
    ("n_after_filters", "Yapısal filtreleri geçen"),
    ("n_after_pharmacophore", "Farmakofor eşiği (> 0,7)"),
    ("n_after_pli", "Bağlanma skoru (< −8 kcal/mol)"),
    ("n_after_novelty", "Yapısal olarak yeni"),
    ("n_after_synthetic", "Sentezlenebilir (SA < 5)"),
    ("n_selected", "Seçilen sentez adayı"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=RESULTS_DIR / "campaign_results.json"
    )
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "funnel.png")
    parser.add_argument("--dark", action="store_true", help="Koyu tema")
    parser.add_argument("--width", type=float, default=10.8, help="Genislik (inc)")
    parser.add_argument("--height", type=float, default=10.8, help="Yukseklik (inc)")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Kampanya sonucu bulunamadi: {args.results}")
        print("Once calistirin: python scripts/04_run_campaign.py")
        return 1

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    report = payload["selection_report"]
    config = payload.get("config", {})

    labels = [label for _, label in STEPS]
    values = [int(report[key]) for key, _ in STEPS]

    theme = DARK if args.dark else LIGHT

    fig, ax = plt.subplots(figsize=(args.width, args.height), dpi=100)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])

    y = range(len(labels))
    # Log eksende 0 cizilemez; en az 1'den baslat
    plot_values = [max(v, 1) for v in values]

    bars = ax.barh(
        list(y),
        plot_values,
        height=0.62,
        color=theme["series"],
        zorder=3,
    )
    # Veri uclarini yuvarlat (dataviz: 4px rounded data-ends)
    for bar in bars:
        bar.set_joinstyle("round")

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=13, color=theme["text_primary"])
    ax.invert_yaxis()
    # Etiketler cubuga degmesin
    ax.tick_params(axis="y", pad=12)

    ax.set_xscale("log")
    ax.set_xlim(1, max(plot_values) * 2.6)

    # Bilimsel gosterim (10^0, 10^1 ...) yerine duz sayilar: gorsel genel
    # okuyucuya gidiyor, us gosterimi gereksiz bir okuma yuku bindiriyor.
    ticks = [1, 10, 100, 1000, 10000]
    ticks = [t for t in ticks if t <= max(plot_values) * 2.6]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:,}".replace(",", ".") for t in ticks])
    ax.minorticks_off()

    # Deger etiketleri - barin hemen sagina
    for index, (bar, value) in enumerate(zip(bars, values)):
        ax.text(
            bar.get_width() * 1.18,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,}".replace(",", "."),
            va="center",
            ha="left",
            fontsize=13,
            fontweight="bold" if index == len(values) - 1 else "normal",
            color=theme["text_primary"],
        )

    # Recessive grid ve eksenler
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(axis="x", colors=theme["text_secondary"], labelsize=11)
    ax.tick_params(axis="y", length=0)

    ax.set_xlabel(
        "Bileşik sayısı (logaritmik ölçek)",
        fontsize=11,
        color=theme["text_secondary"],
        labelpad=10,
    )

    # Baslik
    first, last = values[0], values[-1]
    fig.suptitle(
        f"{first:,}".replace(",", ".") + f" molekülden {last} sentez adayına",
        x=0.025,
        y=0.962,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=theme["text_primary"],
    )
    arms = ", ".join(config.get("arms", []))
    fig.text(
        0.025,
        0.922,
        f"Kuantum–klasik üretici model · seçim hunisi   |   "
        f"{config.get('n_qubits', '?')} qubit QCBM + LSTM · kollar: {arms}",
        fontsize=12.5,
        color=theme["text_secondary"],
        ha="left",
        va="top",
    )

    # Dipnot - kaynak ve sinirlar
    fig.text(
        0.02,
        0.025,
        "Nature Biotechnology 43, 1954–1959 (2025) yönteminin açık kaynak uygulaması · "
        "bağlanma skorları AutoDock Vina\n"
        "Üretilenler hesaplamalı adaylardır; deneysel doğrulama yapılmamıştır.",
        fontsize=9.5,
        color=theme["text_secondary"],
        ha="left",
        va="bottom",
    )

    fig.subplots_adjust(left=0.305, right=0.955, top=0.885, bottom=0.115)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor(), bbox_inches=None)
    print(f"Gorsel kaydedildi: {args.out}")
    print(f"Huni: {first:,} -> {last}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
