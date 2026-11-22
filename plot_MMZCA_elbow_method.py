"""Plot MMZCA mean CM-to-CH hop count versus h_max, with precomputed 95% CIs.

This is a standalone plotting script. It reads the merged JSON file and DOES
NOT rerun MMZCA or recompute confidence intervals.

Default input:  MMZCA_elbow_hmax_0-20_merged.json (beside this script)
Default output: MMZCA_elbow_hmax_0-20_publication.png (beside this script)

Run:
    python plot_MMZCA_elbow_method.py

Optional:
    python plot_MMZCA_elbow_method.py --input /path/to/results.json \
        --output /path/to/figure.png --show
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns


# --- Requested figure formatting ---
label_font = 15
ticks_font = 15
figure_size = (15, 5)
line_width = 2.0
marker_size = 5.5
ci_alpha = 0.18

# Follow the original palette, marker, and line-style used in the elbow figure.
DENSITIES = [
    ("smallsize", "High sparsity"),
    ("midsize", "Medium sparsity"),
    ("largesize", "Low sparsity"),
]
TRANSMISSION_RANGES = (100, 200, 300)
SUBPLOT_LABELS = ("(a)", "(b)", "(c)")


def plot_elbow(input_file: Path, output_file: Path, show: bool = False) -> None:
    with input_file.open("r", encoding="utf-8") as f:
        results = json.load(f)

    if "summary" not in results or not isinstance(results["summary"], list):
        raise ValueError("Expected a JSON object with a 'summary' list.")

    # Keep the original, already-computed 95% CI bounds from the JSON.
    by_key = {}
    for row in results["summary"]:
        key = (row["density"], int(row["transmission_range_m"]), int(row["h_max"]))
        if key in by_key:
            raise ValueError(f"Duplicate summary entry: {key}")
        by_key[key] = row

    sns.set(
        rc={
            "font.family": "serif",
            # Times New Roman is used on systems where it is installed (e.g., Mac).
            # The other fonts avoid missing-font warnings on Linux environments.
            "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
            "text.usetex": False,
        },
        style="whitegrid",
    )

    # The colors are those in the existing Seaborn elbow-method figure.
    colors = dict(zip(TRANSMISSION_RANGES, sns.color_palette("deep", 3)))
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=figure_size, sharey=False
    )
    x_values = sorted({h for (_, _, h), v in by_key.items()
                       if h >= 1 and v.get("mean_average_hops") is not None})
    if not x_values:
        raise ValueError("No valid h_max >= 1 observations found in the JSON.")

    for i, (ax, (density, title)) in enumerate(zip(axes.flatten(), DENSITIES)):
        panel_max = 1.0
        for tr in TRANSMISSION_RANGES:
            rows = [by_key[(density, tr, h)] for h in x_values
                    if (density, tr, h) in by_key
                    and by_key[(density, tr, h)].get("mean_average_hops") is not None]
            if not rows:
                continue

            x = np.asarray([p["h_max"] for p in rows], dtype=int)
            mean = np.asarray([p["mean_average_hops"] for p in rows], dtype=float)
            lower = np.asarray([
                np.nan if p.get("ci95_lower") is None else p["ci95_lower"]
                for p in rows
            ], dtype=float)
            upper = np.asarray([
                np.nan if p.get("ci95_upper") is None else p["ci95_upper"]
                for p in rows
            ], dtype=float)

            # Plot the same solid lines with open circular markers for all TRs.
            sns.lineplot(
                x=x, y=mean, ax=ax, color=colors[tr],
                linewidth=line_width, linestyle="-", marker="o",
                markersize=marker_size, markerfacecolor="none",
                markeredgecolor=colors[tr], markeredgewidth=1.0,
                estimator=None, errorbar=None, sort=False, legend=False,
            )

            ci_valid = np.isfinite(lower) & np.isfinite(upper)
            panel_max = max(panel_max, float(np.nanmax(mean)))
            if np.any(ci_valid):
                panel_max = max(panel_max, float(np.nanmax(upper[ci_valid])))
            if np.any(ci_valid):
                ax.fill_between(
                    x, lower, upper, where=ci_valid, interpolate=False,
                    color=colors[tr], alpha=ci_alpha, linewidth=0,
                )

        ax.set_title(title, fontsize=label_font, pad=10)
        ax.set_xlabel(r"$h_{\max}$", fontsize=label_font)
        ax.set_ylabel(
            "Average hops between CMs\nand associated CHs" if i == 0 else "",
            fontsize=label_font,
        )
        ax.xaxis.set_label_coords(0.5, -0.12)
        ax.set_xlim(min(x_values) - 0.35, max(x_values) + 0.35)
        # h_max=0 is undefined and therefore intentionally omitted.
        # Using every second tick keeps 1--20 legible at 15-point font.
        xticks = [1] + list(range(2, max(x_values) + 1, 2))
        ax.set_xticks(sorted(set(xticks)))
        ax.set_ylim(0.90, panel_max + 0.05 * (panel_max - 0.90))

        for spine in ax.spines.values():
            spine.set_color("k")
            spine.set_linewidth(1.0)

        ax.tick_params(
            axis="both", which="major", direction="in",
            length=5, width=1, colors="black",
            bottom=True, top=True, left=True, right=True,
            labelsize=ticks_font,
        )
        ax.grid(True, which="major", color="0.75", alpha=0.75, linewidth=0.75)
        ax.set_axisbelow(True)
        ax.text(
            0.465, -0.31, SUBPLOT_LABELS[i], transform=ax.transAxes,
            fontsize=label_font, va="bottom",
        )

    # One shared legend instead of three repeated legends.
    handles = [
        Line2D(
            [0], [0], color=colors[tr], linestyle="-", linewidth=line_width,
            marker="o", markersize=7, markerfacecolor="none",
            markeredgecolor=colors[tr], markeredgewidth=1.0,
            label=f"TR = {tr} m",
        )
        for tr in TRANSMISSION_RANGES
    ]
    legend = fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
        ncol=3, fontsize=label_font,
        frameon=True, fancybox=False, framealpha=1.0, edgecolor="black",
        handlelength=3, handletextpad=0.5, columnspacing=1.5,
        borderpad=0.35,
    )
    legend.get_frame().set_linewidth(1.2)

    # Reserve space for the shared legend and for (a), (b), (c).
    fig.tight_layout(rect=(0.0, 0.15, 1.0, 0.96), w_pad=1.1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=500, bbox_inches="tight")
    print(f"Figure saved: {output_file.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=base_dir / "MMZCA_elbow_hmax_0-20_merged.json",
        help="Path to the merged MMZCA elbow-method JSON file",
    )
    parser.add_argument(
        "--output", type=Path,
        default=base_dir / "MMZCA_elbow_hmax_0-20_publication.png",
        help="Output PNG path (saved at 500 DPI)",
    )
    parser.add_argument("--show", action="store_true", help="Display the figure")
    args = parser.parse_args()
    plot_elbow(args.input, args.output, show=args.show)


if __name__ == "__main__":
    main()
