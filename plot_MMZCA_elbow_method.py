"""Plot ONE selectable MMZCA metric versus h_max with its precomputed 95% CIs.

Standalone: reads the existing experiment JSON; does not rerun MMZCA or
recalculate confidence intervals. Supports both the original hop-only JSON and
the newer all-metrics JSON (summary[].metrics).

Run interactively:
    python plot_MMZCA_elbow_method.py

The program lists available metrics, then asks for ONE metric and its y-label.
Line styles and markers are fixed to the original publication figure: solid
lines and open circular markers for the three transmission ranges.

Optional noninteractive usage:
    python plot_MMZCA_elbow_method.py --input /path/to/mmzca_elbow_all_metrics.json \\
        --metric avg_ch_percentage --y-label 'Average CH percentage (%)' \\
        --output /path/to/figure.png --show

Use --list-metrics to inspect metric names without drawing a figure.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns


# --- Requested figure formatting (unchanged from the original) ---
label_font = 15
ticks_font = 15
figure_size = (15, 5)
line_width = 2.0
marker_size = 5.5
ci_alpha = 0.18

DENSITIES = [
    ("smallsize", "High sparsity"),
    ("midsize", "Medium sparsity"),
    ("largesize", "Low sparsity"),
]
TRANSMISSION_RANGES = (100, 200, 300)
SUBPLOT_LABELS = ("(a)", "(b)", "(c)")

METRIC_ALIASES = {
    "mean_average_hops": "average_hops",
    "avg_hops": "average_hops",
    # Optional legacy shorthand, not an actual key in the JSON.
    "CCR_M": "VCSM_CM.CCR_CM",
}


def default_input_path(base_dir: Path) -> Path:
    candidates = [
        base_dir / "elbow_method_results" / "mmzca_elbow_all_metrics.json",
        base_dir / "mmzca_elbow_all_metrics.json",
        base_dir / "MMZCA_elbow_hmax_0-20_merged.json",
        base_dir / '/Users/pouyafirouzmakan/Desktop/mmzca_elbow_all_metrics.json'
    ]
    return next((p for p in candidates if p.is_file()), candidates[0])


def load_results(path: Path) -> tuple[dict, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {path}\n"
            "Specify --input /path/to/mmzca_elbow_all_metrics.json"
        )
    with path.open("r", encoding="utf-8") as file:
        results = json.load(file)
    if not isinstance(results, dict) or not isinstance(results.get("summary"), list):
        raise ValueError("Expected JSON containing a 'summary' list.")

    by_key = {}
    for row in results["summary"]:
        key = (row["density"], int(row["transmission_range_m"]), int(row["h_max"]))
        if key in by_key:
            raise ValueError(f"Duplicate summary entry: {key}")
        by_key[key] = row
    return results, by_key


def available_metrics(results: dict) -> list[str]:
    names = set()
    for row in results["summary"]:
        names.update((row.get("metrics") or {}).keys())
        if "mean_average_hops" in row:
            names.add("average_hops")
    return sorted(names, key=str.casefold)


def resolve_metric(requested: str, choices: list[str]) -> str:
    name = requested.strip()
    if name in METRIC_ALIASES:
        alias = METRIC_ALIASES[name]
        print(f"Metric alias: {name!r} -> {alias!r}")
        name = alias
    if name in choices:
        return name

    direct = [key for key in choices if key.casefold() == name.casefold()]
    if len(direct) == 1:
        return direct[0]
    suffixes = [key for key in choices if key.rsplit(".", 1)[-1].casefold() == name.casefold()]
    if len(suffixes) == 1:
        print(f"Metric: {requested!r} -> {suffixes[0]!r}")
        return suffixes[0]
    if len(suffixes) > 1:
        raise ValueError(f"Ambiguous metric {requested!r}. Choose one of: {suffixes}")
    suggestions = difflib.get_close_matches(name, choices, n=5, cutoff=0.4)
    raise ValueError(
        f"Unknown metric {requested!r}."
        + (f" Closest available: {suggestions}." if suggestions else "")
        + " Use --list-metrics to see all available names."
    )


def get_stats(row: dict, metric: str) -> tuple[float, float, float] | None:
    """Only read precalculated mean and CI: no recalculation of confidence bands."""
    entry = (row.get("metrics") or {}).get(metric)
    if isinstance(entry, dict):
        value = entry.get("mean")
        lower = entry.get("ci95_lower")
        upper = entry.get("ci95_upper")
    elif metric == "average_hops":  # Backwards compatible with hop-only JSON.
        value = row.get("mean_average_hops")
        lower = row.get("ci95_lower")
        upper = row.get("ci95_upper")
    else:
        return None
    if value is None or not math.isfinite(float(value)):
        return None
    return (
        float(value),
        np.nan if lower is None else float(lower),
        np.nan if upper is None else float(upper),
    )


def default_y_label(metric: str) -> str:
    return (
        "Average hops between CMs\nand associated CHs"
        if metric == "average_hops" else metric
    )


def plot_elbow(by_key: dict, metric: str, y_label: str,
               output_file: Path, show: bool = False) -> None:
    sns.set(
        rc={
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "mathtext.default": "it",
            "text.usetex": False,
        },
        style="whitegrid",
    )

    # Keep the same TR palette, solid lines, open circular markers, and CI bands.
    colors = dict(zip(TRANSMISSION_RANGES, sns.color_palette("deep", 3)))
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figure_size, sharey=False)
    x_values = sorted({
        h for (_, _, h), row in by_key.items()
        if h >= 1 and get_stats(row, metric) is not None
    })
    if not x_values:
        raise ValueError(f"No h_max >= 1 observations for metric {metric!r}.")

    for i, (ax, (density, title)) in enumerate(zip(axes.flatten(), DENSITIES)):
        panel_min = float("inf")
        panel_max = -float("inf")
        for tr in TRANSMISSION_RANGES:
            data = []
            for h in x_values:
                row = by_key.get((density, tr, h))
                stats = get_stats(row, metric) if row is not None else None
                if stats is not None:
                    data.append((h, *stats))
            if not data:
                continue

            x = np.asarray([v[0] for v in data], dtype=int)
            mean = np.asarray([v[1] for v in data], dtype=float)
            lower = np.asarray([v[2] for v in data], dtype=float)
            upper = np.asarray([v[3] for v in data], dtype=float)

            sns.lineplot(
                x=x, y=mean, ax=ax, color=colors[tr],
                linewidth=line_width, linestyle="-", marker="o",
                markersize=marker_size, markerfacecolor="none",
                markeredgecolor=colors[tr], markeredgewidth=1.0,
                estimator=None, errorbar=None, sort=False, legend=False,
            )

            ci_valid = np.isfinite(lower) & np.isfinite(upper)
            panel_min = min(panel_min, float(np.min(mean)))
            panel_max = max(panel_max, float(np.max(mean)))
            if np.any(ci_valid):
                panel_min = min(panel_min, float(np.min(lower[ci_valid])))
                panel_max = max(panel_max, float(np.max(upper[ci_valid])))
                ax.fill_between(
                    x, lower, upper, where=ci_valid, interpolate=False,
                    color=colors[tr], alpha=ci_alpha, linewidth=0,
                )

        ax.set_title(title, fontsize=label_font, pad=10)
        ax.set_xlabel(r"maximum allowed depth ($H_{\max}$)", fontsize=label_font)
        ax.set_ylabel(y_label if i == 0 else "", fontsize=label_font)
        ax.xaxis.set_label_coords(0.5, -0.12)
        ax.set_xlim(min(x_values) - 0.35, max(x_values) + 0.35)
        xticks = [1] + list(range(2, max(x_values) + 1, 1))
        ax.set_xticks(sorted(set(xticks)))

        if not math.isfinite(panel_min) or not math.isfinite(panel_max):
            raise ValueError(f"No observations for {metric!r} in {density!r}.")
        if metric == "average_hops":
            # Preserve the original hop figure's y-axis behavior exactly.
            panel_max = max(1.0, panel_max)
            ax.set_ylim(0.90, panel_max + 0.05 * (panel_max - 0.90))
        else:
            # Different metrics can be below 0.9, near zero, or negative.
            # Only the scale changes, not the figure's layout or styling.
            span = panel_max - panel_min
            pad = max(0.05 * span, abs(panel_max) * 0.001, 0.001)
            ax.set_ylim(panel_min - pad, panel_max + pad)

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
    fig.tight_layout(rect=(0.0, 0.15, 1.0, 0.96), w_pad=1.1)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=500, bbox_inches="tight")
    print(f"Figure saved: {output_file.resolve()}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=default_input_path(base_dir),
                        help="Path to all-metrics or original hop-only JSON")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output PNG path (default: metric-specific publication PNG)")
    parser.add_argument("--show", action="store_true", help="Display the figure")
    parser.add_argument("--list-metrics", action="store_true",
                        help="List metrics available in the JSON and exit")
    parser.add_argument("--metric", type=str, default=None,
                        help="ONE metric name; if omitted, ask interactively")
    parser.add_argument("--y-label", type=str, default=None,
                        help="Y-axis label; if omitted, ask interactively")
    args = parser.parse_args()

    results, by_key = load_results(args.input.expanduser().resolve())
    choices = available_metrics(results)
    if args.list_metrics:
        print("\nAvailable metrics in the JSON:\n" + "\n".join("  " + x for x in choices))
        return
    print(f"Loaded {len(by_key)} summary configurations from {args.input}")
    print("Available metrics (exact names or unique suffixes):")
    print("  " + ", ".join(choices))

    requested = args.metric if args.metric is not None else input("\nMetric to plot: ").strip()
    if not requested:
        raise ValueError("Select one metric from the available names.")
    if requested.lstrip().startswith("["):
        raise ValueError("Enter ONE metric name, not a list of metrics.")
    metric = resolve_metric(requested, choices)

    fallback_label = default_y_label(metric)
    y_label = args.y_label
    if y_label is None:
        entered = input(f"Y-axis label [Enter for {fallback_label!r}]: ").strip()
        y_label = entered if entered else fallback_label

    output = (args.output.expanduser().resolve() if args.output is not None
              else base_dir / f"MMZCA_elbow_{re.sub(r'[^A-Za-z0-9_-]+', '_', metric)}_publication.png")
    plot_elbow(by_key, metric, y_label, output, show=args.show)


if __name__ == "__main__":
    main()
