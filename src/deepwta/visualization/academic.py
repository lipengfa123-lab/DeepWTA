"""Small, experiment-agnostic helpers for publication-quality figures.

The functions accept tidy summary frames.  A metric named ``profile_score`` is
expected to have columns such as ``profile_score_mean``,
``profile_score_ci95_low`` and ``profile_score_ci95_high``.  This keeps the
plotting layer independent of any particular experiment runner.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PALETTE = (
    "#0072B2",  # blue
    "#009E73",  # green
    "#E69F00",  # orange
    "#D55E00",  # vermillion
    "#CC79A7",  # purple
    "#56B4E9",  # sky blue
    "#7F7F7F",  # grey
    "#332288",
    "#88CCEE",
    "#44AA99",
)
DEFAULT_MARKERS = ("o", "s", "^", "D", "P", "v", "X", "h", "<", ">")
DEFAULT_LINESTYLES = ("-", "-", "-.", ":", "--", "-.", "--", ":", "-", "--")


def configure_academic_style(*, base_font_size: float = 9.0) -> None:
    """Apply a compact, color-blind-friendly manuscript style globally."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": base_font_size,
            "axes.labelsize": base_font_size + 1,
            "axes.titlesize": base_font_size + 1,
            "legend.fontsize": base_font_size - 1,
            "xtick.labelsize": base_font_size - 0.5,
            "ytick.labelsize": base_font_size - 0.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.2,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def method_style(method: str, index: int = 0, overrides: Mapping[str, Mapping] | None = None) -> dict:
    """Return a deterministic visual identity for a method."""
    style = {
        "color": DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)],
        "marker": DEFAULT_MARKERS[index % len(DEFAULT_MARKERS)],
        "linestyle": DEFAULT_LINESTYLES[index % len(DEFAULT_LINESTYLES)],
    }
    if overrides and method in overrides:
        style.update(dict(overrides[method]))
    return style


def save_academic_figure(
    fig: plt.Figure,
    output_dir: str | Path,
    name: str,
    *,
    formats: Sequence[str] = ("png", "pdf", "svg"),
    close: bool = True,
) -> list[Path]:
    """Save the same figure in raster and editable vector formats."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in formats:
        path = destination / f"{name}.{suffix.lstrip('.')}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    if close:
        plt.close(fig)
    return paths


def _ordered(frame: pd.DataFrame, key: str, values: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    result[key] = result[key].astype(str)
    result[key] = pd.Categorical(result[key], categories=list(values), ordered=True)
    return result.sort_values(key)


def plot_metric_lines(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    metric: str,
    methods: Sequence[str],
    scales: Sequence[str],
    ylabel: str,
    styles: Mapping[str, Mapping] | None = None,
    log_y: bool = False,
    show_ci: bool = True,
) -> plt.Axes:
    """Draw metric-versus-scale lines with optional 95% confidence bands."""
    x = np.arange(len(scales), dtype=float)
    for index, method in enumerate(methods):
        group = summary[summary["method"].astype(str) == method]
        if group.empty:
            continue
        group = _ordered(group, "scale", scales).set_index("scale").reindex(scales)
        mean = pd.to_numeric(group[f"{metric}_mean"], errors="coerce").to_numpy(float)
        style = method_style(method, index, styles)
        ax.plot(x, mean, label=method, **style)
        low_key, high_key = f"{metric}_ci95_low", f"{metric}_ci95_high"
        if show_ci and low_key in group and high_key in group:
            low = pd.to_numeric(group[low_key], errors="coerce").to_numpy(float)
            high = pd.to_numeric(group[high_key], errors="coerce").to_numpy(float)
            finite = np.isfinite(mean) & np.isfinite(low) & np.isfinite(high)
            if finite.any():
                ax.fill_between(x[finite], low[finite], high[finite], color=style["color"], alpha=0.11, linewidth=0)
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(x, [str(value).capitalize() for value in scales])
    ax.set_xlabel("Problem scale")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    return ax


def plot_grouped_metric(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    metric: str,
    methods: Sequence[str],
    scales: Sequence[str],
    ylabel: str,
    styles: Mapping[str, Mapping] | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Axes:
    """Draw grouped bars with asymmetric 95% confidence intervals."""
    x = np.arange(len(scales), dtype=float)
    width = min(0.78 / max(len(methods), 1), 0.18)
    for index, method in enumerate(methods):
        group = summary[summary["method"].astype(str) == method]
        if group.empty:
            continue
        group = _ordered(group, "scale", scales).set_index("scale").reindex(scales)
        mean = pd.to_numeric(group[f"{metric}_mean"], errors="coerce").to_numpy(float)
        low = pd.to_numeric(group[f"{metric}_ci95_low"], errors="coerce").to_numpy(float)
        high = pd.to_numeric(group[f"{metric}_ci95_high"], errors="coerce").to_numpy(float)
        xpos = x + (index - (len(methods) - 1) / 2.0) * width
        style = method_style(method, index, styles)
        ax.bar(xpos, mean, width, color=style["color"], alpha=0.82, label=method, zorder=3)
        error = np.vstack((np.maximum(mean - low, 0), np.maximum(high - mean, 0)))
        ax.errorbar(xpos, mean, yerr=error, fmt="none", ecolor="black", elinewidth=0.55, capsize=1.4, zorder=4)
    ax.set_xticks(x, [str(value).capitalize() for value in scales])
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", linewidth=0.45, alpha=0.45, zorder=0)
    return ax


def plot_metric_grid(
    summary: pd.DataFrame,
    *,
    panels: Sequence[tuple[str, str, bool]],
    methods: Sequence[str],
    scales: Sequence[str],
    styles: Mapping[str, Mapping] | None = None,
    ncols: int = 2,
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a reusable multi-metric line-panel figure."""
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or (7.2, 2.7 * nrows), squeeze=False)
    for ax, (metric, label, log_y) in zip(axes.flat, panels):
        plot_metric_lines(
            ax, summary, metric=metric, methods=methods, scales=scales,
            ylabel=label, styles=styles, log_y=log_y,
        )
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(methods)), frameon=False, fontsize=7,
    )
    fig.subplots_adjust(bottom=0.23, hspace=0.40, wspace=0.30)
    return fig, axes


def plot_quality_runtime_frontier(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    scale: str,
    methods: Sequence[str],
    quality_metric: str = "profile_score_feasible",
    runtime_metric: str = "total_runtime_ms",
    styles: Mapping[str, Mapping] | None = None,
) -> plt.Axes:
    """Plot a quality/runtime trade-off panel for one scale."""
    frame = summary[summary["scale"].astype(str) == scale]
    for index, method in enumerate(methods):
        row = frame[frame["method"].astype(str) == method]
        if row.empty:
            continue
        x = float(row.iloc[0][f"{runtime_metric}_mean"])
        y = float(row.iloc[0][f"{quality_metric}_mean"])
        if not (np.isfinite(x) and x > 0 and np.isfinite(y)):
            continue
        style = method_style(method, index, styles)
        ax.scatter(x, y, s=48, color=style["color"], marker=style["marker"], zorder=3)
        ax.annotate(method, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("End-to-end runtime (ms, log) ↓")
    ax.set_ylabel("Feasible profile score ↓")
    ax.set_title(str(scale).capitalize())
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.45)
    return ax
