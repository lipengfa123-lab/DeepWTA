"""Reusable visualization helpers for DeepWTA experiments."""

from .academic import (
    DEFAULT_PALETTE,
    configure_academic_style,
    method_style,
    plot_grouped_metric,
    plot_metric_grid,
    plot_metric_lines,
    plot_quality_runtime_frontier,
    save_academic_figure,
)

__all__ = [
    "DEFAULT_PALETTE",
    "configure_academic_style",
    "method_style",
    "plot_grouped_metric",
    "plot_metric_grid",
    "plot_metric_lines",
    "plot_quality_runtime_frontier",
    "save_academic_figure",
]
