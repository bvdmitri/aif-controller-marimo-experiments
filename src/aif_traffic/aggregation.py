"""Per-day and experiment summary roll-ups."""

from __future__ import annotations

import pandas as pd

from .parameters import Params


def build_daily_and_summary(
    step: pd.DataFrame,
    params: Params,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group per-step records into per-day and experiment summaries."""
    group_cols = ["seed", "day"] if "seed" in step.columns else ["day"]

    daily = (
        step
        .groupby(group_cols, as_index=False)
        .agg(
            SC=("SC", "first"),
            max_L2=("L2", "max"),
            max_L6=("L6", "max"),
            mean_P_alpha=("P_alpha", "mean"),
            mean_TT_alpha=("TT_alpha", "mean"),
            mean_TT_beta=("TT_beta", "mean"),
            mean_phi2=("phi2", "mean"),
        )
    )

    if "seed" in step.columns:
        summary = (
            daily
            .groupby("seed", as_index=False)
            .agg(
                mean_SC=("SC", "mean"),
                std_SC=("SC", "std"),
                mean_max_L2=("max_L2", "mean"),
                mean_max_L6=("max_L6", "mean"),
                mean_P_alpha=("mean_P_alpha", "mean"),
                mean_phi2=("mean_phi2", "mean"),
            )
        )
    else:
        summary = pd.DataFrame([{
            "mean_SC": daily["SC"].mean(),
            "std_SC": daily["SC"].std(),
            "mean_max_L2": daily["max_L2"].mean(),
            "mean_max_L6": daily["max_L6"].mean(),
            "mean_P_alpha": daily["mean_P_alpha"].mean(),
            "mean_phi2": daily["mean_phi2"].mean(),
        }])

    return daily, summary
