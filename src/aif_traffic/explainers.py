"""Simulation-mechanics explainers rendered at the end of simulation notebooks.

Centralised prose so the per-notebook "How the simulation actually works" cells
share one source of truth. Right now the repository ships only the markdown
landing page (notebook 00), so there are no per-notebook addenda yet -- this
module grows as the experiment notebooks are added, in lockstep with the code
paths they describe.

Public API mirrors the IWAI companion repo:

* ``notebook_explainer(nb_id)`` -- full markdown for an end-of-notebook cell.
* ``explainer_pointer()``        -- short top-of-notebook pointer.
* ``NOTEBOOK_IDS``               -- the canonical set of simulation-notebook IDs.
"""

from __future__ import annotations

# No simulation notebooks yet (only the 00 landing page). Populate as the
# experiment notebooks are designed.
NOTEBOOK_IDS: tuple[str, ...] = ()


def explainer_pointer() -> str:
    return (
        "> **Implementation reference.** Scroll to *How the simulation actually "
        "works* at the end of this notebook for a concise spec of the coupled "
        "traveller and controller loop, against which the code can be verified."
    )


def notebook_explainer(nb_id: str) -> str:
    if nb_id not in NOTEBOOK_IDS:
        raise KeyError(
            f"Unknown notebook id {nb_id!r}. Known ids: {NOTEBOOK_IDS}. "
            "Add an entry here when you add the corresponding simulation notebook."
        )
    raise NotImplementedError  # pragma: no cover - no sim notebooks yet
