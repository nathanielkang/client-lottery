"""Design constants for the missing-experiment battery.

These rows are a separate protocol from the T=50000 frozen grid.
"""

from __future__ import annotations

import numpy as np

# Invitation p. Harmful clients (index 2) drop most often after being invited.
DROPOUT_P = 0.5
DROPOUT_RATES = np.array([0.05, 0.12, 0.65, 0.08, 0.22], dtype=np.float64)

# Three public regimes. Phase 1 is the locked frozen game. Phase 2 compresses
# the mains on a negative intercept. Phase 3 compresses them further on a
# positive intercept, so the last-state effect is not the path average.
PHASE_SCALES = (1.0, 0.40, 0.30)
PHASE_INTERCEPTS = (0.0, -1.80, 2.40)

# Middle phase removes this client from the lottery. Not used inside drift_strong.
ELIGIBILITY_CLIENT = 4

RIDGE_ALPHA = 1.0
N_SIMS = (200, 1000)
SURROGATE_LABEL = "surrogate_not_FedCCEA_paper"

RARE_P = 0.1
RARE_T_CAP = 1_500_000

PAPER_MIN_T = 50_000
PAPER_SEEDS = (0, 1, 2, 3, 4)
# Long enough that p=0.3 still hits every coalition in a smoke draw.
SMOKE_T = 4000
SMOKE_SEEDS = (0, 1)
