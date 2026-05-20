DATA_PATH     = "data/sp500_4yr_train.csv"

# GA
GA_POP        = 150
GA_GENS       = 5   # hard cap — early stopping may terminate before this
GA_PATIENCE   = 5   # stop after this many consecutive stall generations

# Signal processing
EMA_SPAN      = 8
SMA_WINDOW    = 30

# Train / test split
TRAIN_RATIO   = 0.7   # first 70 % of trading days used for GA training (~2.8 yr of 4 yr)

# Gene bounds
# WINDOW_MAX must leave enough bars for valid rolling stats in the training window.
# With TRAIN_RATIO=0.7 and 504 total days → ~353 train days.
# WINDOW_MAX=250 → 353-250=103 usable bars; WINDOW_MIN=130 → 353-130=223 usable bars.
WINDOW_MIN    = 200
WINDOW_MAX    = 350

OPEN_MIN      = 1.5
OPEN_MAX      = 2.5

CLOSE_MIN     = 0.0
CLOSE_MAX     = 0.5

EXIT_STEP_MIN = 0.05
EXIT_STEP_MAX = 0.5

STOP_LOSS_MIN = 2.5   # Gene 9 — dormant in Stage 1
STOP_LOSS_MAX = 3.5

# Fitness
MAG_PENALTY   = 0.5  # applied when |z_start| not in [1, 3]

# Trading rules
MAX_PRIMED_DAYS = 5     # PRIMED state expires if entry not triggered within this many bars
BETA_MIN        = 0.01  # minimum valid hedge ratio (must be positive)
BETA_MAX        = 10.0  # maximum valid hedge ratio
MIN_TRADES      = 1     # minimum trades required for a valid GA solution

# Execution behavior
# If True, entering after recross does not use the day-1 entry guard.
ALLOW_RECROSS_ENTRY_WITHOUT_GUARD = True
# If True, exits use mean reversion to z=0 (or stop-loss), not adaptive close thresholds.
EXIT_ON_ZERO_CROSS_ONLY = False
# If False, momentum filter does not block entries.
ENFORCE_MOMENTUM_FILTER = False
# If False, PRIMED states do not expire by max days.
ENFORCE_PRIMED_EXPIRY = False

OUTPUT_DIR    = "output"

# ── Stage 2 — Monte Carlo / MU ─────────────────────────────────────────────
# Set USE_MONTE_CARLO = True to switch from Stage 1 (historical σ-ROI) to
# Stage 2 (OU-VG Expected Profitability).  Gene 9 stop-loss also activates.
USE_MONTE_CARLO = True   # False = Stage 1 (SU)  |  True = Stage 2 (MU)
MC_N_PATHS      = 50      # simulation paths per chromosome (raise to 500+ for final eval)
MC_N_STEPS      = 90     # forward simulation horizon in trading days (~1 year)
MC_GAMMA        = 0.1     # risk-aversion coeff: obj2 = -(EP - gamma * Var(ROI))
