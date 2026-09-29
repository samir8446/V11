"""Central defaults. Everything user-tunable in the app starts here."""
NOMINAL_AH = 2.0
EOL_AH = 1.4                 # single EOL definition (absolute, on rate-normalised capacity)
EOL_CONSECUTIVE = 3          # EOL = below threshold for k consecutive cycles
HAMPEL_WINDOW = 7            # half-window in cycles
HAMPEL_K = 3.0
BOL_CYCLES = 5               # cycles used for the robust beginning-of-life capacity
PLAUSIBLE_AH = (0.8, 2.3)    # capacities outside are treated as logging artefacts
REF_TEMP_C = 24.0
DEFAULT_RATE_COEF = -0.035   # d ln(C) / dI  (1/A) fleet default, used only when a cell cannot identify it
HORIZONS = (10, 25, 50)      # forecast horizons in cycles
BAND_ALPHA = 0.10            # nominal 90 % bands
SEED = 7
CURVE_POINTS = 60            # points kept per discharge curve for streaming models
