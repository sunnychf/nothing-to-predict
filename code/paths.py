"""Repository-relative default locations. The code lives in <root>/code, so results are
written to <root>/results and model weights are looked for in <root>/weights wherever the
repository is checked out; every script also accepts an environment-variable override
(PILOT_A_OUT, SPEC_OUT, REAL_OUT, REAL_WINDOWS, FINCAST_WEIGHTS), named in its header."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
WEIGHTS_DIR = os.path.join(ROOT, "weights")
