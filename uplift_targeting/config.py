"""Project-wide settings: paths, seeds, and the economics of sending an email."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# Kevin Hillstrom's MineThatData e-mail experiment (2008): 64,000 customers randomly
# assigned to a men's merchandise email, a women's merchandise email, or no email.
DATA_URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
DATA_FILE = DATA_DIR / "hillstrom.csv"
DATA_SHA256 = "0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece"

SEED = 7
TEST_SIZE = 0.4  # uplift evaluation is noisy, so the held-out test set is kept large
N_FOLDS = 5
N_BOOTSTRAP = 2000

# Arm code = position in this tuple. Code 0 is the control group.
ARMS = ("No E-Mail", "Mens E-Mail", "Womens E-Mail")
ARM_LABELS = ("No email", "Men's email", "Women's email")

# Economics. These are assumptions, not facts about the retailer, so every headline
# result is also reported across a range of costs (see the sensitivity analysis).
MARGIN = 0.30  # contribution margin on incremental revenue
COST_PER_EMAIL = 0.10  # all-in cost per email: sending plus unsubscribe / fatigue risk


# --- Criteo replication (python -m uplift_targeting.criteo) -----------------------------
# Criteo Uplift Prediction Dataset v2.1 (Diemert et al. 2018): 13,979,592 users in
# randomized ad experiments, 85% treated. Licensed CC BY-NC-SA 4.0 by Criteo. The link on
# Criteo AI Lab's page redirects to an Azure storage host that no longer resolves, so the
# file comes from Criteo's verified Hugging Face organization. The checksum is the one
# Hugging Face publishes for that file.
CRITEO_URL = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
CRITEO_FILE = DATA_DIR / "criteo-research-uplift-v2.1.csv.gz"
CRITEO_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
CRITEO_ROWS = 13_979_592
CRITEO_RESULTS_DIR = RESULTS_DIR / "criteo"
CRITEO_FIGURES_DIR = FIGURES_DIR / "criteo"
# Fitting all 18 candidates 5 times on the full 8.4-million-user training set would take
# hours, so cross-validation uses a stratified subsample of it. The models picked by
# cross-validation are then refit on the full training set.
CRITEO_CV_USERS = 2_000_000
CRITEO_N_BOOTSTRAP = 2000  # Poisson bootstrap: memory for one resample at a time


def breakeven(margin: float = MARGIN, cost: float = COST_PER_EMAIL) -> float:
    """Incremental revenue an email must generate to pay for itself."""
    return cost / margin
