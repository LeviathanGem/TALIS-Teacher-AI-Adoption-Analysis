import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm

# config and handlers
DATA_PATH = "atgintt4.csv"
DELIM = ";"

COUNTRY_COL = "CNTRY"
WEIGHT_COL_CANDIDATES = ["SCHWGTT", "TCHWGT", "ADJRT24", "IDPOP"]
REPWGT_REGEX = r"^CRWGT\d+$"

# OECD missing codes -> NaN (tweak per codebook if needed)
DEFAULT_MISSING_CODES = {
    7, 8, 9, 97, 98, 99, 997, 998, 999, 9997, 9998, 9999
}

# --- helpers ---
def pick_first_existing(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

def find_cols_by_prefix(df, prefix):
    cols = [c for c in df.columns if c.startswith(prefix)]
    def keyfun(x):
        tail = x[len(prefix):]
        return (len(tail), tail)
    return sorted(cols, key=keyfun)

def find_cols_by_regex(df, pattern):
    rx = re.compile(pattern)
    return [c for c in df.columns if rx.search(c)]

def coerce_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def replace_missing_codes(df, cols, missing_codes=DEFAULT_MISSING_CODES):
    for c in cols:
        df.loc[df[c].isin(missing_codes), c] = np.nan
    return df

def zscore_series(x):
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or np.isnan(sd):
        return (x - mu)
    return (x - mu) / sd

def weighted_mean(x, w):
    mask = np.isfinite(x) & np.isfinite(w)
    if mask.sum() == 0:
        return np.nan
    return np.sum(x[mask] * w[mask]) / np.sum(w[mask])

def weighted_var(x, w):
    mask = np.isfinite(x) & np.isfinite(w)
    if mask.sum() == 0:
        return np.nan
    mu = weighted_mean(x[mask], w[mask])
    return np.sum(w[mask] * (x[mask] - mu) ** 2) / np.sum(w[mask])


# --- load & pick weight ---
df = pd.read_csv(DATA_PATH, sep=DELIM, low_memory=False)
print("Loaded shape:", df.shape)

wcol = pick_first_existing(df.columns, WEIGHT_COL_CANDIDATES)
if wcol is None:
    raise ValueError(f"No weight column found among {WEIGHT_COL_CANDIDATES}. "
                     "Check your extract / codebook.")
print("Using weight column:", wcol)

for req in [COUNTRY_COL, wcol]:
    if req not in df.columns:
        raise ValueError(f"Required column missing: {req}")


# --- adoption outcome: TT4G36 (yes/no) or any TT4G37* = 1 ---
adoption_col = None
if "TT4G36" in df.columns:
    adoption_col = "TT4G36"
else:
    tt4g37_cols = find_cols_by_prefix(df, "TT4G37")
    if len(tt4g37_cols) > 0:
        adoption_col = "AI_ADOPTED"
        # 1=yes 2=no in TALIS, double-check codebook
        df = coerce_numeric(df, tt4g37_cols)
        df = replace_missing_codes(df, tt4g37_cols)

        any_yes = np.zeros(len(df), dtype=bool)
        for c in tt4g37_cols:
            any_yes |= (df[c] == 1)
        df[adoption_col] = any_yes.astype(int)
    else:
        raise ValueError("Could not find a direct adoption variable (TT4G36) "
                         "or TT4G37* use-case items to construct adoption.")

print("Adoption variable:", adoption_col)

# predictor blocks (composites built below)
readiness_cols = find_cols_by_prefix(df, "TT4G33")
digital_att_cols = find_cols_by_prefix(df, "TT4G34")
ai_att_cols = find_cols_by_prefix(df, "TT4G35")
train_digital_cols = find_cols_by_prefix(df, "TT4G24")
train_ai_cols = find_cols_by_prefix(df, "TT4G25")

predictor_groups = {
    "READINESS": readiness_cols,
    "DIGITAL_ATT": digital_att_cols,
    "AI_ATT": ai_att_cols,
    "TRAIN_NEED": train_digital_cols,
    "TRAIN_SKILL": train_ai_cols,
}
for k, cols in predictor_groups.items():
    print(k, "n_cols =", len(cols), "example:", cols[:5])


# --- clean & recode ---
all_model_cols = [COUNTRY_COL, wcol, adoption_col]
for cols in predictor_groups.values():
    all_model_cols.extend(cols)
all_model_cols = [c for c in all_model_cols if c in df.columns]

numeric_cols = [c for c in all_model_cols if c not in [COUNTRY_COL]]
df = coerce_numeric(df, numeric_cols)
df = replace_missing_codes(df, numeric_cols)

# 1=yes 2=no -> 1/0
if df[adoption_col].dropna().isin([1, 2]).all():
    df[adoption_col] = df[adoption_col].map({1: 1, 2: 0})


# --- scales: row mean then z-score ---
def make_scale(df, cols, name, min_items=1):
    if len(cols) == 0:
        df[name] = np.nan
        return df
    x = df[cols].astype(float)
    count_nonmissing = x.notna().sum(axis=1)
    scale_raw = x.mean(axis=1, skipna=True)
    scale_raw[count_nonmissing < min_items] = np.nan
    df[name] = zscore_series(scale_raw.to_numpy())
    return df

df = make_scale(df, readiness_cols, "S_READINESS", min_items=1)
df = make_scale(df, digital_att_cols, "S_DIGITAL_ATT", min_items=1)
df = make_scale(df, ai_att_cols, "S_AI_ATT", min_items=1)
df = make_scale(df, train_digital_cols, "S_TRAIN_NEED", min_items=1)
df = make_scale(df, train_ai_cols, "S_TRAIN_SKILL", min_items=1)

scale_cols = [
    "S_READINESS", "S_DIGITAL_ATT", "S_AI_ATT",
    "S_TRAIN_NEED", "S_TRAIN_SKILL"
]


# --- country adoption rates: Beta-Binomial (prior Beta(1,1), no weights here) ---
alpha0, beta0 = 1.0, 1.0

country_stats = []
for cntry, g in df[[COUNTRY_COL, adoption_col]].dropna().groupby(COUNTRY_COL):
    y = g[adoption_col].sum()
    n = len(g)
    a_post = alpha0 + y
    b_post = beta0 + (n - y)
    p_mean = a_post / (a_post + b_post)
    p_lo = float(pd.Series(np.random.beta(a_post, b_post, size=200000)).quantile(0.025))
    p_hi = float(pd.Series(np.random.beta(a_post, b_post, size=200000)).quantile(0.975))
    country_stats.append((cntry, n, y, p_mean, p_lo, p_hi))

rates = pd.DataFrame(country_stats, columns=["CNTRY", "n", "y", "post_mean", "ci_lo", "ci_hi"])
rates = rates.sort_values("post_mean", ascending=False)
print(rates.head(10))


plt.figure()
plt.errorbar(
    x=rates["post_mean"].to_numpy(),
    y=np.arange(len(rates)),
    xerr=np.vstack([
        rates["post_mean"] - rates["ci_lo"],
        rates["ci_hi"] - rates["post_mean"]
    ]),
    fmt="o"
)
plt.yticks(np.arange(len(rates)), rates["CNTRY"])
plt.xlabel("Posterior mean adoption probability")
plt.title("AI Adoption Rate by Country (Bayesian Beta–Binomial)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# --- weighted logistic (GLM, freq_weights) ---
model_df = df[[COUNTRY_COL, wcol, adoption_col] + scale_cols].copy()
model_df = model_df.dropna(subset=[adoption_col])
model_df = model_df.dropna(subset=scale_cols, how="all")
model_df[scale_cols] = model_df[scale_cols].fillna(0.0)

y = model_df[adoption_col].astype(int).to_numpy()
X = model_df[scale_cols].to_numpy()
X = sm.add_constant(X, has_constant="add")

w = model_df[wcol].astype(float).to_numpy()

glm = sm.GLM(
    y, X,
    family=sm.families.Binomial(),
    freq_weights=w
)
res = glm.fit()
print(res.summary())


# --- MAP + Laplace (L2 prior, Hessian -> cov) ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_map_laplace(X, y, l2=1.0, max_iter=50, tol=1e-6):
    n, d = X.shape
    beta = np.zeros(d)
    for _ in range(max_iter):
        p = sigmoid(X @ beta)
        grad = X.T @ (p - y) + l2 * beta
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + l2 * np.eye(d)
        step = np.linalg.solve(H, grad)
        beta_new = beta - step

        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Laplace covariance approx = H^{-1} at MAP
    p = sigmoid(X @ beta)
    W = p * (1 - p)
    H = X.T @ (X * W[:, None]) + l2 * np.eye(d)
    cov = np.linalg.inv(H)
    return beta, cov

beta_map, cov_map = logistic_map_laplace(X, y, l2=1.0)
se = np.sqrt(np.diag(cov_map))
coef_table = pd.DataFrame({
    "term": ["const"] + scale_cols,
    "beta_map": beta_map,
    "approx_se": se,
    "z": beta_map / se
})
print("\nMAP + Laplace approx (rough Bayesian posterior):")
print(coef_table)

# Linear algebra demonstration: eigen-decomposition of covariance
eigvals = np.linalg.eigvalsh(cov_map)
print("\nCovariance eigenvalues (showing conditioning / uncertainty directions):")
print(np.sort(eigvals))


#readiness by country (weighted mean)
cntry_summary = []
for cntry, g in df[[COUNTRY_COL, wcol, "S_READINESS"]].dropna().groupby(COUNTRY_COL):
    x = g["S_READINESS"].to_numpy()
    ww = g[wcol].to_numpy()
    cntry_summary.append((cntry, weighted_mean(x, ww)))

cs = pd.DataFrame(cntry_summary, columns=["CNTRY", "wmean_readiness"]).sort_values("wmean_readiness", ascending=False)

plt.figure()
plt.barh(cs["CNTRY"], cs["wmean_readiness"])
plt.xlabel("Weighted mean (z-scored readiness)")
plt.title("Digital Readiness by Country (Teacher sample, weighted)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# --- correlation of scales ---
corr = model_df[scale_cols].corr()

plt.figure()
plt.imshow(corr.to_numpy(), aspect="auto")
plt.xticks(range(len(scale_cols)), scale_cols, rotation=45, ha="right")
plt.yticks(range(len(scale_cols)), scale_cols)
plt.colorbar()
plt.title("Correlation among Construct Scales")
plt.tight_layout()
plt.show()
