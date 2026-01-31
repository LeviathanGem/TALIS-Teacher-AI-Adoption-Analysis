# TALIS-Teacher-AI-Adoption-Analysis
Analysis of OECD TALIS teacher survey data: who uses AI in teaching, by country, and what predicts it. Uses Beta–Binomial rates by country, weighted logistic regression, and a MAP+Laplace logistic fit. Outputs are tables (adoption rates, GLM summary, MAP coefficients) and three figures.

## Data

- **Source:** TALIS teacher questionnaire (TT4G* items), semicolon-delimited CSV(s).
- **Outcome:** AI adoption — either the single yes/no item `TT4G36` (used AI in last 12 months) or a binary built from `TT4G37*` (used in at least one AI use case). Coded 1 = yes, 0 = no.
- **Predictors (composite scales, z-scored):**
  - **S_READINESS** — digital/tech readiness (TT4G33*).
  - **S_DIGITAL_ATT** — attitudes toward digital tools (TT4G34*).
  - **S_AI_ATT** — attitudes toward AI (TT4G35*).
  - **S_TRAIN_NEED** — training needs (TT4G24*).
  - **S_TRAIN_SKILL** — skills/training (TT4G25*).
- **Weights:** Script picks the first available of `SCHWGTT`, `TCHWGT`, `ADJRT24`, `IDPOP` so estimates can be interpreted for the teacher population when weights are used.
- **Missing:** OECD-style codes (e.g. 97, 98, 99, 998, 999) are turned into NaN before modeling.

---

## What the script does

1. Loads CSV(s), picks weight column, defines adoption and predictor blocks.
2. Cleans and recodes (numeric, missing codes, 1/2 → 1/0 for adoption).
3. Builds the five scales (row mean of items, then z-score).
4. **Country-level adoption:** Beta–Binomial with prior Beta(1,1); for each country you get posterior mean adoption probability and 95% credible interval.
5. **Weighted logistic (GLM):** Adoption ~ scales, with `freq_weights`; MLE.
6. **MAP + Laplace:** Same logistic model with L2 prior; coefficients and approximate SE from Hessian inverse; eigenvalues of that covariance for conditioning/uncertainty.
7. **Figures:** adoption-by-country (error bars = credible interval), digital readiness by country (weighted mean), correlation heatmap of the five scales.

---

## Visualizations

| Figure | What it is | How to read it |
|--------|------------|----------------|
| **AI Adoption Rate by Country** | Horizontal error-bar plot: one point per country, x = posterior mean adoption probability, error bars = 95% credible interval. | Countries further right have higher estimated adoption; longer bars mean more uncertainty (often smaller n). |
| **Digital Readiness by Country** | Horizontal bar chart: one bar per country, length = weighted mean of z-scored readiness. | Longer bars = higher average self-reported digital readiness in that country (weighted). |
| **Correlation among Construct Scales** | Heatmap of the 5×5 correlation matrix of S_READINESS, S_DIGITAL_ATT, S_AI_ATT, S_TRAIN_NEED, S_TRAIN_SKILL. | Strong positive (red) = scales move together; helps spot redundancy or multicollinearity before interpreting logistic coefficients. |

---

## Data meaning and interpretation

- **Adoption (TT4G36 / TT4G37*):** Measures whether the teacher has used AI in teaching (e.g. in the last 12 months or in specific use cases). The country plot shows where adoption is high vs low and how precise the estimates are.
- **Scales:** All five are standardized (z-scores), so in the logistic model a one-unit change is “one standard deviation” on that scale. Positive coefficient = higher score on that scale is associated with higher probability of adoption.
- **GLM summary:** Coefficients, standard errors, and (if available) tests for the weighted logistic model. Use this for “which factors predict adoption?” in the population sense when weights are correct.
- **MAP + Laplace table:** Same predictors, L2-regularized fit; approximate posterior uncertainty via Hessian. Eigenvalues of the (approximate) posterior covariance indicate which directions are most/least informed by the data (small eigenvalues = flat posterior in that direction).
- **Readiness-by-country plot:** Complements adoption: countries can have high readiness but lower adoption (or the reverse), which is useful for policy or further modeling.

---

## Outcomes you get

- **Printed:** Combined data shape; weight column used; adoption variable name; predictor group sizes; top-10 countries by adoption rate; full GLM summary; MAP coefficient table; covariance eigenvalues.
- **Plots (from `plt.show()`):**  
  1) Adoption by country (Beta–Binomial);  
  2) Digital readiness by country (weighted);  
  3) Correlation heatmap of scales.

---

## How to run

Put your TALIS teacher CSV(s) in the same folder as `portfolio.py`. Set `DATA_PATH` to a single filename (e.g. `"atgintt4.csv"`) or adjust the script to loop over a list of files and concatenate. Then:

```bash
python portfolio.py
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `statsmodels`.
