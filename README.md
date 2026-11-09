# Phishing URL Detection: EDA, Parameterized Logistic Regression & SHAP Explainability

**Course:** M115 – AI & Intelligent Systems for Cybersecurity (ENSIASD Taroudant, Ibn Zohr University)
**Instructor:** Prof. Sara EL-ATEIF | **Academic year:** 2026–2027
**Author:** [IMOURIGUEN Zakaria]

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/zack404-sketch/AI-labs/blob/main/LAB_1.ipynb)

---

## 1. Project overview

This lab covers the full lifecycle of a supervised classification model applied to a cybersecurity problem: **detecting phishing URLs**. The work goes from raw data inspection to a model that can be explained to a SOC (Security Operations Center) analyst.

The main steps are:

1. Security-oriented exploratory data analysis (EDA).
2. A leakage-free preprocessing pipeline.
3. A comparison of five Logistic Regression configurations (no regularization, L2, L1, class-balanced).
4. Explanation of the model with SHAP (global and per-alert).
5. Tuning of operational decision thresholds for a SOC.

## 2. Dataset

**PhiUSIIL Phishing URL Dataset** (Prasad & Chandra, *Computers & Security*, 2024):
**235,795 URLs** described by **56 columns** (lexical, network and HTML features). Data quality check: **0 missing values, 0 duplicate rows**.

> The CSV file is not included in this repository because of its size. It can be downloaded from the UCI Machine Learning Repository or from Kaggle (search "PhiUSIIL Phishing URL Dataset").

**Important label trap:** in the raw file, `label = 1` means *Legitimate* and `label = 0` means *Phishing*. In a SOC the positive class must be the attack, so I created `IsPhishing = 1 - label` (1 = Phishing, 0 = Legitimate). With this choice, Precision and Recall measure real threat detection.

| Class | Count | Proportion |
|---|---|---|
| Phishing (1) | 100,945 | 42.81 % |
| Legitimate (0) | 134,850 | 57.19 % |

The imbalance is mild, so no resampling was needed.

## 3. What I did, step by step

### Part 1: Exploratory Data Analysis (Exercises 1 to 3)

- **Exercise 1 – Data quality:** loaded the CSV with pandas and checked shape, data types, missing values, duplicates and summary statistics (`URLLength`, `LineOfCode`, `NoOfSelfRef`). Created the `IsPhishing` target and checked class proportions (42.81 % phishing / 57.19 % legitimate).
- **Exercise 2 – Visualization:**
  - count plot of the class balance;
  - KDE plots of `URLLength`, `CharContinuationRate` and `NoOfSelfRef` per class (clipped at the 99th percentile for readability);
  - bar chart of the prevalence (%) of the binary indicators per class:

| Indicator | Legitimate (%) | Phishing (%) |
|---|---|---|
| `IsHTTPS` | 100.00 | 49.22 |
| `HasPasswordField` | 13.85 | 5.39 |
| `HasCopyrightInfo` | 80.82 | 5.73 |
| `HasSocialNet` | 79.46 | 0.51 |

- **Exercise 3 – Cleanup and correlations:** dropped the raw text columns (`FILENAME`, `URL`, `Domain`, `TLD`, `Title`), the original `label` and the pre-computed `URLSimilarityIndex`, so the model learns from real lexical and HTML features instead of memorizing identifiers. The final matrix has **49 features** (`X` shape: 235,795 × 49). Computed the Pearson correlation of every feature with the target and plotted the top 15 positive and top 15 negative ones.

### Part 2: Leakage-free preprocessing (Exercise 4)

- Stratified **75 / 25** train/test split (`random_state=42`): **176,846 training URLs / 58,949 test URLs**.
- `StandardScaler` **fitted on the training set only**, then applied to the test set with the training mean and standard deviation (check: `URLLength` has mean 0.0 and std 1.0 on the training set).
- Kept column names and indices in DataFrames so that weights and SHAP plots stay readable.

Why this matters: fitting the scaler before the split would let test-set statistics leak into the training pipeline and make the evaluation too optimistic.

### Part 3: Parameterized Logistic Regression (Exercises 5 to 7)

**Exercise 5 – five configurations**, all evaluated at threshold τ = 0.50:

| Model | Penalty | C | Notes |
|---|---|---|---|
| M1 | None | n/a | Unregularized baseline (`lbfgs`) |
| M2 | L2 (Ridge) | 0.001 | Strong regularization |
| M3 | L2 (Ridge) | 1.0 | Standard balance |
| M4 | L1 (Lasso) | 0.01 | `liblinear`, automatic feature selection |
| M5 | L2 (Ridge) | 1.0 | `class_weight='balanced'` |

For each model I recorded Precision, Recall, F1-Score, PR-AUC, False Positives, False Negatives and the number of zeroed weights (|w| < 1e-4).

**Exercise 6 – L1 regularization path:** trained L1 models for `C ∈ [0.002, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]` and plotted the number of active features against `C` (log scale).

| C | Active features | Zeroed features |
|---|---|---|
| 0.002 | 22 | 27 |
| 0.01 | 30 | 19 |
| 0.05 | 45 | 4 |
| 0.1 | 47 | 2 |
| 0.5 | 48 | 1 |
| 1.0 | 49 | 0 |
| 5.0 | 49 | 0 |

The weaker the regularization (larger `C`), the more features survive; from `C = 1.0` onward L1 keeps all 49.

**Exercise 7 – coefficients and Odds Ratios:** extracted the weights of M3, computed `exp(w)` for each feature, added the sparse L1 weight from M4, and listed the top 8 phishing indicators (w > 0) and top 8 benign indicators (w < 0). Because features are standardized, an Odds Ratio gives the change in the odds of phishing for a one-standard-deviation increase of the feature.

**Top 8 phishing indicators (w > 0):**

| Feature | Weight L2 (w) | Odds Ratio | L1 weight (M4) |
|---|---|---|---|
| `LetterRatioInURL` | 4.7228 | 112.49 | 2.1726 |
| `DegitRatioInURL` | 3.6057 | 36.81 | 1.8036 |
| `SpacialCharRatioInURL` | 3.5981 | 36.53 | 1.9106 |
| `LargestLineLength` | 2.5575 | 12.90 | 0.0848 |
| `NoOfOtherSpecialCharsInURL` | 2.0834 | 8.03 | 0.0000 |
| `NoOfLettersInURL` | 1.6801 | 5.37 | 0.0000 |
| `CharContinuationRate` | 1.4202 | 4.14 | 0.0000 |
| `NoOfDegitsInURL` | 0.9718 | 2.64 | 0.0000 |

**Top 8 benign indicators (w < 0):**

| Feature | Weight L2 (w) | Odds Ratio | L1 weight (M4) |
|---|---|---|---|
| `IsHTTPS` | −6.9024 | 0.0010 | −3.5777 |
| `NoOfJS` | −3.6536 | 0.0259 | −1.2017 |
| `NoOfSelfRef` | −3.1249 | 0.0439 | −0.8227 |
| `NoOfImage` | −2.7981 | 0.0609 | −0.1505 |
| `NoOfExternalRef` | −2.6414 | 0.0713 | −0.0157 |
| `LineOfCode` | −2.3676 | 0.0937 | 0.0000 |
| `HasSocialNet` | −2.1533 | 0.1161 | −1.0887 |
| `DomainLength` | −1.9117 | 0.1478 | −0.0585 |

### Part 4: Explainability and SOC thresholds (Exercises 8 to 10)

- **Exercise 8 – global SHAP:** `shap.LinearExplainer` on M3 with the training set as background (SHAP subsamples it to 100 rows internally), computed on the first 1,000 test URLs. Produced the bar plot (mean |SHAP|) and the beeswarm plot (direction of impact). Top features by mean |SHAP| (log-odds units): `IsHTTPS` (5.19), `LetterRatioInURL` (3.73), `SpacialCharRatioInURL` (3.02), `DegitRatioInURL` (2.21), `NoOfJS` (2.21), `HasSocialNet` (2.15).
- **Exercise 9 – local alert triage:** selected one True Positive, one True Negative and one False-Positive-like case and plotted their SHAP waterfall plots, using unscaled feature values so that an analyst can read real URL metrics.
  - **True Positive** (sample 2): true label = phishing, P(Phishing) = 99.99 %.
  - **True Negative** (sample 0): true label = legitimate, P(Phishing) = 0.00 %.
  - **False-Positive-like** (sample 330): legitimate URL with P(Phishing) = 31.58 % (see section 6).
- **Exercise 10 – threshold tuning:** compared the confusion matrix, Precision, Recall and F1-Score of M3 at:
  - τ = 0.20 (threat hunting mode),
  - τ = 0.50 (default mode),
  - τ = 0.85 (automated firewall blocking mode).

## 4. Results

All metrics are computed on the 58,949-URL test set. PR-AUC is displayed as 1.0 at 4 decimals for all models (the models are almost perfectly separating the classes).

| Model | Precision | Recall | F1 | PR-AUC | FP | FN | Zeroed weights |
|---|---|---|---|---|---|---|---|
| M1: Unregularized | 0.9996 | 0.9990 | 0.9993 | ≈ 1.0 | 11 | 25 | 0 / 49 |
| M2: L2, C = 0.001 | 0.9985 | 0.9958 | 0.9971 | ≈ 1.0 | 39 | 105 | 0 / 49 |
| M3: L2, C = 1.0 | 0.9996 | 0.9989 | 0.9992 | ≈ 1.0 | 10 | 28 | 0 / 49 |
| M4: L1, C = 0.01 | 0.9993 | 0.9971 | 0.9982 | ≈ 1.0 | 18 | 72 | 24 / 49 |
| M5: L2, C = 1.0, balanced | 0.9995 | 0.9990 | 0.9992 | ≈ 1.0 | 13 | 25 | 0 / 49 |

**Thresholds (M3):**

| τ | Mode | Precision | Recall | F1 | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|
| 0.20 | Threat hunting | 0.9988 | 0.9994 | 0.9991 | 25,221 | 33,682 | 31 | 15 |
| 0.50 | Default | 0.9996 | 0.9989 | 0.9992 | 25,208 | 33,703 | 10 | 28 |
| 0.85 | Automated blocking | 1.0000 | 0.9975 | 0.9987 | 25,172 | 33,713 | 0 | 64 |

**Main findings:**

- **Best performance / sparsity trade-off: M4 (L1, C = 0.01).** It removes **24 of 49** features (about half) with a very small loss: F1 = 0.9982 vs 0.9992 for M3, PR-AUC ≈ 1.0 for both. The price is more missed phishing (FN = 72 vs 28). M3 remains the most accurate model overall, while M4 is the better choice for a real-time firewall where every HTML feature costs parsing time.
- **Strong regularization hurts:** M2 (C = 0.001) is the worst model (FP = 39, FN = 105), so too much shrinkage underfits even on this easy problem.
- **`class_weight='balanced'` (M5) has a small effect** because the imbalance is mild (43 % / 57 %): compared with M3, FN goes from 28 to 25 (slightly better Recall) and FP from 10 to 13 (slightly lower Precision), the expected direction.
- **Strongest phishing indicators (Odds Ratio per +1 std):** `LetterRatioInURL` (OR = 112.5), `DegitRatioInURL` (36.8), `SpacialCharRatioInURL` (36.5).
- **Strongest benign indicators:** `IsHTTPS` (OR = 0.001), `NoOfJS` (0.026), `NoOfSelfRef` (0.044).
- **SHAP is consistent with the weights:** the top mean |SHAP| features are `IsHTTPS`, `LetterRatioInURL`, `SpacialCharRatioInURL`, `DegitRatioInURL` and `NoOfJS`, i.e. the same group as the largest |w|.
- **Recommended policy:** block automatically at τ = 0.85 (Precision = 1.0000, FP = 0 on the test set); send URLs with 0.20 ≤ P(Phishing) < 0.85 to an analyst queue (τ = 0.20 gives the best Recall = 0.9994, with FN = 15); use τ = 0.50 as the default reference.

## 5. Key concepts learned

- **Label alignment:** the positive class must be the attack, otherwise Precision and Recall lose their security meaning.
- **Identifier columns cause overfitting:** raw URLs and domains are memorized and do not generalize to new domains.
- **HTTPS is not a trust signal:** 49.22 % of phishing URLs use HTTPS (vs 100 % of legitimate ones). Free certificates mean many phishing sites use HTTPS, so "HTTPS ⇒ legitimate" would miss about half of the attacks. In this dataset, `IsHTTPS` is a strong *model* feature only because every legitimate URL has it; absence of HTTPS is a strong phishing signal, but presence proves nothing.
- **Scaling and regularization:** without standardization, L1/L2 penalties depend on the units of the features.
- **Data leakage:** every transformation must be learned on the training set only.
- **L1 vs L2:** L1 performs automatic feature selection (24 of 49 weights zeroed at C = 0.01; useful for a fast real-time firewall); L2 shrinks weights without zeroing them (0 / 49).
- **`class_weight='balanced'`:** gives more weight to the minority class and usually trades False Negatives for False Positives.
- **Global weight vs local SHAP value:** `w_j` is the same for all URLs, while `φ_j(x) = w_j (x_j − E[x_j])` depends on the specific URL and explains a single alert.
- **Two-threshold SOC policy:** a high threshold for automatic blocking limits False Positives, and an intermediate band goes to human review.
- **Odds Ratio interpretation:** with standardized features, `exp(w_j)` is the multiplicative change in the odds of phishing for +1 standard deviation (e.g. OR = 112.5 for `LetterRatioInURL`).

## 6. Difficulties and how I solved them

- **Label inversion:** noticed that `label = 1` means legitimate and created `IsPhishing` to fix it.
- **Notebook errors from execution order:** several `NameError` messages came from running cells out of order or from a session restart. Solution: run the notebook from top to bottom (*Run all*). For the same reason, the Exercise 6 cell rebuilds the data (loading, split, scaling) so that it can run on its own.
- **File path:** the CSV path differs between environments (`/content/` on Colab, `/kaggle/input/...` on Kaggle). I adapted the path in `pd.read_csv`.
- **Slow L1 path (Exercise 6):** `liblinear` with L1 and a large `C` converges slowly on nearly separable data. To keep the runtime reasonable I loosened the solver (`tol=1e-2`, `max_iter=100`) while still using the full training set. A side effect: at `C = 0.01` the path shows 30 active features, while M4 in Exercise 5 (default tolerance, `max_iter=500`) keeps 25 (24 zeroed). The difference comes from the solver tolerance, not from a different model.
- **No strict false positive in the sample:** the 1,000-URL sample contained no legitimate URL with P(Phishing) ≥ 0.50, so I used the legitimate URL with the highest phishing probability (sample 330, P = 31.58 %) as the closest case to a false alarm. (On the full test set, M3 has 10 real false positives at τ = 0.50.)
  - Waterfall reading (log-odds, f(x) = −0.773 vs baseline E[f(X)] = −1.959): the features pushing toward phishing are `HasSocialNet = 0` (+1.95), `HasCopyrightInfo = 0` (+1.21), `DomainTitleMatchScore = 0` (+1.10), `URLCharProb = 0.048` (+1.10) and `HasDescription = 0` (+1.06). The site has no social links, no copyright notice and no title/domain match, which is typical of a phishing page, but `IsHTTPS = 1` (−3.01) and `LetterRatioInURL = 0.462` (−2.08) keep the score below 0.50.
  - Triage note: the legitimate URL was flagged with P = 31.58 %, which lands in the analyst band (0.20 ≤ P < 0.85) but not in automatic blocking. The missing social, copyright and title-match features pushed the score up. The analyst should verify and whitelist it, and such minimalist legitimate sites should be added to the training data.
- **`HasPasswordField` surprise:** I expected phishing pages to contain more password fields, but in this dataset the indicator is *less* frequent in phishing (5.39 %) than in legitimate URLs (13.85 %). It is therefore not a standalone phishing signal here, and the lab's intuition should be checked against the data.

## 7. Tools and libraries

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, SHAP. Developed in Jupyter notebooks on Google Colab / Kaggle.

## 8. How to run

1. Open the notebook with the "Open In Colab" badge (or upload it to Kaggle).
2. Download the dataset and upload `PhiUSIIL_Phishing_URL_Dataset.csv` (or add it as input on Kaggle and adapt the path).
3. Install SHAP if needed: `!pip install shap`.
4. Run all cells from top to bottom.

## 9. Repository structure

```
AI-labs/
├── README.md
└── LAB_1.ipynb          # Lab 1 notebook (code, plots, answers, synthesis)
```

