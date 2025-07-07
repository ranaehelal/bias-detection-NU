# Uncovering Bias and Explaining Decisions in a Text-Based Job Screening Model

## 📌 Nile University Internship – 48-Hour Challenge  
**Track:** Bias Detection & Explainability in AI  
**Prepared by:** Rana Helal

---

## 1. 📊 Dataset Description

The dataset contains **1,500 job applicant profiles** represented using **11 structured features**:

- **Numerical Features:**
  - `Age`, `ExperienceYears`, `PreviousCompanies`, `DistanceFromCompany`, `InterviewScore`, `SkillScore`, `PersonalityScore`

- **Categorical Features:**
  - `Gender` (0 = Female, 1 = Male)  
  - `EducationLevel` (1 = Primary to 4 = Graduate)  
  - `RecruitmentStrategy` (1 to 3)

- **Target Variable:**
  - `HiringDecision` (0 = Not Hire, 1 = Hire)

There were no missing values.  
To simulate **bias**, only **30% of female candidates** were included in the training set, causing **intentional representation imbalance**.

---

## 2. 🧠 Modeling Approach

### Data Preparation
- Train/Test split: **70% / 30%**
- Numerical features scaled with `StandardScaler`

### Model
- Classifier: `LogisticRegression(max_iter=1000)` (via scikit-learn)

### Performance
| Metric       | Value     |
|--------------|-----------|
| Accuracy     | 84.7%     |
| Precision (Hire) | 78%   |
| Recall (Hire)    | 72%   |
| F1-Score (Hire)  | 75%   |

Despite high accuracy, the model showed **lower recall on “Hire”**, suggesting possible unfairness.

---

## 3. ⚖️ Fairness Analysis

Fairness metrics were computed using **Fairlearn** using `Gender` as the sensitive attribute.

| Metric                     | Value     |
|----------------------------|-----------|
| Accuracy (Female)          | 83.4%     |
| Accuracy (Male)            | 86.0%     |
| Demographic Parity Diff.   | 0.104     |

A **parity difference of 0.104** indicates **gender-based bias** in predictions.

---

## 4. 🧾 Explainability with SHAP

**SHAP (SHapley Additive Explanations)** was used to explain model predictions.

- 5 samples selected (3 predicted "Hire", 2 predicted "Not Hire")
- Top influential features:  
  `InterviewScore`, `SkillScore`, `PersonalityScore`
- `Gender` was **not a top feature**, but its **indirect influence** through correlated features is possible.

**Waterfall plots** provided insight into how each feature pushed a prediction toward “Hire” or “Not Hire”.

---

## 5. 🛠️ Bias Mitigation

Applied **ExponentiatedGradient** method (with **Demographic Parity** constraint) from Fairlearn:

| Metric                 | Value       |
|------------------------|-------------|
| Accuracy (Post-Mitigation) | **86.3%** |
| Demographic Parity Diff.   | **0.088** |

Fairness improved **while maintaining or even improving accuracy**.

---

## 6. ✅ Summary and Conclusions

This project was part of **Nile University’s AI Internship Challenge**, simulating a real-world bias auditing case.

Key Steps:
- Bias identification from **imbalanced training data**
- Fairness measurement using **standard group metrics**
- **SHAP** explanations to interpret predictions
- **Bias mitigation** via constrained optimization

Results confirm that **AI fairness** is achievable without harming performance, when supported by strong methodology.

---

## 🧪 Extras

- Code is fully documented in a **Jupyter notebook**
- Includes:
  - Correlation heatmaps
  - Gender distribution plots
  - SHAP visualizations
- Designed for **Google Colab** or **local execution**

---

## 📁 Sample from Dataset

| Age | Gender | EducationLevel | ExperienceYears | PreviousCompanies | DistanceFromCompany | InterviewScore | SkillScore | PersonalityScore | RecruitmentStrategy | HiringDecision |
|-----|--------|----------------|------------------|-------------------|----------------------|----------------|-------------|------------------|----------------------|----------------|
| 26  | 1      | 2              | 0                | 3                 | 26.78               | 48             | 78          | 91               | 1                    | 1              |
| 39  | 1      | 4              | 12               | 3                 | 25.86               | 35             | 68          | 80               | 2                    | 1              |
| 48  | 0      | 2              | 3                | 2                 | 9.92                | 20             | 67          | 13               | 2                    | 0              |
| 30  | 0      | 1              | 6                | 1                 | 43.10               | 23             | 52          | 85               | 2                    | 0              |
| 27  | 0      | 3              | 14               | 4                 | 31.70               | 54             | 50          | 50               | 1                    | 1              |

---

