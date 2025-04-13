# Customer-Churn-Prediction
Churn prediction project using ensemble machine learning techniques (Random Forest, AdaBoost, Gradient Boosting) with preprocessing, SMOTE for class balancing, and feature importance analysis. Achieved 85.8% accuracy.

---

## 📂 Dataset

- **Source**: [Kaggle – Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records**: 7043
- **Features**: 20 attributes including customer account data, services subscribed, and churn status

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- imbalanced-learn (SMOTE)

---

## 🚀 Project Workflow

1. **Data Cleaning**  
   - Removed irrelevant columns (e.g. `customerID`)  
   - Handled missing values in numeric and categorical fields

2. **Encoding & Scaling**  
   - Encoded categorical variables using `LabelEncoder`  
   - Scaled numerical features using `StandardScaler`

3. **Class Imbalance Handling**  
   - Applied `SMOTE` to balance the `Churn` class distribution

4. **Model Training & Ensemble**  
   - Trained: `RandomForest`, `GradientBoosting`, and `AdaBoost`  
   - Combined models using `VotingClassifier` (soft voting)

5. **Evaluation**  
   - Evaluated using `accuracy`, `precision`, `recall`, and classification report  
   - Plotted feature importance using `RandomForest`

---

## 📊 Results

| Metric      | Value    |
|-------------|----------|
| Accuracy    | 85.80%   |
| Precision   | ~86%     |
| Recall      | ~86%     |

---

## 📌 Key Takeaways

- Churn is strongly influenced by **contract type**, **monthly charges**, and **tenure**
- Ensemble modeling significantly improved performance over individual classifiers
- SMOTE effectively handled class imbalance to reduce false negatives

---
