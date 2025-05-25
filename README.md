# 🩺 DiabetesDetectAI

This machine learning project aims to predict diabetes using patient health data. By applying classification models on a Kaggle dataset, we achieve highly accurate and reliable results, assisting early diagnosis and supporting better healthcare outcomes.

## 📊 Dataset

- **Source:** [Kaggle - Healthcare Diabetes Dataset](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes/data)
- **Records:** 2,768 patients
- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (0 = No diabetes, 1 = Diabetes)

## 🔍 Data Preprocessing

- Removed outliers using quantile clipping
- Standardized data with `StandardScaler`
- Addressed imbalance using SMOTE (Synthetic Minority Oversampling Technique)
- Removed non-informative columns (ID)
- Replaced physiologically invalid values with column means

## 📈 Exploratory Data Analysis (EDA)

- Count plots and histograms for feature distributions
- Correlation heatmaps to identify relationships
- Pairplots to visually distinguish class distributions
- Boxplots to spot and treat outliers

## 🤖 Models Implemented

- **Random Forest**
- **Support Vector Machine (SVM)**
- **Decision Tree**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**

## 🏆 Results

| Model           | Test Accuracy |
|----------------|---------------|
| KNN (weighted) | **99.64%**     |
| Random Forest  | 99.46%        |
| Decision Tree  | 99.04%        |
| SVM            | 99.04%        |
| Naive Bayes    | 79.60%        |

## 📌 Best Model: KNN (Weighted)

- Weighted distance metric with `n_neighbors = 2`
- Balanced precision and recall across both classes
- Outperformed all other models on generalization

## 📁 Project Structure

├── 100661485_Diabetes_prediction.ipynb # Main implementation
├── 100661485_Diabetes_prediction_report.docx # Detailed report
└── README.md # Project documentation


## 📚 Tools & Libraries

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- imbalanced-learn (SMOTE)

## 👨‍🎓 Author

**Student ID:** 100661485  
**Institution:** School of Computing and Engineering  
**Module:** Data Mining and AI

## 📖 References

- Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Oversampling Technique
- Bonaccorso, G. (2018). *Machine Learning Algorithms*
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Kaggle Dataset](https://www.kaggle.com/datasets/nanditapore/healthcare-diabetes/data)

---

