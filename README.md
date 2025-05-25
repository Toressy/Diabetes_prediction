Diabetes Prediction Using Machine Learning

This project explores various machine learning models to predict diabetes using healthcare data. By pre-processing, analyzing, and modeling the dataset from Kaggle, we aim to accurately classify whether a patient is diabetic or not.

📊 Dataset

Source: Kaggle - Healthcare Diabetes Dataset
Rows: 2,768
Features:
  Pregnancies
  Glucose
  BloodPressure
  SkinThickness
  Insulin
  BMI
  DiabetesPedigreeFunction
  Age
  Outcome (target variable: 0 = non-diabetic, 1 = diabetic)
  
⚙️ Project Workflow
    Data Preprocessing
        Removal of outliers and noisy values
        Feature scaling using StandardScaler
        SMOTE applied for handling class imbalance
        Removal of non-informative columns (e.g., ID)
    Exploratory Data Analysis (EDA)
        Visualization using seaborn and matplotlib
        Distribution plots, boxplots, and correlation heatmap
        Analysis of class imbalance and key predictive features
Model Training & Evaluation
    Models implemented:
        Decision Tree
        Random Forest
        Support Vector Machine (SVM)
        k-Nearest Neighbors (KNN)
        Naive Bayes
    Evaluation metrics:
        Accuracy, Precision, Recall, F1-Score
        Confusion Matrices
        Hyperparameter tuning with GridSearchCV and RandomizedSearchCV

        
🏆 Results

Model	Test Accuracy
  KNN (Weighted)	99.64%
  Random Forest	99.46%
  Decision Tree	99.04%
  SVM	99.04%
  Naive Bayes	79.60%
📌 Best model: KNN with weighted distance and n_neighbors=2


📁 File Structure
├── 100661485_Diabetes_prediction.ipynb    # Jupyter notebook with full implementation
├── 100661485_Diabetes_prediction_report.docx # Detailed project report
└── README.md                              # Project documentation

🧠 Key Insights
Proper data pre-processing significantly boosts model performance.
Ensemble and distance-based models (Random Forest, KNN) outperform probabilistic models (Naive Bayes).
Addressing class imbalance with SMOTE is crucial for accurate diabetes prediction.

🛠️ Tools & Libraries
  Python (Pandas, NumPy, Matplotlib, Seaborn)
  Scikit-learn
  imbalanced-learn (SMOTE)
