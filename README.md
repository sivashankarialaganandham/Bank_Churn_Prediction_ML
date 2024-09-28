# Bank Churn Prediction Using Machine Learning Techniques

![bank churn](https://github.com/sivashankarialaganandham/Bank_Churn_Prediction_ML/blob/main/bank%20churn%20image.jpg)

## Introduction
This project addresses the challenge of predicting customer churn for a multinational bank using machine learning. The goal is to leverage various algorithms to develop a predictive model and evaluate its performance using key metrics. The project explores multiple algorithms such as Logistic Regression, Decision Trees, and Random Forest, with a focus on data preprocessing, model training, and interpretation of results. This project showcases the full data science lifecycle, from data collection and exploration to scrubbing, modeling, and interpretation of outcomes.

## Project Overview
The bank churn prediction dataset, sourced from Kaggle, contains 175,028 records and 25 features related to customer demographics, account status, and product information. The target variable Exited indicates whether a customer churned (1) or stayed with the bank (0). The project aims to:
- Clean and preprocess the data
- Explore the data through visualization
- Build and evaluate machine learning models to predict churn
- Interpret the models' predictions and derive actionable insights for the bank

# Methodology
The project follows the OSEMN data science process:
Obtain: The dataset is loaded and inspected for structure and content.
1. **Scrub**: Data cleaning steps include detecting missing values, outliers, and transforming categorical data.
2. **Explore**: Exploratory Data Analysis (EDA) is conducted to visualize trends and distributions.
3. **Model**: Machine learning models are built, including Logistic Regression, Decision Trees, and Random Forest.
4. **Interpret**: Models are evaluated based on accuracy, precision, recall, and F1-score, among other metrics.

## Data Preprocessing Steps
- **Data Scrubbing**: Outliers were identified using the Z-score, and missing values were imputed with mean values.
- **Categorical Encoding**: One-hot encoding was used to transform categorical variables into numeric form.
- **Data Balancing**: Since the dataset was imbalanced, an undersampling technique was applied to balance the target classes.
- **Standardization**: Features were standardized using StandardScaler to improve model performance.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) was applied to reduce multicollinearity between independent variables.

## Tools & Technologies
- Programming Language: Python
- Libraries:
    - Data manipulation: pandas, NumPy
    - Visualization: matplotlib, seaborn
    - Machine Learning: scikit-learn
    - Dimensionality Reduction: PCA

## Data Description
The dataset contains customer details like:
- **Demographics**: Age, Gender, Country of Residence
- **Account Information**: Credit Score, Tenure, Balance, and IsActiveMember
- **Product Information**: Number of Products, HasCreditCard, and IsActiveMember
The target variable Exited indicates customer churn (1 for churned, 0 for retained).

## Modeling
Several machine learning models were built to predict churn, including:
- **Logistic Regression**
- **Decision Trees**
- **Random Forest**
The models were trained and tested using a 70-30 train-test split. Random Forest outperformed the other models with an accuracy of 78.3% and a sensitivity of 69.3%.

## Evaluation Metrics
The models were evaluated using:
- **Accuracy**: Proportion of correctly classified customers.
- **Precision**: Proportion of true churners among predicted churners.
- **Recall (Sensitivity)**: Proportion of actual churners correctly identified.
- **F1 Score**: Harmonic mean of precision and recall.
- **AUC**: Area Under the Receiver Operating Characteristic (ROC) curve.

## Conclusion
This project successfully demonstrates the prediction of customer churn using machine learning. Random Forest provided the best results in terms of accuracy and sensitivity. In this analysis, Random Forest proved to be the best-performing model, achieving an accuracy of 78.3% and a sensitivity of 69.3%. Future enhancements could involve testing additional models and tuning hyperparameters to further improve predictive performance.
