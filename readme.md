# Loan Approval Classifier

## Project Description

This a project on building machine learning model based on a loan applicant's Loan Approval. The dataset is used here is [Kaggle Dataset](https://www.kaggle.com/datasets/burak3ergun/loan-data-set) of loan applicants containing varity of features like `applicant demographics`, `financial information` and `loan characteristics`. By analyzing these features, the model proves predictions that help to assess the likelihood of approving a loan.


### Problem Statement

In financial institutions, determining whether a loan application will be approved is a crucial task. The manual review process can be time-consuming and subjective. By implementing a predictive model, loan processing can be automated, and decisions can be made more consistently and efficiently.

The objective of this project is to create a machine learning model that can accurately predict loan approvals based on the features in the dataset.

### Project Workflow

1. **Data Understanding**:
   - The dataset includes several variables related to the applicants and their loans, such as:
     - Demographics (gender, marital status, education level, etc.)
     - Applicant income
     - Co-applicant income
     - Loan amount and tenure
     - Credit history
     - Property area
     - Loan status (approved or rejected)
   
2. **Exploratory Data Analysis (EDA)**:
   - Data exploration is conducted to understand the distributions, relationships, and patterns in the dataset.
   - Visualizations are used to uncover insights, such as correlations between features and loan approval status, outlier detection, and more.

3. **Data Preprocessing**:
   - Handling missing values, encoding categorical variables, and normalizing or scaling numerical data to prepare the dataset for machine learning.

4. **Model Selection**:
   - Multiple classification algorithms are explored, including:
     - Logistic Regression
     - Decision Trees Classifier
     - Random Forest Classifier
     - Support Vector Classifier
     - KNN
     - Bagging Classifier
     - Extra Trees Classifier
     - Gradient Boosting Classifier
     - XGBoosting Classifier
     - Gaussian, Multinomial and Bernoulli Classifier



5. **Model Evaluation**:
   - The models are evaluated using performance metrics like accuracy, precision. Precision is mattered most in this scenario (Data is imbalanced).
   - Hyperparameter tuning is performed to improve the model's performance.


### Technologies Used

- **Python** for programming and data analysis.
- **Jupyter Notebook** for the project implementation and visualization.
- **Pandas and NumPy** for data manipulation.
- **Matplotlib and Seaborn** for visualizations and EDA.
- **Scikit-learn** for machine learning models, evaluation, and preprocessing.

### Conclusion

This project provides a comprehensive machine learning solution for loan approval prediction. By leveraging historical data and powerful classification models, it enables financial institutions to automate and streamline the loan approval process while maintaining accuracy and consistency.