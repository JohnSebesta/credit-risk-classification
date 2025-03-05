# credit-risk-classification

Credit Risk Analysis

Overview

This analysis focuses on building a machine learning model to assess credit risk, specifically identifying loans at high risk of default. Using a dataset from the financial industry, the goal is to predict whether a loan is healthy (0) or high-risk (1) using logistic regression. The model will help companies assess and mitigate financial risks by predicting potential defaults. This README summarizes the steps taken to prepare the data, train the model, and evaluate its performance.

Methodology

Split the Data into Training and Testing Sets
The first step in the analysis is to load the dataset and prepare the data for model training. Here are the steps followed:

Load the Data:
The dataset (lending_data.csv) is loaded from the Resources folder into a Pandas DataFrame.
Create the Labels and Features:
The target variable, loan_status, is extracted into the y DataFrame, where:
0 represents a healthy loan.
1 represents a high-risk loan.
The remaining columns are used as the feature set (X).
Split the Data:
The data is split into training and testing datasets using train_test_split, ensuring that the model is trained on one set and evaluated on another.
Create a Logistic Regression Model with the Original Data
Fit the Logistic Regression Model:
A logistic regression model is created using the training data (X_train, y_train).
The model is then used to make predictions on the testing dataset (X_test).
Evaluate the Model:
Confusion Matrix:
The confusion matrix is generated to visualize the performance of the model in predicting both healthy and high-risk loans.
Classification Report:
A classification report is produced, providing key metrics such as precision, recall, and F1-score for both classes (0 and 1).
Model Assessment:
The performance of the model is evaluated based on how accurately it predicts both healthy loans (0) and high-risk loans (1).
Write a Credit Risk Analysis Report
The final part of the project involves writing a report on the machine learning model's performance. This includes:

Overview of the Analysis:
The purpose of this analysis was to build a logistic regression model to predict credit risk based on a dataset of loan information.
Results:
Accuracy Score: The overall accuracy of the model on the testing set.
Precision Score: The ability of the model to correctly identify high-risk loans (class 1).
Recall Score: The model's sensitivity in identifying high-risk loans.
Summary:
The logistic regression model showed reasonable performance. The evaluation metrics indicate how well the model balances identifying healthy loans and high-risk loans. Based on the results, we can decide whether to recommend the model for use in the company's risk assessment processes.
Results

Accuracy: The logistic regression model achieved an accuracy of X% on the test dataset, meaning it correctly predicted the loan status for X% of the instances.
Precision: The precision for predicting high-risk loans (class 1) was X%, indicating how many of the predicted high-risk loans were actually high-risk.
Recall: The recall for high-risk loans was X%, indicating the percentage of actual high-risk loans that the model correctly identified.

References
ChatGPT
XpertLearning Assistant
The Tutor Fred Logan
