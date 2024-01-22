 # MachineLearning
 This repostory working on Machine Learning technic with Python


- [Data source](https://bilkav.com/makine-ogrenmesi-egitimi)
- [Education](https://www.btkakademi.gov.tr/portal/course/python-ile-makine-ogrenmesi-11800)
- [DATA MINING: CONCEPTS AND TECHNIQUES Book](https://www.academia.edu/22412092/DATA_MINING_CONCEPTS_AND_TECHNIQUES_3RD_EDITION)

- we need some installations:
    - pip install pandas
    - pip install numpy
    - pip install matplotlib
    - pip install scikit-learn
    - pip install statsmodels
    - pip install xlrd

## Some Important Concepts for Machine Learning

- Prediction: It predicts past, faulty or missing values in our data and also predicts the future.
- Forecasting: used to predict values outside the sample.

## Project Search

### [1-Preprocessing](https://github.com/GALACICEK/MachineLearning/blob/main/1-Preprocessing/)

- [1-data_preprocessing.py](https://github.com/GALACICEK/MachineLearning/blob/main/1-Preprocessing/1-data_preprocessing.py)
    - Loading data sets
    - Data preprocessing
    - Finding NAN values
    - Encoder categoric (Ordinal,nominal) var to Numeric var
    - numpy transform to dataframe
    - Split test and train variables
    - Datas Scaler with Standard

### [2-Predictions](https://github.com/GALACICEK/MachineLearning/blob/main/2-Predictions)

- [1-Simple_Linear_Regression.py](https://github.com/GALACICEK/MachineLearning/blob/main/2-Predictions/1-Simple_Linear_Regression.py)
    - Preprocessing
    - Modelling
    - from tahmin values predicted Y_test values
    - Visualization
    - Comment: We have drawn the line y = α+βx+ε closest to the selected values.

- [2-Multiple_Linear_Regression.py](https://github.com/GALACICEK/MachineLearning/blob/main/2-Predictions/2-Multiple_Linear_Regression.py)
    - Multiple Linear Regression Summary
    - Preprocessing
    - Modelling
    - Backward Elimination

- [3-Polynomial_Regression.py](https://github.com/GALACICEK/MachineLearning/blob/main/2-Predictions/3-Polynomial_Regression.py)
    - Preprocessing
    - Dataframe Slicing And Transform to Array
    - Linear Regression
    - Polynomial Regression
        - Non-linear Regression
        - Second  Degree Polynomial
        - Fourth Degree Polynomial
    - Visualization
    - Predictions

- [4-Support_Vector_Regression.py](https://github.com/GALACICEK/MachineLearning/blob/main/2-Predictions/4-Support_Vector_Regression.py)
    - Preprocessing
    - Dataframe Slicing And Transform to Array
    - Data Scaler
    - SVR kernel = rbf
    - SVR kernel = linear
    - SVR kernel = poly 
    - Visualization
    - Predictions


- [5-Decision_Tree.py](https://github.com/GALACICEK/MachineLearning/blob/main/2-Predictions/5-Decision_Tree.py)
    - Preprocessing
    - Dataframe Slicing And Transform to Array
    - Decision Tree
    - Visualization
    - Tree Shape Visualization
    - Predictions

- [6-Random_Forest.py](https://github.com/GALACICEK/
MachineLearning/blob/main/2-Predictions/6-Random_Forest.py)
    - Preprocessing
    - Dataframe Slicing And Transform to Array
    - Random Forest
    - Visualization
    - Predictions

- [7-Evaluation_of_Predictions.py](https://github.com/GALACICEK/MachineLearning/blob/main/2-Predictions/7-Evaluation_of_Predictions.py)
    - Preprocessing
    - Dataframe Slicing And Transform to Array
    - Linear Regression
        - Visualization
        - R2 Score
    - Polynomial Regression
        - Second Degree Polynomial
        - Visualization
        - R2 Score
    - SVR Regression
        - Datas Scaler
        - SVR kernel='rbf'
        - Visualization
        - R2 Score
    - Decision Tree
        - Visualization
        - R2 Score
     - Random Forest
        - Visualization
        - R2 Score
    - OutPuts R2 Score Regression Models

- [8-Regression_Example.py](https://github.com/GALACICEK/MachineLearning/blob/main/2-Predictions/8-Regression_Example.py)
    - Linear Regression
    - Polynomial Regression
    - SVR Regression
    - Decision Tree
    - Random Forest

### [3-Classifications](https://github.com/GALACICEK/MachineLearning/blob/main/3-Classifications/)

- [1-Logistic_Regression.py](https://github.com/GALACICEK/MachineLearning/blob/main/3-Classifications/1-Logistic_Regression.py)
    - Sigmoid function
    - Loading data sets
    - Split test and train variables
    - Datas Scaler
    - Logistic Regression
    - Confusion Matrix

- [2-K-NN.py](https://github.com/GALACICEK/MachineLearning/blob/main/3-Classifications/2-K-NN.py)
    - Sigmoid function
    - Loading data sets
    - Split test and train variables
    - Datas Scaler
    - KNN
    - Confusion Matrix

- [3-Support_Vector_Machine.py](https://github.com/GALACICEK/MachineLearning/blob/main/3-Classifications/3-Support_Vector_Machine.py)
    - Loading data sets
    - Split test and train variables
    - Datas Scaler
    - SVC kernel='linear'
        - Confusion Matrix
    - SVC kernel='rbf'
        - Confusion Matrix
    - SVC kernel='poly'
        - Confusion Matrix

- [4-NaiveBayes.py](https://github.com/GALACICEK/MachineLearning/blob/main/3-Classifications/4-NaiveBayes.py)
    - Loading data sets
    - Split test and train variables
    - Gaussian Naive Bayes
         - Confusion Matrix
    - Multinominal Naive Bayes
         - Confusion Matrix
    - Bernoulli Naive Bayes
         - Confusion Matrix

- [5-Decission_Tree.py](https://github.com/GALACICEK/MachineLearning/blob/main/3-Classifications/5-Decission_Tree.py)
    - Quinlan's ID3 Algoritm
    - Loading data sets
    - Split test and train variables
    - Datas Scaler
    - Decision Tree Classifier 'entropy'
        - Tree Shape Visualization
        - Confusion Matrix
    - Decision Tree Classifier 'gini'
        - Tree Shape Visualization
        - Confusion Matrix

- [6-Random_Forest.py](https://github.com/GALACICEK/MachineLearning/blob/main/3-Classifications/6-Random_Forest.py)
    - Loading data sets
    - Split test and train variables
    - Datas Scaler
    - Random Forest Classifier 'entropy'
        - Confusion Matrix
    - Random Forest Classifier 'gini'
        - Confusion Matrix

- [7-Classification_Example.py](https://github.com/GALACICEK/MachineLearning/blob/main/3-Classifications/7-Classification_Example.py)
    - Loading data sets
    - Label Encoding
    -  2D scatter plot 
    -  3D scatter plot 
    - Split test and train variables
    - Datas Scaler
    - Logistic Regression
        - Logistic Regression Calculation Metrics
            - Confusion Matrix
            - Recall Score
            - F1 Score
            - Classification Report
    - KNN
        - KNN Calculation Metrics
            - Confusion Matrix
            - Recall Score
            - F1 Score
            - Classification Report
    - SVC kernel='rbf'
        - SVC kernel='rbf' Calculation Metrics
            - Confusion Matrix
            - Recall Score
            - F1 Score
            - Classification Report
    - Gaussian Naive Bayes
        - Gaussian Naive Bayes Calculation Metrics
            - Confusion Matrix
            - Recall Score
            - F1 Score
            - Classification Report
    - Decision Tree Classifier
        - LDecision Tree Classifier Calculation Metrics
            - Confusion Matrix
            - Recall Score
            - F1 Score
            - Classification Report
    - Random Forest Classifier
        - Random Forest Calculation Metrics
            - Confusion Matrix
            - Recall Score
            - F1 Score
            - Classification Report

