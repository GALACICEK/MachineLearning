 # MachineLearning
 This repostory working on Machine Learning technic with Python


- [Data source](https://bilkav.com/makine-ogrenmesi-egitimi)
- [Education](https://www.btkakademi.gov.tr/portal/course/python-ile-makine-ogrenmesi-11800)
- [DATA MINING: CONCEPTS AND TECHNIQUES Book](https://www.academia.edu/22412092/DATA_MINING_CONCEPTS_AND_TECHNIQUES_3RD_EDITION)

- create your vitual environment:
    - python -m venv myenv
    - Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process (for permission)
    - .\myenv\Scripts\activate
    - python -m pip install --upgrade pip


- we need some installations:
    - pip install pandas
    - pip install numpy
    - pip install matplotlib
    - pip install scipy
    - pip install scikit-learn
    - pip install statsmodels
    - pip install graphviz
    - pip install pyarrow
    - pip install xlrd
    - pip install nltk
    - pip install pydot
    - pip install tensorflow
    - pip install --upgrade https://files.pythonhosted.org/packages/8a/8e/0ad1eff787bf13f8dca87472414fbdfb73ea53f5a1a1c20489cfccfb7717/tensorflow-2.17.0-cp310-cp310-win_amd64.whl , 
    this package "tensorflow-2.17.0-cp310-cp310-win_amd64.whl" 
    install package files according your tensorflow version  and choose your win,mac or something else in [this](https://pypi.org/project/tensorflow/#files)
    - pip install xgboost

    



!!Checked requirements.txt files for package versions.

- pip install -r requirements.txt
- pip check
    - Check compatibility by running the command


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

- [6-Random_Forest.py](https://github.com/GALACICEK/MachineLearning/blob/main/2-Predictions/6-Random_Forest.py)
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

### [4-Clustering](https://github.com/GALACICEK/MachineLearning/blob/main/4-Clustering/)

- [1-K-Means.py](https://github.com/GALACICEK/MachineLearning/blob/main/4-Clustering/1-K-Means.py)
    - Loading data sets
    - K-Means Clustering
    - Find Optimum Cluster number
    - Plot Optimum Clusters

- [2-Hierarchical.py](https://github.com/GALACICEK/MachineLearning/blob/main/4-Clustering/2-Hierarchical.py)
    - Loading data sets
    - Agglomerative Clustering
    - Scater Clustering
    - Dendogram

### [5-Association_Rule_Mining](https://github.com/GALACICEK/MachineLearning/tree/main/5-Association_Rule_Mining)

- [1-Apriori_Algorithm.py](https://github.com/GALACICEK/MachineLearning/blob/main/5-Association_Rule_Mining/1-Apriori_Algorithm.py)
    - Loading data sets
    - Using apriori lib
    - Using my_functions

- [2-Eclat_Algorithm.py](https://github.com/GALACICEK/MachineLearning/blob/main/5-Association_Rule_Mining/2-Eclat_Algorithm.py)
    - Loading data sets
    - Using my_functions

### [6-Reinforced_Learning](https://github.com/GALACICEK/MachineLearning/tree/main/6-Reinforced_Learning)


- [1-Upper_Confidence_Bound.py](https://github.com/GALACICEK/MachineLearning/blob/main/6-Reinforced_Learning/1-Upper_Confidence_Bound.py)
    - Loading data sets
    - Random Selection
    - UCB
 

- [2-Thompson_Sample.py](https://github.com/GALACICEK/MachineLearning/blob/main/5-Association_Rule_Mining/2-Thompson_Sample.py)
    - Loading data sets

### [7-NLP](https://github.com/GALACICEK/MachineLearning/tree/main/7-NLP)

- [7.1-NLP.py](https://github.com/GALACICEK/MachineLearning/tree/main/7-NLP/7.1NLP.py)
    - Preprocessing
        - Loading data sets
        - filtering alphanumeric data and punctuation
        - Convert Lowercase and Uppercase
        - NOTE : Sparks Matrices
    - Feature Extraction (Bag of Words BOW)
    - Machine Learning
        - Split test and train variables
        - Gaussian Naive Bayes
        - Confusion Matrix

### [8-Deep_Learning](https://github.com/GALACICEK/MachineLearning/tree/main/8-Deep_Learning)

- [8.1ANN.py](https://github.com/GALACICEK/MachineLearning/tree/main/8-Deep_Learning/8.1ANN.py)
    - Preprocessing
        - Loading data sets
        - Encoder Categoric -> Numeric
        - Split test and train variables
        - Datas Scaler
    - ANN
        - Input Layer: 11 extentions 
        - First Hidden  Layer: 6 neuron
        - Second Hidden  Layer: 6 neuron
        - Output Layer: 1
        - Confusion Matrix 

- [8.2XGBoost.py](https://github.com/GALACICEK/MachineLearning/tree/main/8-Deep_Learning/8.2XGBoost.py)
    - Preprocessing
        - Loading data sets
        - Encoder Categoric -> Numeric
        - Split test and train variables
    - XGBoost
        - Confusion Matrix 

### [9-DimetionReduction](https://github.com/GALACICEK/MachineLearning/tree/main/9-DimetionReduction)

- [1-PCA.py](https://github.com/GALACICEK/MachineLearning/tree/main/9-DimetionReduction/1-PCA.py)

    - Preprocessing
        - Loading data sets
        - Split test and train variables
        - Datas Scaler
    - PCA
        - Logistic Regression
        - Predicts
        - Confusion Matrix 

- [2-LDA.py](https://github.com/GALACICEK/MachineLearning/tree/main/9-DimetionReduction/2-LDA.py)
    - Preprocessing
        - Loading data sets
        - Split test and train variables
        - Datas Scaler
    - LDA
        - Logistic Regression
        - Predicts
        - Confusion Matrix 

- [3-PCA&LDA.md](https://github.com/GALACICEK/MachineLearning/tree/main/9-DimetionReduction/3-PCA&LDA.md)
    - PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis) 
    - Comparing PCA (Principal Component Analysis) and LDA (Linear Discriminant Analysis) conclusions


### [10-ModelSelection](https://github.com/GALACICEK/MachineLearning/tree/main/10-ModelSelection)


- [1-k_foldCrossValidation.py](https://github.com/GALACICEK/MachineLearning/tree/main/10-ModelSelection/1-k_foldCrossValidation.py)
    - Preprocessing
        - Loading data sets
        - Split test and train variables
        - Datas Scaler
    - SVM
        - Predicts
        - Confusion Matrix
    - k-fold Cross Validation

- [2-GridSearch.py](https://github.com/GALACICEK/MachineLearning/tree/main/10-ModelSelection/2-GridSearch.py)
    - Preprocessing
        - Loading data sets
        - Split test and train variables
        - Datas Scaler
    - SVM
        - Predicts
        - Confusion Matrix
    - Grid Search


### [11-ModelSaving](https://github.com/GALACICEK/MachineLearning/tree/main/11-ModelSaving)


- [1.pickle.py](https://github.com/GALACICEK/MachineLearning/tree/main/11-ModelSaving/1.pickle.py)
    - Preprocessing
        - Loading data sets
        - Split test and train variables
    - Linear Regression
    - Pickle
        - Save the fitted model
        -  Loading the fitting model on file
