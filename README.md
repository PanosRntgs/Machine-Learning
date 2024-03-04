This repository contains a collection of machine learning projects covering various topics, datasets, and techniques. Each project is implemented using Python and may include notebooks, scripts, and datasets.

Project List:

**MNIST Dataset Exploration and Clustering Analysis:**

Load the MNIST dataset and select digits 7, 8, and 9.
Apply Principal Component Analysis (PCA) to reduce dimensionality.
Visualize the data using the first two principal components.
Perform clustering with K-Means and evaluate the silhouette scores to determine the optimal number of clusters.
Plot representative images for each cluster.


**Housing Price Prediction with Random Forest Regression:**

Load the housing dataset and perform data cleaning and preprocessing.
Encode categorical variables and scale numerical features.
Use Grid Search with Random Forest Regressor to find the best hyperparameters.
Train a Random Forest Regressor model with the optimized parameters.
Identify and visualize the most important features affecting house prices.


**Stacking Ensemble Classifier for MNIST Digit Recognition:**

Load the MNIST dataset and split it into training and test sets.
Apply PCA to reduce dimensions and preserve variance.
Train various classifiers including Decision Tree, Random Forest, AdaBoost, LinearSVC, and Logistic Regression.
Build a Stacking Ensemble Classifier using the trained base classifiers.
Evaluate the Stacking Classifier's performance on the test set and compare it with individual classifiers.


**Iris Species Classification with Linear SVM and Logistic Regression:**

Load the Iris dataset and select relevant features for classification.
Visualize the data distribution in a scatter plot.
Train linear SVM classifiers with different values of C and evaluate accuracy.
Train a Logistic Regression model and evaluate its accuracy.
Plot a probability contour plot to visualize the decision boundaries.
Predict the probability of Iris Setosa for a given sample.

**Neural Network Digit Classifier with Early Stopping**

This code repository contains a Python script for training a neural network model to classify handwritten digits using the MNIST dataset. 
The model is built using TensorFlow and Keras, with early stopping implemented to prevent overfitting. 
The provided script loads and preprocesses the data, constructs the neural network architecture, trains the model, and evaluates its performance on a separate test set.
