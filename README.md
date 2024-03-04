This repository contains a collection of machine learning projects covering various topics, datasets, and techniques. Each project is implemented using Python and may include notebooks, scripts, and datasets.

Project List:

**MNIST Dataset Exploration and Clustering Analysis:**

This repository contains a python script explores the MNIST dataset, focusing on digits 7, 8, and 9. 
It applies Principal Component Analysis (PCA) for dimensionality reduction and visualizes the data in two dimensions. 
Clustering using K-Means is performed to identify optimal clusters, followed by the visualization of representative images for each cluster.


**Housing Price Prediction with Random Forest Regression:**

This repository contains a python script that predicts housing prices using Random Forest Regression. 
It involves data cleaning, encoding categorical variables, and scaling numerical features. 
Grid Search is used to find the best hyperparameters for the Random Forest Regressor, and the model is trained with optimized parameters. 
Important features affecting house prices are identified and visualized.


**Stacking Ensemble Classifier for MNIST Digit Recognition:**

This repository contains a python script that builds a Stacking Ensemble Classifier for MNIST digit recognition. 
It begins by splitting the dataset into training and test sets and applying PCA for dimensionality reduction. 
Various classifiers, including Decision Tree, Random Forest, AdaBoost, LinearSVC, and Logistic Regression, are trained. 
A Stacking Ensemble Classifier is then constructed using the trained base classifiers, and its performance is evaluated on the test set, comparing it with individual classifiers.


**Iris Species Classification with Linear SVM and Logistic Regression:**

This repository contains a python script that classifies Iris species using Linear SVM and Logistic Regression. 
It selects relevant features, visualizes data distribution, and trains linear SVM classifiers with different regularization parameters. 
A Logistic Regression model is also trained and evaluated. 
Probability contour plots visualize decision boundaries, and the probability of Iris Setosa for a given sample is predicted.

**Neural Network Digit Classifier with Early Stopping**

This repository contains a python script for training a neural network model to classify handwritten digits using the MNIST dataset. 
The model is built using TensorFlow and Keras, with early stopping implemented to prevent overfitting. 
The provided script loads and preprocesses the data, constructs the neural network architecture, trains the model, and evaluates its performance on a separate test set.
