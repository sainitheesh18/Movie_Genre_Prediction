# Movie_Genre_Prediction

This repository contains code for a movie genre prediction model using Spark's MLlib and ML APIs. The model is built to classify movies into genres based on their plot descriptions. The project uses various machine learning techniques, including Random Forest, Count Vectorizer, HashingTF, IDF, and Word2Vec.

## Prerequisites
Before running the code, ensure you have the following installed:
Python 3.x
Apache Spark (with PySpark)
Pandas
NumPy

## Project Structure

### Data Files
train.csv: Contains the training dataset.
test.csv: Contains the test dataset.
mapping.csv: Contains the genre mapping data.

### Code Files
Main Script: The script that runs the model training and prediction.

### Code Overview
Initialization: The script sets up the Spark environment, initializes SparkContext and SparkSession, and imports necessary libraries for data manipulation and machine learning.
Data Loading: It loads the training, test, and genre mapping data from CSV files into Pandas DataFrames and then converts them into Spark DataFrames for further processing.
Data Preprocessing: The plot descriptions are processed using tokenization and stop words removal. This converts raw text into a format suitable for feature extraction.
Feature Extraction: Various techniques like Count Vectorizer, HashingTF, and Word2Vec are used to convert the processed text into numerical features that can be fed into machine learning models.
Model Training and Prediction: A Random Forest Classifier is trained on the extracted features. The model then predicts the genres for the test data. The results are transformed into readable genre labels and saved to CSV files.
Output: The predictions are saved into CSV files, which include the movie IDs and their corresponding predicted genres.
