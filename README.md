# Resume Parser - NLP Model

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Extraction](#feature-extraction)
  - [Model Building](#model-building)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
This project aims to build a Natural Language Processing (NLP) model for parsing resumes and categorizing them into different job roles. The model takes a resume text as input and predicts the most suitable job category for the candidate based on the content of the resume.

## Dataset
The dataset used for this project is the "UpdatedResumeDataSet.csv" file, which contains resume text data along with the corresponding job category labels. The dataset consists of 962 resume samples distributed across 25 different job categories.

## Methodology

### Data Preprocessing
- Cleaned the resume text by removing URLs, hashtags, mentions, punctuations, and extra whitespace using regular expressions.
- Encoded the target variable (job category) using LabelEncoder.

### Exploratory Data Analysis (EDA)
- Analyzed the distribution of job categories using a pie chart.
- Visualized the count of resumes in each job category using a countplot.

### Feature Extraction
- Converted the cleaned resume text into a matrix of TF-IDF features using TfidfVectorizer.
- Limited the number of features to 1500 to avoid overfitting.

### Model Building
- Split the data into training and testing sets using train_test_split.
- Trained a KNeighborsClassifier using OneVsRestClassifier for multi-class classification.

### Model Evaluation
- Evaluated the model's performance on the training and testing sets using accuracy score.
- Generated a classification report to analyze precision, recall, and F1-score for each job category.

## Results
The KNeighborsClassifier achieved the following performance metrics:
- Training Set Accuracy: 0.99
- Testing Set Accuracy: 0.99

The classification report showed high precision, recall, and F1-score values for most of the job categories, indicating the model's effectiveness in predicting the correct job role based on the resume text.

## Requirements
- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Usage
1. Clone the repository: git clone https://github.com/Akalbir17resume-parser.git

2. Install the required dependencies: `pip install -r requirements.txt`

3. Open the Jupyter Notebook `Resume Parser.ipynb` and run the cells to preprocess the data, build the model, and evaluate its performance.

## License
This project is licensed under the [MIT License](LICENSE).

