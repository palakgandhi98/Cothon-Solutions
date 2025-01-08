# Web Traffic Analysis for Predicting Website Growth

This repository contains a Jupyter notebook for a time-series analysis and forecasting task using Wikipedia page views data. The notebook includes data preprocessing, exploratory data analysis, anomaly detection, and time-series forecasting using various models.

## Objective

The primary objective of this project is to analyze and forecast the number of views for Wikipedia pages using time-series data. Specific goals include:

1. **Data Preprocessing**: Clean and prepare the dataset for analysis.
2. **Exploratory Data Analysis (EDA)**: Understand the trends, patterns, and correlations within the data.
3. **Anomaly Detection**: Identify and handle anomalies in the data to improve the accuracy of forecasting models.
4. **Time-Series Forecasting**: Apply and evaluate different time-series forecasting models to predict future page views accurately.
5. **Model Evaluation**: Compare the performance of different models using metrics such as Root Mean Squared Error (RMSE) to determine the best-performing model.


## Table of Contents

- [Web Traffic Analysis for Predicting Website Growth](#web-traffic-analysis-for-predicting-website-growth)
  - [Objective](#objective)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Dependencies](#dependencies)
  - [Data](#data)
    - [Data Description](#data-description)
    - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Anomaly Detection](#anomaly-detection)
  - [Time-Series Forecasting](#time-series-forecasting)
    - [ARMA Model](#arma-model)
    - [ARIMA Model](#arima-model)
    - [Exponential Smoothing Model](#exponential-smoothing-model)
  - [Model Evaluation](#model-evaluation)
  - [Conclusion](#conclusion)
  - [Contributing](#contributing)
  - [Acknowledgments](#acknowledgments)
  - [Stay Connected:](#stay-connected)

## Introduction

This project aims to analyze and forecast the number of views for Wikipedia pages using time-series data. The analysis includes detecting anomalies, making the data stationary, and applying various forecasting models to predict future page views.

## Dependencies

The following Python libraries are required to run the notebook:

- pandas
- numpy
- matplotlib
- seaborn
- re
- datetime
- sklearn
- statsmodels
- warnings
- calendar
- pickle

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

## Data

The dataset used in this project is `train_1.csv`, which contains the number of views for different Wikipedia pages over a period. The data is preprocessed by filling missing values and reshaping the DataFrame for analysis.

 Dataset: [Kaggle Web Traffic Time Series Forecasting](#https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/data)
### Data Description

The dataset used in this project is `train_1.csv`, which contains the number of views for different Wikipedia pages over a period.

- **File**: `train_1.csv`
- **Columns**:
  - `Page`: The name of the Wikipedia page.
  - Multiple date columns (e.g., `2016-01-01`, `2016-01-02`, etc.): Each column represents the number of views for the corresponding date.

### Data Preprocessing

1. **Missing Values**: Any missing values in the dataset are filled with zeros.
2. **Reshaping**: The dataset is reshaped using the `pd.melt` function to convert it from a wide format to a long format. This results in a DataFrame with columns `Page`, `Date`, and `Views`.
3. **Date Indexing**: The `Date` column is set as the index of the DataFrame and converted to a datetime format for time-series analysis.



## Exploratory Data Analysis

The exploratory data analysis (EDA) includes:

- Average number of views per day
- Average number of views per month
- Total number of views based on the language of the Wikipedia webpage
- Total number of views based on the day of the week
- Identifying the top 5 pages with the maximum number of views
- Correlation between the pages

## Anomaly Detection

Anomalies in the data are detected using the Isolation Forest algorithm. The anomalies are removed, and missing values are filled using a rolling mean of 30 days.

## Time-Series Forecasting

### ARMA Model

The ARMA (AutoRegressive Moving Average) model is used to forecast the number of views. The model is trained on the training data, and predictions are made for the test data. The forecasted values are compared with the actual values to evaluate the model's performance.

### ARIMA Model

The ARIMA (AutoRegressive Integrated Moving Average) model is an extension of the ARMA model that includes differencing to make the data stationary. The model is trained and evaluated similarly to the ARMA model.

### Exponential Smoothing Model

The Exponential Smoothing model is used for forecasting. The model is trained on the training data, and predictions are made for the test data. The forecasted values are compared with the actual values to evaluate the model's performance.

## Model Evaluation

The performance of the models is evaluated using the Root Mean Squared Error (RMSE). The RMSE values for each model are calculated and compared to determine the best-performing model.

## Conclusion

The notebook provides a comprehensive analysis and forecasting of Wikipedia page views using time-series data. The models are evaluated, and the best-performing model is identified based on the RMSE values.

---

You can run the notebook using Jupyter Notebook or JupyterLab. Ensure that you have all the required dependencies installed before running the notebook.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## Acknowledgments

- Special thanks to the contributors of the dataset and the libraries used in this project.
- The project is inspired by various data science and machine learning resources available online.

## Stay Connected:
 * [![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=fff)](https://www.github.com/palakgandhi98)
 * [![LinkedIn](https://img.shields.io/badge/Linkedin-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/palakgandhi98)

Let's build something amazing together!