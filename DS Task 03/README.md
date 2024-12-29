# Sales Forecasting Model

This project focuses on building a sales forecasting model to predict future sales based on historical data. Accurate sales forecasting is crucial for businesses to optimize inventory, manage supply chains, and improve decision-making. The notebook provides a comprehensive step-by-step process, including data preprocessing, feature engineering, model building, and evaluation.

---

## Objectives
The main goals of this project are:
- To analyze sales trends and patterns from historical data.
- To preprocess the data and create meaningful features for forecasting.
- To build and evaluate a machine learning model that predicts future sales.

---

## Dataset Description

- **train.csv**: Training data with time series of features (`store_nbr`, `family`, `onpromotion`) and target sales.
- **test.csv**: Test data with the same features as the training data. Predict the target sales for the dates in this file.
- **sample_submission.csv**: Sample submission file in the correct format.
- **stores.csv**: Store metadata, including city, state, type, and cluster.
- **oil.csv**: Daily oil price, including values during both the train and test data timeframes.
- **holidays_events.csv**: Holidays and Events with metadata.

## Key Features
- **Data Loading and Preprocessing**:
  - Handles missing values, cleans data, and transforms raw datasets.
  - Performs exploratory data analysis (EDA) to uncover trends and patterns.
- **Feature Engineering**:
  - Generates time-based features (e.g., year, month, day) and aggregates statistical information to improve the model.
- **Modeling**:
  - Implements one or more machine learning models (e.g., Linear Regression, Decision Trees, XGBoost) or time-series methods (e.g., ARIMA).
- **Evaluation**:
  - Evaluates model performance using metrics such as RMSE, MAE, and MAPE.
  - Visualizes predictions against actual values to validate model accuracy.


## Requirements
To run the notebook, ensure you have the following dependencies installed:
- `Python` 3.7+
- `Jupyter Notebook`
- `kaggle`
- `Pandas`
- `NumPy`
- `Scikit-learn`
- `scipy`
- `Matplotlib`
- `statsmodels`
- `Seaborn`
- (Optional) `XGBoost` or other `time-series` libraries like Statsmodels for `ARIMA`

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/retail-customer-segmentation-recommendation.git
```

2. Navigate to the project directory:

```bash
cd retail-customer-segmentation-recommendation
```

3. Open the Jupyter notebook:

```bash
jupyter notebook DS_Task_02_customer_segmentation_recommendation_system.ipynb
```

4. Run the cells in the notebook to execute the project code and analysis.

## Setup
1. Install the Kaggle API client:
   
   ```bash
   !pip install kaggle
   ```

2. Upload your Kaggle API key:
   - Go to https://www.kaggle.com//account
   - Click "Create New API Token"
   - Upload the downloaded kaggle.json file
   - Move the kaggle.json file to the correct location and set the correct permissions:
   ```bash
   !mkdir -p ~/.kaggle
   !mv kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ``` 
3. Download the dataset:
   ```bash
   !kaggle competitions download -c store-sales-time-series-forecasting
   ```
4. Unzip the downloaded file:
   ```bash
   !unzip store-sales-time-series-forecasting.zip
   ```
---
## Data Preprocessing
- Load the datasets using pandas.
- Merge the datasets (holidays_events.csv, stores.csv, oil.csv, and train.csv) based on relevant columns.
- Handle missing values and duplicates.
- Perform exploratory data analysis (EDA) to understand the data.
--- 

## Analysis
1. Does the type of stores affect the store sales?
   - Use ANOVA test.
2. Which family is having the highest sales?
   - Use a pie chart to visualize the distribution of sales by family.
3. Does promotion able to improve the sales?
   - Use Pearson correlation test.
4. Which city is having the most number of customers?
   - Use a count plot to visualize the distribution of sales by city.
5. Which state is having the most number of customers?
   - Use a count plot to visualize the distribution of sales by state.
6. Which of the stores has the highest sales?
   - Use a bar plot to visualize the total sales by store.
7. Which month is having the most sales, and least sales?
   - Use a line chart to visualize the monthly sales trend.
---

## Time Series Analysis
- **Autocorrelation**: Plot ACF and PACF to identify significant lag values.
- **Differencing Technique**: Transform the time series data to stationary.
- **Stationarity Test**: Use ADF and KPSS tests to check for stationarity. 
---
## Modeling
- **Autoregressive Integrated Moving Average (ARIMA) Model**:
   - Fit the *ARIMA* model to the differenced time series.
   - Make predictions and evaluate the model using *MAE*, *MSE*, and *RMSE*.
---
## Submission
- Create a submission file in the correct format and save it as *mysubmission.csv.*
---
## Conclusion
The project successfully predicts future sales using time-series forecasting techniques. The ARIMA model provides a reasonable prediction with evaluation metrics such as MAE, MSE, and RMSE.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## Acknowledgments

- Special thanks to the contributors of the dataset and the libraries used in this project.
- The project is inspired by various data science and machine learning resources available online.

## Stay Connected:
 * [![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=fff)](https://www.github.com/palakgandhi98)
 * [![LinkedIn](https://img.shields.io/badge/Linkedin-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/palakgandhi98)

Let's build something amazing together!