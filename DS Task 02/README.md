# Retail Customer Segmentation and Recommendation System

This repository contains an end-to-end project for retail customer segmentation and recommendation system. The project aims to understand customer purchasing behaviors, segment customers into distinct clusters, and build a recommendation system to suggest products to customers based on their cluster preferences.

---
## Project Overview

The project involves the following steps:

1. **Data Cleaning and Transformation**:
   - Handling missing values.
   - Removing duplicate entries.
   - Correcting anomalies in product codes and descriptions.
   - Treating cancelled transactions.

2. **Feature Engineering**:
   - Creating RFM (Recency, Frequency, Monetary) features.
   - Engineering product diversity and behavioral features.
   - Incorporating geographic and cancellation insights.
   - Extracting seasonality and trend features.

3. **Dimensionality Reduction**:
   - Applying Principal Component Analysis (PCA) to reduce the dimensionality of the feature space.

4. **Clustering**:
   - Applying K-means clustering to segment customers into distinct clusters.
   - Evaluating the clustering quality using silhouette analysis, Calinski-Harabasz score, and Davies-Bouldin score.

5. **Cluster Analysis and Profiling**:
   - Analyzing the characteristics of each cluster.
   - Profiling each cluster to identify key traits.

6. **Recommendation System**:
   - Developing a recommendation system to suggest products to customers based on their cluster preferences.
---

## Dataset

The dataset used in this project is a [retail transaction dataset](https://archive.ics.uci.edu/dataset/352/online+retail) containing information about customer purchases, including details such as invoice number, stock code, description, quantity, unit price, customer ID, and country.

## Dependencies

The project uses the following Python libraries:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `yellowbrick`
- `plotly`
- `tabulate`

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

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

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Special thanks to the contributors of the dataset and the libraries used in this project.
- The project is inspired by various data science and machine learning resources available online.

## Stay Connected:
 * [![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=fff)](https://www.github.com/palakgandhi98)
 * [![LinkedIn](https://img.shields.io/badge/Linkedin-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/palakgandhi98)

Let's build something amazing together!