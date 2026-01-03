# 🌧️ Rainfall Prediction using Machine Learning

This project predicts whether it will rain or not based on weather parameters such as pressure, humidity, wind speed, cloud cover, and sunshine using a **Random Forest Classifier**.

The model includes data preprocessing, exploratory data analysis, class imbalance handling, and hyperparameter tuning using **GridSearchCV** to achieve optimal performance.

---

## 📂 Project Structure

```text
.
├── Rainfall.csv
├── rainfall.py
├── requirements.txt
└── README.md
```
### ⚙️ Technologies 
```
Python

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn
```

### 📊 Dataset

The dataset contains daily weather observations.

Target variable: rainfall

yes → 1

no → 0

Missing values are handled using:

Mode for categorical features

Median for numerical features

Place the dataset file as Rainfall.csv in the project root directory.

### 🔧 Installation

Clone the repository:
```
git clone https://github.com/RDharanish24/rainfall_prediction
cd rainfall_prediction
```


Install the required dependencies:
```
pip install -r requirements.txt
```
### 🚀 How to Run

Launch Jupyter Notebook:
```
jupyter notebook rainfall.py

```
Run all cells sequentially to:

Preprocess the data

Perform EDA

Train the model

Evaluate performance

Make predictions

### 🧠 Model Workflow

Data Cleaning and Preprocessing

Exploratory Data Analysis (EDA)

Feature Selection

Handling Class Imbalance (Downsampling)

Train-Test Split

Hyperparameter Tuning using GridSearchCV

Model Evaluation

### 📈 Model Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Cross-Validation Score (5-fold)

### 📌 Future Improvements

Use SMOTE for class imbalance handling

Feature importance visualization

Try additional ML models (XGBoost, Logistic Regression)

Deploy the model using Flask or FastAPI
