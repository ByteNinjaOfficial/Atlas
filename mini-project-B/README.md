# 🤖 Mini Project B — Titanic Survival Prediction & API

## Project Overview

This project builds and evaluates Machine Learning models to predict whether a Titanic passenger survived the disaster. It extends the Exploratory Data Analysis (EDA) from [Mini Project A](../mini-project-A/) by applying the insights gathered there to guide feature selection, preprocessing, and model development.

The best-performing model is served via a **FastAPI** REST API, allowing real-time survival predictions through HTTP requests.

---

## Objectives

- Load and prepare the Titanic dataset for machine learning.
- Engineer features (`FamilySize`, `IsAlone`, `Title`) based on EDA insights.
- Build a robust preprocessing pipeline using `sklearn.pipeline.Pipeline` and `ColumnTransformer`.
- Train and compare **9 classification models**.
- Evaluate models using **Accuracy, Precision, Recall, F1-Score, ROC-AUC, 5-Fold Cross-Validation**, and **Training Time**.
- Automatically save the best model using `joblib`.
- Deploy the model as a **FastAPI REST API** with prediction, health check, and home endpoints.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | `Titanic-Dataset.csv` (891 rows × 12 columns) |
| **Target Variable** | `Survived` (0 = Died, 1 = Survived) |
| **Missing Values** | `Age`: 177 (19.9%), `Cabin`: 687 (77.1%), `Embarked`: 2 (0.2%) |
| **Memory Usage** | 83.7 KB |

### Features Used After Preprocessing

| Feature | Type | Description |
|---|---|---|
| `Pclass` | Categorical | Passenger class (1, 2, 3) |
| `Sex` | Categorical | Gender (male, female) |
| `Age` | Numeric | Age (median imputed) |
| `SibSp` | Numeric | Number of siblings/spouses aboard |
| `Parch` | Numeric | Number of parents/children aboard |
| `Fare` | Numeric | Ticket fare |
| `Embarked` | Categorical | Port of embarkation (C, Q, S) |
| `FamilySize` | Numeric | `SibSp + Parch + 1` |
| `IsAlone` | Numeric | `1` if `FamilySize == 1`, else `0` |
| `Title` | Categorical | Extracted from `Name` (Mr, Mrs, Miss, Master, Rare) |

---

## Feature Engineering

```python
# Family Size
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# Is Alone
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Extract Title from Name
df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)

# Simplify rare titles
df["Title"] = df["Title"].replace(
    ['Lady','Countess','Capt','Col','Don','Dr','Major',
     'Rev','Sir','Jonkheer','Dona'], 'Rare'
)
df["Title"] = df["Title"].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'})
```

---

## Preprocessing Pipeline

The pipeline handles all transformations automatically:

- **Numeric Features** (`Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`, `IsAlone`):
  - Missing value imputation using **median**
  - Feature scaling using **StandardScaler**

- **Categorical Features** (`Sex`, `Embarked`, `Title`, `Pclass`):
  - Missing value imputation using **most frequent**
  - **OneHotEncoding** with unknown handling

---

## Models Evaluated

9 classification algorithms were trained and compared:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier (200 estimators)
4. Extra Trees Classifier (200 estimators)
5. Gradient Boosting Classifier
6. AdaBoost Classifier
7. K-Nearest Neighbors (KNN)
8. Support Vector Machine (SVM with `probability=True`)
9. Gaussian Naive Bayes

---

## 📊 Model Evaluation Results

| # | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV Accuracy | Training Time (s) |
|---|---|---|---|---|---|---|---|---|
| 1 | **Logistic Regression** | **0.8547** | **0.8413** | **0.7681** | **0.8030** | **0.8787** | 0.8227 | 0.0323 |
| 2 | Support Vector Machine | 0.8436 | 0.8254 | 0.7536 | 0.7879 | 0.8451 | **0.8339** | 0.0622 |
| 3 | Gaussian Naive Bayes | 0.8212 | 0.7606 | 0.7826 | 0.7714 | 0.8582 | 0.8047 | 0.0135 |
| 4 | K-Nearest Neighbors | 0.8212 | 0.7937 | 0.7246 | 0.7576 | 0.8530 | 0.8160 | 0.0120 |
| 5 | Gradient Boosting | 0.8156 | 0.8103 | 0.6812 | 0.7402 | 0.8505 | 0.8294 | 0.1638 |
| 6 | AdaBoost | 0.8101 | 0.7397 | 0.7826 | 0.7606 | 0.8547 | 0.8137 | 0.1214 |
| 7 | Random Forest | 0.8101 | 0.7692 | 0.7246 | 0.7463 | 0.8234 | 0.7957 | 0.3182 |
| 8 | Extra Trees | 0.8101 | 0.7692 | 0.7246 | 0.7463 | 0.8098 | 0.7812 | 0.2716 |
| 9 | Decision Tree | 0.8045 | 0.7576 | 0.7246 | 0.7407 | 0.7785 | 0.7654 | 0.0120 |

> **Best Model: Logistic Regression** — Highest test accuracy (85.5%), highest ROC-AUC (87.9%), and best F1-Score (80.3%) with the fastest training time.

The best model was automatically saved as `model.pkl` and is loaded by the FastAPI application.

---

## 🚀 FastAPI Prediction API

The project includes a fully functional **FastAPI** REST API (`app.py`) that loads the saved model pipeline and serves real-time predictions.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Home — Returns project info and status |
| `GET` | `/health` | Health check — Returns `{"status": "healthy"}` |
| `POST` | `/predict` | Predict — Accepts passenger data, returns survival prediction |

### Request Schema (`POST /predict`)

```json
{
  "Pclass": 1,
  "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
  "Sex": "female",
  "Age": 38,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 71.2833,
  "Embarked": "C"
}
```

### Response Example

```json
{
  "prediction": 1,
  "result": "Survived",
  "survival_probability": 95.18,
  "engineered_features": {
    "FamilySize": 2,
    "IsAlone": 0,
    "Title": "Mrs"
  }
}
```

### Running the API

```bash
cd mini-project-B
uvicorn app:app --reload --port 8001
```

Then open the interactive Swagger docs at: `http://127.0.0.1:8001/docs`

---

## 📸 API Screenshots

### Swagger UI — Landing Page
![Swagger UI Landing Page](Screenshots/Landing%20page.png)

### GET `/` — Home Endpoint
![GET Home Endpoint](Screenshots/GET%20Home.png)

### GET `/health` — Health Check
![GET Health Check](Screenshots/GET%20Health.png)

### POST `/predict` — Prediction Endpoint
![POST Predict Endpoint](Screenshots/POST%20Predict.png)

### POST `/predict` — Successful Prediction Response
![Successful Prediction Response](Screenshots/successful%20prediction%20response.png)

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Visualization |
| Scikit-Learn | ML pipeline, models, metrics, cross-validation |
| Joblib | Model serialization |
| FastAPI | REST API framework |
| Pydantic | Request/response validation |
| Uvicorn | ASGI server |
| Jupyter Notebook | Interactive model training |

---

## Project Structure

```text
mini-project-B/
├── training.ipynb       # Jupyter Notebook — ML training pipeline
├── app.py               # FastAPI prediction API
├── model.pkl            # Serialized best model (Logistic Regression)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── Screenshots/
    ├── Landing page.png                    # Swagger UI overview
    ├── GET Home.png                        # GET / endpoint
    ├── GET Health.png                      # GET /health endpoint
    ├── POST Predict.png                    # POST /predict endpoint
    └── successful prediction response.png  # Prediction response output
```

---

## Installation & Usage

1. **Install Dependencies:**
   ```bash
   cd mini-project-B
   pip install -r requirements.txt
   ```

2. **Run the Training Notebook:**
   ```bash
   jupyter notebook training.ipynb
   ```

3. **Start the API Server:**
   ```bash
   uvicorn app:app --reload --port 8001
   ```

4. **Access the API Docs:**
   Open `http://127.0.0.1:8001/docs` in your browser.
