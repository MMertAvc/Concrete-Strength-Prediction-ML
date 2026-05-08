# Concrete Compressive Strength: Regression & Classification with Machine Learning

## Project Overview

Concrete is the most critical material in civil engineering. Its compressive strength is a highly nonlinear function of age and its ingredients. This project applies both **Machine Learning** and **Deep Learning** approaches to:
- **Predict** the exact compressive strength of concrete (**Regression**)
- **Classify** concrete into structural tiers and identify eco-friendly mixtures (**Classification**)

## Dataset

1,030 observations, 8 input features, 1 target.

| Feature | Description |
|---|---|
| Cement | kg/m³ |
| Blast Furnace Slag | kg/m³ |
| Fly Ash | kg/m³ |
| Water | kg/m³ |
| Superplasticizer | kg/m³ |
| Coarse Aggregate | kg/m³ |
| Fine Aggregate | kg/m³ |
| Age | days |
| **Strength** (target) | MPa |

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)

## Models Implemented

### Regression
| Model | R² |
|---|---|
| XGBoost Regressor | **0.923** |
| Gradient Boosting | 0.881 |
| Random Forest | ~0.92 |
| Decision Tree | 0.811 |
| Extra Tree | 0.811 |
| KNeighbors | 0.755 |
| AdaBoost | 0.737 |
| MLP (Neural Network - sklearn) | 0.717 |
| Linear / Ridge / Lasso / ElasticNet | ~0.628 |
| SVR | improved with scaling |
| SGD | improved with scaling |
| **Deep Learning (Keras ANN)** | competitive |

### Classification

Concrete is labeled by strength tier: `non-structural`, `residential`, `commercial`, `high-strength`.  
A `Green` flag marks eco-friendly mixes (high slag + fly ash or high plasticizer).

| Model | Accuracy |
|---|---|
| Random Forest | **0.87** |
| Gradient Boosting | 0.87 |
| AdaBoost | ~0.85 |
| Decision Tree | 0.85 |
| **Deep Learning (Keras ANN)** | 0.84 |
| Logistic Regression | 0.78 (scaled) |
| KNN | 0.73 (scaled) |
| SVM | 0.65 (scaled) |

## Technologies

- Python 3.x
- pandas, NumPy
- scikit-learn
- XGBoost
- TensorFlow / Keras

## How to Run

```bash
git clone https://github.com/yourusername/concrete-strength-prediction.git
cd concrete-strength-prediction
pip install -r requirements.txt
jupyter notebook "ANN -  DEEP Learning - Concrete Compressive Strength - Regression  & Classification.ipynb"
```

## Key Findings

- **XGBoost** and **Gradient Boosting** are top performers for regression (R² ≈ 0.92).
- **Random Forest** and **Gradient Boosting** lead classification (Accuracy ≈ 87%).
- **Cement content**, **water ratio**, and **curing age** are the most influential features.
- Proper **feature scaling** (StandardScaler) is essential for SVR, KNN, SGD, and neural networks.
- Deep Learning (Keras ANN) is competitive with gradient-boosted ensembles on this tabular dataset.
