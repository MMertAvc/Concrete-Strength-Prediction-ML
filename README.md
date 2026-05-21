# Concrete Compressive Strength — Regression & Classification with ML

A comprehensive machine learning and deep learning project that predicts concrete compressive strength through regression and classifies concrete into structural tiers through multi-class classification.

---

## English

### About

Concrete compressive strength is one of the most critical properties in civil engineering, yet it is a highly nonlinear function of its ingredients and curing age. This project applies a wide range of ML algorithms — from classical ensemble methods to deep neural networks — to both predict the exact strength (regression) and categorize concrete into practical structural tiers (classification).

### Features

- **Regression:** Predict exact MPa compressive strength
- **Classification:** Categorize concrete as non-structural / residential / commercial / high-strength
- **Eco-friendly flagging:** Identifies environmentally friendly mixes (high slag, fly ash, or plasticizer content)
- Comprehensive model comparison across 10+ algorithms
- Feature importance analysis

### Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)

| Feature | Unit |
|---|---|
| Cement | kg/m³ |
| Blast Furnace Slag | kg/m³ |
| Fly Ash | kg/m³ |
| Water | kg/m³ |
| Superplasticizer | kg/m³ |
| Coarse Aggregate | kg/m³ |
| Fine Aggregate | kg/m³ |
| Age | days |
| **Strength (target)** | MPa |

- **Observations:** 1,030
- **Features:** 8 input + 1 target

### Model Architecture / Tech Stack

**Regression Results:**

| Model | R² Score |
|---|---|
| XGBoost Regressor | **0.923** |
| Random Forest | ~0.920 |
| Gradient Boosting | 0.881 |
| Decision Tree | 0.811 |
| Extra Tree | 0.811 |
| KNeighbors | 0.755 |
| AdaBoost | 0.737 |
| MLP (sklearn) | 0.717 |
| Linear / Ridge / Lasso / ElasticNet | ~0.628 |
| Deep Learning ANN (Keras) | competitive |

**Classification Results:**

| Model | Accuracy |
|---|---|
| Random Forest | **0.87** |
| Gradient Boosting | 0.87 |
| AdaBoost | ~0.85 |
| Decision Tree | 0.85 |
| Deep Learning ANN (Keras) | 0.84 |
| Logistic Regression | 0.78 |
| KNN | 0.73 |
| SVM | 0.65 |

**Tech Stack:** Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow/Keras

### Key Findings

- XGBoost and Gradient Boosting are top performers for regression (R² ≈ 0.92)
- Cement content, water ratio, and curing age are the most influential features
- Proper feature scaling (StandardScaler) is essential for SVR, KNN, SGD, and neural networks
- Deep Learning ANN is competitive with gradient-boosted ensembles on this tabular dataset

### How to Run

```bash
git clone https://github.com/MMertAvc/Concrete-Strength-Prediction-ML.git
cd Concrete-Strength-Prediction-ML
pip install -r requirements.txt
jupyter notebook "ANN - DEEP Learning - Concrete Compressive Strength - Regression & Classification.ipynb"
```

### Requirements

```
tensorflow
keras
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
```

---

## Türkçe

### Hakkında

Beton basınç dayanımı, inşaat mühendisliğinde en kritik özelliklerden biridir; ancak malzeme bileşenlerinin ve kür süresinin son derece doğrusal olmayan bir fonksiyonudur. Bu proje, klasik topluluk yöntemlerinden derin sinir ağlarına kadar geniş bir ML algoritması yelpazesini uygulayarak hem tam dayanımı tahmin eder (regresyon) hem de betonu pratik yapısal kategorilere ayırır (sınıflandırma).

### Özellikler

- **Regresyon:** Tam MPa basınç dayanımını tahmin etme
- **Sınıflandırma:** Betonu yapısal olmayan / konut / ticari / yüksek dayanımlı olarak kategorilendirme
- **Çevre dostu işaretleme:** Çevreci karışımları tespit etme (yüksek cüruf, uçucu kül veya plastifiye içeriği)
- 10+ algoritma üzerinde kapsamlı model karşılaştırması
- Özellik önemi analizi

### Veri Seti

**Kaynak:** [UCI Makine Öğrenmesi Deposu](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)

| Özellik | Birim |
|---|---|
| Çimento | kg/m³ |
| Yüksek Fırın Cürufu | kg/m³ |
| Uçucu Kül | kg/m³ |
| Su | kg/m³ |
| Süperpiastifiye | kg/m³ |
| İri Agrega | kg/m³ |
| İnce Agrega | kg/m³ |
| Yaş | gün |
| **Dayanım (hedef)** | MPa |

- **Gözlem Sayısı:** 1.030
- **Özellikler:** 8 giriş + 1 hedef

### Model Mimarisi / Teknoloji Yığını

**Regresyon Sonuçları:**

| Model | R² Skoru |
|---|---|
| XGBoost Regressor | **0.923** |
| Random Forest | ~0.920 |
| Gradient Boosting | 0.881 |
| Karar Ağacı | 0.811 |
| Extra Tree | 0.811 |
| KNeighbors | 0.755 |
| AdaBoost | 0.737 |
| MLP (sklearn) | 0.717 |
| Doğrusal / Ridge / Lasso / ElasticNet | ~0.628 |
| Derin Öğrenme ANN (Keras) | rekabetçi |

**Sınıflandırma Sonuçları:**

| Model | Doğruluk |
|---|---|
| Random Forest | **0.87** |
| Gradient Boosting | 0.87 |
| AdaBoost | ~0.85 |
| Karar Ağacı | 0.85 |
| Derin Öğrenme ANN (Keras) | 0.84 |
| Lojistik Regresyon | 0.78 |
| KNN | 0.73 |
| SVM | 0.65 |

**Teknoloji Yığını:** Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow/Keras

### Temel Bulgular

- XGBoost ve Gradient Boosting regresyon için en iyi performansı göstermektedir (R² ≈ 0.92)
- Çimento içeriği, su oranı ve kür süresi en etkili özelliklerdir
- SVR, KNN, SGD ve sinir ağları için uygun özellik ölçeklendirmesi (StandardScaler) gereklidir
- Derin Öğrenme ANN, bu tablo veri kümesinde gradient-boosted topluluk yöntemleriyle rekabet edebilir

### Nasıl Çalıştırılır

```bash
git clone https://github.com/MMertAvc/Concrete-Strength-Prediction-ML.git
cd Concrete-Strength-Prediction-ML
pip install -r requirements.txt
jupyter notebook "ANN - DEEP Learning - Concrete Compressive Strength - Regression & Classification.ipynb"
```

### Gereksinimler

```
tensorflow
keras
scikit-learn
xgboost
pandas
numpy
matplotlib
seaborn
```
