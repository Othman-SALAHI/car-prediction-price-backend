# 🚗 Car Price Prediction Web App (Morocco) 🇲🇦

A Flask-based web application that predicts the price of used cars in Morocco using a trained **CatBoostRegressor** machine learning model. This app helps users estimate car prices based on features like brand, model, year, fuel type, and more.

### 🔗 Live Demo
👉 [https://car-prediction-price-maroc.vercel.app](https://car-prediction-price-maroc.vercel.app)

---

## 📌 Features

- Predict car prices instantly using machine learning
- Built with Python, Flask, and CatBoost
- Clean and responsive web UI (HTML/CSS)
- Moroccan market-specific data
- Lightweight and easy to deploy

---

## 🧠 Model Info

- **Algorithm**: CatBoost Regressor
- **Evaluation**:
  - R² Score: ~0.XX
  - MAE: ~XXX MAD
  - RMSE: ~XXX MAD
- **Training Data**: Cleaned dataset of used car listings in Morocco
- **Trained with**: 700 iterations, learning rate 0.04, depth 7

Model is saved as `catboost_model.pkl` and loaded at runtime to make predictions.

---

## 🛠️ Tech Stack

- `Flask` – lightweight Python web framework
- `CatBoost` – gradient boosting library by Yandex
- `scikit-learn`, `joblib`, `numpy`, `pandas` – for data processing & evaluation
- `HTML/CSS` – for UI
- `Vercel` – for deployment

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
