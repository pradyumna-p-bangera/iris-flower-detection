# Iris Flower Species Classifier

A full-stack machine learning web application built with **Flask** and **Scikit-Learn** that classifies Iris flower species (*Setosa*, *Versicolor*, and *Virginica*) based on sepal and petal measurements.

## Project Structure
- `train_model.py` — Loads the Iris dataset, trains the classification model, and exports `model.pkl`.
- `app.py` — Flask backend serving the web UI and `/predict` JSON API endpoint.
- `templates/index.html` — Interactive frontend interface for entering flower measurements and viewing predictions.
- `requirements.txt` — Python dependencies required to run the project.

## Input Features & Classes
### Features
- **Sepal Length** (cm)
- **Sepal Width** (cm)
- **Petal Length** (cm)
- **Petal Width** (cm)

### Predicted Species
- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt