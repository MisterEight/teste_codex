# Eye Color Predictor

This project demonstrates a simple machine learning pipeline using a decision tree
classifier to predict a person's eye color based on the sport they play.
It includes a small Flask web application where a user can input their sport
and receive a predicted eye color. Each prediction is stored in `user_data.csv`.

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python train_model.py
```

This will create `model.pkl`.

3. Run the web application:

```bash
python app.py
```

Navigate to `http://localhost:5000` in a browser to use the form.

## Files

- `data.csv` – sample training data.
- `train_model.py` – script to train and save the decision tree model.
- `app.py` – Flask application serving the prediction form and storing user data.
- `templates/index.html` – HTML template for the front-end form.
- `user_data.csv` – generated file storing user submissions and predictions.


