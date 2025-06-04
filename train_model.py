import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import joblib

DATA_PATH = 'data.csv'
MODEL_PATH = 'model.pkl'

def main():
    df = pd.read_csv(DATA_PATH)

    sport_enc = LabelEncoder()
    X = sport_enc.fit_transform(df['sport']).reshape(-1, 1)

    eye_enc = LabelEncoder()
    y = eye_enc.fit_transform(df['eye_color'])

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    joblib.dump({'model': clf, 'sport_enc': sport_enc, 'eye_enc': eye_enc}, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

if __name__ == '__main__':
    main()
