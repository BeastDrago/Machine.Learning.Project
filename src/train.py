import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from .utils import load_data

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models'
MODEL_DIR.mkdir(exist_ok=True)

def build_pipeline():
    numeric_features = ['study_hours','failed_subjects','absences','hours_wasted','grasp','sleep','motivation']
    categorical_features = ['gender','wifi','parental_support','extra_classes']
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    clf = Pipeline(steps=[('pre', preprocessor),
                          ('clf', RandomForestClassifier(n_estimators=150, random_state=42))])
    return clf

def train_and_save(path=None, model_path=None):
    df = load_data(path)
    X = df.drop(columns=['pass'])
    y = df['pass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, stratify=y, random_state=42)
    clf = build_pipeline()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print('Test accuracy:', acc)
    print(classification_report(y_test, preds))
    if model_path is None:
        model_path = MODEL_DIR / 'model.pkl'
    joblib.dump(clf, model_path)
    print(f'Model saved to {model_path}')
    return clf

if __name__ == '__main__':
    train_and_save()