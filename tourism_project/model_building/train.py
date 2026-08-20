
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

CATEGORICAL_COLUMNS = [
    "TypeofContact", "Occupation", "Gender",
    "ProductPitched", "MaritalStatus", "Designation",
]
NUMERIC_COLUMNS = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]

MODEL_OUTPUT_PATH = "tourism_project/deployment/best_model.joblib"

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

def load_splits():
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()
    return Xtrain, Xtest, ytrain, ytest

def build_pipeline():
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        (StandardScaler(), NUMERIC_COLUMNS),
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    return make_pipeline(preprocessor, model)

def train_and_log():
    Xtrain, Xtest, ytrain, ytest = load_splits()
    pipeline = build_pipeline()

    with mlflow.start_run():
        grid_search = GridSearchCV(
            pipeline, PARAM_GRID, cv=3, scoring="accuracy", n_jobs=-1
        )
        grid_search.fit(Xtrain, ytrain)

        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)

        ypred = best_model.predict(Xtest)
        accuracy = accuracy_score(ytest, ypred)
        report = classification_report(ytest, ypred, output_dict=True)

        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", report["weighted avg"]["precision"])
        mlflow.log_metric("test_recall", report["weighted avg"]["recall"])
        mlflow.log_metric("test_f1", report["weighted avg"]["f1-score"])

        print("Best parameters:", grid_search.best_params_)
        print(f"Test accuracy: {accuracy:.4f}")
        print(classification_report(ytest, ypred))

        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        joblib.dump(best_model, MODEL_OUTPUT_PATH)
        print(f"Model saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    train_and_log()
