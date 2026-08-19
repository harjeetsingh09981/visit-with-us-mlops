
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET_COLUMN = "ProdTaken"
COLUMNS_TO_DROP = ["Unnamed: 0", "CustomerID"]

def prepare_data():
    df = pd.read_csv(DATA_PATH)

    df = df.drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data preparation complete.")
    print(f"Xtrain shape: {Xtrain.shape}")
    print(f"Xtest shape: {Xtest.shape}")
    print(f"ytrain shape: {ytrain.shape}")
    print(f"ytest shape: {ytest.shape}")

if __name__ == "__main__":
    prepare_data()
