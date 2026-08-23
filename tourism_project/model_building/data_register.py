
import os
import pandas as pd

# Path to the dataset inside the repo — no external data store needed
DATA_PATH = "tourism_project/data/tourism.csv"

# Expected schema — used to catch a corrupted or swapped dataset early
EXPECTED_COLUMNS = [
    "Unnamed: 0", "CustomerID", "ProdTaken", "Age", "TypeofContact",
    "CityTier", "DurationOfPitch", "Occupation", "Gender",
    "NumberOfPersonVisiting", "NumberOfFollowups", "ProductPitched",
    "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

def register_dataset():
    # A clear message if the CSV was never uploaded/committed
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Confirm the schema matches what the rest of the pipeline expects
    missing_columns = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Print a short summary as proof the dataset was registered correctly
    print("Dataset registered successfully.")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values per column:\n{df.isnull().sum()}")

if __name__ == "__main__":
    register_dataset()
