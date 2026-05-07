import pandas as pd

def preprocess_data(df):

    # Remove customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df.fillna(0, inplace=True)

    # Convert target
    df["Churn"] = df["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Convert categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df