import pandas as pd
from solution import bias
from solution import GroupTimeSeriesSplit
from solution import mape
from solution import smape
from solution import wape


def main():
    # Data loading
    df_path = "../datasets/data_train_sql.csv"
    df = pd.read_csv(df_path, parse_dates=["monday"])

    y = df.pop("y")

    # monday or product_name as a groups for validation?
    df.drop(..., axis=1, inplace=True)
    groups = df.pop(...)

    X = df

    # Validation loop
    cv = GroupTimeSeriesSplit(
        n_splits=5,
        max_train_size=None,
        test_size=None,
        gap=0,
    )

    for train_idx, test_idx in cv.split(X, y, groups):
        # Split train/test
        ...

        # Fit model
        model = best_model()
        # Predict and print metrics
        y_pred = model.predict(X_test)
        ...


if __name__ == "__main__":
    main()
