import pandas as pd

def fillna_with_mean(df: pd.DataFrame, target: str, group: str) -> pd.DataFrame:
    """fillna with mean """
    df = df.copy()
    df[target] = df.groupby(group)[target].transform(lambda x: x.fillna(int(x.mean())))
    return df





