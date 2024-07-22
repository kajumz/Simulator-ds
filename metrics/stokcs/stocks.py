import numpy as np
import pandas as pd

#def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    #"""manipulation with dataframe"""
    #df = df.copy()
    #a = df['stock'] > round(df['gmv'] / df['price'].astype(int), 0)
    #for i in range(len(a)):
        #if a[i]:
            #df['gmv'][i] = df['price'][i] * round(df['gmv'][i] / df['price'][i].astype(int), 0)
        #else:
            #df['gmv'][i] = df['price'][i] * df['stock'][i]

    #return df
def limit_gmv(df: pd.DataFrame) -> pd.DataFrame:
    """manipulation with dataframe"""
    df = df.copy()
    df['gmv'] = df['price'] * np.minimum(df['stock'], np.floor(df['gmv'] / df['price']))
    return df

