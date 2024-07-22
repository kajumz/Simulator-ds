import pandas as pd
import numpy as np
from scipy.stats import linregress

def elasticity_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(['sku']) # it returns a groupby object that contains information about groups
    list_elasticity = []
    for sku, group_df in grouped:
        price_g = group_df['price']
        qty_g = group_df['qty']
        qty_reg = np.log1p(qty_g)
        mod = linregress(price_g, qty_reg)
        r = mod.rvalue ** 2
        list_elasticity.append({'sku' : int(sku), 'elasticity' : r})
    result = pd.DataFrame(list_elasticity, columns=['sku', 'elasticity'])


    return result
