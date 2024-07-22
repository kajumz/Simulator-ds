import numpy as np
import pandas as pd

def agg_comp_price(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    result_df = pd.DataFrame()
    group = X.groupby(['sku', 'agg'])
    for group_name, group_data in group:
        if group_name[1] == 'med':
            group_data['comp_price'] = np.median(group_data['comp_price'])
        elif group_name[1] == 'min':
            group_data['comp_price'] = np.min(group_data['comp_price'])
        elif group_name[1] == 'max':
            group_data['comp_price'] = np.max(group_data['comp_price'])
        elif group_name[1] == 'avg':
            group_data['comp_price'] = np.average(group_data['comp_price'])
        elif group_name[1] == 'rnk':
            group_data['comp_price'] = group_data['comp_price'].iloc[0]
        local_data = group_data.head(1).copy()
        if local_data['comp_price'].iloc[0] is not None \
                and 0.8 * local_data['base_price'].iloc[0] \
                <= local_data['comp_price'].iloc[0] \
                <= 1.2 * local_data['base_price'].iloc[0]:

            local_data['new_price'] = local_data['comp_price']
        else:
            local_data['new_price'] = local_data['base_price']
        result_df = pd.concat([result_df, local_data])

    result_df = result_df.drop('rank', axis=1).reset_index(drop=True)

    return result_df

