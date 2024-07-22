import pandas as pd

ev = pd.read_csv('events.csv')
co = pd.read_csv('ad_costs.csv')

def last_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate last touch attribution"""
    events = events.loc[events['is_purchased'] == 1, :]
    last_touch_df = events.groupby('user_id').last()


    #last_touch_df = last_touch_df.sort_values(by='week', ascending=True)
    attribution = pd.pivot_table(last_touch_df, values='gmv', index=['week', 'user_id'], columns='channel',
                                 fill_value=0).reset_index()
    attribution['total_gmv'] = attribution.iloc[:, 2:].sum(axis=1).astype(int)
    # print(attribution.loc[attribution['user_id'] == 296, :])
    att = pd.DataFrame({'week': attribution['week'],
                        'user_id': attribution['user_id'],
                        'social_media': attribution['social_media'].astype(int),
                        'mobile_ads': attribution['mobile_ads'].astype(int),
                        'bloggers': attribution['bloggers'].astype(int),
                        'context_ads': attribution['context_ads'].astype(int),
                        'total_gmv': attribution['total_gmv']})
    att = att.sort_values(by=['week', 'user_id'], ascending=True)

    return att

def roi(attribution: pd.DataFrame, ad_costs: pd.DataFrame) -> pd.DataFrame:
    """Calculate ROI"""
    # YOUR CODE HERE
    merged_data = pd.merge(attribution, ad_costs, on='channel', how='left')
    merged_data['total_gmv'] = merged_data.iloc[:, 2:-2].sum(axis=1)
    merged_data['total_costs'] = merged_data['costs'] * merged_data['total_gmv'] / merged_data['total_gmv'].sum()

    # Calculate ROI percentage
    merged_data['roi%'] = ((merged_data['total_gmv'] - merged_data['total_costs']) / merged_data['total_costs']) * 100

    # Extract relevant columns for the final result
    roi_result = merged_data[['channel', 'total_gmv', 'total_costs', 'roi%']].round(0)

    return roi_result

at = last_touch_attribution(ev)
print(at)
#print(roi(at, co))


