import pandas as pd


def last_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate last touch attribution"""
    last_touch_df = events.groupby(['week', 'user_id', 'channel']).agg(
        {'gmv': 'sum'}).reset_index()
    last_touch_df = last_touch_df.sort_values('week', ascending=False).drop_duplicates('user_id')

    attribution = pd.pivot_table(last_touch_df, values='gmv', index=['week', 'user_id'], columns='channel',
                                 fill_value=0).reset_index()
    attribution['total_gmv'] = attribution.iloc[:, 2:].sum(axis=1)
    # print(attribution.loc[attribution['user_id'] == 296, :])
    att = pd.DataFrame({'week': attribution['week'],
                        'user_id': attribution['user_id'],
                        'social_media': attribution['social_media'],
                        'mobile_ads': attribution['mobile_ads'],
                        'bloggers': attribution['bloggers'],
                        'context_ads': attribution['context_ads'],
                        'total_gmv': attribution['total_gmv']})
    att = att.sort_values('user_id')
    return att


def first_touch_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate first touch attribution"""
    # YOUR CODE HERE
    pass


def linear_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate linear attribution"""
    # YOUR CODE HERE
    pass


def u_shaped_attribution(events: pd.DataFrame) -> pd.DataFrame:
    """Calculate U-Shaped attribution"""
    # YOUR CODE HERE
    pass

def roi(attribution: pd.DataFrame, ad_costs: pd.DataFrame) -> pd.DataFrame:
    """Calculate ROI"""
    # YOUR CODE HERE
    pass




