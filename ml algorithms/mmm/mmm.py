import pandas as pd


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




