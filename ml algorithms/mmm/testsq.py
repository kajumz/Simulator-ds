import pandas as pd

data = pd.read_csv('events.csv')
#print(data)
#last_touch_df = data.groupby(['week', 'user_id', 'channel']).agg(
#    {'gmv': 'sum'}).reset_index()
last_touch_df = data.groupby('user_id').last()
#last_touch_df['user_id'] = [x for x in range(1, 301)]
#print(last_touch_df)
last_touch_df = last_touch_df.sort_values('week', ascending=False)#.drop_duplicates('user_id')
#print(last_touch_df)
attribution = pd.pivot_table(last_touch_df, values='gmv', index=['week', 'user_id'], columns='channel',
                             fill_value=0).reset_index()
attribution['total_gmv'] = attribution.iloc[:, 2:].sum(axis=1)
attribution = attribution.sort_values('user_id')
attribution.drop(['channels'], axis=0)
print(attribution)
#print(attribution.loc[attribution['user_id'] == 296, :])
#att = pd.DataFrame({'week': attribution['week'],
#                                'user_id': attribution['user_id'],
#                                'social_media': attribution['social_media'],
#                                'mobile_ads': attribution['mobile_ads'],
#                                'bloggers': attribution['bloggers'],
#                                'context_ads': attribution['context_ads'],
#                                'total_gmv': attribution['total_gmv']})
#print(att)
