import pandas as pd

data = pd.read_csv('events.csv')
#print(data)
last_event = data.groupby('user_id').last()
last_event['user_id'] = [x for x in range(1, 301)]
#print(last_event.loc[last_event['user_id'] == 112])
print(last_event)