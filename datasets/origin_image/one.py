import pandas as pd

df_data = pd.read_csv('fold-5.csv')
df_data['source'] = df_data.apply(lambda x: x['imgnames'].split('/')[0], axis=1)
df_data = df_data[['imgnames', 'labels', 'new_labels', 'fold', 'source']]
df_data.to_csv('fold-5_name.csv')

