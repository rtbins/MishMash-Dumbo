import pandas as pd

a = pd.read_csv('Aaron_Eckhart_0001_jpg.csv')


a.loc[a['id']==2, 'accuracy'] = 50

print('sds', a.loc[a['id']==2])

a.to_csv('sdd.csv', index=False)