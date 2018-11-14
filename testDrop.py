import pandas as pd
import numpy as np 

df = pd.read_csv('ACIDATA.csv',sep=',',header=0, encoding='unicode_escape')
#df = df.fillna(-1)
#df = df.drop(columns=['NEW_USED'])
df = df.loc[:, df.columns.intersection(['BRAND_MODEL'])]
df = df.dropna()
total_rows=len(df.axes[0])
print(df)
print(total_rows)
#df = df.to_csv('ACIExp.csv')
