import pandas as pd
import numpy as np 

df = pd.read_csv('ACIDATA2.csv',sep=',',header=0, encoding='TIS-620')
#df = df.fillna(-1)
#df = df.drop(columns=['NOTIFY_DATE','EFF_DATE','EXP_DATE','OUTBOUND_DATE','HP_NO','NEW_USED','BRAND_MODEL','PERZIP','PERADD2','COM_EFF_Y','COM_EFF_M','COM_EXP_Y','COM_EXP_M','MAILADD2','MAILZIP','BIRTH_Y','INST_AMOUNT'])
#df = df.loc[:, df.columns.intersection(['BRAND_MODEL'])]
#df = df.dropna()
total_rows=len(df.axes[0])
print(df)
print(total_rows)
df = df.to_csv('ACIFull.csv', sep=',', index=False)
print("----------------------END------------------------------")