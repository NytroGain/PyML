import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('ACIDATA.csv',sep=',',header=0, encoding='unicode_escape')
#df = df.fillna(-1)

#df = df.dropna()
#df = df.loc[:, df.columns.intersection(['SEX','AGE','BRAND_MODEL'])] #ตัดให้เหลือเฉพาะคอลัมน์ที่สนใจ
#total_rows=len(df.axes[0]) #นับจำนวน row

print("------------------------------Show Info------------------------------")

#ab = df.info()
#ab = ab.to_csv('ACIExpInfo.csv', index=False)
#print(df)
#print(total_rows)
#df = df.to_csv('ACIExpIXTest.csv', index=False) #export โดยไม่เอาเลข row
print("-------------------------------End Process-------------------------------")

print("-------------------------------Correlation Test-------------------------------")

cov = df.cov()
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
pic = sns.heatmap(cov, linewidths=.5, annot=True)
plt.show()