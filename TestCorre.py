import pandas as pd
import numpy as np 
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
df = pd.read_csv('DropNewUsed.csv',sep=',',header=0, encoding='unicode_escape')
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

#print("------------------------------Mean Calculate------------------------------")

# mean of the specific column
#df.loc[:,"Score1"].mean()
print("-------------------------------End Process-------------------------------")

print("-------------------------------Correlation Test-------------------------------")

corr = df.corr()
f, ax = plt.subplots(figsize=(25, 25))

colormap = sns.diverging_palette(255, 10, as_cmap=True)
pic = sns.heatmap(corr,cmap=colormap, linewidths=.5, annot=True, fmt=".2f")

plt.show()