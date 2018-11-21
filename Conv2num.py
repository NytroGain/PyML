import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('FINALDROPf.csv',sep=',',header=0, encoding='UTF-8')
print(df)
#df = df.fillna(-1)
#df = df.drop(['NEW_USED'], axis = 1)
#df = df.dropna()
#print(df)
#df = df.to_csv('DropNewUsed.csv', index=False)
#total_rows=len(df.axes[0])
#-----------------------Factorize-----------------------
df.APP_CODE = pd.factorize(df.APP_CODE)[0]
df.APP_CODE = pd.factorize(df.APP_CODE)[0]
df.SEX = pd.factorize(df.SEX)[0]
df.STATUS = pd.factorize(df.STATUS)[0]
df.BRAND = pd.factorize(df.BRAND)[0]
df.PRO_RES = pd.factorize(df.PRO_RES)[0]
df.OCCUP = pd.factorize(df.OCCUP)[0]
df.OCCUP_SUB = pd.factorize(df.OCCUP_SUB)[0]
df.COM_TYPE_COVERAGE = pd.factorize(df.COM_TYPE_COVERAGE)[0]
df.COM_INS_CODE = pd.factorize(df.COM_INS_CODE)[0]
df.OCCUP_SUB = pd.factorize(df.INS_PAY_TYPE)[0]


print("After transform")
print(df)

#---------------EXPORT------------------------------
df = df.to_csv('FileConverted.csv', sep=',', index=False)
#df = df.loc[:, df.columns.intersection(['SEX','AGE','BRAND_MODEL'])] #ตัดให้เหลือเฉพาะคอลัมน์ที่สนใจ
#total_rows=len(df.axes[0]) #นับจำนวน row

#print("------------------------------Show Info------------------------------")
#corr = df.corr()
#f, ax = plt.subplots(figsize=(20, 20))
#cmap = sns.diverging_palette(9, 9, as_cmap=True)
#pic = sns.heatmap(corr, annot=True)
#plt.show()
#___
#ab = df.info()
#ab = ab.to_csv('ACIExpInfo.csv', index=False)
#print(df)
#print(total_rows)
#df = df.to_csv('ACIExpIXTest.csv', index=False) #export โดยไม่เอาเลข row
print("-------------------------------End Process-------------------------------")