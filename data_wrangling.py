# Data Wrangling - Pandas DataFrame                                      #

import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 18)
df = pd.read_csv(r'single_family_home_values.csv')    # zillow file
print(df.head(2))
print(df.tail(3))
print(type(df))
print(df.shape)
print(df.info())
print(df.describe())
print(df.fillna(df.mean()))
print(df.estimated_value.head(10))
print(df.estimated_value.describe())
print(df[['estimated_value', 'yearBuilt', 'priorSaleAmount']].tail(10))
print(df[df.estimated_value<=1000000].shape)
print(df[df.estimated_value<800000].shape)
sn.boxplot(df.estimated_value)
plt.show()

# plt.show(df.estimated_value)
#filter out noise (outliers) and slice the data frame    #
print(df[(df.estimated_value<=1000000) & (df.yearBuilt>2013) & (df.zipcode==80209)])

# Data Wrangling - Pandas Data Frames                #
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
desired_width = 400
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 20)
df = pd.read_csv(r'single_family_home_values.csv')    # zillow file
print(df[['estimated_value', 'yearBuilt', 'priorSaleAmount']].tail(10))
print(sn.boxplot(df.estimated_value))
print(df[df.estimated_value<=1000000].shape)
print(df[df.estimated_value<800000].shape)
sn.boxplot(df.estimated_value)
plt.show()
df2 = df[(df.estimated_value<=1000000) & (df.lastSaleAmount<=1000000)]
print(df2.shape)
sn.boxplot(df2.estimated_value)
plt.show()
df2.estimated_value.hist()
plt.show()
#Slice the dataframe    #
print(df[(df.estimated_value<=1000000) & (df.yearBuilt>2013) & (df.zipcode==80209)])
sn.pairplot(df2[['lastSaleAmount', 'estimated_value', 'zipcode']], hue='zipcode')
plt.show()
sn.stripplot(x=df2.zipcode, y=df.estimated_value)
plt.show()
sn.violinplot(x=df2.zipcode, y=df2.estimated_value)
plt.show()
df['lastSaleDate2'] = 1
df['lastSaleDate2'] = pd.to_datetime(df.lastSaleDate)
print(df.info())                                          # returns df columns
print(df.lastSaleDate2.dt.year.head(2))                    # returns lastSaleDate2 as year format
print(df.lastSaleDate2.dt.month.head(2))                    # returns lastSaleDate2 as month format
print(df.lastSaleDate2.dt.week.head(2))                    # returns lastSaleDate2 as week format
print(df.lastSaleDate2.dt.day.head(2))                    # returns lastSaleDate2 as day format
print(df.lastSaleDate2.dt.dayofweek.head(2))              # returns lastSaleDate2 as dayofweek format
# df.corr() correlation coefficients
print(df.corr())
print(df2.corr().loc['estimated_value', :].sort_values(ascending=False))
#GroupBy
df3 = df.groupby('zipcode').estimated_value.median().reset_index()      # reset_index() shifts pd series to df
df3.columns = ('zipcode', 'zip_median_value')
#Merge
df4 = pd.merge(df, df3, on = 'zipcode', how = 'left')
print(df4.head(2))