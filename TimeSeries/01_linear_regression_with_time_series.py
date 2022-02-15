"""
1. Linear Regression With Time Series
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression

plot_params = {
    'color': '0.75',
    'style': '.-',
    'markeredgecolor': '0.25',
    'markerfacecolor': '0.25',
    'legend': False,
    }
data_dir = Path('data/course')
comp_dir = Path('data/competition')

#Book sales dataset
book_sales = pd.read_csv(
    data_dir / 'book_sales.csv',
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)
book_sales['Time'] = np.arange(len(book_sales.index)) #dummy-time feature
book_sales['Lag_1'] = book_sales['Hardcover'].shift(1) #lag feature
book_sales = book_sales.reindex(columns=['Hardcover', 'Time', 'Lag_1'])

#Ar dataset
ar = pd.read_csv(data_dir / 'ar.csv')

#Store sales dataset
dtype = {
    'store_nbr': 'category',
    'family': 'category',
    'sales': 'float32',
    'onpromotion': 'uint64',
}
store_sales = pd.read_csv(
    comp_dir / 'train.csv',
    dtype=dtype,
    parse_dates=['date'],
    infer_datetime_format=True,
)
store_sales = store_sales.set_index('date').to_period('D')
store_sales = store_sales.set_index(['store_nbr', 'family'], append=True)
average_sales = store_sales.groupby('date').mean()['sales']

#Plot regression 'model' with single dummy-time feature
fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=book_sales, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=book_sales, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales')

#Different sign of coefficient in lag-feature
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
ax1.plot(ar['ar1']) #positive lag coefficient
ax1.set_title('Series 1')
ax2.plot(ar['ar2']) #negative lag coefficient
ax2.set_title('Series 2')

#Simple linear model based only on dummy-time feature
average_sales.head()
df = average_sales.to_frame() 
time = np.arange(len(df.index)) #dummy-time feature
df['time'] = time 
X = df.loc[:, ['time']]  # features
y = df.loc[:, 'sales']  # target
model = LinearRegression()
model.fit(X, y)
#Plot prediction vs labels
y_pred = pd.Series(model.predict(X), index=X.index)
ax = y.plot(**plot_params, alpha=0.5)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Total Store Sales')

#Simple linear model based only on lag feature
df = average_sales.to_frame()
lag_1 = df['sales'].shift(1)
df['lag_1'] = lag_1  # add to dataframe
X = df.loc[:, ['lag_1']].dropna()  # features
y = df.loc[:, 'sales']  # target
y, X = y.align(X, join='inner')  # drop corresponding values in target
model = LinearRegression().fit(X, y)
y_pred = pd.Series(model.predict(X), index=X.index)
#Plot prediction vs labels
fig, ax = plt.subplots()
ax.plot(X['lag_1'], y, '.', color='0.25')
ax.plot(X['lag_1'], y_pred)
ax.set(aspect='equal', ylabel='sales', xlabel='lag_1', title='Lag Plot of Average Sales')