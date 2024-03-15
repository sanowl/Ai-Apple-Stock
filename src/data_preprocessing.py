import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/raw/appledata.csv', parse_dates=['date'])
data.set_index('date', inplace=True)

data.dropna(inplace=True)

label_encoder = LabelEncoder()
data['target'] = label_encoder.fit_transform(data['target'])


data['return'] = data['close'].pct_change()
data['ema_ratio'] = data['ema_50'] / data['ema_100']
data['sma_ratio'] = data['sma_50'] / data['sma_100']
data['rsi_diff'] = data['rsi_14'] - data['rsi_7']
data['cci_diff'] = data['cci_14'] - data['cci_7']

for lag in [1, 2, 3]:
    data[f'return_lag{lag}'] = data['return'].shift(lag)
    data[f'rsi_14_lag{lag}'] = data['rsi_14'].shift(lag)
    data[f'cci_14_lag{lag}'] = data['cci_14'].shift(lag)

X = data.drop(['target'], axis=1)
y = data['target']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)


train_size = 0.8
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=train_size, shuffle=False)


X_train.to_csv('data/processed/X_train.csv')
X_test.to_csv('data/processed/X_test.csv')
y_train.to_csv('data/processed/y_train.csv')
y_test.to_csv('data/processed/y_test.csv')