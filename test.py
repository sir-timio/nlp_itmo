import pandas as pd
import time

df_down = pd.read_csv('train_down.csv')
df_up = pd.read_csv('train_up.csv')

train = df_down[['message_a', 'message_b']]
y = df_down['target'].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    train, y, train_size=0.8, random_state=42)

X_train['target'] = y_train

start = time.time()

from model import Model

model = Model()
pred = model.one_fit_predict(X_train, X_test)

print(f'time: {time.time()-start}')

from sklearn.metrics import accuracy_score

print(f'score: {accuracy_score(pred, y_test)}')






