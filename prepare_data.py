import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('/home/rufael.marew/Documents/ data_iiit.csv')
df.dropna(axis=0,inplace=True)

df = df.sample(n=200000 ,random_state=42)

train_df, test_df = train_test_split(df, test_size=0.1,random_state=0)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_df.to_csv(r'./dataset/train_1.txt', header=None, index=None, sep='\t', mode='a')
test_df.to_csv(r'./dataset/test_1.txt', header=None, index=None, sep='\t', mode='a')
