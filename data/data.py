import pandas as pd

df = pd.read_csv('data/kagle/train.txt', sep='\t')
df.to_csv('data/train.csv', index=False)
print(df.head())