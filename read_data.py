import pandas as pd

df = pd.read_csv("sentiment/combined/paired/train_paired.tsv", delimiter='\t')
print(df.shape, df.columns)

print(df.head())
