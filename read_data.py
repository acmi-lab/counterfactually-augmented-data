import pandas as pd

df = pd.read_csv("sentiment/combined/paired/train_paired.tsv", delimiter='\t')
print(df.head())
df["label"] = (df["Sentiment"] == "Positive").astype('int')
df = df[["label", "Text"]]
print(df.head())
df.to_csv("sentiment/combined/paired/train_preprocessed.csv", index=False)

df = pd.read_csv("sentiment/combined/paired/dev_paired.tsv", delimiter='\t')
print(df.head())
df["label"] = (df["Sentiment"] == "Positive").astype('int')
df = df[["label", "Text"]]
print(df.head())
df.to_csv("sentiment/combined/paired/dev_preprocessed.csv", index=False)

df = pd.read_csv("sentiment/combined/paired/test_paired.tsv", delimiter='\t')
print(df.head())
df["label"] = (df["Sentiment"] == "Positive").astype('int')
df = df[["label", "Text"]]
print(df.head())
df.to_csv("sentiment/combined/paired/test_preprocessed.csv", index=False)

