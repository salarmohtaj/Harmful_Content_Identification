import pandas as pd
import os
import numpy as np
data_dir = "../Data/hate_speech"

data_name = "rawData.csv"

df = pd.read_csv(os.path.join(data_dir,data_name))
print(df.shape)
print(df.columns)

print(df["task_1"].value_counts())

l = df["text"].apply(len)
print(len(l))
l = l.to_numpy()
print(l.mean())
print(l.min())
print(l.max())