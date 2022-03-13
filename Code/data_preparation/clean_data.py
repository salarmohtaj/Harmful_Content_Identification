import pandas as pd
import os
from sklearn.model_selection import KFold

#data_dir = "../../Data/hate_speech"
#data_name = "rawData.csv"

# df = pd.read_csv(os.path.join(data_dir, data_name))
# df = df[['text', 'task_1']]
# df = df.rename(columns={"task_1": "label"})
# print(df.columns)
# df['label'] = df['label'].str.replace('HOF', 'True')
# df['label'] = df['label'].str.replace('NOT', 'False')
#
#
# print(df["label"].value_counts())
#
# data_dir = "../../Data/hate_speech/final_data"
# data_name = "final_data.csv"
# try:
#     os.makedirs(data_dir)
# except OSError as e:
#     pass

data_dir = "../../Data/fake_news"
data_name = "merged_dataset.csv"
df = pd.read_csv(os.path.join(data_dir, data_name), sep="\t")
df = df[['text', 'label']]
#df = df.rename(columns={"task_1": "label"})
print(df.columns)
df['label'] = df['label'].str.replace('fake', 'True')
df['label'] = df['label'].str.replace('real', 'False')


print(df["label"].value_counts())

data_dir = "../../Data/fake_news/final_data"
data_name = "final_data.csv"
try:
    os.makedirs(data_dir)
except OSError as e:
    pass

df.to_csv(os.path.join(data_dir, data_name), sep="\t", index=False)
kf5 = KFold(n_splits=5, shuffle=True)
j = 1
for train_index, test_index in kf5.split(df):
    try:
        os.makedirs(os.path.join(data_dir, str(j)))
    except OSError as e:
        pass
    Train = df.iloc[train_index]
    Test = df.iloc[test_index]
    Train.to_csv(os.path.join(data_dir, str(j), "train.tsv"), sep="\t", index=False)
    Test.to_csv(os.path.join(data_dir, str(j), "test.tsv"), sep="\t", index=False)
    j += 1
