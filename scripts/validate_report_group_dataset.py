import pandas as pd

bird_groups_df = pd.read_csv('C:/Users/Carl/Documents/GitHub/SCI1402-Projet/datasets/Birds_by_Report_Group.csv', header=0, sep=',')

print(bird_groups_df.count())

print(bird_groups_df.isnull().sum())