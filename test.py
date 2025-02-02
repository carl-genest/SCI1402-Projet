import pandas as pd

bird_groups_df = pd.read_csv('C:/Users/Carl/Documents/GitHub/SCI1402-Projet/datasets/Birds_by_Report_Group.csv', header=0, sep=',')

# Display count of non-null values per column
print(bird_groups_df.count())

# Display total number of missing values per column
print(bird_groups_df.isnull().sum())

"""
print('start')

df = pd.read_csv('C:/Users/Carl/Documents/GitHub/SCI1402-Projet/datasets/bbs50-can/bbs50-can_naturecounts_data.txt', header=0, sep='\t', usecols=['ScientificName', 'Order', 'StateProvince', 'YearCollected', 'ObservationCount', 'DecimalLatitude', 'DecimalLongitude', 'CommonName'])

filtered_df = df[df["YearCollected"] > 1996]

filtered_df.to_csv('C:/Users/Carl/Documents/GitHub/SCI1402-Projet/datasets/bbs50-can/bbs50-can_naturecounts_filtered_data.txt', sep='\t', index=False)

print('finish')
"""
"""
attributes = [
    "ScientificName", "Order", "StateProvince", "DecimalLatitude", "DecimalLongitude", 
    "YearCollected", "ObservationCount", "CommonName"
]

for attribute in attributes:
    null_count = df[attribute].isna().sum()
    print(f"Null values in '{attribute}': {null_count}")
"""