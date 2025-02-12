import pandas as pd

df = pd.read_csv('datasets/bbs50-can/bbs50-can_naturecounts_data.txt', header=0, sep='\t', usecols=['ScientificName', 'Order', 'StateProvince', 'YearCollected', 'ObservationCount', 'DecimalLatitude', 'DecimalLongitude', 'CommonName'])

attributes = [
    "ScientificName", "Order", "StateProvince", "DecimalLatitude", "DecimalLongitude", 
    "YearCollected", "ObservationCount", "CommonName"
]

for attribute in attributes:
    null_count = df[attribute].isna().sum()
    print(f"Null values in '{attribute}': {null_count}")