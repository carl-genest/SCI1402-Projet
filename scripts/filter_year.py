import pandas as pd

df = pd.read_csv('datasets/bbs50-can/bbs50-can_naturecounts_data.txt', header=0, sep='\t', usecols=['ScientificName', 'Order', 'StateProvince', 'YearCollected', 'ObservationCount', 'DecimalLatitude', 'DecimalLongitude', 'CommonName'])

filtered_df = df[df["YearCollected"] > 1996]

filtered_df.to_csv('datasets/bbs50-can/bbs50-can_naturecounts_filtered_data.txt', sep='\t', index=False)