# Script pour éliminer les entrées en surplus où une espèce est associée à plus d'un groupe

import pandas as pd


bird_groups_df = pd.read_csv('data/Birds_by_Report_Group.csv', header=0, sep=',')

bird_groups_df_sorted = bird_groups_df.sort_values(by="Scientific name")

conflicting_groups = bird_groups_df_sorted.groupby("Scientific name")["Report Group"].nunique()
conflicting_species = conflicting_groups[conflicting_groups > 1].index

filtered_df = bird_groups_df[~((bird_groups_df_sorted["Scientific name"].isin(conflicting_species)) & 
                               (bird_groups_df_sorted["Report Group"] == "Bird of Prey"))] # Insérer le nom d'un groupe

filtered_df.to_csv('data/Birds_by_Report_Group_Unique.csv', index=False)