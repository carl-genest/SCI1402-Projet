import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress, shapiro, zscore, skew, kurtosis
import plotly.graph_objects as go
import plotly.express as px


df = pd.read_csv('data/bbs50-can_naturecounts_filtered_data.txt', header=0, sep='\t', usecols=['ScientificName', 'Order', 'StateProvince', 'YearCollected', 'ObservationCount', 'DecimalLatitude', 'DecimalLongitude', 'CommonName'])

bird_groups_df = pd.read_csv('data/Birds_by_Report_Group_Unique.csv', header=0, sep=',', usecols=['Common name', 'Scientific name', 'Trend', 'Goal', 'Report Group'])

merged_df = df.merge(bird_groups_df[['Scientific name', 'Report Group']], 
                     left_on='ScientificName', 
                     right_on='Scientific name', 
                     how='left')

filtered_df = merged_df.dropna(subset=['Report Group'])

summed_df = filtered_df.groupby(["Report Group", "ScientificName", "YearCollected"], as_index=False).agg({"ObservationCount": "sum"})





"""
one_slope_by_report_group = {}
for group, sub_df in summed_df.groupby("Report Group"): 
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        sub_df["YearCollected"], sub_df["ObservationCount"]
    )
    one_slope_by_report_group[group] = slope

# Extract the slopes from the slopes dictionary
slopes_list = [slope for slope in one_slope_by_report_group.values()]

# Perform the Shapiro-Wilk test for normality
stat, p_value = shapiro(slopes_list)

# Output the p-value of the test
print(f"Shapiro-Wilk Test p-value: {p_value}")




# Perform a Q-Q plot using the normal distribution
qq = stats.probplot(slopes_list, dist="norm", plot=None)

# Create a Plotly figure
fig = go.Figure()

# Scatter plot for the observed vs. expected quantiles
fig.add_scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Observed vs Theoretical')

# Add the line representing the normal distribution
x = np.array([qq[0][0][0], qq[0][0][-1]])
fig.add_scatter(
    x=x,
    y=qq[1][1] + qq[1][0] * x,
    mode='lines',
    name='Theoretical Line',
    line=dict(color='red', dash='dash')
)

# Update layout for the plot
fig.update_layout(
    title="Q-Q Plot for Slopes",
    xaxis_title="Theoretical Quantiles",
    yaxis_title="Observed Quantiles",
    template="plotly_dark",
    showlegend=False
)

fig.show()
"""




selected_bird_group = "Grassland Bird"
slopes_by_report_group = {}

for group, sub_df in summed_df.groupby(["Report Group", "ScientificName"]):
    if len(sub_df) >= 2:
        slope, intercept, r_value, p_value, std_err = linregress(sub_df["YearCollected"], sub_df["ObservationCount"])
        if group[0] not in slopes_by_report_group:
                slopes_by_report_group[group[0]] = []
        slopes_by_report_group[group[0]].append(slope)

# Extract slopes for Bird group and other groups
group_slopes = slopes_by_report_group.get(selected_bird_group, [])
nongroup_slopes = [slope for group, slopes in slopes_by_report_group.items() if group != selected_bird_group for slope in slopes]

"""
print(np.std(other_slopes))
print(np.mean(other_slopes))
print(len(other_slopes))

z_scores_other = zscore(other_slopes)
print(f"Outliers in Other: {np.where(np.abs(z_scores_other) > 3)}")  # Check z-scores greater than 3
"""

# Filter out outliers based on z-scores
z_scores = zscore(group_slopes)
non_outliers_indices = np.abs(z_scores) <= 3
group_slopes = np.array(group_slopes)[non_outliers_indices]

z_scores = zscore(nongroup_slopes)
non_outliers_indices = np.abs(z_scores) <= 3
nongroup_slopes = np.array(nongroup_slopes)[non_outliers_indices]

sh_stat_group, sh_p_group = shapiro(group_slopes)
sh_stat_nongroup, sh_p_nongroup = shapiro(nongroup_slopes)



# Create a DataFrame for box plot
box_plot_data = []
if group_slopes.any():
    box_plot_data.extend([(selected_bird_group, slope) for slope in group_slopes])
if nongroup_slopes.any():
    box_plot_data.extend([("Remaining Group", slope) for slope in nongroup_slopes])

df_slopes = pd.DataFrame(box_plot_data, columns=["Report Group", "Slope"])

# Apply a signed log transformation
df_slopes["Transformed Slope"] = np.sign(df_slopes["Slope"]) * np.log1p(abs(df_slopes["Slope"]))

# Create a box plot
fig_box = px.box(df_slopes, x="Report Group", y="Transformed Slope", points="all",
                 title="Distribution of Population Trends (Slopes) by Report Group",
                 labels={"Transformed Slope": "Log-Adjusted Trend Slope", "Report Group": "Bird Report Group"},
                 template="plotly_white")

# Show box plot
fig_box.show()



"""

# Generate Q-Q plot points


# Create Q-Q plots for both groups
fig = go.Figure()

# Q-Q plot for selected bird group
if len(group_slopes) > 1:
    q_theoretical, q_sample = qq_plot_data(group_slopes)
    q_theoretical, q_sample = filter_finite_values(q_theoretical, q_sample) 
    fig.add_trace(go.Scatter(x=q_theoretical, y=q_sample, mode='markers', name=f'Q-Q {selected_bird_group}'))

# Q-Q plot for non-group birds
if len(nongroup_slopes) > 1:
    q_theoretical_nongroup, q_sample_nongroup = qq_plot_data(nongroup_slopes)
    q_theoretical_nongroup, q_sample_nongroup = filter_finite_values(q_theoretical_nongroup, q_sample_nongroup)
    fig.add_trace(go.Scatter(x=q_theoretical_nongroup, y=q_sample_nongroup, mode='markers', name='Q-Q Other Groups'))

# Determine the actual range of plotted points
line_min = min(q_sample.min(), q_sample_nongroup.min(), q_theoretical.min(), q_theoretical_nongroup.min())
line_max = max(q_sample.max(), q_sample_nongroup.max(), q_theoretical.max(), q_theoretical_nongroup.max())

# Adjust reference line to fit within data range
fig.add_trace(go.Scatter(x=[line_min, line_max], y=[line_min, line_max], mode='lines', name='Reference Line', line=dict(color='black', width=2)))

# Layout
fig.update_layout(
    title='Q-Q Plot of Slopes', 
    xaxis_title='Theoretical Quantiles', 
    yaxis_title='Sample Quantiles', 
    template='plotly_white',
    xaxis=dict(range=[min(q_theoretical.min(), q_theoretical_nongroup.min())-5, max(q_theoretical.max(), q_theoretical_nongroup.max())+5])
)

# Show plot
fig.show()

"""



"""
if len(grassland_slopes) > 1 and len(other_slopes) > 1:
    # Kolmogorov-Smirnov test for normality
    stat_grass, p_grass = ks_2samp(grassland_slopes, np.random.normal(np.mean(grassland_slopes), np.std(grassland_slopes), len(grassland_slopes)))
    stat_other, p_other = ks_2samp(other_slopes, np.random.normal(np.mean(other_slopes), np.std(other_slopes), len(other_slopes)))

    print(f"Normality test p-values: Grassland={p_grass:.4f}, Other={p_other:.4f}")

    if p_grass > 0.05 and p_other > 0.05:  # Both distributions are normal
        # Bartlett’s test (assumes normality)
        stat_bartlett, p_bartlett = stats.bartlett(grassland_slopes, other_slopes)
        print(f"Bartlett’s Test: p-value = {p_bartlett:.4f}")

        if p_bartlett < 0.05: 
            print("Variances are significantly different (Bartlett’s test). Use Welch’s t-test.")
            t_stat, p_value = stats.ttest_ind(grassland_slopes, other_slopes, equal_var=False)  # Welch's t-test
            test_used = "Welch’s t-test"
        else:
            print("Variances are similar (Bartlett’s test). You may use a standard t-test.")
            t_stat, p_value = stats.ttest_ind(grassland_slopes, other_slopes, equal_var=True)   # Student’s t-test 
            test_used = "Standard t-test"
        
    else: 
        print("At least one group is not normally distributed. Using Mann-Whitney U test.")
        t_stat, p_value = stats.mannwhitneyu(grassland_slopes, other_slopes, alternative='two-sided')  
        test_used = "Mann-Whitney U test"

    print(f"{test_used} result: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")

    if p_value < 0.05:
        print("Significant difference between 'Grassland Bird' trends and other groups.")
    else:
        print("No significant difference in trends.")

else:
    print("Not enough data to perform the test.")

"""
"""
f_stat, p_val = stats.f_oneway(*slopes_by_report_group.values())
print(f"ANOVA F-statistic: {f_stat}, p-value: {p_val}")

"""


