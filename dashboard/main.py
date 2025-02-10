import os
import requests
from urllib.parse import quote
import numpy as np
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm, linregress, zscore, skew, kurtosis, shapiro, bartlett, ttest_ind, mannwhitneyu


def read_chunks_into_dataframe(output_dir="data/data_split_files"):
    chunk_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("data_split_file_")])

    df_list = [pd.read_csv(chunk_files[0], sep="\t", header=0)]
    
    for chunk in chunk_files[1:]:
        df_list.append(pd.read_csv(chunk, sep="\t", header=None, names=df_list[0].columns))
    
    final_df = pd.concat(df_list, ignore_index=True)
    
    return final_df

def get_api_bird_data(sci_name, api_key):
    base_url = "https://nuthatch.lastelm.software/v2/birds"
    
    encoded_name = quote(sci_name)
    
    url = f"{base_url}?sciName={encoded_name}&operator=AND"

    headers = {
        "API-Key": api_key,
        "accept": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

def qq_plot_data(data):
    quantiles = np.linspace(0, 1, len(data))
    theoretical_quantiles = norm.ppf(quantiles, np.mean(data), np.std(data))
    sorted_data = np.sort(data)
    return theoretical_quantiles, sorted_data

def filter_finite_values(q_theoretical, q_sample):
    mask = np.isfinite(q_theoretical) & np.isfinite(q_sample)
    return q_theoretical[mask], q_sample[mask]


df = read_chunks_into_dataframe()

bird_groups_df = pd.read_csv('data/Birds_by_Report_Group.csv', header=0, sep=',')
bird_groups_unique_df = pd.read_csv('data/Birds_by_Report_Group_Unique.csv', header=0, sep=',')

scientific_names_list = sorted(df["ScientificName"].unique())
global_bird_name = "Poecile atricapillus"
year_list = sorted(df["YearCollected"].unique(), reverse=True)
global_year = year_list[0]
bird_groups_list = sorted(bird_groups_unique_df["Report Group"].unique())
global_bird_group = "Grassland Bird"

nuthatch_api_key = "accfdbb6-92ec-4836-948e-54500763bd96"

app = Dash(external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Bird Species Observation Dashboard",
                    className="text-center my-4"
                ),
                width=12
            )
        ),

        dbc.Row(
            dbc.Col(
                html.H5(
                    "Select a species",
                    className="text-center mb-2"
                ),
                width=12
            )
        ),
        
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="species-dropdown",
                    options=[{"label": name, "value": name} for name in scientific_names_list],
                    value=global_bird_name,
                    placeholder="Select a species",
                    className="text-dark"
                ),
                width=6
            ),
            className="mb-4 justify-content-center"
        ),

        dbc.Row(
            dbc.Col(
                dcc.Loading(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4(id="bird-name", className="card-title mb-2 text-info"),
                                html.P(id="bird-report-group", className="mb-1"),
                                html.P(id="bird-family", className="mb-1"),
                                html.P(id="bird-order", className="mb-1"),
                                html.P(id="bird-length", className="mb-1"),
                                html.P(id="bird-wingspan", className="mb-1"),
                                html.P(id="report-trend", className="mb-1"),
                                html.P(id="report-goal", className="mb-1"),
                            ]
                        ),
                        className="mb-4 shadow"
                    ),
                    id="loading-bird-card"
                ),
                width=6
            ),
            className="justify-content-center"
        ),
        
        dcc.Loading(
            dbc.Row(id="bird-images", className="mb-4"),
            id="loading-bird-images"
        ),
        
        dbc.Row(
            dbc.Col(
                dcc.Loading(
                    dcc.Graph(
                        id="observation-graph",
                        style={
                            'width': '80%', 
                            'margin': '0 auto'
                        }
                    ),
                    id="loading-observation-graph"
                ),
                width=12
            ),
            className="mb-4",
            style={'backgroundColor': 'black'}
        ),
        
        dbc.Row(
            dbc.Col(
                html.H5(
                    "Select a year",
                    className="text-dark text-center mb-2 mt-4"
                ),
                width=12
            ),
            style={'backgroundColor': 'white'}
        ),

        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[{"label": year, "value": year} for year in year_list],
                    value=global_year,
                    placeholder="Select a year",
                    className="text-dark"
                ), 
                width=3
            ),
            className="justify-content-center",
            style={'backgroundColor': 'white'}
        ),

        dbc.Row(
            dbc.Col(
                dcc.Loading(           
                    dcc.Graph(
                        id="species-map",
                        style={"height": "700px", "width": "100%"}
                    ),
                    id="loading-species-map"       
                ),
                width=12
            ),
            style={'backgroundColor': 'white'}
        ),

        dbc.Row(
            dbc.Col(
                html.H5(
                    "Select a bird group",
                    className="text-center mb-2 mt-4"
                ),
                width=12
            )
        ),
        
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="bird-group-dropdown",
                    options=[{"label": group, "value": group} for group in bird_groups_list],
                    value=global_bird_group,
                    placeholder="Select a bird group",
                    className="text-dark"
                ),
                width=5
            ),
            className="mb-4 justify-content-center"
        ),

        dbc.Row(
            dbc.Col(
                dcc.Loading(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Bird Group Analysis", className="card-title mb-2 text-info"),
                                html.P(id="group-len", className="mt-4 mb-1"),
                                html.P(id="group-mean", className="mb-1"),
                                html.P(id="group-std", className="mb-1"),
                                html.P(id="group-skew", className="mb-1"),
                                html.P(id="group-kurtosis", className="mb-1"),
                                html.P(id="group-norm-pval", className="mb-1"),
                                html.P(id="nongroup-len", className="mt-4 mb-1"),
                                html.P(id="nongroup-mean", className="mb-1"),
                                html.P(id="nongroup-std", className="mb-1"),
                                html.P(id="nongroup-skew", className="mb-1"),
                                html.P(id="nongroup-kurtosis", className="mb-1"),
                                html.P(id="nongroup-norm-pval", className="mb-1"),
                                html.P(id="norm_interpretation", className="mt-4 mb-1"),
                                html.P(id="variance_test_pval", className="mb-1"),
                                html.P(id="variance_interpretation", className="mb-1"),
                            ]
                        ),
                        className="mb-4 shadow"
                    ),
                    id="loading-bird-group-card"
                ),
                width=5
            ),
            className="justify-content-center"
        ),

        dbc.Row(
            dbc.Col(
                dcc.Loading(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Hypothesis test", className="card-title mb-2 text-info"),
                                html.P(id="null_hypo_def", className="mt-4 mb-1"),
                                html.P(id="alt_hypo_def", className="mb-1"),
                                html.P(id="null_hypo_test_name", className="mt-4 mb-1"),
                                html.P(id="null_hypo_test_t_stat", className="mb-1"),
                                html.P(id="null_hypo_test_p_val", className="mb-1"),
                                html.P(id="null_hypo_test_interpretation", className="mt-4 mb-1"),
                            ]
                        ),
                        className="mb-4 shadow"
                    ),
                    id="loading-null-hypo-card"
                ),
                width=5
            ),
            className="justify-content-center"
        ),

        dbc.Row(
            [
                dbc.Col(
                    dcc.Loading(           
                        dcc.Graph(
                            id="qqplot",
                            style={"height": "500px", "width": "100%"}
                        ),
                        id="loading-qqplot"       
                    ),
                    width=6
                ),
                dbc.Col(
                    dcc.Loading(           
                        dcc.Graph(
                            id="fig_box",
                            style={"height": "500px", "width": "100%"}
                        ),
                        id="loading-fig-box"       
                    ),
                    width=6
                )
            ],
            style={'backgroundColor': 'white'},
            className="justify-content-center"
        ),

        dbc.Row(
            dbc.Col(
                html.P(
                    "Développé par Carl Genest dans le cadre du cours SCI1420 Projet en science des données de l'Université Téluq",
                    className="text-center mb-2 mt-4"
                ),
                width=12
            ),
            className="justify-content-center"
        ),
    ],
    fluid=True
)

@callback(
    [
        Output("bird-name", "children"),
        Output("bird-report-group", "children"),
        Output("bird-family", "children"),
        Output("bird-order", "children"),
        Output("bird-length", "children"),
        Output("bird-wingspan", "children"),
        Output("report-trend", "children"),
        Output("report-goal", "children"),
        Output("bird-images", "children"),
        Output("observation-graph", "figure"),
        Output("year-dropdown", "options")
    ],
    Input("species-dropdown", "value")
)
def update_graph(selected_species):
    global_bird_name = selected_species
    species_filtered_df = df[df["ScientificName"] == selected_species]
    year_options = sorted(species_filtered_df["YearCollected"].unique(), reverse=True)
    
    bird_groups_for_species = bird_groups_df[bird_groups_df["Scientific name"] == selected_species]
    
    if not bird_groups_for_species.empty:
        report_groups_string = ', '.join(bird_groups_for_species['Report Group'].astype(str))
        bird_group_row = bird_groups_for_species.iloc[0]
    else:
        report_groups_string = None
        bird_group_row = None

    common_name = f"{species_filtered_df.iloc[0]['CommonName'].capitalize()}"
    order = f"Order: {species_filtered_df.iloc[0]['Order'].capitalize()}"

    report_group = f"Group: {report_groups_string}" if report_groups_string is not None else ""
    report_trend = f"2024 Report Trend: {bird_group_row.get('Trend', 'Unknown')}" if bird_group_row is not None else ""
    report_goal = f"2024 Report Goal: {bird_group_row.get('Goal', 'Unknown')}" if bird_group_row is not None else ""

    api_bird_data = get_api_bird_data(selected_species, nuthatch_api_key)

    if api_bird_data and "entities" in api_bird_data and isinstance(api_bird_data["entities"], list) and len(api_bird_data["entities"]) > 0:
        first_entity = api_bird_data["entities"][0]  

        family = f"Family: {first_entity.get('family')}" if first_entity.get("family") else ""
        length_min = first_entity.get('lengthMin')
        length_max = first_entity.get('lengthMax')
        wingspan_min = first_entity.get("wingspanMin")
        wingspan_max = first_entity.get("wingspanMax")

        if length_min and length_max:
            length = f"Length: {length_min} - {length_max} cm"
        else:
            length = ""

        if wingspan_min and wingspan_max:
            wingspan = f"Wingspan: {wingspan_min} - {wingspan_max} cm"
        else:
            wingspan = ""
        
        images = first_entity.get("images", [])
        image_cards = [
            dbc.Col(dbc.Card(dbc.CardImg(src=img, top=True), className="mb-3"), width=4)
            for img in images[:1]
        ]
        image_cards_list = dbc.Row(image_cards, className="justify-content-center")
    else: 
        family = ""
        length = ""
        wingspan = ""
        image_cards_list = ""

    
    
    summed_df = species_filtered_df.groupby("YearCollected", as_index=False)[["ObservationCount"]].sum()

    if len(summed_df) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(summed_df["YearCollected"], summed_df["ObservationCount"])
        summed_df["Regression"] = summed_df["YearCollected"] * slope + intercept
    else:
        slope, intercept, r_value, p_value, std_err = None, None, None, None, None
        summed_df["Regression"] = None 

    bar_fig = go.Figure()

    bar_fig.add_trace(
        go.Bar(
            x=summed_df["YearCollected"],
            y=summed_df["ObservationCount"],
            name="Observations",
            marker_color="lightskyblue",
            text=summed_df["ObservationCount"],
            textposition="outside" 
        )
    )

    if summed_df["Regression"].notnull().all():
        bar_fig.add_trace(
            go.Scatter(
                x=summed_df["YearCollected"],
                y=summed_df["Regression"],
                mode="lines",
                name="Linear Regression",
                line=dict(color="red")
            )
        )

    bar_fig.update_layout(
        title={
            "text": f"Observations and Trend for {selected_species}",
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        },
        xaxis_title="Year",
        yaxis_title="Total Observations",
        template="plotly_dark",

        xaxis=dict(
            tickformat=".0f"
        ),

        annotations=[
            dict(
                x=1.14, 
                y=0.5, 
                xref="paper", 
                yref="paper",
                text=f"Slope: {slope:.2f}<br>R: {r_value:.2f}" if slope is not None else f"",
                showarrow=False,
                font=dict(size=14, color="white"),
                bgcolor="rgba(0, 0, 0, 0.6)",
                bordercolor="white",
                borderwidth=1
            )
        ]
    )

    return common_name, report_group, family, order, length, wingspan, report_trend, report_goal, image_cards_list, bar_fig, year_options


@callback(
    Output('year-dropdown', 'value'),
    Input('year-dropdown', 'options'))
def update_selected_year(options):
    return options[0]


@callback(
    Output("species-map", "figure"),
    [Input("year-dropdown", "value")]
)
def update_map(selected_year):
    year_filtered_df = df[(df["ScientificName"] == global_bird_name) & (df["YearCollected"] == selected_year)]

    map_fig = px.scatter_geo(
        year_filtered_df,
        lat="DecimalLatitude",
        lon="DecimalLongitude",
        color="ObservationCount",
        size="ObservationCount",
        hover_name="ScientificName",
        hover_data=["YearCollected", "ObservationCount"],
        title=f"Observation Locations for {global_bird_name} in Canada in {selected_year}",
        projection="natural earth",
        scope="north america",
        template="plotly",
    )
    map_fig.update_geos(
        showcoastlines=True, 
        coastlinecolor="black", 
        projection_scale=1.75,
        center={"lat": 58.0000, "lon": -95.0000}
    )
    map_fig.update_layout(
        title={
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        }
    )

    return map_fig


@callback(
    [
        Output("group-len", "children"),
        Output("group-mean", "children"),
        Output("group-std", "children"),
        Output("group-skew", "children"),
        Output("group-kurtosis", "children"),
        Output("group-norm-pval", "children"),
        Output("nongroup-len", "children"),
        Output("nongroup-mean", "children"),
        Output("nongroup-std", "children"),
        Output("nongroup-skew", "children"),
        Output("nongroup-kurtosis", "children"),
        Output("nongroup-norm-pval", "children"),
        Output("norm_interpretation", "children"),
        Output("variance_test_pval", "children"),
        Output("variance_interpretation", "children"),
        Output("null_hypo_def", "children"),
        Output("alt_hypo_def", "children"),
        Output("null_hypo_test_name", "children"),
        Output("null_hypo_test_t_stat", "children"),
        Output("null_hypo_test_p_val", "children"),
        Output("null_hypo_test_interpretation", "children"),
        Output("qqplot", "figure"),
        Output("fig_box", "figure"),
    ],
    Input("bird-group-dropdown", "value")
)
def update_bird_group_card(selected_bird_group):
    global_bird_group = selected_bird_group

    merged_df = df.merge(bird_groups_unique_df[['Scientific name', 'Report Group']], 
                    left_on='ScientificName', 
                    right_on='Scientific name', 
                    how='left')

    filtered_df = merged_df.dropna(subset=['Report Group'])

    summed_df = filtered_df.groupby(["Report Group", "ScientificName", "YearCollected"], as_index=False).agg({"ObservationCount": "sum"})

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

    # Filter out outliers based on z-scores
    z_scores = zscore(group_slopes)
    non_outliers_indices = np.abs(z_scores) <= 3
    group_slopes = np.array(group_slopes)[non_outliers_indices]

    z_scores = zscore(nongroup_slopes)
    non_outliers_indices = np.abs(z_scores) <= 3
    nongroup_slopes = np.array(nongroup_slopes)[non_outliers_indices]

    group_mean_val = np.mean(group_slopes)
    group_std_val = np.std(group_slopes)
    group_len = f"Group size : {len(group_slopes)}"
    group_mean = f"Mean: {group_mean_val:.4f}"
    group_std = f"Std: {group_std_val:.4f}"
    group_skew = f"Skew: {skew(group_slopes):.4f}"
    group_kurtosis = f"Kurtosis: {kurtosis(group_slopes):.4f}"


    nongroup_mean_val = np.mean(nongroup_slopes)
    nongroup_std_val = np.std(nongroup_slopes)
    nongroup_len = f"Remaining group size : {len(nongroup_slopes)}"
    nongroup_mean = f"Mean: {nongroup_mean_val:.4f}"
    nongroup_std = f"Std: {nongroup_std_val:.4f}"
    nongroup_skew = f"Skew: {skew(nongroup_slopes):.4f}"
    nongroup_kurtosis = f"Kurtosis: {kurtosis(nongroup_slopes):.4f}"

    sh_stat_group, sh_p_group = shapiro(group_slopes)
    sh_stat_nongroup, sh_p_nongroup = shapiro(nongroup_slopes)
    group_norm_pval = f"Normality test p-value: {sh_p_group}"
    nongroup_norm_pval = f"Normality test p-value: {sh_p_nongroup}"

    if sh_p_group > 0.05 and sh_p_nongroup > 0.05:  # Both distributions are normal
        # Bartlett’s test (assumes normality)
        norm_interpretation = "Both distributions are normal; Bartlett's test apply for variance"
        stat_bartlett, p_bartlett = bartlett(group_slopes, nongroup_slopes)
        variance_test_pval = f"Bartlett's Test: p-value = {p_bartlett:.10f}"

        if p_bartlett < 0.05: 
            #Variances are significantly different (Bartlett’s test). Use Welch’s t-test.
            variance_interpretation = "Variances are significantly different; Welch's t-test apply"
            t_stat, p_value = ttest_ind(group_slopes, nongroup_slopes, equal_var=False)  # Welch's t-test
            test_used = "Welch’s t-test"
        else:
            #Variances are similar (Bartlett’s test). You may use a standard t-test.
            variance_interpretation = "Variances are similar; Student's t-test apply"
            t_stat, p_value = ttest_ind(group_slopes, nongroup_slopes, equal_var=True)   # Student’s t-test 
            test_used = "Standard t-test"
        
    else: 
        norm_interpretation = "At least one group is not normally distributed; Mann-Whitney U test apply"
        variance_test_pval = ""
        variance_interpretation = ""

        t_stat, p_value = mannwhitneyu(group_slopes, nongroup_slopes, alternative='two-sided')  
        test_used = "Mann-Whitney U test"

    null_hypo_def = f"H0: Differences between the {selected_bird_group} group slopes and the remaining group slopes are not significant"
    alt_hypo_def = f"H1: There is significant difference between the {selected_bird_group} group slopes and the remaining group slopes"
    null_hypo_test_name = f"{test_used} results"
    null_hypo_test_t_stat = f"T-statistic: {t_stat}"
    null_hypo_test_p_val = f"P-value: {p_value}"

    if p_value < 1e-10:
        null_hypo_test_interpretation = "Test does not return a consistent p-value; This could be due to outliers, skewness, or distribution shape"
    elif p_value < 0.05:
        null_hypo_test_interpretation = f"H0 rejected; Significant difference found between {selected_bird_group} trends and remaining group."
    else:
        null_hypo_test_interpretation = "H0 not rejected; No significant difference found in trends."

    # Create Q-Q plots for both groups
    qqfig = go.Figure()

    # Q-Q plot for selected bird group
    if len(group_slopes) > 1:
        q_theoretical, q_sample = qq_plot_data(group_slopes)
        q_theoretical, q_sample = filter_finite_values(q_theoretical, q_sample) 
        qqfig.add_trace(go.Scatter(x=q_theoretical, y=q_sample, mode='markers', name=f'Q-Q {selected_bird_group}'))

    # Q-Q plot for non-group birds
    if len(nongroup_slopes) > 1:
        q_theoretical_nongroup, q_sample_nongroup = qq_plot_data(nongroup_slopes)
        q_theoretical_nongroup, q_sample_nongroup = filter_finite_values(q_theoretical_nongroup, q_sample_nongroup)
        qqfig.add_trace(go.Scatter(x=q_theoretical_nongroup, y=q_sample_nongroup, mode='markers', name='Q-Q Other Groups'))

    # Determine the actual range of plotted points
    line_min = min(q_sample.min(), q_sample_nongroup.min(), q_theoretical.min(), q_theoretical_nongroup.min())
    line_max = max(q_sample.max(), q_sample_nongroup.max(), q_theoretical.max(), q_theoretical_nongroup.max())

    # Adjust reference line to fit within data range
    qqfig.add_trace(go.Scatter(x=[line_min, line_max], y=[line_min, line_max], mode='lines', name='Reference Line', line=dict(color='black', width=2)))

    # Layout
    qqfig.update_layout(
        title='Q-Q Plot of Slopes', 
        xaxis_title='Theoretical Quantiles', 
        yaxis_title='Sample Quantiles', 
        template='plotly_white',
        xaxis=dict(range=[min(q_theoretical.min(), q_theoretical_nongroup.min())-5, max(q_theoretical.max(), q_theoretical_nongroup.max())+5])
    )

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

    return group_len, group_mean, group_std, group_skew, group_kurtosis, group_norm_pval, nongroup_len, nongroup_mean, nongroup_std, nongroup_skew, nongroup_kurtosis, nongroup_norm_pval, norm_interpretation, variance_test_pval, variance_interpretation, null_hypo_def, alt_hypo_def, null_hypo_test_name, null_hypo_test_t_stat, null_hypo_test_p_val, null_hypo_test_interpretation, qqfig, fig_box


if __name__ == '__main__':
    app.run(debug=True)
