import os
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from scipy.stats import linregress
import requests
from urllib.parse import quote

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

df = read_chunks_into_dataframe()

bird_groups_df = pd.read_csv('data/Birds_by_Report_Group.csv', header=0, sep=',')

scientific_names_list = sorted(df["ScientificName"].unique())
global_bird_name = "Poecile atricapillus"
year_list = sorted(df["YearCollected"].unique(), reverse=True)
global_year = year_list[0]
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
                                html.H4(id="bird-name", className="card-title mb-2 text-primary"),
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
    
    bird_group_for_species = bird_groups_df[bird_groups_df["Scientific name"] == selected_species]
    
    if not bird_group_for_species.empty:
        bird_group_row = bird_group_for_species.iloc[0]
    else:
        bird_group_row = None

    common_name = f"{species_filtered_df.iloc[0]['CommonName'].capitalize()}"
    order = f"Order: {species_filtered_df.iloc[0]['Order'].capitalize()}"

    report_group = f"Group: {bird_group_row.get('Report Group', 'Unknown')}" if bird_group_row is not None else ""
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

    
    
    grouped_df = species_filtered_df.groupby("YearCollected", as_index=False)[["ObservationCount"]].sum()

    if len(grouped_df) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(grouped_df["YearCollected"], grouped_df["ObservationCount"])
        grouped_df["Regression"] = grouped_df["YearCollected"] * slope + intercept
    else:
        slope, intercept, r_value, p_value, std_err = None, None, None, None, None
        grouped_df["Regression"] = None 

    bar_fig = go.Figure()

    bar_fig.add_trace(
        go.Bar(
            x=grouped_df["YearCollected"],
            y=grouped_df["ObservationCount"],
            name="Observations",
            marker_color="lightskyblue",
            text=grouped_df["ObservationCount"],
            textposition="outside" 
        )
    )

    if grouped_df["Regression"].notnull().all():
        bar_fig.add_trace(
            go.Scatter(
                x=grouped_df["YearCollected"],
                y=grouped_df["Regression"],
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


if __name__ == '__main__':
    app.run(debug=True)
