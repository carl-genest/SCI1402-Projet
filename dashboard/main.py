from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from scipy.stats import linregress
import requests
from urllib.parse import quote

# Import Data
df = pd.read_csv('C:/Users/Carl/Documents/GitHub/SCI1402-Projet/datasets/bbs50-can/bbs50-can_naturecounts_filtered_data.txt', header=0, sep='\t')

bird_groups_df = pd.read_csv('C:/Users/Carl/Documents/GitHub/SCI1402-Projet/datasets/Birds_by_Report_Group.csv', header=0, sep=',')

scientific_names_list = sorted(df["ScientificName"].unique())
default_selected_bird_name = "Poecile atricapillus"
nuthatch_api_key = "accfdbb6-92ec-4836-948e-54500763bd96"

# Initialize the Dash app
app = Dash(external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Species Observation Dashboard",
                    className="text-center my-4"
                ),
                width=12
            )
        ),

        # Dropdown Menu Header
        dbc.Row(
            dbc.Col(
                html.H5(
                    "Select a species",
                    className="text-center mb-2"
                ),
                width=12
            )
        ),
        
        # Dropdown Menu
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="species-dropdown",
                    options=[{"label": name, "value": name} for name in scientific_names_list],
                    value=default_selected_bird_name,  # Default value
                    placeholder="Select a species",
                    className="text-dark"
                ),
                width=6  # Adjust column width
            ),
            className="mb-4 justify-content-center"
        ),

        # Bird Details Card
        dbc.Row(
            dbc.Col(
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
                width=6
            ),
            className="justify-content-center"
        ),
        
        # Bird Images
        dbc.Row(id="bird-images", className="mb-4"),
        
        # Graph for Observations
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id="observation-graph",
                    style={
                        'width': '80%', 
                        'margin': '0 auto'
                    }
                ),
                width=12
            ),
            className="mb-4",
            style={'backgroundColor': 'black'}
        ),
        
        # Graph for Map Visualization
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id="species-map",
                    style={"height": "700px", "width": "100%"}
                ),
                width=12
            ),
            style={'backgroundColor': 'white'}
        ),
    ],
    fluid=True
)

def get_basic_bird_data(sci_name, api_key):
    base_url = "https://nuthatch.lastelm.software/v2/birds"
    
    # Encode the scientific name for a URL
    encoded_name = quote(sci_name)
    
    # Construct the full URL with the encoded name
    url = f"{base_url}?sciName={encoded_name}&operator=AND"

    # Set up headers with API key
    headers = {
        "API-Key": api_key,
        "accept": "application/json"
    }

    # Make the request
    response = requests.get(url, headers=headers)

    # Check for success and return JSON data
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Callback to update the graph based on dropdown selection
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
        Output("species-map", "figure")
    ],
    Input("species-dropdown", "value")
)
def update_graph(selected_species):
    species_first_row = df[df['ScientificName'] == selected_species].iloc[0]
    bird_group_row = bird_groups_df[bird_groups_df["Scientific name"] == selected_species].iloc[0]

    common_name = f"{species_first_row['CommonName'].capitalize()}"
    report_group = f"Group: {bird_group_row['Report Group']}"
    order = f"Order: {species_first_row['Order'].capitalize()}"
    report_trend = f"2024 Report Trend: {bird_group_row['Trend']}"
    report_goal = f"2024 Report Goal: {bird_group_row['Goal']}"

    basic_bird_data = get_basic_bird_data(selected_species, nuthatch_api_key)

    # Extract and print the bird's common name safely
    if basic_bird_data and "entities" in basic_bird_data and isinstance(basic_bird_data["entities"], list) and len(basic_bird_data["entities"]) > 0:
        first_entity = basic_bird_data["entities"][0]  # Get the first entity

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
        
        # Bird images
        images = first_entity.get("images", [])
        image_cards = [
            dbc.Col(dbc.Card(dbc.CardImg(src=img, top=True), className="mb-3"), width=4)
            for img in images[:1]  # Show up to 1 image
        ]
        image_cards_list = dbc.Row(image_cards, className="justify-content-center")
    else: 
        family = ""
        length = ""
        wingspan = ""
        image_cards_list = ""

    # Filter data for the selected species
    species_filtered_df = df[df["ScientificName"] == selected_species]

    # Get the most recent year in YearCollected for the selected species
    min_year = species_filtered_df["YearCollected"].min()
    max_year = species_filtered_df["YearCollected"].max()

    # Filter further to keep only entries from the latest year
    year_filtered_df = species_filtered_df[species_filtered_df["YearCollected"] == max_year]
    
    # Group data by YearCollected and sum only the ObservationCount column
    grouped_df = species_filtered_df.groupby("YearCollected", as_index=False)[["ObservationCount"]].sum()

    # Linear regression for ObservationCount over YearCollected
    if len(grouped_df) > 1:  # Ensure there are enough data points
        slope, intercept, r_value, p_value, std_err = linregress(grouped_df["YearCollected"], grouped_df["ObservationCount"])
        grouped_df["Regression"] = grouped_df["YearCollected"] * slope + intercept
    else:
        slope, intercept, r_value, p_value, std_err = None, None, None, None, None
        grouped_df["Regression"] = None  # No regression line if insufficient data

    # Create the figure
    bar_fig = go.Figure()

    # Add bar trace
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

    # Add regression line trace if applicable
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
        title=f"Observations and Trend for {selected_species}",
        xaxis_title="Year",
        yaxis_title="Total Observations",
        template="plotly_dark",

        # Limit x-axis to only show the year with data
        xaxis=dict(
            tickformat=".0f"
        ),

        # Add annotation for slope and r-value
        annotations=[
            dict(
                x=1.14,  # Places annotation slightly outside the right border
                y=0.5,  # Middle of the chart
                xref="paper",  # Relative to the figure, not data points
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

    # Create the map visualization
    map_fig = px.scatter_geo(
        year_filtered_df,
        lat="DecimalLatitude",
        lon="DecimalLongitude",
        color="ObservationCount",
        size="ObservationCount",
        hover_name="ScientificName",
        hover_data=["YearCollected", "ObservationCount"],
        title=f"Observation Locations for {selected_species} in Canada in {max_year}",
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

    return common_name, report_group, family, order, length, wingspan, report_trend, report_goal, image_cards_list, bar_fig, map_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
