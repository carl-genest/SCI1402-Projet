from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
from scipy.stats import linregress

# Import Data
df = pd.read_csv('C:/Users/Carl/Documents/GitHub/SCI1402-Projet/datasets/bbs50-can/bbs50-can_naturecounts_data.txt', header=0, sep='\t')

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
        
        # Dropdown Menu
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="species-dropdown",
                    options=[{"label": name, "value": name} for name in sorted(df["ScientificName"].unique())],
                    value=df["ScientificName"].unique()[0],  # Default value
                    placeholder="Select a species",
                    className="text-dark"
                ),
                width=6  # Adjust column width
            ),
            className="mb-4"
        ),
        
        # Graph for Observations
        dbc.Row(
            dbc.Col(
                dcc.Graph(id="observation-graph"),
                width=12
            ),
            className="mb-4"
        ),
        
        # Graph for Map Visualization
        dbc.Row(
            dbc.Col(
                dcc.Graph(id="species-map"),
                width=12
            )
        ),
    ],
    fluid=True
)

# Callback to update the graph based on dropdown selection
@callback(
    [Output("observation-graph", "figure"),
     Output("species-map", "figure")],
    Input("species-dropdown", "value")
)
def update_graph(selected_species):
    # Filter data for the selected species
    filtered_df = df[df["ScientificName"] == selected_species]
    
    # Group data by YearCollected and sum only the ObservationCount column
    grouped_df = filtered_df.groupby("YearCollected", as_index=False)[["ObservationCount"]].sum()

    # Linear regression for ObservationCount over YearCollected
    if len(grouped_df) > 1:  # Ensure there are enough data points
        slope, intercept, r_value, p_value, std_err = linregress(grouped_df["YearCollected"], grouped_df["ObservationCount"])
        grouped_df["Regression"] = grouped_df["YearCollected"] * slope + intercept
    else:
        grouped_df["Regression"] = None  # No regression line if insufficient data

    # Create the figure
    bar_fig = go.Figure()

    # Add bar trace
    bar_fig.add_trace(
        go.Bar(
            x=grouped_df["YearCollected"],
            y=grouped_df["ObservationCount"],
            name="Observations",
            marker_color="lightskyblue"
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
    )

    # Create the map visualization
    map_fig = px.scatter_geo(
        filtered_df,
        lat="DecimalLatitude",
        lon="DecimalLongitude",
        color="ObservationCount",
        size="ObservationCount",
        hover_name="ScientificName",
        hover_data=["YearCollected", "ObservationCount"],
        title=f"Observation Locations for {selected_species} in Canada",
        projection="natural earth",
        scope="north america",
        template="plotly",
    )
    map_fig.update_geos(showcoastlines=True, coastlinecolor="black", projection_scale=5)

    return bar_fig, map_fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
