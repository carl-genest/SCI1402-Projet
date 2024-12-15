from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd

# Import Data
df = pd.read_csv('C:/Users/Carl/Documents/GitHub/SCI1402-Projet/datasets/bbs50-can/bbs50-can_naturecounts_data.txt', header=0, sep='\t')

# Initialize the Dash app
app = Dash()

# App Layout
app.layout = [
    html.H1(children='Species Observation Dashboard', style={'textAlign':'center'}),
    
    # Dropdown Menu
    dcc.Dropdown(
        id="species-dropdown",
        options=[{"label": name, "value": name} for name in df["ScientificName"].unique()],
        value=df["ScientificName"].unique()[0],  # Default value
        placeholder="Select a species",
    ),

    # Graph
    dcc.Graph(id='observation-graph'),

    # Graph for Map Visualization
    dcc.Graph(id="species-map")
]

# Callback to update the graph based on dropdown selection
@callback(
    [Output("observation-graph", "figure"),
     Output("species-map", "figure")],
    Input('species-dropdown', 'value')
)
def update_graph(selected_species):
    # Filter data for the selected species
    filtered_df = df[df["ScientificName"] == selected_species]
    
    # Group data by YearCollected and sum only the ObservationCount column
    grouped_df = filtered_df.groupby("YearCollected", as_index=False)[["ObservationCount"]].sum()
    
    # Create the figure
    bar_fig  = px.bar(
        grouped_df,
        x="YearCollected",
        y="ObservationCount",
        title=f"Observations for {selected_species}",
        labels={"ObservationCount": "Total Observations", "YearCollected": "Year"},
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
