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


# Fonction pour lire les fichiers de données en morceaux et les concaténer en un DataFrame
def read_chunks_into_dataframe(output_dir="data/data_split_files"):
    chunk_files = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith("data_split_file_")])

    df_list = [pd.read_csv(chunk_files[0], sep="\t", header=0)]
    
    for chunk in chunk_files[1:]:
        df_list.append(pd.read_csv(chunk, sep="\t", header=None, names=df_list[0].columns))
    
    final_df = pd.concat(df_list, ignore_index=True)
    
    return final_df

# Fonction pour récupérer les données d'un oiseau à partir de l'API Nuthatch
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

# Fonction pour générer les quantiles théoriques pour le QQ-plot
def qq_plot_data(data):
    quantiles = np.linspace(0, 1, len(data))
    theoretical_quantiles = norm.ppf(quantiles, np.mean(data), np.std(data))
    sorted_data = np.sort(data)
    return theoretical_quantiles, sorted_data

# Filtrer les valeurs finies pour éviter les problèmes de calcul
def filter_finite_values(q_theoretical, q_sample):
    mask = np.isfinite(q_theoretical) & np.isfinite(q_sample)
    return q_theoretical[mask], q_sample[mask]

# Chargement des données dans un DataFrame
df = read_chunks_into_dataframe()

# Chargement des fichiers contenant des informations sur les groupes d'oiseaux
bird_groups_df = pd.read_csv('data/Birds_by_Report_Group.csv', header=0, sep=',')
bird_groups_unique_df = pd.read_csv('data/Birds_by_Report_Group_Unique.csv', header=0, sep=',')

# Création des listes pour les menus déroulants
scientific_names_list = sorted(df["ScientificName"].unique())
init_bird_name = "Poecile atricapillus"
year_list = sorted(df["YearCollected"].unique(), reverse=True)
init_year = year_list[0]
bird_groups_list = sorted(bird_groups_unique_df["Report Group"].unique())
init_bird_group = "Grassland Bird"

# Clé API pour accéder aux données des oiseaux
nuthatch_api_key = "accfdbb6-92ec-4836-948e-54500763bd96"

# Création de l'application Dash avec un thème Bootstrap
app = Dash(external_stylesheets=[dbc.themes.DARKLY])

# Définition de la mise en page du tableau de bord
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
        
        # Sélection d'une espèce d'oiseau
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="species-dropdown",
                    options=[{"label": name, "value": name} for name in scientific_names_list],
                    value=init_bird_name,
                    placeholder="Select a species",
                    className="text-dark"
                ),
                width=6
            ),
            className="mb-4 justify-content-center"
        ),

        # Affichage des informations sur l'oiseau sélectionné
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
        
        # Affichage de l'image de l'oiseau sélectionné
        dcc.Loading(
            dbc.Row(id="bird-images", className="mb-4"),
            id="loading-bird-images"
        ),
        
        # Affichage du graphique des observations par année pour l'espèce sélectionnée
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

        # Sélection de l'année pour filter les observations représentées sur la carte
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="year-dropdown",
                    options=[{"label": year, "value": year} for year in year_list],
                    value=init_year,
                    placeholder="Select a year",
                    className="text-dark"
                ), 
                width=3
            ),
            className="justify-content-center",
            style={'backgroundColor': 'white'}
        ),

        # Affichage de la carte des observations d'une année sélectionnée pour l'espèce sélectionnée
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
        
        # Sélection d'un groupe d'espèces 
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id="bird-group-dropdown",
                    options=[{"label": group, "value": group} for group in bird_groups_list],
                    value=init_bird_group,
                    placeholder="Select a bird group",
                    className="text-dark"
                ),
                width=5
            ),
            className="mb-4 justify-content-center"
        ),

        # Affichage des statistiques produites sur les pentes de tendance démographique pour le groupe d'espèces sélectionné
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

        # Affichage des résultats du test d'hypothèse selon le groupe d'espèces sélectionné
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
                # Affichage du graphique quartile-quartile pour évaluer la normalité de la distribution des pentes du groupe d'espèces sélectionné
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

                # Affichage du diagramme de boîtes pour évaluer la répartition des pentes du groupe d'espèces sélectionné comparativement au reste des pentes
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


# Callback pour mettre à jour les informations affichées sur l'oiseau sélectionné
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
    # Filtrage des données pour l'espèce sélectionnée
    species_filtered_df = df[df["ScientificName"] == selected_species]
    
    # Obtention des années uniques de collecte des observations, triées en ordre décroissant
    year_options = sorted(species_filtered_df["YearCollected"].unique(), reverse=True)
    
    # Filtrage des groupes d'oiseaux associés à l'espèce sélectionnée
    bird_groups_for_species = bird_groups_df[bird_groups_df["Scientific name"] == selected_species]
    
    # Récupération des informations des groupes
    if not bird_groups_for_species.empty:
        report_groups_string = ', '.join(bird_groups_for_species['Report Group'].astype(str))
        bird_group_row = bird_groups_for_species.iloc[0]
    else:
        report_groups_string = None
        bird_group_row = None

    # Récupération des informations générales sur l'espèce
    common_name = f"{species_filtered_df.iloc[0]['CommonName'].capitalize()}"
    order = f"Order: {species_filtered_df.iloc[0]['Order'].capitalize()}"

    # Récupération des informations sur le groupe et la tendance du rapport 2024 si disponibles
    report_group = f"Group: {report_groups_string}" if report_groups_string is not None else ""
    report_trend = f"2024 Report Trend: {bird_group_row.get('Trend', 'Unknown')}" if bird_group_row is not None else ""
    report_goal = f"2024 Report Goal: {bird_group_row.get('Goal', 'Unknown')}" if bird_group_row is not None else ""

    # Appel à une API externe pour récupérer des données complémentaires sur l'espèce
    api_bird_data = get_api_bird_data(selected_species, nuthatch_api_key)

    # Vérification de la validité des données renvoyées par l'API
    if api_bird_data and "entities" in api_bird_data and isinstance(api_bird_data["entities"], list) and len(api_bird_data["entities"]) > 0:
        first_entity = api_bird_data["entities"][0]  

        # Récupération des informations taxonomiques et biologiques
        family = f"Family: {first_entity.get('family')}" if first_entity.get("family") else ""
        length_min = first_entity.get('lengthMin')
        length_max = first_entity.get('lengthMax')
        wingspan_min = first_entity.get("wingspanMin")
        wingspan_max = first_entity.get("wingspanMax")

        # Formatage des informations sur la taille et l'envergure
        if length_min and length_max:
            length = f"Length: {length_min} - {length_max} cm"
        else:
            length = ""

        if wingspan_min and wingspan_max:
            wingspan = f"Wingspan: {wingspan_min} - {wingspan_max} cm"
        else:
            wingspan = ""
        
        # Récupération de l'image associée à l'espèce et affichage sous forme de carte
        images = first_entity.get("images", [])
        image_cards = [
            dbc.Col(dbc.Card(dbc.CardImg(src=img, top=True), className="mb-3"), width=4)
            for img in images[:1]
        ]
        image_cards_list = dbc.Row(image_cards, className="justify-content-center")
    else: 
        # Si aucune donnée valide n'est retournée par l'API, on retourne des valeurs vides
        family = ""
        length = ""
        wingspan = ""
        image_cards_list = ""

    # Regroupement des observations par année et calcul du total des observations
    summed_df = species_filtered_df.groupby("YearCollected", as_index=False)[["ObservationCount"]].sum()

    # Vérification que les données sont suffisantes pour effectuer une régression linéaire
    if len(summed_df) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(summed_df["YearCollected"], summed_df["ObservationCount"])
        summed_df["Regression"] = summed_df["YearCollected"] * slope + intercept
    else:
        # Si une seule année est disponible, impossible de faire une régression
        slope, intercept, r_value, p_value, std_err = None, None, None, None, None
        summed_df["Regression"] = None 

    # Création du graphique à barres pour afficher les observations
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

    # Ajout de la régression linéaire si elle a été calculée
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

    # Mise en forme du graphique avec titre et annotations
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

    # Retourne les informations nécessaires à l'affichage dans l'interface utilisateur
    return common_name, report_group, family, order, length, wingspan, report_trend, report_goal, image_cards_list, bar_fig, year_options


# Callback pour mettre à jour la valeur du menu déroulant selon les nouvelles options produites par le callback précédent
@callback(
    Output('year-dropdown', 'value'),
    Input('year-dropdown', 'options'))
def update_selected_year(options):
    return options[0]


# Callback pour mettre à jour la carte selon la valeur du callback précédent ou selon une sélection manuelle de l'année
@callback(
    Output("species-map", "figure"),
    [
        Input("species-dropdown", "value"),
        Input("year-dropdown", "value")
    ]
)
def update_map(selected_species, selected_year):
    # Filtrage des données pour ne conserver que les observations de l'espèce sélectionnée pour l'année donnée
    year_filtered_df = df[(df["ScientificName"] == selected_species) & (df["YearCollected"] == selected_year)]

    # Création d'une carte interactive des observations avec Plotly
    map_fig = px.scatter_geo(
        year_filtered_df,  # Données filtrées
        lat="DecimalLatitude",  # Coordonnées latitude
        lon="DecimalLongitude",  # Coordonnées longitude
        color="ObservationCount",  # Coloration des points en fonction du nombre d'observations
        size="ObservationCount",  # Taille des points proportionnelle au nombre d'observations
        hover_name="ScientificName",  # Nom affiché au survol
        hover_data=["YearCollected", "ObservationCount"],  # Informations supplémentaires affichées au survol
        title=f"Observation Locations for {selected_species} in Canada in {selected_year}",  # Titre de la carte
        projection="natural earth",  # Projection cartographique
        scope="north america",  # Affichage limité à l'Amérique du Nord
        template="plotly",  # Style de la carte
    )

    # Personnalisation de l'affichage de la carte
    map_fig.update_geos(
        showcoastlines=True,  # Affichage des côtes
        coastlinecolor="black",  # Couleur des côtes
        projection_scale=1.75,  # Échelle de la projection (zoom)
        center={"lat": 58.0000, "lon": -95.0000}  # Centrage de la carte sur le Canada
    )

    # Mise en forme du titre pour qu'il soit centré
    map_fig.update_layout(
        title={
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top"
        }
    )

    # Retourne la carte interactive
    return map_fig


# Callback pour mettre à jour les affichages liés à la sélection du groupe d'espèces
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
    # Fusion des données principales avec les groupes d'oiseaux pour ajouter la colonne "Report Group"
    merged_df = df.merge(bird_groups_unique_df[['Scientific name', 'Report Group']], 
                    left_on='ScientificName', 
                    right_on='Scientific name', 
                    how='left')

    # Suppression des lignes où "Report Group" est manquant
    filtered_df = merged_df.dropna(subset=['Report Group'])

    # Regroupement des données par groupe d'oiseaux, espèce et année, en faisant la somme des observations
    summed_df = filtered_df.groupby(["Report Group", "ScientificName", "YearCollected"], as_index=False).agg({"ObservationCount": "sum"})

    # Calcul de la pente de la régression linéaire pour chaque espèce dans chaque groupe
    slopes_by_report_group = {}
    for group, sub_df in summed_df.groupby(["Report Group", "ScientificName"]):
        if len(sub_df) >= 2:
            slope, intercept, r_value, p_value, std_err = linregress(sub_df["YearCollected"], sub_df["ObservationCount"])
            if group[0] not in slopes_by_report_group:
                    slopes_by_report_group[group[0]] = []
            slopes_by_report_group[group[0]].append(slope)

    # Extraction des pentes pour le groupe sélectionné et les autres groupes
    group_slopes = slopes_by_report_group.get(selected_bird_group, [])
    nongroup_slopes = [slope for group, slopes in slopes_by_report_group.items() if group != selected_bird_group for slope in slopes]

    # Suppression des valeurs aberrantes en utilisant le score Z
    z_scores = zscore(group_slopes)
    non_outliers_indices = np.abs(z_scores) <= 3
    group_slopes = np.array(group_slopes)[non_outliers_indices]

    z_scores = zscore(nongroup_slopes)
    non_outliers_indices = np.abs(z_scores) <= 3
    nongroup_slopes = np.array(nongroup_slopes)[non_outliers_indices]

    # Calcul des statistiques pour le groupe sélectionné
    group_mean_val = np.mean(group_slopes)
    group_std_val = np.std(group_slopes)
    group_len = f"Group size : {len(group_slopes)}"
    group_mean = f"Mean: {group_mean_val:.4f}"
    group_std = f"Std: {group_std_val:.4f}"
    group_skew = f"Skew: {skew(group_slopes):.4f}"
    group_kurtosis = f"Kurtosis: {kurtosis(group_slopes):.4f}"

    # Calcul des statistiques pour les autres groupes
    nongroup_mean_val = np.mean(nongroup_slopes)
    nongroup_std_val = np.std(nongroup_slopes)
    nongroup_len = f"Remaining group size : {len(nongroup_slopes)}"
    nongroup_mean = f"Mean: {nongroup_mean_val:.4f}"
    nongroup_std = f"Std: {nongroup_std_val:.4f}"
    nongroup_skew = f"Skew: {skew(nongroup_slopes):.4f}"
    nongroup_kurtosis = f"Kurtosis: {kurtosis(nongroup_slopes):.4f}"

    # Test de normalité de Shapiro-Wilk
    sh_stat_group, sh_p_group = shapiro(group_slopes)
    sh_stat_nongroup, sh_p_nongroup = shapiro(nongroup_slopes)
    group_norm_pval = f"Normality test p-value: {sh_p_group}"
    nongroup_norm_pval = f"Normality test p-value: {sh_p_nongroup}"

    # Choix du test statistique approprié en fonction de la normalité des distributions
    if sh_p_group > 0.05 and sh_p_nongroup > 0.05: 
        # Si les deux distributions sont normales, le test Bartlett s'applique pour évaluer la variance
        norm_interpretation = "Both distributions are normal; Bartlett's test apply for variance"
        stat_bartlett, p_bartlett = bartlett(group_slopes, nongroup_slopes)
        variance_test_pval = f"Bartlett's Test: p-value = {p_bartlett:.10f}"

        if p_bartlett < 0.05: 
            # Si les variances sont significativement différentes, on applique le test de Welch
            variance_interpretation = "Variances are significantly different; Welch's t-test apply"
            t_stat, p_value = ttest_ind(group_slopes, nongroup_slopes, equal_var=False)  # Welch's t-test
            test_used = "Welch’s t-test"
        else:
            # Si les variances sont similaires, on applique le t-test de Student
            variance_interpretation = "Variances are similar; Student's t-test apply"
            t_stat, p_value = ttest_ind(group_slopes, nongroup_slopes, equal_var=True)   # Student’s t-test 
            test_used = "Standard t-test"
        
    else: 
        # Si les deux distributions ne sont pas normales, on applique le test de Mann-Whitney
        norm_interpretation = "At least one group is not normally distributed; Mann-Whitney U test apply"
        variance_test_pval = ""
        variance_interpretation = ""

        t_stat, p_value = mannwhitneyu(group_slopes, nongroup_slopes, alternative='two-sided')  
        test_used = "Mann-Whitney U test"

    # Définition des hypothèses et affichage des résultats
    null_hypo_def = f"H0: Differences between the {selected_bird_group} group slopes and the remaining group slopes are not significant"
    alt_hypo_def = f"H1: There is significant difference between the {selected_bird_group} group slopes and the remaining group slopes"
    null_hypo_test_name = f"{test_used} results"
    null_hypo_test_t_stat = f"T-statistic: {t_stat}"
    null_hypo_test_p_val = f"P-value: {p_value}"

    # Interprétation du test statistique
    if p_value < 1e-10:
        null_hypo_test_interpretation = "Test does not return a consistent p-value; This could be due to outliers, skewness, or distribution shape"
    elif p_value < 0.05:
        null_hypo_test_interpretation = f"H0 rejected; Significant difference found between {selected_bird_group} trends and remaining group."
    else:
        null_hypo_test_interpretation = "H0 not rejected; No significant difference found in trends."

    # Création des Q-Q plots pour les deux groupes
    qqfig = go.Figure()

    if len(group_slopes) > 1:
        # Calcul des quantiles théoriques et des quantiles de l'échantillon pour le groupe sélectionné
        q_theoretical, q_sample = qq_plot_data(group_slopes)
        # Filtrage des valeurs infinies
        q_theoretical, q_sample = filter_finite_values(q_theoretical, q_sample) 
        # Ajout d'un nuage de points représentant le graphique Q-Q du groupe sélectionné
        qqfig.add_trace(go.Scatter(x=q_theoretical, y=q_sample, mode='markers', name=f'Q-Q {selected_bird_group}'))

    if len(nongroup_slopes) > 1:
        # Calcul des quantiles théoriques et des quantiles de l'échantillon pour les autres groupes
        q_theoretical_nongroup, q_sample_nongroup = qq_plot_data(nongroup_slopes)
        # Filtrage des valeurs infinies
        q_theoretical_nongroup, q_sample_nongroup = filter_finite_values(q_theoretical_nongroup, q_sample_nongroup)
        # Ajout d'un nuage de points représentant le graphique Q-Q des autres groupes
        qqfig.add_trace(go.Scatter(x=q_theoretical_nongroup, y=q_sample_nongroup, mode='markers', name='Q-Q Other Groups'))

    # Détermination des valeurs minimales et maximales pour l'axe des quantiles
    line_min = min(q_sample.min(), q_sample_nongroup.min(), q_theoretical.min(), q_theoretical_nongroup.min())
    line_max = max(q_sample.max(), q_sample_nongroup.max(), q_theoretical.max(), q_theoretical_nongroup.max())

    # Ajout d'une ligne de référence (diagonale y = x) pour comparer les quantiles
    qqfig.add_trace(go.Scatter(x=[line_min, line_max], y=[line_min, line_max], mode='lines', name='Reference Line', line=dict(color='black', width=2)))

    # Mise en page du graphique
    qqfig.update_layout(
        title='Q-Q Plot of Slopes',
        xaxis_title='Theoretical Quantiles', 
        yaxis_title='Sample Quantiles', 
        template='plotly_white',  # Thème du graphique
        # Ajustement de l'axe des x pour inclure une marge autour des valeurs extrêmes
        xaxis=dict(range=[min(q_theoretical.min(), q_theoretical_nongroup.min())-5, max(q_theoretical.max(), q_theoretical_nongroup.max())+5])
    )

    # Création d'un boxplot des tendances par groupe
    # Ajout des pentes à une liste
    box_plot_data = []
    if group_slopes.any():
        box_plot_data.extend([(selected_bird_group, slope) for slope in group_slopes])
    if nongroup_slopes.any():
        box_plot_data.extend([("Remaining Group", slope) for slope in nongroup_slopes])

    # Création d'un DataFrame à partir des données collectées
    df_slopes = pd.DataFrame(box_plot_data, columns=["Report Group", "Slope"])

    # Application d'une transformation logarithmique signée pour ajuster la distribution des valeurs
    # La transformation log1p (logarithme de (1 + valeur absolue)) est utilisée pour éviter les problèmes avec les valeurs nulles ou négatives
    df_slopes["Transformed Slope"] = np.sign(df_slopes["Slope"]) * np.log1p(abs(df_slopes["Slope"]))

    # Création d'un diagramme de boîtes pour visualiser la distribution des pentes transformées
    fig_box = px.box(
        df_slopes, 
        x="Report Group", 
        y="Transformed Slope", 
        points="all", # Affichage des points individuels en plus du boxplot
        title="Distribution of Population Trends (Slopes) by Report Group",
        labels={"Transformed Slope": "Log-Adjusted Trend Slope", "Report Group": "Bird Report Group"},
        template="plotly_white" # Utilisation d'un thème clair
    ) 

    # Retourne les variables utiles à l'affichage
    return group_len, group_mean, group_std, group_skew, group_kurtosis, group_norm_pval, nongroup_len, nongroup_mean, nongroup_std, nongroup_skew, nongroup_kurtosis, nongroup_norm_pval, norm_interpretation, variance_test_pval, variance_interpretation, null_hypo_def, alt_hypo_def, null_hypo_test_name, null_hypo_test_t_stat, null_hypo_test_p_val, null_hypo_test_interpretation, qqfig, fig_box


# Lancement de l'application si le fichier est exécuté directement
if __name__ == '__main__':
    app.run(debug=True)
