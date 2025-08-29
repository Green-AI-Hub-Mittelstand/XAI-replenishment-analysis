import dash
from dash import Dash, html, dcc, Output, Input, State
from dash.dependencies import ALL
import dash_bootstrap_components as dbc
import os
from pages import functions as fn

# Ensure the "pages" folder exists
PAGES_FOLDER = os.path.join(os.path.dirname(__file__), "pages")
if not os.path.exists(PAGES_FOLDER):
    raise FileNotFoundError(f"Error: The 'pages' folder does not exist at {PAGES_FOLDER}. Please create it.")

def create_navbar():
    return dbc.NavbarSimple(
        brand=html.Span(id={"type":"i18n", "key":"navbar.brand"}),
        color="darkgreen",
        dark=True,
        children=[
            dbc.NavItem(dbc.NavLink(html.Span(id={"type":"i18n", "key":"navbar.home"}), href="/")),
            dbc.NavItem(dbc.NavLink(html.Span(id={"type":"i18n", "key":"navbar.produkte"}), href="/product_usage")),
            dbc.NavItem(dbc.NavLink(html.Span(id={"type":"i18n", "key":"navbar.baureihen"}), href="/baureihe")),
            dbc.NavItem(dbc.NavLink(html.Span(id={"type":"i18n", "key":"navbar.planung"}), href="/planning")),
            dbc.NavItem(dbc.NavLink(html.Span(id={"type":"i18n", "key":"navbar.simulation"}), href="/simulation")),
            dbc.NavItem(dbc.NavLink(html.Span(id={"type": "i18n", "key": "navbar.model_testing"}), href="/article")),
            dbc.NavItem(dbc.NavLink(html.Span(id={"type":"i18n", "key":"navbar.settings"}), href="/settings")),
            dbc.NavItem(dbc.NavLink(html.Span(id={"type":"i18n", "key":"navbar.others"}), href="/others")),
        ]
    )

#  Initialize Dash App with Multi-Page Support
app = Dash(__name__, use_pages=True, external_stylesheets=[dbc.themes.BOOTSTRAP,"assets/article_page_style.css"],suppress_callback_exceptions=True)
server = app.server
# Storage for the de-anonymized button in the settings page
de_anonymized_store = dcc.Store(id='de-anonymized-store', data = {'de-anonymized': False})
language_store = dcc.Store(id='language-store', data={'language': 'de'}, storage_type='session')

# TBD how do we know that home.py will be shown here?
app.layout = dbc.Container([
    create_navbar(),
    html.Hr(),
    dash.page_container,
    de_anonymized_store,
    language_store,
], fluid=True)

@app.callback(
    Output('language-store', 'data'),
    Input('language-Switch', 'value'),
)

def update_language_switch(is_en):
    """
    Update the language store based on the switch value.
    """
    return "en" if is_en else "de"

@app.callback(
    Output({"type": "i18n", "key": ALL}, 'children'),
    Input('language-store', 'data'),
)

def update_translations(language):
    out = []
    fn.change_language(language)
    fn.translate_column_explanations()
    for out_id in dash.callback_context.outputs_list:
        key = out_id['id']['key']
        translated = fn.translate(language, key)
        out.append(translated)
    return out

if __name__ == '__main__':
    app.run(debug=True)
