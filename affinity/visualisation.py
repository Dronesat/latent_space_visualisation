from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import json
import pandas as pd

from affinity_vega_loader import AffinityVegaLoadHTML

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# Use VegaSpecExtractor to get the dataset
extractor = AffinityVegaLoadHTML("plt_latent_embed_epoch_30_tsne.html")
dataset = extractor.get_dataset("data-68149f6c5c3929511c2d24005b767d92")

df = pd.DataFrame(dataset)

fig = px.scatter(
    df,
    x="emb-x",
    y="emb-y",
    color="id",              
    hover_name="filename",   
    custom_data=["filename"] 
)

fig.update_layout(clickmode='event+select')

fig.update_traces(marker_size=12)

app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure=fig
    ),

    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data', style=styles['pre']),
        ], className='three columns'),

        html.Div([
            dcc.Markdown("""
                **Selection Data**

                Choose the lasso or rectangle tool in the graph's menu
                bar and then select points in the graph.

                Note that if `layout.clickmode = 'event+select'`, selection data also
                accumulates (or un-accumulates) selected data if you hold down the shift
                button while clicking.
            """),
            html.Pre(id='selected-data', style=styles['pre']),
        ], className='three columns')

    ])
])

@callback(
    Output('click-data', 'children'),
    Input('basic-interactions', 'clickData'))
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

@callback(
    Output('selected-data', 'children'),
    Input('basic-interactions', 'selectedData'))
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)

if __name__ == '__main__':
    app.run(debug=True)
