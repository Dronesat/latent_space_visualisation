import plotly.express as px
import plotly.graph_objects as go
from affinity_vega_loader import AffinityVegaLoadHTML
import pandas as pd

# Use VegaSpecExtractor to get the dataset
loader = AffinityVegaLoadHTML("plt_latent_embed_epoch_30_tsne.html")
dataset = loader.get_dataset("data-68149f6c5c3929511c2d24005b767d92")

df = pd.DataFrame(dataset)

#fig = px.scatter(
    #df,
    #x="emb-x",
    #y="emb-y",
    #color="id",
    #hover_data=['meta','filename']  # Add 'meta' to hover data       
    #hover_name="filename",   
    #custom_data=["meta"] 
#)
scatter_fig = go.FigureWidget(
    data=[go.Scatter(
        #df,
        x=df['emb-x'],
        y=df['emb-y'],
        #color=df['id'],
        #hover_data=['meta','filename'], 
        mode='markers',
        marker=dict(size=10, color='blue'),
        selected=dict(marker=dict(color='red')),
        unselected=dict(marker=dict(opacity=0.3)),
        #text='id',
        #customdata=df['meta']
    )]
)
#df = px.data.iris()

#fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
#                 size='petal_length', hover_data=['petal_width'])
fig.show()