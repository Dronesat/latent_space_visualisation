from python_code.vega.affinity_vega_loader import VegaSpecExtractor  

extractor = VegaSpecExtractor("plt_latent_embed_epoch_30_tsne.html")
dataset = extractor.get_dataset("data-68149f6c5c3929511c2d24005b767d92")

for entry in dataset:
    print(entry["id"], entry["emb-x"], entry["emb-y"])

import plotly.graph_objects as go

import numpy as np
np.random.seed(1)

x = np.random.rand(100)
y = np.random.rand(100)

f = go.FigureWidget([go.Scatter(x=x, y=y, mode='markers')])

scatter = f.data[0]
colors = ['#a3a7e4'] * 100
scatter.marker.color = colors
scatter.marker.size = [10] * 100
f.layout.hovermode = 'closest'


# create our callback function
def update_point(trace, points, selector):
    c = list(scatter.marker.color)
    s = list(scatter.marker.size)
    for i in points.point_inds:
        c[i] = '#bae2be'
        s[i] = 20
        with f.batch_update():
            scatter.marker.color = c
            scatter.marker.size = s


scatter.on_click(update_point)

f.show()