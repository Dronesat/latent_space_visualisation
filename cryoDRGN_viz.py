'''
CryoDRGN Visualization Module
Import from CryoDRGN Jypiter Notebook
'''
import pandas as pd
import numpy as np
import pickle
import subprocess
import os, sys

from cryodrgn import analysis
from cryodrgn import utils
from cryodrgn import dataset
from cryodrgn import ctf
import cryodrgn.config
                
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py
from ipywidgets import interact, interactive, HBox, VBox
from scipy.spatial.transform import Rotation as RR
#py.init_notebook_mode()
from IPython.display import FileLink, FileLinks
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html

# Specify the workdir and the epoch number (0-based index) to analyze
WORKDIR = os.path.expandvars('${HOME}/myproject/cryodrgn_dataset/CryoDRGN/job002/train_128')
EPOCH = 24

print(os.path.abspath(WORKDIR))


'''Load Results'''  
# Load z
with open(f'{WORKDIR}/z.{EPOCH}.pkl','rb') as f:
    z = pickle.load(f)
    z_logvar = pickle.load(f)

# Load UMAP
umap = utils.load_pkl(f'{WORKDIR}/analyze.{EPOCH}/umap.pkl')
# or run UMAP
#umap = analysis.run_umap(z)

# Load kmeans
KMEANS = 20
kmeans_labels = utils.load_pkl(f'{WORKDIR}/analyze.{EPOCH}/kmeans{KMEANS}/labels.pkl')
kmeans_centers = np.loadtxt(f'{WORKDIR}/analyze.{EPOCH}/kmeans{KMEANS}/centers.txt')
# Or re-run kmeans with the desired number of classes
#kmeans_labels, kmeans_centers = analysis.cluster_kmeans(z, 20)

# Get index for on-data cluster center
kmeans_centers, centers_ind = analysis.get_nearest_point(z, kmeans_centers)

''' Load Dataset '''
# Load configuration file
config = cryodrgn.config.load(f'{WORKDIR}/config.yaml')
#print(config)

# Load poses
if config['dataset_args']['do_pose_sgd']:
    pose_pkl = f'{WORKDIR}/pose.{EPOCH}.pkl'
    with open(pose_pkl,'rb') as f:
        rot, trans = pickle.load(f)
else:
    pose_pkl = config['dataset_args']['poses']
    rot, trans = utils.load_pkl(pose_pkl)

# Convert rotation matrices to euler angles
euler = RR.from_matrix(rot).as_euler('zyz', degrees=True)

# Load index filter
ind_orig = config['dataset_args']['ind']
if ind_orig is not None:
    ind_orig = utils.load_pkl(ind_orig)
    if len(rot) > len(ind_orig):
        print(f'Filtering poses from {len(rot)} to {len(ind_orig)}')
        rot = rot[ind_orig]
        trans = trans[ind_orig]
        euler = euler[ind_orig]

# load input particles, first time just to get total number of particles
particles = dataset.ImageDataset(
    config['dataset_args']['particles'], lazy=True, ind=ind_orig,
    datadir=config['dataset_args']['datadir']
)
N_orig = particles.src.orig_n

# Load CTF
ctf_params = utils.load_pkl(config['dataset_args']['ctf'])
if ind_orig is not None:
    print(f'Filtering ctf parameters from {len(ctf_params)} to {len(ind_orig)}')
    ctf_params = ctf_params[ind_orig]
ctf.print_ctf_params(ctf_params[0])

'''Training Loss'''
loss = analysis.parse_loss(f'{WORKDIR}/run.log')
import plotly.express as px

df_loss = pd.DataFrame({
    'Epoch': list(range(len(loss))),  # x-axis
    'Loss': loss                      # y-axis
})

fig_loss = go.Figure()
fig_loss.add_trace(
    go.Scatter(x=df_loss['Epoch'], y=df_loss['Loss'], mode='lines', name='Loss')
)
fig_loss.update_layout(
    title="Training Loss",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    height=400, width=700
)


'''PCA of latent space with density profiles'''
pc, pca = analysis.run_pca(z)

df_pca = pd.DataFrame({
    'PC1': pc[:, 0],
    'PC2': pc[:, 1]
})

# Create scatter plot with marginal histograms
fig_pca = px.scatter(
    df_pca,
    x='PC1',
    y='PC2',
    title='PCA of latent space with density profiles',
    marginal_x='histogram',
    marginal_y='histogram',
    opacity=0.5,      
)

fig_pca.update_traces(marker=dict(size=2, color='blue'), selector=dict(mode='markers'))
fig_pca.update_layout(height=700, width=700)

'''PCA of latent space hex-binned'''
# Create subplots: 2 rows x 2 cols
fig_hex_marginal = make_subplots(
    rows=2, cols=2,
    column_widths=[0.8, 0.2],
    row_heights=[0.2, 0.8],
    specs=[[{"type": "histogram"}, None],
           [{"type": "histogram2d"}, {"type": "histogram"}]],
    horizontal_spacing=0.02,
    vertical_spacing=0.02
)

# Top marginal histogram (PC1)
fig_hex_marginal.add_trace(
    go.Histogram(
        x=df_pca['PC1'],
        nbinsx=50,
        marker=dict(color='blue'),
        showlegend=False
    ),
    row=1, col=1
)

# Right marginal histogram (PC2) 
fig_hex_marginal.add_trace(
    go.Histogram(
        y=df_pca['PC2'],
        nbinsy=50,
        marker=dict(color='blue'),
        showlegend=False
    ),
    row=2, col=2
)

# Hexbin (2D histogram) in the main plot
fig_hex_marginal.add_trace(
    go.Histogram2d(
        x=df_pca['PC1'],
        y=df_pca['PC2'],
        colorscale='Blues',
        colorbar=dict(title='Count'),
        nbinsx=100,
        nbinsy=100,
        showscale=True,
        opacity=1
    ),
    row=2, col=1
)

# Update layout
fig_hex_marginal.update_layout(
    title='PCA Hexbin with Marginal Histograms',
    height=700, width=700,
    bargap=0.05,
    showlegend=False
)
fig_hex_marginal.update_xaxes(title_text="PC1", row=2, col=1)
fig_hex_marginal.update_yaxes(title_text="PC2", row=2, col=1)
fig_hex_marginal.update_xaxes(showticklabels=False, row=1, col=1)
fig_hex_marginal.update_yaxes(showticklabels=False, row=2, col=2)

'''UMAP of latent space with density profiles'''
df_umap = pd.DataFrame({
    'UMAP1': umap[:, 0],
    'UMAP2': umap[:, 1]
})

fig_umap = px.scatter(
    df_umap,
    x='UMAP1',
    y='UMAP2',
    title='UMAP of latent space with density profiles',
    marginal_x='histogram',
    marginal_y='histogram',
    opacity=0.1,  # similar to alpha=.1 in sns
)
fig_umap.update_traces(marker=dict(size=2, color='green'), selector=dict(mode='markers'))
fig_umap.update_layout(height=700, width=700)

'''K-means centres on PCA of latent space'''
K = len(set(kmeans_labels))
c = pca.transform(kmeans_centers) # transform to view with PCs
cluster_colors = px.colors.qualitative.Plotly
#print(f'kmeans:{kmeans_labels}')
#print(f'Number of clusters: {K}')
#print(f'Cluster centers in PCA space: {c}')
# Create a scatter plot for each cluster
fig_kmeans_pca = go.Figure()

for k in range(K):
    mask = (kmeans_labels == k) #NumPy boolean array | mask = [False True ... False]
    print(f'mask = {mask}')
    fig_kmeans_pca.add_trace(
        go.Scatter(
            x=pc[mask, 0], #PC1 for cluster k
            y=pc[mask, 1], #PC2 for cluster k 
            mode='markers',
            marker=dict(size=4, color=cluster_colors[k % len(cluster_colors)], opacity=0.5),
            name=f'Cluster {k+1}',
            showlegend=True
        )
    )

# fig_kmeans_pca.add_trace(
#     go.Scatter(
#     x=pc[:,0],
#     y=pc[:,1],
#     mode='markers',
#     marker=dict(size=4, color='blue', opacity=0.5),
#     name='All Points',
#     showlegend=True
# )
# )

# with open("debug_output.txt", "w") as f:
#     print(f'pc:\n{pc}', file=f)
#     print(f'x={pc[:,0]}', file=f)
#     print(f'y={pc[:,1]}', file=f)

# Add cluster centers
fig_kmeans_pca.add_trace(
    go.Scatter(
        x=c[:, 0],
        y=c[:, 1],
        mode='markers+text',
        marker=dict(size=10, color='black', symbol='circle-dot'),
        text=[str(i+1) for i in range(K)],
        textposition='top center',
        name='Centers',
        showlegend=True
    )
)

fig_kmeans_pca.update_layout(
    title='K-means Centres on PCA of Latent Space',
    xaxis_title='PC1',
    yaxis_title='PC2',
    height=700, width=700
)

'''K-means centres on UMAP of latent space'''

# Create a scatter plot for each clutter
fig_kmeans_umap = go.Figure()

for k in range(K):
    mask = (kmeans_labels == k)
    fig_kmeans_umap.add_trace(
        go.Scatter(
            x=umap[mask,0],
            y=umap[mask,1],
            mode='markers',
            marker=dict(size=4, color=cluster_colors[k % len(cluster_colors)], opacity=0.5),
            name=f'Cluster {k+1}',
            showlegend=True
        )
    )

# Add cluster centers
# fig_kmeans_umap.add_trace(
#     go.Scatter(
#         x=centers_ind[:,0],
#         y=centers_ind[:,1],
#         mode='markers+text',
#         marker=dict(size=10, color='black', symbol='circle-dot'),
#         text=[str(i+1) for i in range(K)],
#         textposition='top center',
#         name='Centers',
#         showlegend=True
#     )
# )

fig_kmeans_umap.update_layout(
    title='K-means Centres on UMAP of Latent Space',
    xaxis_title='UMAP1',
    yaxis_title='UMAP2',
    height=700, width=700
)

# Dash App
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("CryoDRGN Visualization", style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(figure=fig_loss),
        dcc.Graph(figure=fig_pca),
        dcc.Graph(figure=fig_hex_marginal),
        dcc.Graph(figure=fig_umap),
        dcc.Graph(figure=fig_kmeans_pca),
        dcc.Graph(figure=fig_kmeans_umap),
    ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
])

if __name__ == '__main__':
    app.run(debug=True)