import os
import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
import yaml
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as RR
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import json
from bs4 import BeautifulSoup

import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html

from cryodrgn_libs import analysis
from cryodrgn_libs.dataset import ImageDataset
from affinity.affinity_vega_loader import AffinityVegaLoadHTML

class LatentSpace(ABC):
    """
    Abstract class for heterogeneous reconstruction algorithm
    """
    def __init__(self, workdir: str, epoch: int):
        self.workdir = workdir
        self.epoch = epoch
        self.latent_space_coords = None
        self.embedding2d = None
        self.df_latent_space_coords = None
        self.df_embeddings = None
        self.df_labels = None
        self.kmeans_labels = None
        self.gmm_labels = None
        self.nearest_neighbors_labels = None

    # Utility Functions 
    def load_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
  
    def load_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    @abstractmethod
    def load_build_df_latent_space_coords(self):
        """Load latent space coordinates from file"""
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def load_build_df_embeddings(self):
        """Load 2D embedding from file (UMAP, t-SNE, PCA, ...)"""
        raise NotImplementedError("Subclasses must implement this method.")

    # refractor from analysis.py    use sklearn nearest neighbors function instead
    def get_nearest_point(self, data: np.ndarray, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        For each point in 'queries', find the closest point in 'data'.
        Returns:
            nearest_points: The closest points in 'data' for each query 
            nearest_indices: The indices in 'data' of the closest points
        """
        distances = cdist(queries, data)
        nearest_indices = np.argmin(distances, axis=1)
        nearest_points = data[nearest_indices]
        return nearest_points, nearest_indices
    
    def run_pca_fit(self, z: np.ndarray) -> PCA:
        """
        Fit a PCA model to the data z and return the fitted PCA object
        """
        pca = PCA(z.shape[1])
        pca.fit(z)
        return pca
    
    def run_pca_transform(self, z: np.ndarray, pca: PCA) -> np.ndarray:
        """
        Transform the data z using a fitted PCA model
        """
        pc = pca.transform(z)
        return pc
    
    def run_umap_fit(self, z: np.ndarray, **kwargs):
        """
        Fit a UMAP model to the data z and return the fitted UMAP object
        """
        import umap
        reducer = umap.UMAP(**kwargs)
        reducer.fit(z)
        return reducer
    
    def run_umap_transform(self, z: np.ndarray, reducer):
        """
        Transform the data z using a fitted UMAP model 
        """
        z_embedded = reducer.transform(z)
        return z_embedded
    
    # Copy from analysis.py
    # Scilearn tsne doesn't have a separate fit/tranform functions
    def run_tsne(z: np.ndarray, n_components: int = 2, perplexity: float = 1000 ) -> np.ndarray:
        if len(z) > 10000:
            warnings.warn(
                f"WARNING: {len(z)} datapoints > 10000. This may take a while."
            )
        z_embedded = TSNE(n_components=n_components, perplexity=perplexity).fit_transform(z)
        return z_embedded
    
    # Copy from analysis.py
    def cluster_kmeans(self,z: np.ndarray, K: int, on_data: bool = True, reorder: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster z by K means clustering
        Returns cluster labels, cluster centers
        If reorder=True, reorders clusters according to agglomerative clustering of cluster centers
        """
        kmeans = KMeans(n_clusters=K, random_state=0, max_iter=10)
        labels = kmeans.fit_predict(z)
        centers = kmeans.cluster_centers_

        # use kmeans_predict instead of get_nearest_point()
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        centers_ind = None
        if on_data:
            centers, centers_ind = self.get_nearest_point(z, centers) # use sklearn nearest neighbors function instead
        
    def build_df_latent_space_coords(self):
        """
        Build a base dataframe for latent space coords 
        Checks that columns exist and Dataframe is not empty
        """
        n_dimensions = self.latent_space_coords.shape[1]
        columns = [f'lat{i+1}' for i in range (n_dimensions)]         
        self.df_latent_space_coords = pd.DataFrame(
            self.latent_space_coords,
            columns = columns
        )  # load straight awayy no shape
        lat_cols = [col for col in self.df_latent_space_coords.columns if col.startswith('lat') and col[3:].isdigit()]
        # Check columns exist
        if not lat_cols:
            raise ValueError(f"Missing latent columns (lat0,1,..)")
        # Check Dataframe is not empty
        if self.df_latent_space_coords.empty:
            raise ValueError('Latent Dataframe is empty (no rows)')
        # Check for msissing value
        if self.df_latent_space_coords[lat_cols].isnull().any().any():
            raise ValueError("Some rows have missing values")
    
    def build_df_embeddings(self, embedding_type: str = "umap"):
        """
        Build a base dataframe for embedding coords (PCA, TSNE, UMAP)
        Checks that columns exist and Dataframe is not empty
        """
        # Embedding n-D or Embedding 2D ??? for now n-D
        n_dimensions = self.embedding2d.shape[1]
        columns = [f'{embedding_type}{i+1}' for i in range(n_dimensions)]
        self.df_embeddings = pd.DataFrame(
            self.embedding2d,
            columns=columns
        )
        # Check column exist 
        if not all(col in self.df_embeddings.columns for col in columns):
            raise ValueError(f'Missing expected columns: {columns}')
        # Check dataframe is not empty
        if self.df_embeddings.empty:
            # Make a warning and try run_tsne or run_umap
            raise ValueError('Embedding dataframe is empty (no rows)')
        # Check for missing values
        if self.df_embeddings[columns].isnull().any().any():
            raise ValueError('Some rows have missing values')

    # Will move to specific algorithm (cryodrgn)
    def build_df_labels(self):
        """
        Build a dataframe for clustering labels (Kmeans, GMM, Nearest Neighbor)
        Each column is a clustering algorithm's labels
        """
        data = {}
        # Add algorithm labels (Kmeans, GMM, NN) if provided
        if self.kmeans_labels is not None:
            data['kmeans'] = self.kmeans_labels
        if self.gmm_labels is not None:
            data['gmm'] = self.gmm_labels
        if self.nearest_neighbors_labels is not None:
            data['nn'] = self.nearest_neighbors_labels

        # Check if labels are provided
        if not data:
            raise ValueError('No clustering labels provided')
        
        df_labels = pd.DataFrame(data)

        # Check if Dataframe is empty
        if df_labels.empty:
            raise ValueError('Labels Dataframe is empty')
        self.df_labels = df_labels

    def check_df_size(self, df):
        """
        Ensure the given dataframe has the same number of rows as df_latent_space_coords
        """ 
        if len(df) != len(self.df_latent_space_coords):
            raise ValueError('Dataframe does not have the same number of rows as df_latent_space_coords')
    
    # Don't see the use of this function bc df_latent already in init and df_embedding depends on df_latent
    def build_df_latent_embs(self):
        """
        Enforce that both df_latent_space_coords and df_embeddings are present 
        and have the same number of rows
        Returns a tuple (df_latent_space_coords, df_embeddings)
        """
        if self.df_latent_space_coords is None:
            raise ValueError('df_latent_space_coords does not exist')
        if self.df_embedding is None:
            raise ValueError('df_embeddings does not exist')
        self.check_df_size(self.df_embeddings)
        return self.df_latent_space_coords, self.df_embeddings 
    
    def plotly_graph(self, df: pd.DataFrame, title: str, x_axis: str, y_axis: str):
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=df[x_axis], y=df[y_axis], mode='lines')
        )
        figure.update_layout(
            title=title,
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            height=400, width=700
)
        return figure

class CryoDRGNLatent(LatentSpace):
    def __init__(self, workdir: str, epoch: int, kmeans_cluster_number: int, latent_space_coords_path: str, embedding2d_path: str):
        super().__init__(workdir, epoch)
        self.kmeans_cluster_number = kmeans_cluster_number
        self.kmeans_labels = None
        self.kmeans_centers = None
        self.rotation = None
        self.translation = None
        self.config = None
        self.latent_space_coords_path = latent_space_coords_path
        self.embedding2d_path = embedding2d_path
        self.training_loss = None # Training loss values (Numpy array)
        self.pc = None # Principle components (PCA-transformed latent coords)
        self.K = None # Number of K-means clusters (scalar)
        self.c = None # K-means cluster centers in PCA space
        self.df_kmeans_centers = None
        self.load_build_df_latent_space_coords()
        self.load_kmeans()
    
    def parse_training_loss_file(self, file: str) -> np.ndarray:
        """
        Parse the total loss values from a log file.
        """
        losses = []
        with open(file) as f:
            # Searches for lines containing "total loss =" and extractes the numeric value
            for line in f:
                if "total loss =" in line:
                    before, seperate, after = line.partition("total loss =")
                    value = after.split()[0].strip(";")
                    losses.append(float(value))
        return np.asarray(losses).astype(np.float32)
    
    def run_analysis(self):
        '''Training Loss'''
        self.training_loss = self.parse_training_loss_file(f'{self.workdir}/run.log')

        '''PCA of latent space with density profiles'''
        pca = self.run_pca_fit(self.latent_space_coords)
        self.pc = self.run_pca_transform(self.latent_space_coords, pca)

        '''K-means centres on PCA of latent space'''
        self.K = self.kmeans_centers.shape[0]
        self.c = pca.transform(self.kmeans_centers) # transform to view with PCs 
    
    def load_build_df_latent_space_coords(self):
        self.latent_space_coords = self.load_pickle(self.latent_space_coords_path) #numpy.ndarray
        self.build_df_latent_space_coords()

    def load_build_df_embeddings(self):
        self.embedding2d = self.load_pickle(self.embedding2d_path) #numpy.ndarray
        self.build_df_embeddings("umap")

        df_pca = pd.DataFrame({
            'pc1': self.pc[:, 0],
            'pc2': self.pc[:, 1]
        }) 
        self.check_df_size(df_pca)
        self.df_embeddings = pd.concat([self.df_embeddings, df_pca], axis = 1)

    def load_kmeans(self):
        kmeans_labels_path = f'{self.workdir}/analyze.{self.epoch}/kmeans{self.kmeans_cluster_number}/labels.pkl'
        kmeans_centers_path = f'{self.workdir}/analyze.{self.epoch}/kmeans{self.kmeans_cluster_number}/centers.txt'
        self.kmeans_labels = self.load_pickle(kmeans_labels_path) # in the df_labels
        self.kmeans_centers = np.loadtxt(kmeans_centers_path)

    def build_df_kmeans_centers(self):
        df_kmeans_centers_lat = pd.DataFrame(
            self.kmeans_centers,
            columns=[f'k_lat{i+1}' for i in range(self.kmeans_centers.shape[1])]
        )
        
        # Kmeans labels dataframe (this will come from predict function) predict similar to get_nearest_point
        df_kmeans_pc = pd.DataFrame(
            self.c, 
            columns=[f'k_pc{i+1}' for i in range(self.c.shape[1])]
        )
        self.df_kmeans_centers = pd.concat([df_kmeans_centers_lat, df_kmeans_pc], axis = 1)

    def build_df_labels(self):
        """
        Build a dataframe for clustering labels (Kmeans, GMM, Nearest Neighbor)
        Each column is a clustering algorithm's labels
        """
        data = {}
        # Add algorithm labels (Kmeans, GMM, NN) if provided
        if self.kmeans_labels is not None:
            data['kmeans'] = self.kmeans_labels
        if self.gmm_labels is not None:
            data['gmm'] = self.gmm_labels
        if self.nearest_neighbors_labels is not None:
            data['nn'] = self.nearest_neighbors_labels

        # Check if labels are provided
        if not data:
            raise ValueError('No clustering labels provided')
        
        df_labels = pd.DataFrame(data)

        # Check if Dataframe is empty
        if df_labels.empty:
            raise ValueError('Labels Dataframe is empty')
        self.df_labels = df_labels

    # Ask Joel if we need this???  -----------------
    def load_cryodrgn_config(self):
        """
        Load configuration from a file path or return the config object if already loaded.
        Supports .yaml/.yml (recommended) and /.pkl (deprecated)
        """
        config_path = f'{self.workdir}/config.yaml'
        if isinstance(config_path, str):
            extension = os.path.splitext(config_path)[-1].lower()
            if extension in {'.yml', '.yaml'}:
                self.config = self.load_yaml(config_path)
            elif extension == '.pkl':
                warnings.warn(
                    "Loading configuration from a .pkl file is deprecated. Please use .yaml instead.",
                    DeprecationWarning
                )
                self.config = self.load_pickle(config_path)
            else:
                raise RuntimeError(f"Unrecognized config extension: {extension}")
            
    def load_poses(self):
        """
        Load particle poses (rotations and translations) according to config
        """
        if self.config['dataset_args']['do_pose_sgd']:
            pose_pickle = f'{self.workdir}/pose.{self.epoch}.pkl'
            with open(pose_pickle, 'rb') as f:
                rotation, translation = pickle.load(f)
        else: 
            pose_pickle = self.config['dataset_args']['poses']
            rotation, translation = self.load_pickle(pose_pickle)

        self.rotation = rotation
        self.translation = translation
    
    def euler_angles(self):
        # Convert loaded rotation matrices to Euler angles
        return RR.from_matrix(self.rotation).as_euler('zyz', degrees=True)
    
    def summary(self): # Copy from Jupyter Notebook viz
        # load input particles, first time just to get total number of particles
        ind_orig = None  
        particles = ImageDataset(
            self.config['dataset_args']['particles'], lazy=True, ind=ind_orig,
            datadir = self.config['dataset_args']['datadir']
        )
        N_orig = particles.src.orig_n
        print("Original number of particles:", N_orig)

        # Load CTF
        ctf_params = self.load_pickle(self.config['dataset_args']['ctf'])
        if ind_orig is not None:
            print(f'Filtering ctf parameters from {len(ctf_params)} to {len(ind_orig)}')
            ctf_params = ctf_params[ind_orig]
        print("First CTF parameter:", ctf_params[0])
    # Ask ---------------------------------
    
class AffinityLatent(LatentSpace): 
    def __init__(self, workdir: str, epoch: int, affinity_html_path: str):
        super().__init__(workdir, epoch)
        self.epoch = epoch
        self.standard_deviation = None
        self.pose = None
        self.metadata = None
        self.affinity_html_path = affinity_html_path
    
    def load_affinity_dataset(self, html_file: str) -> dict:
        # read HTML file
        with open(html_file, "r", encoding="utf-8") as file:
            html = file.read()

        # parse HTML and extract the script containing json
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script")

        # loop script tags to find "var spec ="
        spec_script = None
        for script in scripts:
            if script.string and "var spec =" in script.string:
                spec_script = script.string
                break

        # extract JSON string part of "var spec = {...};"
        start_index = spec_script.find("var spec =") + len("var spec =")
        end_index = spec_script.find("var embedOpt")
        spec_json_str = spec_script[start_index:end_index].strip().rstrip(";")

        # parse into dictionary
        spec_dict = json.loads(spec_json_str)

        # get the first dataset
        datasets = spec_dict["datasets"]
        first_dataset_key = next(iter(datasets))
        dataset = datasets[first_dataset_key]

        return dataset
    
    def extract_coords(self, affinity_dataset: dict, keys: list) -> np.ndarray:
        """
        Helper function to extract numpy N-dimension arrays (np.ndarray) 
        from dataset (dictionary) given a list of keys
        """
        coords = []
        for entry in affinity_dataset:
            key_values = []
            for k in keys:
                key_values.append(entry[k])
            entry_array = np.array(key_values)
            coords.append(entry_array)
        return np.vstack(coords).astype(np.float32)

    def load_latent_space_coords(self):
        affinity_dataset = self.load_affinity_dataset(self.affinity_html_path)
        self.latent_space_coords = self.extract_coords(
            affinity_dataset, ["lat0", "lat1", "lat3", "lat4", "lat5", "lat6"])
        return self.latent_space_coords

    def load_embedding2d(self):
        affinity_dataset = self.load_affinity_dataset(self.affinity_html_path)
        self.embedding2d = self.extract_coords(
            affinity_dataset, ["emb-x", "emb-y"])
        return self.embedding2d
        
    def load_standard_deviation(self):
        affinity_dataset = self.load_affinity_dataset(self.affinity_html_path)
        self.standard_deviation = self.extract_coords(
            affinity_dataset, ["std-0", "std-1", "std-2", "std-3", "std-4", "std-5", "std-6"])
        return self.standard_deviation
    
    def load_pose(self):
        affinity_dataset = self.load_affinity_dataset(self.affinity_html_path)
        self.pose = self.extract_coords(affinity_dataset, ["pos0"])
        return self.pose

    def load_metadata(self):
        affinity_dataset = self.load_affinity_dataset(self.affinity_html_path)
        coords = []
        for entry in affinity_dataset:
            entry_array = np.array([
                entry["id"], entry["meta"], entry["avg"], entry["mode"], entry["image"]
            ])
            coords.append(entry_array)
        self.metadata = np.vstack(coords)
        return self.metadata    
    
    def build_dataframe(self):
        affinity_dataset = self.load_affinity_dataset(self.affinity_html_path)
        # Start with the base dataframe
        #df_base = super().build_df_latent_emb()

        df_affinity = pd.DataFrame(affinity_dataset)
        return df_affinity

def run_affinity():
    workdir = os.path.expandvars('${HOME}/myproject/cryodrgn_dataset/CryoDRGN/job002/python_code/affinity')
    affinity_job = AffinityLatent(
        epoch=30,
        workdir=workdir,
        affinity_html_path=f'{workdir}/plt_latent_embed_epoch_30_tsne.html',
    )
    affinity_job.load_latent_space_coords()
    print('Latent Space Coords:', affinity_job.latent_space_coords.shape)

    affinity_job.load_embedding2d()
    print('Embedding 2D: ', affinity_job.embedding2d.shape)

    affinity_job.load_standard_deviation()
    print('Standard Deviation: ', affinity_job.standard_deviation.shape)

    affinity_job.load_pose()
    print('Position: ', affinity_job.pose.shape)

    affinity_job.load_metadata()
    print('Metadata: ', affinity_job.metadata.shape)

    print('Done')

    #affinity_loader = AffinityVegaLoader("affinity/plt_latent_embed_epoch_30_tsne.html")
    #dataset = affinity_loader.get_dataset("data-68149f6c5c3929511c2d24005b767d92")
    #print(dataset["emb-x"])
    #for entry in dataset:
        #entry_str = entry["id"], entry["emb-x"], entry["emb-y"]
        #print(entry_str)
        #print(entry['emb-x'])
    
    #dataframe = pd.DataFrame(dataset)
    #print(dataframe)
    #dataframe.to_csv("affinity-dataset.csv",encoding='utf-8',index=False)

def run_cryodrgn():
    workdir = os.path.expandvars('${HOME}/myproject/cryodrgn_dataset/CryoDRGN/job002/train_128')
    epoch = 24
    cryo_job = CryoDRGNLatent(
        workdir=workdir,
        epoch=epoch,
        kmeans_cluster_number=20,
        latent_space_coords_path=f'{workdir}/z.{epoch}.pkl',
        embedding2d_path=f'{workdir}/analyze.{epoch}/umap.pkl'
    )
    cryo_job.run_analysis() 
    cryo_job.load_build_df_embeddings()
    cryo_job.build_df_kmeans_centers()
    cryo_job.build_df_labels()

    os.makedirs("./cryodrgn_output", exist_ok=True)
    cryo_job.df_latent_space_coords.to_csv("./cryodrgn_output/df_latent.csv",encoding='utf-8',index=False)
    cryo_job.df_embeddings.to_csv("./cryodrgn_output/df_embeddings.csv",encoding='utf-8',index=False)
    cryo_job.df_labels.to_csv("./cryodrgn_output/df_labels.csv",encoding='utf-8',index=False)
    cryo_job.df_kmeans_centers.to_csv("./cryodrgn_output/df_kmeans_centers.csv",encoding='utf-8',index=False)

    x_axis = 'Epoch'
    y_axis = 'Loss'
    title = 'Training Loss'
    df_training_loss = pd.DataFrame({
        x_axis: list(range(len(cryo_job.training_loss))), 
        y_axis: cryo_job.training_loss                      
    })

    fig = cryo_job.plotly_graph(df_training_loss, title, x_axis, y_axis)

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H1("CryoDRGN Training Loss", style={'textAlign':'center'}),
        dcc.Graph(figure=fig)
    ])
    app.run(debug=True)

def affinity_dataframe():
    workdir = os.path.expandvars('${HOME}/myproject/cryodrgn_dataset/CryoDRGN/job002/python_code/affinity')
    affinity_job = AffinityLatent(
        epoch=30,
        workdir=workdir,
        affinity_html_path=f'{workdir}/plt_latent_embed_epoch_30_tsne.html',
    )
    affinity_job.build_dataframe().to_csv("affinity-dataframe.csv",encoding='utf-8',index=False)
    print(affinity_job.build_dataframe())


if __name__ == "__main__":
    run_cryodrgn()
    #run_affinity()
    #affinity_dataframe()
    
    
    
