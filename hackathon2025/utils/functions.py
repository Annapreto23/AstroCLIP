# @title [Run me]
%pylab inline
import pyarrow
pyarrow.PyExtensionType.set_auto_load(True)
from tqdm import tqdm
import networkx as nx
from pyvis.network import Network
from IPython.display import HTML
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact
from IPython.display import display


def graph_from_embeddings(embeddings, images, k=5, symmetrize=True):
    """
    Builds a k-NN graph adjacency matrix using scikit-learn (cosine distance).

    Returns a sparse adjacency matrix (numpy array or CSR).
    """
    # Convert to NumPy and normalize (cosine similarity = dot of L2-normalized vectors)
    emb_np = embeddings.cpu().numpy()

    # scikit-learn uses cosine *distance*, so we negate similarity in behavior
    A = kneighbors_graph(emb_np, n_neighbors=k, metric='cosine', mode='connectivity', include_self=False)

    if symmetrize:
        A = A.maximum(A.T)  # make it symmetric (undirected)

    G = nx.Graph()
    A = A.tocoo()

    for i in range(A.shape[0]):
        G.add_node(i, image_tensor=images[i].permute(1, 2, 0).cpu().numpy() if images is not None else None)

    for i, j, w in zip(A.row, A.col, A.data):
        G.add_edge(i, j, weight=w)

    return G

def encode_image_base64(image_array, scale=1.):
    fig = plt.figure(figsize=(scale, scale), dpi=100)
    plt.axis("off")
    plt.imshow(image_array)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def draw_interactive_graph_colab(graph):
    net = Network(height='750px', width='100%', bgcolor='#000000', font_color='white', notebook=False, cdn_resources='remote')

    # Encode each image as base64 and use as node icon
    for node_id, data in graph.nodes(data=True):
        img_url = encode_image_base64(data['image_tensor'])
        net.add_node(
            int(node_id),
            shape="image",
            image=img_url,
            title=f"Galaxy {node_id}",
            size=50
        )

    for src, tgt, data in graph.edges(data=True):
        net.add_edge(int(src), int(tgt), value=float(data['weight']) * 5,
                     color='rgba(255, 255, 255, 0.5)')

    # Generate the HTML string (without writing to file)
    html_str = net.generate_html()

    # Display directly in Colab
    return HTML(html_str)
