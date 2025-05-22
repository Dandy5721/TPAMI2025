import numpy as np
import nibabel.freesurfer as fs
from scipy import sparse
import matplotlib.pyplot as plt
import torch

# === Step 1: Load the FreeSurfer pial surface mesh ===
pial_path = "lh_r.pial"
vertices, faces = fs.read_geometry(pial_path)
num_vertices = vertices.shape[0]
print(f"Vertices: {num_vertices}, Faces: {faces.shape[0]}")

# === Step 2: Construct Euclidean-weighted adjacency matrix based on mesh topology ===
def build_weighted_adjacency(vertices, faces):
    row = []
    col = []
    data = []

    for tri in faces:
        i, j, k = tri
        edges = [(i, j), (j, k), (k, i)]

        for a, b in edges:
            dist = np.linalg.norm(vertices[a] - vertices[b])
            row.extend([a, b])
            col.extend([b, a])
            data.extend([dist, dist])

    adj = sparse.coo_matrix((data, (row, col)), shape=(vertices.shape[0], vertices.shape[0]))
    return adj.tocsr()

adjacency = build_weighted_adjacency(vertices, faces)

# === Step 3: Print matrix statistics ===
print(f"Adjacency shape: {adjacency.shape}")
print(f"Number of non-zero entries: {adjacency.nnz}")
sparsity = 1.0 - adjacency.nnz / (num_vertices ** 2)
print(f"Sparsity: {sparsity:.6f}")

# === Step 4: Save the sparse adjacency matrix to disk ===
sparse.save_npz("lh_r_mesh_weighted_adjacency.npz", adjacency)

# === Step 5: Visualize the top-left 1000x1000 block of the adjacency matrix ===
sub_adj = adjacency[:1000, :1000].toarray()
plt.figure(figsize=(8, 8))
plt.imshow(sub_adj, cmap='inferno', interpolation='nearest')
plt.title("Weighted Adjacency (1000x1000)")
plt.colorbar()
plt.savefig("weighted_adj_top1000x1000.svg", dpi=300)
plt.close()

# === Step 6: Construct symmetrically normalized graph Laplacian: L = I - D^{-1/2} A D^{-1/2} ===
def build_normalized_laplacian(A):
    d = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=(d != 0))
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    L = sparse.identity(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return L

laplacian = build_normalized_laplacian(adjacency)
sparse.save_npz("lh_r_mesh_laplacian.npz", laplacian)
print("Normalized Laplacian saved.")

# === Step 7: Export adjacency matrix in PyTorch Geometric format ===
def convert_to_edge_index(adj):
    coo = adj.tocoo()
    edge_index = np.vstack((coo.row, coo.col)).astype(np.int64)
    edge_weight = coo.data.astype(np.float32)
    return edge_index, edge_weight

edge_index_np, edge_weight_np = convert_to_edge_index(adjacency)
torch.save({
    "edge_index": torch.tensor(edge_index_np),
    "edge_weight": torch.tensor(edge_weight_np)
}, "lh_r_pyg_graph.pt")

print("Saved PyTorch Geometric graph as lh_r_pyg_graph.pt")
