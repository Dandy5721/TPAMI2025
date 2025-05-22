import os
import numpy as np
from scipy.sparse import load_npz
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import expm_multiply
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def load_subject_data(subject_id, data_root, amy_dir, adj_dir):
    """
    Load tau0, tau1, amyloid, and Laplacian matrix for a given subject.
    - data_root: contains subject_id.csv with tau0 and tau1 in the first two rows
    - amy_dir: contains subject_id.csv with a single line of amyloid values
    - adj_dir: contains subject_id.npz storing sparse adjacency matrix
    """
    tau_path = os.path.join(data_root, f"{subject_id}.csv")
    amy_path = os.path.join(amy_dir, f"{subject_id}.csv")
    adj_path = os.path.join(adj_dir, f"{subject_id}.npz")

    if not (os.path.isfile(tau_path) and os.path.isfile(amy_path) and os.path.isfile(adj_path)):
        return None

    try:
        tau_data = np.loadtxt(tau_path, delimiter=',')
        if tau_data.shape[0] < 2:
            return None
        u_t = tau_data[0]
        u1 = tau_data[1]

        amy_data = np.loadtxt(amy_path, delimiter=',')
        v_t = amy_data[0] if amy_data.ndim == 1 else amy_data[0, :]

        A = load_npz(adj_path)
        L = laplacian(A, normed=False)
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}")
        return None

    return u_t, u1, v_t, L


def predict_tau(u_t, L, beta, delta_t):
    """
    Predict tau at t+1 using sparse diffusion:
    u(t+1) = expm(-beta * L * delta_t) @ u(t)
    """
    return expm_multiply(-beta * L * delta_t, u_t)


def grid_search_diffusion(data_root, amy_dir, adj_dir, beta_list,
                          alpha=0.5, delta_t=1.0, max_samples=None):
    """
    Perform grid search over beta values for sparse diffusion model.
    Input:
        - data_root: directory with tau files
        - amy_dir: directory with amyloid files
        - adj_dir: directory with .npz adjacency matrices
        - beta_list: list of beta diffusion rates
        - alpha: weighting factor between tau and amyloid
        - delta_t: time step
        - max_samples: optional limit on number of samples
    """
    subject_ids = [f[:-4] for f in os.listdir(data_root) if f.endswith('.csv')]
    samples = []
    for sid in subject_ids:
        sample = load_subject_data(sid, data_root, amy_dir, adj_dir)
        if sample is not None:
            samples.append((sid, sample))
        if max_samples and len(samples) >= max_samples:
            break
    print(f"✅ Loaded {len(samples)} valid samples")

    best = (None, None, np.inf)
    results = {}

    for beta in beta_list:
        maes = []
        rmses = []
        for _, (u_t, u1, v_t, L) in samples:
            x_fused = alpha * u_t + (1 - alpha) * v_t
            u_pred = predict_tau(x_fused, L, beta, delta_t)
            maes.append(mean_absolute_error(u1, u_pred))
            rmses.append(root_mean_squared_error(u1, u_pred))
        mean_mae, std_mae = np.mean(maes), np.std(maes)
        mean_rmse, std_rmse = np.mean(rmses), np.std(rmses)
        results[beta] = {
            'mae_mean': mean_mae, 'mae_std': std_mae,
            'rmse_mean': mean_rmse, 'rmse_std': std_rmse
        }
        print(f"🧪 beta = {beta:.3f} -> MAE = {mean_mae:.4f} ± {std_mae:.4f}, "
              f"RMSE = {mean_rmse:.4f} ± {std_rmse:.4f}")
        if mean_mae < best[2]:
            best = (beta, alpha, mean_mae)

    print(f"\n🏆 Best parameters: beta={best[0]}, alpha={best[1]} -> MAE={best[2]:.4f}")
    return results


if __name__ == "__main__":
    # ✅ Replace with your actual folder paths
    data_root = "/your/path/to/tau_data"             # contains subject_id.csv (tau0, tau1)
    amy_dir = "/your/path/to/amyloid_data"           # contains subject_id.csv (amyloid)
    adj_dir = "/your/path/to/adjacency_matrices"     # contains subject_id.npz (sparse matrix)

    beta_list = [0.1, 0.5, 1.0, 2.0]  # diffusion rates to test
    grid_search_diffusion(
        data_root, amy_dir, adj_dir,
        beta_list=beta_list,
        alpha=0.5, delta_t=1.0,
        max_samples=10
    )
