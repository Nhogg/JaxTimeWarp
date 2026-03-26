import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

@partial(jax.jit, static_argnames=['R'])
def compute_dtw_matrix_single(X, Y, R=10):
    """Computes DTW accum. cost matrix for a single pair of sequences."""
    C = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    N, M = C.shape

    # Sakoe-Chiba Band
    i_indices = jnp.arange(N)[:, None]
    j_indices = jnp.arange(M)[None, :]
    valid_band_mask = jnp.abs(i_indices - j_indices) <= R 
    C_banded = C + jnp.where(valid_band_mask, 0.0, jnp.inf)

    D_init = jnp.full((M + 1,), jnp.inf).at[0].set(0.0)

    def row_scan(prev_row, current_cost_row):
        def col_scan(left_val, j):
            up_val = prev_row[j + 1]
            diag_val = prev_row[j]
            min_prev = jnp.minimum(jnp.minimum(up_val, left_val), diag_val)
            current_val = current_cost_row[j] + min_prev
            return current_val, current_val

        _, new_row_core = jax.lax.scan(col_scan, jnp.inf, jnp.arange(M))
        new_row = jnp.concatenate([jnp.array([jnp.inf]), new_row_core])
        return new_row, new_row_core

    _, D_matrix = jax.lax.scan(row_scan, D_init, C_banded)
    return D_matrix

batch_compute_dtw = jax.vmap(compute_dtw_matrix_single, in_axes=(0, 0, None))

def compute_dtw_path(D_matrix):
    """
    Trace back the optimal DTW path through the accum. cost matrix.

    Args:
        D_matrix: np.ndarray of shape (N, M) containing accumulated costs.

    Returns:
        np.ndarray of shape (K, 2) representing the warping path coords.
    """
    N, M = D_matrix.shape
    i, j = N - 1, M - 1

    path = [(i, j)]

    # Walk backwards until we hit the starting corner (0, 0)
    while i > 0 or j > 0:
        if i == 0:
            # Top edge reached, can only move left
            j -= 1
        elif j == 0:
            # Left edge reached, can only move up
            i -= 1
        else:
            # Look at three valid previous steps
            cost_up = D_matrix[i - 1, j]
            cost_left = D_matrix[i, j - 1]
            cost_diag = D_matrix[i - 1, j - 1]

            # Find cheapest path
            min_cost = min(cost_up, cost_left, cost_diag)

            if min_cost == cost_diag:
                i -= 1
                j -= 1
            elif min_cost == cost_up:
                i -= 1
            else:
                j -= 1

        path.append((i, j))

    # Path was built backwards, reverse
    path.reverse()

    return np.array(path)




def test_dtw_on_dummy_data():
    B, N, M, DOF = 32, 20, 30, 7 # Batch size 32, Human 20-steps, Robot 30-steps
    
    # Generate smooth sine wave motions for the "Human" joints
    t_human = np.linspace(0, 2 * np.pi, N)
    # Shape: [B, N, 7]
    human_batch = np.sin(t_human)[None, :, None] * np.random.randn(B, 1, DOF) 
    
    # Generate the "Robot" data by evaluating the same waves over a longer time (M)
    t_robot = np.linspace(0, 2 * np.pi, M)
    robot_batch = np.sin(t_robot)[None, :, None] * np.random.randn(B, 1, DOF)
    
    # Add slight execution noise
    robot_batch += np.random.normal(0, 0.05, size=(B, M, DOF))
    
    # Convert to JAX arrays and push to GPU
    j_human = jnp.array(human_batch)
    j_robot = jnp.array(robot_batch)
    
    print(f"Running DTW on batches: Human {j_human.shape}, Robot {j_robot.shape}")
    
    # Output will be [32, 20, 30] - 32 independent DP matrices
    D_matrices = batch_compute_dtw(j_human, j_robot, 15)
    
    # Pull back to CPU for the traceback
    D_matrices_cpu = np.array(D_matrices)
    
    print(f"Successfully computed {B} DP matrices. Shape: {D_matrices_cpu.shape}")
    
    path_0 = compute_dtw_path(D_matrices_cpu[0])
    human_1d = np.array(j_human[0, :, 0])
    robot_1d = np.array(j_robot[0, :, 0])

    plot_dtw_alignment(human_1d, robot_1d, path_0)



def plot_dtw_alignment(seq_a, seq_b, path):
    """
    3 panel dtw plot

    Args:
        seq_a: 1D np.array (e.g., Human joint trajectory)
        seq_b: 1D np.array (e.g., Robot joint trajectory)
        path: np.array of shape (K, 2) containing the warping path indices
    """
    fig = plt.figure(figsize=(14, 8))
    
    # Top Left: Original Time Series
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(seq_a, label='Human (Series A)', color='blue')
    ax1.plot(seq_b, label='Robot (Series B)', color='orange', linestyle='--')
    ax1.set_title("Original Time Series")
    ax1.legend()
    
    # Top Right: The Warping Path Matrix
    ax2 = plt.subplot(2, 2, 2)
    # Plot the path
    ax2.plot(path[:, 0], path[:, 1], marker='o', color='green', markersize=4)
    ax2.set_title("Shortest Path (Warping Path)")
    ax2.set_xlabel("Human Index")
    ax2.set_ylabel("Robot Index")
    ax2.grid(True)
    
    # Bottom: Point-to-Point Alignment
    ax3 = plt.subplot(2, 1, 2)
    ax3.plot(seq_a, marker='o', color='blue', label='Human (Series A)')
    ax3.plot(seq_b, marker='x', color='orange', linestyle='--', label='Robot (Series B)')
    
    # Draw the gray alignment lines mapping the timesteps
    for (i, j) in path:
        # Plot a line from (x=i, y=seq_a[i]) to (x=j, y=seq_b[j])
        ax3.plot([i, j], [seq_a[i], seq_b[j]], color='gray', alpha=0.4, linewidth=1)
        
    ax3.set_title("Point-to-Point Comparison After DTW Alignment")
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig("dtw_alignment_test.png", dpi=150)
    print("Saved plot to 'dtw_alignment_test.png'")


def main():
    test_dtw_on_dummy_data()

if __name__ == "__main__":
    main()
