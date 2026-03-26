import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def compute_dtw_matrix(X, Y):
    # X shape: (N, 7) -> human traj.
    # Y shape: (M, 7) -> robot traj.

    # Parallelized distance matrix computation (C)
    # Broadcasts to (N, M, 7) then sums squared differences to yield (N, M)
    C = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2. axis=-1)

    # Define Sakoe-Chiba Band Radius (path stray distance)
    R = 10

    # Coordinate grids for matrix indices
    i_indices = jnp.arrange(N)[:, None]
    j_indices = jnp.arrange(M)[None, :]

    # Build boolean mask (True if inside band, false if in corners
    # Condition: |i - j| <= R 
    valid_band_mask = jnp.abs(i_indices - j_indices) <= R

    # Convert mask to penalty tensor
    # 0.0 added cost for valid cells, jnp.inf added cost for invalid cells
    band_penalty = jnp.where(valid_band_mask, 0.0, jnp.inf)

    # Apply constraint to cost matrix
    C_banded = C + band_pentalty
   
    N, M = C.shape

    # Double scan DP table
    # Pad previous row with infinity to avoid diagonal checks
    def row_scan(prev_row, current_cost_row):

        # Traverse columns of current row
        def col_scan(left_val, j):
            up_val = prev_row[j + 1]    # D[i - 1, j]
            diag_val = prev_row[j]      # D[i - 1, j - 1]

            min_prev = jnp.minimum(jnp.minimum(up_val, left_val), diag_val)
            current_val = current_cost_row[j] + min_prev

            return current_val, current_val

        _, new_row_core = jax.lax.scan(col_scan, jnp.inf, jnp.arrange(M))

        new_row = jnp.concatenate([jnp.array[jnp.inf]), new_row_core])

        return new_row, new_row_core
    
    # Rerun outser scan  over all rows of C
    _, D_matrix = jax.lax.scan(row_scan, D_init, C)

    return D_matrix[-1, -1], D_matrix

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
            min_cost = min(cost_p, cost_left, cost_diag)

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


