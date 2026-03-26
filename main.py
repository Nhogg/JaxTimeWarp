import jax
import jax.numpy as jnp

@jax.jit
def compute_dtw_matrix(X, Y):
    # X shape: (N, 7) -> human traj.
    # Y shape: (M, 7) -> robot traj.

    # Parallelized distance matrix computation (C)
    # Broadcasts to (N, M, 7) then sums squared differences to yield (N, M)
    C = jnp.sum((X[:, None, :] - Y[None, :, :]) ** 2. axis=-1)
   
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
