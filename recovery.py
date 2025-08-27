import numpy as np
import scipy.sparse as sp






import numpy as np
import scipy.sparse as sp
from collections import deque

def recover_matrix(recover_from, edges, n, B, vertex_colors,
                   reconstruct_sparsely=False, dtype=np.float32,
                   fail_on_cycle=True, verify=False):
    """
    Parameters
    ----------
    recover_from : {'DISTANCE_TWO','STAR','ACYCLIC','ACYCLIC_FOR_INDIRECT_RECOVERY'}
    edges : (m,2) int array
        Undirected edges; each edge should appear once (u < v not required).
    n : int
        Number of vertices.
    B : (n, C) array
        Compressed matrix B = A @ S where S[:, c] = 1{color==c}.
        I.e., B[i, c] = sum_j A[i, j] * 1{color[j]==c}.
    vertex_colors : (n,) int array
        0-based color id for each vertex.
    reconstruct_sparsely : bool
        If True, builds via COO
    dtype : numpy dtype
    fail_on_cycle : bool
        If True, raise if a two-colored induced subgraph contains a cycle
        (which would violate acyclic coloring assumptions).
    verify : bool
        If True, verify that A_rec @ S == B (within fp tolerance).

    Returns
    -------
    A_rec : (n, n) array (dtype)
    """
    u = edges[:, 0].astype(np.intp)
    v = edges[:, 1].astype(np.intp)
    ii = np.arange(n, dtype=np.intp)
    colors = np.asarray(vertex_colors, dtype=np.intp)
    C = int(colors.max()) + 1 if len(colors) else 0

    if recover_from == 'DISTANCE_TWO':
        if reconstruct_sparsely:
            rows = np.concatenate([ii, u, v])
            cols = np.concatenate([ii, v, u])
            vals = np.concatenate([
                B[ii, colors],        # diagonal
                B[u, colors[v]],      # off-diagonal (u,v)
                B[v, colors[u]],      # off-diagonal (v,u)
            ])
            A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            A_rec[ii, ii] = B[ii, colors]
            A_rec[u, v] = B[u, colors[v]]
            A_rec[v, u] = B[v, colors[u]]
        return A_rec.astype(dtype, copy=False)

    elif recover_from == 'STAR':
        # color statistics per row (how many neighbors per color)
        color_stat = np.zeros((n, C), dtype=np.int32)
        np.add.at(color_stat, (u, colors[v]), 1)
        np.add.at(color_stat, (v, colors[u]), 1)

        diag_vals = B[ii, colors]
        col_u = colors[u]
        col_v = colors[v]

        unique_uv = (color_stat[u, col_v] == 1)
        val_uv = np.where(unique_uv, B[u, col_v], B[v, col_u])

        unique_vu = (color_stat[v, col_u] == 1)
        val_vu = np.where(unique_vu, B[v, col_u], B[u, col_v])

        if reconstruct_sparsely:
            rows = np.concatenate([ii, u, v])
            cols = np.concatenate([ii, v, u])
            vals = np.concatenate([diag_vals, val_uv, val_vu])
            A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            A_rec[ii, ii] = diag_vals
            A_rec[u, v] = val_uv
            A_rec[v, u] = val_vu

        return A_rec.astype(dtype, copy=False)

    elif recover_from in ('ACYCLIC', 'ACYCLIC_FOR_INDIRECT_RECOVERY'):
        # Special case: single color => direct/diagonal only (matches ColPack)
        if C <= 1:
            A_rec = np.zeros((n, n), dtype=dtype)
            if C == 1:
                A_rec[ii, ii] = B[ii, colors]
            return A_rec

        # Work copy of B. We will peel contributions from specific (c1,c2) pairs only.
        B_work = B.astype(dtype, copy=True)

        if reconstruct_sparsely:
            rows = [ii]
            cols = [ii]
            vals = [B[ii, colors].astype(dtype, copy=False)]
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            # Diagonals are always direct from B[:, color]
            A_rec[ii, ii] = B[ii, colors]

        # Precompute per-edge color pairs (unordered)
        col_u = colors[u]
        col_v = colors[v]
        pair_uv = np.stack([np.minimum(col_u, col_v), np.maximum(col_u, col_v)], axis=1)
        unique_pairs = np.unique(pair_uv, axis=0)

        # Build adjacency once for the whole graph (as Python lists for speed)
        # We'll reuse it per color pair by filtering.
        adj_all = [[] for _ in range(n)]
        for a, b in edges:
            adj_all[a].append(b)
            adj_all[b].append(a)

        # Helper to process each two-colored forest via leaf peeling
        def peel_pair(c1, c2):
            # Gather vertices and edges belonging to this pair
            mask = ( (col_u == c1) & (col_v == c2) ) | ( (col_u == c2) & (col_v == c1) )
            if not np.any(mask):
                return 0  # nothing to do

            e_u = u[mask]
            e_v = v[mask]

            # Nodes participating in this pair
            nodes = np.unique(np.concatenate([e_u, e_v]))
            if nodes.size == 0:
                return 0

            # Build per-pair degree and adjacency (filtered to the pair)
            in_pair = np.zeros(n, dtype=bool)
            in_pair[nodes] = True

            # Degree in the pair subgraph
            deg = np.zeros(n, dtype=np.int32)
            # We *could* restrict adjacency by checking neighbor color in {c1,c2}
            # and that the edge is indeed in this pair (saves a map lookup).
            for a, b in zip(e_u, e_v):
                deg[a] += 1
                deg[b] += 1

            # Leaf queue (only nodes in this pair with deg==1)
            q = deque(int(x) for x in nodes if deg[x] == 1)
            recovered_edges = 0

            while q:
                x = q.popleft()
                if deg[x] != 1:
                    continue  # might have been updated

                # Find the unique neighbor y of x within this pair
                y = -1
                # Scan neighbors, but accept only those that (i) are in_pair, and (ii) edge (x,y) is of colors (c1,c2)
                cx = colors[x]
                for nb in adj_all[x]:
                    if not in_pair[nb]:
                        continue
                    cy = colors[nb]
                    if ( (cx == c1 and cy == c2) or (cx == c2 and cy == c1) ):
                        if deg[nb] > 0:
                            y = nb
                            break
                if y < 0:
                    # Isolated or already removed
                    deg[x] = 0
                    continue

                cx = colors[x]
                cy = colors[y]

                # Leaf substitution:
                # If x has color c1, its opposite column is c2 (and vice versa)
                if cx == c1 and cy == c2:
                    a_xy = B_work[x, c2]
                    # place value
                    if reconstruct_sparsely:
                        rows.append(np.array([x, y], dtype=np.intp))
                        cols.append(np.array([y, x], dtype=np.intp))
                        vals.append(np.array([a_xy, a_xy], dtype=dtype))
                    else:
                        A_rec[x, y] = a_xy
                        A_rec[y, x] = a_xy
                    # subtract from neighbor's opposite column
                    B_work[y, c1] -= a_xy
                else:
                    # x has color c2, y has c1
                    a_xy = B_work[x, c1]
                    if reconstruct_sparsely:
                        rows.append(np.array([x, y], dtype=np.intp))
                        cols.append(np.array([y, x], dtype=np.intp))
                        vals.append(np.array([a_xy, a_xy], dtype=dtype))
                    else:
                        A_rec[x, y] = a_xy
                        A_rec[y, x] = a_xy
                    B_work[y, c2] -= a_xy

                # Remove x from the pair-graph
                deg[x] = 0
                if deg[y] > 0:
                    deg[y] -= 1
                    if deg[y] == 1:
                        q.append(y)

                recovered_edges += 1

            # If anything left with deg>0, we hit a cycle in this 2-colored subgraph.
            if fail_on_cycle and np.any(deg[nodes] > 0):
                raise ValueError(
                    f"Detected a cycle in the two-colored subgraph (colors {c1},{c2}). "
                    "This violates acyclic coloring assumptions; cannot complete indirect recovery."
                )

            return recovered_edges

        # Process each unordered color pair
        for c1, c2 in unique_pairs:
            if c1 == c2:
                continue  # proper vertex coloring: no edges expected; skip
            peel_pair(int(c1), int(c2))

        if reconstruct_sparsely:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            vals = np.concatenate(vals)
            A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        # else A_rec already filled

        if verify:
            # Verify A_rec @ S == B within tolerance, without explicitly forming S.
            # B_hat[:, c] = sum_{j: color[j]==c} A_rec[:, j]
            B_hat = np.zeros_like(B_work)
            for c in range(C):
                cols_c = (colors == c)
                if np.any(cols_c):
                    B_hat[:, c] = A_rec[:, cols_c].sum(axis=1)
            err = np.max(np.abs(B_hat - B))
            if not np.isfinite(err) or err > 1e-4 * (np.linalg.norm(B, ord=np.inf) + 1e-12):
                raise AssertionError(f"Verification failed: ||A S - B||_∞ = {err:.3e}")
        return A_rec.astype(dtype, copy=False)

    else:
        raise ValueError(f"Unknown recover_from: {recover_from}")
