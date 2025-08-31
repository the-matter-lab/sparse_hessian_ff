import numpy as np
import scipy.sparse as sp
from collections import deque

def recover_matrix(
    recover_from,
    edges,
    n,
    B,
    vertex_colors,
    *,
    reconstruct_sparsely=False,
    dtype=np.float32,
    fail_on_cycle=True,
    verify=False,
    acyclic_components=None,   # <- NEW: pass the dict from your C++ binding
    adjacency=None             # optional prebuilt adjacency (list-of-lists), else built from edges
):
    """
    Parameters
    ----------
    recover_from : {'DISTANCE_TWO','STAR','ACYCLIC','ACYCLIC_INDIRECT'}
    edges : (m,2) int array
        Undirected edges, each listed once (order u<v not required).
    n : int
    B : (n, C) array
        Y = A @ S where S[:, c] = 1{color==c}.
    vertex_colors : (n,) int array
        0-based color id per vertex.
    reconstruct_sparsely : bool
        If True, build via COO pieces and densify at the end.
    dtype : numpy dtype
    fail_on_cycle : bool
        If True, raise if a two-colored subgraph contains a cycle.
    verify : bool
        If True, check that A_rec @ S == B (∞-norm small).
    acyclic_components : dict or None
        The preprocessing returned by your binding when using
        "ACYCLIC_FOR_INDIRECT_RECOVERY".
        Expected keys: "color_pairs": [(c1,c2), ...],
                        "edges": [[(u,v), ...], ...],
                        "vertices": [[u,v,...], ...]
        If present, the acyclic branch uses this directly (faster).
    adjacency : list[list[int]] or None
        Optional global adjacency (0..n-1 -> neighbors). If None, built from edges.

    Returns
    -------
    A_rec : (n, n) array (dtype)
    """
    edges = np.asarray(edges, dtype=np.intp)
    # 1) drop self-loops if any
    edges = edges[edges[:, 0] != edges[:, 1]]
    # 2) canonicalize each pair so (min,max)
    edges = np.sort(edges, axis=1)
    # 3) deduplicate
    edges = np.unique(edges, axis=0)
    # 4) unpack
    u = edges[:, 0]
    v = edges[:, 1]
    ii = np.arange(n, dtype=np.intp)
    colors = np.asarray(vertex_colors, dtype=np.intp)
    C = int(colors.max()) + 1 if colors.size else 0

    # ---------- DISTANCE-TWO (unchanged) ----------
    if recover_from == 'DISTANCE_TWO':
        if reconstruct_sparsely:
            rows = np.concatenate([ii, u, v])
            cols = np.concatenate([ii, v, u])
            vals = np.concatenate([
                B[ii, colors],        # diagonal
                B[u, colors[v]],      # off-diagonal (u,v)
                B[v, colors[u]],      # off-diagonal (v,u)
            ])
            A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr().toarray()
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            A_rec[ii, ii] = B[ii, colors]
            A_rec[u, v] = B[u, colors[v]]
            A_rec[v, u] = B[v, colors[u]]
        return A_rec.astype(dtype, copy=False)

    # ---------- STAR (direct) (unchanged) ----------
    elif recover_from == 'STAR':
        # color stats per row: count neighbors in each color
        col_stat = np.zeros((n, C), dtype=np.int32)
        np.add.at(col_stat, (u, colors[v]), 1)
        np.add.at(col_stat, (v, colors[u]), 1)

        diag_vals = B[ii, colors]
        col_u, col_v = colors[u], colors[v]

        unique_uv = (col_stat[u, col_v] == 1)
        val_uv = np.where(unique_uv, B[u, col_v], B[v, col_u])

        unique_vu = (col_stat[v, col_u] == 1)
        val_vu = np.where(unique_vu, B[v, col_u], B[u, col_v])

        if reconstruct_sparsely:
            rows = np.concatenate([ii, u, v])
            cols = np.concatenate([ii, v, u])
            vals = np.concatenate([diag_vals, val_uv, val_vu])
            A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr().toarray()
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            A_rec[ii, ii] = diag_vals
            A_rec[u, v] = val_uv
            A_rec[v, u] = val_vu

        return A_rec.astype(dtype, copy=False)

    # ---------- ACYCLIC (indirect / substitution) ----------
    elif recover_from in ('ACYCLIC', 'ACYCLIC_INDIRECT', 'ACYCLIC_FOR_INDIRECT_RECOVERY'):
        # Trivial: one color -> diagonal only
        if C <= 1:
            A_rec = np.zeros((n, n), dtype=dtype)
            if C == 1:
                A_rec[ii, ii] = B[ii, colors]
            return A_rec

        # Working copy of B for in-place subtraction during peeling
        B_work = B.astype(dtype, copy=True)

        if reconstruct_sparsely:
            rows = [ii]
            cols = [ii]
            vals = [B[ii, colors].astype(dtype, copy=False)]
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            A_rec[ii, ii] = B[ii, colors]

        # Global adjacency (used only by the fallback path)
        if adjacency is None:
            adj_all = [[] for _ in range(n)]
            for a, b in edges:
                adj_all[a].append(int(b))
                adj_all[b].append(int(a))
        else:
            adj_all = adjacency

        # Helper: peel one 2-colored component given its node/edge lists
        def peel_component(comp_vertices, comp_edges, c1, c2):
            # Build per-component degree + adjacency
            deg = {int(x): 0 for x in comp_vertices}
            adj = {int(x): [] for x in comp_vertices}
            for (a, b) in comp_edges:
                a = int(a); b = int(b)
                deg[a] += 1
                deg[b] += 1
                adj[a].append(b)
                adj[b].append(a)

            q = deque([x for x in deg if deg[x] == 1])

            while q:
                x = q.popleft()
                if deg[x] != 1:
                    continue
                # find its unique neighbor y in this component
                y = next((nb for nb in adj[x] if deg.get(nb, 0) > 0), None)
                if y is None:
                    deg[x] = 0
                    continue

                cx, cy = colors[x], colors[y]
                # Leaf substitution (opposite color):
                # A[x,y] = B_work[x, color(y)], and subtract from B_work[y, color(x)]
                a_xy = B_work[x, cy] if cx == c1 or cx == c2 else B_work[x, cy]
                if reconstruct_sparsely:
                    rows.append(np.array([x, y], dtype=np.intp))
                    cols.append(np.array([y, x], dtype=np.intp))
                    vals.append(np.array([a_xy, a_xy], dtype=dtype))
                else:
                    A_rec[x, y] = a_xy
                    A_rec[y, x] = a_xy

                B_work[y, cx] -= a_xy  # subtract contribution on the opposite column at y

                # Remove x (and edge x-y) from the component
                deg[x] = 0
                deg[y] -= 1
                if deg[y] == 1:
                    q.append(y)

            if fail_on_cycle:
                # If any vertex in component still has deg>0, there was a cycle
                if any(deg[vx] > 0 for vx in deg):
                    raise ValueError(
                        f"Cycle detected in 2-colored component (colors {c1},{c2}); "
                        "violates acyclic coloring assumptions."
                    )

        # Fast path: use preprocessed components from the binding
        if acyclic_components is not None and len(acyclic_components.get("edges", [])) > 0:
            comp_color_pairs = acyclic_components["color_pairs"]
            comp_edges_list  = acyclic_components["edges"]
            comp_vertices_list = acyclic_components["vertices"]

            # Optional: sanity—ensure each component only uses its two colors
            # (we trust ColPack; skip for speed)

            for (c1, c2), comp_edges, comp_vertices in zip(
                comp_color_pairs, comp_edges_list, comp_vertices_list
            ):
                if not comp_edges:
                    continue
                peel_component(comp_vertices, comp_edges, int(c1), int(c2))

        else:
            # Fallback: rebuild per-pair forests on the fly (previous implementation)
            col_u, col_v = colors[u], colors[v]
            pair_uv = np.stack([np.minimum(col_u, col_v), np.maximum(col_u, col_v)], axis=1)
            unique_pairs = np.unique(pair_uv, axis=0)

            for c1, c2 in unique_pairs:
                if c1 == c2:
                    continue
                mask = ((col_u == c1) & (col_v == c2)) | ((col_u == c2) & (col_v == c1))
                e_u = u[mask]; e_v = v[mask]
                if e_u.size == 0:
                    continue
                comp_edges = list(map(tuple, np.stack([e_u, e_v], axis=1)))
                comp_vertices = np.unique(np.concatenate([e_u, e_v])).tolist()
                peel_component(comp_vertices, comp_edges, int(c1), int(c2))

        if reconstruct_sparsely:
            rows = np.concatenate(rows)
            cols = np.concatenate(cols)
            vals = np.concatenate(vals)
            A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr().toarray()

        if verify:
            # B_hat[:, c] = sum_{j: color[j]==c} A_rec[:, j]
            B_hat = np.zeros_like(B, dtype=dtype)
            for c in range(C):
                cols_c = (colors == c)
                if np.any(cols_c):
                    B_hat[:, c] = A_rec[:, cols_c].sum(axis=1)
            err_inf = np.max(np.abs(B_hat - B))
            if not np.isfinite(err_inf) or err_inf > 1e-4 * (np.linalg.norm(B, ord=np.inf) + 1e-12):
                raise AssertionError(f"Verification failed: ||A S - B||_inf = {err_inf:.3e}")

        return A_rec.astype(dtype, copy=False)

    else:
        raise ValueError(f"Unknown recover_from: {recover_from}")
