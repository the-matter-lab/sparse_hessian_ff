# import numpy as np
# import scipy.sparse as sp
# from collections import deque

# def recover_matrix(
#     recover_from,
#     edges,
#     n,
#     B,
#     vertex_colors,
#     *,
#     reconstruct_sparsely=False,
#     dtype=np.float32,
#     fail_on_cycle=True,
#     verify=False,
#     acyclic_components=None,   # <- NEW: pass the dict from your C++ binding
#     adjacency=None             # optional prebuilt adjacency (list-of-lists), else built from edges
# ):
#     """
#     Parameters
#     ----------
#     recover_from : {'DISTANCE_TWO','STAR','ACYCLIC','ACYCLIC_INDIRECT'}
#     edges : (m,2) int array
#         Undirected edges, each listed once (order u<v not required).
#     n : int
#     B : (n, C) array
#         Y = A @ S where S[:, c] = 1{color==c}.
#     vertex_colors : (n,) int array
#         0-based color id per vertex.
#     reconstruct_sparsely : bool
#         If True, build via COO pieces and densify at the end.
#     dtype : numpy dtype
#     fail_on_cycle : bool
#         If True, raise if a two-colored subgraph contains a cycle.
#     verify : bool
#         If True, check that A_rec @ S == B (∞-norm small).
#     acyclic_components : dict or None
#         Expected keys: "color_pairs": [(c1,c2), ...],
#                         "edges": [[(u,v), ...], ...],
#                         "vertices": [[u,v,...], ...]
#         If present, the acyclic branch uses this directly (faster).
#     adjacency : list[list[int]] or None
#         Optional global adjacency (0..n-1 -> neighbors). If None, built from edges.

#     Returns
#     -------
#     A_rec : (n, n) array (dtype)
#     """
#     edges = np.asarray(edges, dtype=np.intp)
#     # 1) drop self-loops if any
#     edges = edges[edges[:, 0] != edges[:, 1]]
#     # 2) canonicalize each pair so (min,max)
#     edges = np.sort(edges, axis=1)
#     # 3) deduplicate
#     edges = np.unique(edges, axis=0)
#     # 4) unpack
#     u = edges[:, 0]
#     v = edges[:, 1]
#     ii = np.arange(n, dtype=np.intp)
#     colors = np.asarray(vertex_colors, dtype=np.intp)
#     C = int(colors.max()) + 1 if colors.size else 0

#     # ---------- DISTANCE-TWO (unchanged) ----------
#     if recover_from == 'DISTANCE_TWO':
#         if reconstruct_sparsely:
#             rows = np.concatenate([ii, u, v])
#             cols = np.concatenate([ii, v, u])
#             vals = np.concatenate([
#                 B[ii, colors],        # diagonal
#                 B[u, colors[v]],      # off-diagonal (u,v)
#                 B[v, colors[u]],      # off-diagonal (v,u)
#             ])
#             A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
#             return A_rec.astype(dtype, copy=False)
#         else:
#             A_rec = np.zeros((n, n), dtype=dtype)
#             A_rec[ii, ii] = B[ii, colors]
#             A_rec[u, v] = B[u, colors[v]]
#             A_rec[v, u] = B[v, colors[u]]
#             return A_rec.astype(dtype, copy=False)

#     # ---------- STAR (direct) (unchanged) ----------
#     elif recover_from == 'STAR':
#         # color stats per row: count neighbors in each color
#         col_stat = np.zeros((n, C), dtype=np.int32)
#         np.add.at(col_stat, (u, colors[v]), 1)
#         np.add.at(col_stat, (v, colors[u]), 1)

#         diag_vals = B[ii, colors]
#         col_u, col_v = colors[u], colors[v]

#         unique_uv = (col_stat[u, col_v] == 1)
#         val_uv = np.where(unique_uv, B[u, col_v], B[v, col_u])

#         unique_vu = (col_stat[v, col_u] == 1)
#         val_vu = np.where(unique_vu, B[v, col_u], B[u, col_v])

#         if reconstruct_sparsely:
#             rows = np.concatenate([ii, u, v])
#             cols = np.concatenate([ii, v, u])
#             vals = np.concatenate([diag_vals, val_uv, val_vu])
#             A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
#             return A_rec.astype(dtype, copy=False)
#         else:
#             A_rec = np.zeros((n, n), dtype=dtype)
#             A_rec[ii, ii] = diag_vals
#             A_rec[u, v] = val_uv
#             A_rec[v, u] = val_vu
#             return A_rec.astype(dtype, copy=False)

#     # ---------- ACYCLIC (indirect / substitution) ----------
#     elif recover_from in ('ACYCLIC', 'ACYCLIC_INDIRECT', 'ACYCLIC_FOR_INDIRECT_RECOVERY'):
#         # Trivial: one color -> diagonal only
#         if C <= 1:
#             A_rec = np.zeros((n, n), dtype=dtype)
#             if C == 1:
#                 A_rec[ii, ii] = B[ii, colors]
#             return A_rec

#         # Working copy of B for in-place subtraction during peeling
#         B_work = B.astype(dtype, copy=True)

#         if reconstruct_sparsely:
#             rows = [ii]
#             cols = [ii]
#             vals = [B[ii, colors].astype(dtype, copy=False)]
#         else:
#             A_rec = np.zeros((n, n), dtype=dtype)
#             A_rec[ii, ii] = B[ii, colors]

#         # Global adjacency (used only by the fallback path)
#         if adjacency is None:
#             adj_all = [[] for _ in range(n)]
#             for a, b in edges:
#                 adj_all[a].append(int(b))
#                 adj_all[b].append(int(a))
#         else:
#             adj_all = adjacency

#         # Helper: peel one 2-colored component given its node/edge lists
#         def peel_component(comp_vertices, comp_edges, c1, c2):
#             # Build per-component degree + adjacency
#             deg = {int(x): 0 for x in comp_vertices}
#             adj = {int(x): [] for x in comp_vertices}
#             for (a, b) in comp_edges:
#                 a = int(a); b = int(b)
#                 deg[a] += 1
#                 deg[b] += 1
#                 adj[a].append(b)
#                 adj[b].append(a)

#             q = deque([x for x in deg if deg[x] == 1])

#             while q:
#                 x = q.popleft()
#                 if deg[x] != 1:
#                     continue
#                 # find its unique neighbor y in this component
#                 y = next((nb for nb in adj[x] if deg.get(nb, 0) > 0), None)
#                 if y is None:
#                     deg[x] = 0
#                     continue

#                 cx, cy = colors[x], colors[y]
#                 # Leaf substitution (opposite color):
#                 # A[x,y] = B_work[x, color(y)], and subtract from B_work[y, color(x)]
#                 a_xy = B_work[x, cy] if cx == c1 or cx == c2 else B_work[x, cy]
#                 if reconstruct_sparsely:
#                     rows.append(np.array([x, y], dtype=np.intp))
#                     cols.append(np.array([y, x], dtype=np.intp))
#                     vals.append(np.array([a_xy, a_xy], dtype=dtype))
#                 else:
#                     A_rec[x, y] = a_xy
#                     A_rec[y, x] = a_xy

#                 B_work[y, cx] -= a_xy  # subtract contribution on the opposite column at y

#                 # Remove x (and edge x-y) from the component
#                 deg[x] = 0
#                 deg[y] -= 1
#                 if deg[y] == 1:
#                     q.append(y)

#             if fail_on_cycle:
#                 # If any vertex in component still has deg>0, there was a cycle
#                 if any(deg[vx] > 0 for vx in deg):
#                     raise ValueError(
#                         f"Cycle detected in 2-colored component (colors {c1},{c2}); "
#                         "violates acyclic coloring assumptions."
#                     )

#         # Fast path: use preprocessed components from the binding
#         if acyclic_components is not None and len(acyclic_components.get("edges", [])) > 0:
#             comp_color_pairs = acyclic_components["color_pairs"]
#             comp_edges_list  = acyclic_components["edges"]
#             comp_vertices_list = acyclic_components["vertices"]

#             # Optional: sanity—ensure each component only uses its two colors
#             # (we trust ColPack; skip for speed)

#             for (c1, c2), comp_edges, comp_vertices in zip(
#                 comp_color_pairs, comp_edges_list, comp_vertices_list
#             ):
#                 if not comp_edges:
#                     continue
#                 peel_component(comp_vertices, comp_edges, int(c1), int(c2))

#         else:
#             # Fallback: rebuild per-pair forests on the fly (previous implementation)
#             col_u, col_v = colors[u], colors[v]
#             pair_uv = np.stack([np.minimum(col_u, col_v), np.maximum(col_u, col_v)], axis=1)
#             unique_pairs = np.unique(pair_uv, axis=0)

#             for c1, c2 in unique_pairs:
#                 if c1 == c2:
#                     continue
#                 mask = ((col_u == c1) & (col_v == c2)) | ((col_u == c2) & (col_v == c1))
#                 e_u = u[mask]; e_v = v[mask]
#                 if e_u.size == 0:
#                     continue
#                 comp_edges = list(map(tuple, np.stack([e_u, e_v], axis=1)))
#                 comp_vertices = np.unique(np.concatenate([e_u, e_v])).tolist()
#                 peel_component(comp_vertices, comp_edges, int(c1), int(c2))

#         if reconstruct_sparsely:
#             rows = np.concatenate(rows)
#             cols = np.concatenate(cols)
#             vals = np.concatenate(vals)
#             A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

#         if verify:
#             # B_hat[:, c] = sum_{j: color[j]==c} A_rec[:, j]
#             B_hat = np.zeros_like(B, dtype=dtype)
#             for c in range(C):
#                 cols_c = (colors == c)
#                 if np.any(cols_c):
#                     col_sum = A_rec[:, cols_c].sum(axis=1)
#                     # Handle sparse matrix result (returns matrix) vs dense (returns array)
#                     B_hat[:, c] = np.asarray(col_sum).ravel()
#             err_inf = np.max(np.abs(B_hat - B))
#             if not np.isfinite(err_inf) or err_inf > 1e-4 * (np.linalg.norm(B, ord=np.inf) + 1e-12):
#                 raise AssertionError(f"Verification failed: ||A S - B||_inf = {err_inf:.3e}")

#         return A_rec.astype(dtype, copy=False)

#     else:
#         raise ValueError(f"Unknown recover_from: {recover_from}")



import numpy as np
import scipy.sparse as sp
from collections import deque

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

def _is_torch_tensor(x):
    return _HAS_TORCH and isinstance(x, torch.Tensor)

def recover_matrix(
    recover_from,
    edges,
    n,
    Y,
    vertex_colors,
    *,
    reconstruct_sparsely=False,
    dtype=np.float32,
    fail_on_cycle=True,
    verify=False,
    acyclic_components=None,   # dict from your C++ binding
    adjacency=None             # optional prebuilt adjacency (list-of-lists), else built from edges
):
    """
    Parameters
    ----------
    recover_from : {'DISTANCE_TWO','STAR','ACYCLIC','ACYCLIC_INDIRECT','ACYCLIC_FOR_INDIRECT_RECOVERY'}
    edges : (m,2) int array
        Undirected edges, each listed once (order u<v not required).
    n : int
    Y : (n, C) array
        Y = A @ S where S[:, c] = 1{color==c}.  If Y is torch.Tensor, PyTorch backend is used.
    vertex_colors : (n,) int array
        0-based color id per vertex.
    reconstruct_sparsely : bool
        NumPy backend: build via COO pieces and return CSR (or dense if False).
        PyTorch backend: ignored (returns dense torch.Tensor).
    dtype : numpy dtype (NumPy backend). For torch, dtype is inferred from Y.
    fail_on_cycle : bool
        If True, raise if a two-colored subgraph contains a cycle (acyclic methods).
    verify : bool
        If True, check that A_rec @ S == Y (∞-norm small).
    acyclic_components : dict or None
        Keys: "color_pairs": [(c1,c2), ...],
              "edges": [[(u,v), ...], ...],
              "vertices": [[u,v,...], ...]
    adjacency : list[list[int]] or None
        Optional global adjacency (0..n-1 -> neighbors). If None, built from edges.

    Returns
    -------
    A_rec : (n, n)
        NumPy backend: np.ndarray or scipy.sparse.csr_matrix (if reconstruct_sparsely=True)
        PyTorch backend: torch.Tensor (dense)
    """

    use_torch = _is_torch_tensor(Y)

    # --- canonicalize graph/color data once (NumPy for structure) ---
    edges_np = np.asarray(edges, dtype=np.intp)
    edges_np = edges_np[edges_np[:, 0] != edges_np[:, 1]]
    edges_np = np.sort(edges_np, axis=1)
    edges_np = np.unique(edges_np, axis=0)
    u = edges_np[:, 0] if edges_np.size else np.array([], dtype=np.intp)
    v = edges_np[:, 1] if edges_np.size else np.array([], dtype=np.intp)
    colors_np = np.asarray(vertex_colors, dtype=np.intp)
    ii = np.arange(n, dtype=np.intp)
    C = int(colors_np.max()) + 1 if colors_np.size else 0

    # -------------------------- PyTorch backend --------------------------
    if use_torch:
        device = Y.device
        t_dtype = Y.dtype
        colors = torch.as_tensor(colors_np, device=device, dtype=torch.long)

        def _verify(A):
            if not verify:
                return
            S = torch.zeros((n, C), dtype=t_dtype, device=device)
            S[torch.arange(n, device=device), colors] = 1
            err_inf = torch.max(torch.abs(A @ S - Y)).item()
            if not np.isfinite(err_inf) or err_inf > 1e-4 * (float(torch.linalg.vector_norm(Y, ord=float('inf'))) + 1e-12):
                raise AssertionError(f"Verification failed: ||A S - Y||_inf = {err_inf:.3e}")

        if recover_from == 'DISTANCE_TWO':
            A_rec = torch.zeros((n, n), dtype=t_dtype, device=device)
            if n > 0 and C > 0:
                ar = torch.arange(n, device=device)
                A_rec[ar, ar] = Y[ar, colors]
                if u.size:
                    u_t = torch.as_tensor(u, device=device)
                    v_t = torch.as_tensor(v, device=device)
                    A_rec[u_t, v_t] = Y[u_t, colors[v_t]]
                    A_rec[v_t, u_t] = Y[v_t, colors[u_t]]
            _verify(A_rec)
            return A_rec

        elif recover_from == 'STAR':
            A_rec = torch.zeros((n, n), dtype=t_dtype, device=device)
            if n > 0 and C > 0:
                ar = torch.arange(n, device=device)
                A_rec[ar, ar] = Y[ar, colors]
                if u.size:
                    u_t = torch.as_tensor(u, device=device)
                    v_t = torch.as_tensor(v, device=device)
                    col_u = colors[u_t]
                    col_v = colors[v_t]
                    col_stat = torch.zeros((n, C), dtype=torch.int32, device=device)
                    col_stat.index_put_((u_t, col_v), torch.ones_like(u_t, dtype=torch.int32), accumulate=True)
                    col_stat.index_put_((v_t, col_u), torch.ones_like(v_t, dtype=torch.int32), accumulate=True)
                    unique_uv = (col_stat[u_t, col_v] == 1)
                    val_uv = torch.where(unique_uv, Y[u_t, col_v], Y[v_t, col_u])
                    unique_vu = (col_stat[v_t, col_u] == 1)
                    val_vu = torch.where(unique_vu, Y[v_t, col_u], Y[u_t, col_v])
                    A_rec[u_t, v_t] = val_uv
                    A_rec[v_t, u_t] = val_vu
            _verify(A_rec)
            return A_rec

        elif recover_from in ('ACYCLIC', 'ACYCLIC_INDIRECT', 'ACYCLIC_FOR_INDIRECT_RECOVERY'):
            if C <= 1:
                A_rec = torch.zeros((n, n), dtype=t_dtype, device=device)
                if C == 1 and n > 0:
                    ar = torch.arange(n, device=device)
                    A_rec[ar, ar] = Y[ar, colors]
                return A_rec

            Y_work = Y.clone()     # keep it differentiable
            A_rec = torch.zeros((n, n), dtype=t_dtype, device=device)
            if n > 0:
                ar = torch.arange(n, device=device)
                A_rec[ar, ar] = Y_work[ar, colors]

            def peel_component(comp_vertices, comp_edges, c1, c2):
                deg = {int(x): 0 for x in comp_vertices}
                adj = {int(x): [] for x in comp_vertices}
                for (a, b) in comp_edges:
                    a = int(a); b = int(b)
                    deg[a] += 1; deg[b] += 1
                    adj[a].append(b); adj[b].append(a)

                q = deque([x for x in deg if deg[x] == 1])

                while q:
                    x = q.popleft()
                    if deg[x] != 1:
                        continue
                    y = next((nb for nb in adj[x] if deg.get(nb, 0) > 0), None)
                    if y is None:
                        deg[x] = 0; continue

                    cx = int(colors_np[x]); cy = int(colors_np[y])
                    a_xy = Y_work[x, cy]
                    A_rec[x, y] = a_xy
                    A_rec[y, x] = a_xy
                    Y_work[y, cx] = Y_work[y, cx] - a_xy

                    deg[x] = 0
                    deg[y] -= 1
                    if deg[y] == 1:
                        q.append(y)

                if fail_on_cycle and any(deg[vx] > 0 for vx in deg):
                    raise ValueError(f"Cycle detected in 2-colored component (colors {c1},{c2})")

            if acyclic_components is not None and len(acyclic_components.get("edges", [])) > 0:
                for (c1, c2), comp_edges, comp_vertices in zip(
                    acyclic_components["color_pairs"],
                    acyclic_components["edges"],
                    acyclic_components["vertices"]
                ):
                    if comp_edges:
                        peel_component(comp_vertices, comp_edges, int(c1), int(c2))
            else:
                if u.size:
                    col_u = colors_np[u]; col_v = colors_np[v]
                    pair_uv = np.stack([np.minimum(col_u, col_v), np.maximum(col_u, col_v)], axis=1)
                    unique_pairs = np.unique(pair_uv, axis=0)
                    for c1, c2 in unique_pairs:
                        if c1 == c2: continue
                        mask = ((col_u == c1) & (col_v == c2)) | ((col_u == c2) & (col_v == c1))
                        e_u = u[mask]; e_v = v[mask]
                        if e_u.size == 0: continue
                        comp_edges = list(map(tuple, np.stack([e_u, e_v], axis=1)))
                        comp_vertices = np.unique(np.concatenate([e_u, e_v])).tolist()
                        peel_component(comp_vertices, comp_edges, int(c1), int(c2))

            _verify(A_rec)
            return A_rec

        else:
            raise ValueError(f"Unknown recover_from: {recover_from}")

    # -------------------------- NumPy / SciPy backend --------------------------
    Y = np.asarray(Y)
    edges = edges_np
    colors = colors_np
    ii = np.arange(n, dtype=np.intp)
    C = int(colors.max()) + 1 if colors.size else 0
    u = edges[:, 0] if edges.size else np.array([], dtype=np.intp)
    v = edges[:, 1] if edges.size else np.array([], dtype=np.intp)

    if recover_from == 'DISTANCE_TWO':
        if reconstruct_sparsely:
            rows = np.concatenate([ii, u, v]) if u.size else ii
            cols = np.concatenate([ii, v, u]) if u.size else ii
            vals = [Y[ii, colors]]
            if u.size:
                vals += [Y[u, colors[v]], Y[v, colors[u]]]
            vals = np.concatenate(vals)
            A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
            return A_rec.astype(dtype, copy=False)
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            if n > 0:
                A_rec[ii, ii] = Y[ii, colors]
                if u.size:
                    A_rec[u, v] = Y[u, colors[v]]
                    A_rec[v, u] = Y[v, colors[u]]
            return A_rec.astype(dtype, copy=False)

    elif recover_from == 'STAR':
        col_stat = np.zeros((n, C), dtype=np.int32)
        if u.size:
            np.add.at(col_stat, (u, colors[v]), 1)
            np.add.at(col_stat, (v, colors[u]), 1)

        diag_vals = Y[ii, colors]
        if u.size:
            col_u, col_v = colors[u], colors[v]
            unique_uv = (col_stat[u, col_v] == 1)
            val_uv = np.where(unique_uv, Y[u, col_v], Y[v, col_u])
            unique_vu = (col_stat[v, col_u] == 1)
            val_vu = np.where(unique_vu, Y[v, col_u], Y[u, col_v])

        if reconstruct_sparsely:
            rows = [ii]; cols = [ii]; vals = [diag_vals]
            if u.size:
                rows += [u, v]; cols += [v, u]; vals += [val_uv, val_vu]
            rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
            A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
            return A_rec.astype(dtype, copy=False)
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            if n > 0:
                A_rec[ii, ii] = diag_vals
                if u.size:
                    A_rec[u, v] = val_uv
                    A_rec[v, u] = val_vu
            return A_rec.astype(dtype, copy=False)

    elif recover_from in ('ACYCLIC', 'ACYCLIC_INDIRECT', 'ACYCLIC_FOR_INDIRECT_RECOVERY'):
        if C <= 1:
            A_rec = np.zeros((n, n), dtype=dtype)
            if C == 1 and n > 0:
                A_rec[ii, ii] = Y[ii, colors]
            return A_rec

        Y_work = Y.astype(dtype, copy=True)

        if reconstruct_sparsely:
            rows = [ii]; cols = [ii]; vals = [Y[ii, colors].astype(dtype, copy=False)]
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            if n > 0:
                A_rec[ii, ii] = Y[ii, colors]

        def peel_component(comp_vertices, comp_edges, c1, c2):
            deg = {int(x): 0 for x in comp_vertices}
            adj = {int(x): [] for x in comp_vertices}
            for (a, b) in comp_edges:
                a = int(a); b = int(b)
                deg[a] += 1; deg[b] += 1
                adj[a].append(b); adj[b].append(a)

            q = deque([x for x in deg if deg[x] == 1])

            while q:
                x = q.popleft()
                if deg[x] != 1:
                    continue
                y = next((nb for nb in adj[x] if deg.get(nb, 0) > 0), None)
                if y is None:
                    deg[x] = 0; continue

                cx, cy = colors[x], colors[y]
                a_xy = Y_work[x, cy]
                if reconstruct_sparsely:
                    rows.append(np.array([x, y], dtype=np.intp))
                    cols.append(np.array([y, x], dtype=np.intp))
                    vals.append(np.array([a_xy, a_xy], dtype=dtype))
                else:
                    A_rec[x, y] = a_xy
                    A_rec[y, x] = a_xy

                Y_work[y, cx] -= a_xy

                deg[x] = 0
                deg[y] -= 1
                if deg[y] == 1:
                    q.append(y)

            if fail_on_cycle and any(deg[vx] > 0 for vx in deg):
                raise ValueError(f"Cycle detected in 2-colored component (colors {c1},{c2})")

        if acyclic_components is not None and len(acyclic_components.get("edges", [])) > 0:
            for (c1, c2), comp_edges, comp_vertices in zip(
                acyclic_components["color_pairs"],
                acyclic_components["edges"],
                acyclic_components["vertices"]
            ):
                if comp_edges:
                    peel_component(comp_vertices, comp_edges, int(c1), int(c2))
        else:
            if u.size:
                col_u, col_v = colors[u], colors[v]
                pair_uv = np.stack([np.minimum(col_u, col_v), np.maximum(col_u, col_v)], axis=1)
                unique_pairs = np.unique(pair_uv, axis=0)
                for c1, c2 in unique_pairs:
                    if c1 == c2: continue
                    mask = ((col_u == c1) & (col_v == c2)) | ((col_u == c2) & (col_v == c1))
                    e_u = u[mask]; e_v = v[mask]
                    if e_u.size == 0: continue
                    comp_edges = list(map(tuple, np.stack([e_u, e_v], axis=1)))
                    comp_vertices = np.unique(np.concatenate([e_u, e_v])).tolist()
                    peel_component(comp_vertices, comp_edges, int(c1), int(c2))

        if reconstruct_sparsely:
            rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
            A_rec = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

        if verify:
            Y_hat = np.zeros_like(Y, dtype=dtype)
            for c in range(C):
                cols_c = (colors == c)
                if np.any(cols_c):
                    col_sum = A_rec[:, cols_c].sum(axis=1)
                    Y_hat[:, c] = np.asarray(col_sum).ravel()
            err_inf = np.max(np.abs(Y_hat - Y))
            if not np.isfinite(err_inf) or err_inf > 1e-4 * (np.linalg.norm(Y, ord=np.inf) + 1e-12):
                raise AssertionError(f"Verification failed: ||A S - Y||_inf = {err_inf:.3e}")

        return A_rec.astype(dtype, copy=False)

    else:
        raise ValueError(f"Unknown recover_from: {recover_from}")
