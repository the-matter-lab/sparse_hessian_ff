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

            # On CUDA, scalar tensor updates inside Python loops are very expensive.
            # Use a CPU staging buffer for recovery and copy back once at the end.
            use_cpu_fast_path = (device.type == "cuda")
            if use_cpu_fast_path:
                if t_dtype in (torch.float16, torch.float32, torch.bfloat16):
                    np_dtype = np.float32
                else:
                    np_dtype = np.float64
                Y_work_np = Y.detach().cpu().numpy().astype(np_dtype, copy=True)
                A_rec_np = np.zeros((n, n), dtype=np_dtype)
                if n > 0:
                    A_rec_np[ii, ii] = Y_work_np[ii, colors_np]
                op_x = []
                op_y = []
                op_cx = []
                op_cy = []
            else:
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
                next_pos = {x: 0 for x in deg}

                q = deque([x for x in deg if deg[x] == 1])

                while q:
                    x = q.popleft()
                    if deg[x] != 1:
                        continue
                    y = None
                    adj_x = adj[x]
                    p = next_pos[x]
                    while p < len(adj_x):
                        nb = adj_x[p]
                        if deg.get(nb, 0) > 0:
                            y = nb
                            break
                        p += 1
                    next_pos[x] = p
                    if y is None:
                        deg[x] = 0; continue

                    cx = int(colors_np[x]); cy = int(colors_np[y])
                    if use_cpu_fast_path:
                        a_xy = Y_work_np[x, cy]
                        A_rec_np[x, y] = a_xy
                        A_rec_np[y, x] = a_xy
                        Y_work_np[y, cx] -= a_xy
                        op_x.append(x)
                        op_y.append(y)
                        op_cx.append(cx)
                        op_cy.append(cy)
                    else:
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
                # Coalesce cached components by color-pair to reduce Python/autograd
                # overhead from many tiny peel_component calls.
                pair_to_edges = {}
                pair_to_vertices = {}
                for (c1, c2), comp_edges, comp_vertices in zip(
                    acyclic_components["color_pairs"],
                    acyclic_components["edges"],
                    acyclic_components["vertices"]
                ):
                    if not comp_edges:
                        continue
                    c1 = int(c1)
                    c2 = int(c2)
                    if c1 > c2:
                        c1, c2 = c2, c1
                    key = (c1, c2)
                    if key not in pair_to_edges:
                        pair_to_edges[key] = []
                        pair_to_vertices[key] = []
                    pair_to_edges[key].extend(comp_edges)
                    pair_to_vertices[key].extend(comp_vertices)

                for (c1, c2), merged_edges in pair_to_edges.items():
                    if merged_edges:
                        peel_component(pair_to_vertices[(c1, c2)], merged_edges, c1, c2)
            else:
                if u.size:
                    col_u = colors_np[u]
                    col_v = colors_np[v]
                    cmin = np.minimum(col_u, col_v)
                    cmax = np.maximum(col_u, col_v)
                    valid = (cmin != cmax)
                    if np.any(valid):
                        u_valid = u[valid]
                        v_valid = v[valid]
                        cmin_valid = cmin[valid]
                        cmax_valid = cmax[valid]
                        pair_code = cmin_valid * C + cmax_valid
                        order = np.argsort(pair_code, kind='mergesort')

                        start = 0
                        while start < order.size:
                            code = pair_code[order[start]]
                            end = start + 1
                            while end < order.size and pair_code[order[end]] == code:
                                end += 1
                            idx = order[start:end]
                            e_u = u_valid[idx]
                            e_v = v_valid[idx]
                            comp_edges = list(map(tuple, np.stack([e_u, e_v], axis=1)))
                            comp_vertices = np.unique(np.concatenate([e_u, e_v])).tolist()
                            c1 = int(code // C)
                            c2 = int(code % C)
                            peel_component(comp_vertices, comp_edges, c1, c2)
                            start = end

            if use_cpu_fast_path:
                A_fast = torch.as_tensor(A_rec_np, dtype=t_dtype, device=device)
                if Y.requires_grad:
                    colors_t = torch.as_tensor(colors_np, dtype=torch.long)
                    op_x_t = torch.as_tensor(op_x, dtype=torch.long)
                    op_y_t = torch.as_tensor(op_y, dtype=torch.long)
                    op_cx_t = torch.as_tensor(op_cx, dtype=torch.long)
                    op_cy_t = torch.as_tensor(op_cy, dtype=torch.long)

                    class _AcyclicRecoverCpuAutograd(torch.autograd.Function):
                        @staticmethod
                        def forward(ctx, Y_in, colors_in, op_x_in, op_y_in, op_cx_in, op_cy_in, A_in):
                            ctx.save_for_backward(colors_in, op_x_in, op_y_in, op_cx_in, op_cy_in)
                            ctx.n = Y_in.shape[0]
                            ctx.c = Y_in.shape[1]
                            return A_in

                        @staticmethod
                        def backward(ctx, grad_A):
                            colors_in, op_x_in, op_y_in, op_cx_in, op_cy_in = ctx.saved_tensors
                            n = ctx.n
                            c = ctx.c

                            gA = grad_A.detach().cpu().numpy()
                            gY_work = np.zeros((n, c), dtype=gA.dtype)

                            op_x_np = op_x_in.cpu().numpy()
                            op_y_np = op_y_in.cpu().numpy()
                            op_cx_np = op_cx_in.cpu().numpy()
                            op_cy_np = op_cy_in.cpu().numpy()

                            for k in range(op_x_np.shape[0] - 1, -1, -1):
                                xk = int(op_x_np[k])
                                yk = int(op_y_np[k])
                                cxk = int(op_cx_np[k])
                                cyk = int(op_cy_np[k])
                                ga = gA[xk, yk] + gA[yk, xk] - gY_work[yk, cxk]
                                gY_work[xk, cyk] += ga

                            grad_Y = gY_work
                            colors_np_local = colors_in.cpu().numpy()
                            grad_Y[np.arange(n), colors_np_local] += np.diag(gA)

                            grad_Y_t = torch.as_tensor(grad_Y, dtype=grad_A.dtype, device=grad_A.device)
                            return grad_Y_t, None, None, None, None, None, None

                    A_rec = _AcyclicRecoverCpuAutograd.apply(Y, colors_t, op_x_t, op_y_t, op_cx_t, op_cy_t, A_fast)
                else:
                    A_rec = A_fast
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
            A_rec = A_rec.astype(dtype, copy=False)
            return A_rec
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            if n > 0:
                A_rec[ii, ii] = Y[ii, colors]
                if u.size:
                    A_rec[u, v] = Y[u, colors[v]]
                    A_rec[v, u] = Y[v, colors[u]]
            A_rec = A_rec.astype(dtype, copy=False)
            return A_rec

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
            A_rec = A_rec.astype(dtype, copy=False)
            return A_rec
        else:
            A_rec = np.zeros((n, n), dtype=dtype)
            if n > 0:
                A_rec[ii, ii] = diag_vals
                if u.size:
                    A_rec[u, v] = val_uv
                    A_rec[v, u] = val_vu
            A_rec = A_rec.astype(dtype, copy=False)
            return A_rec

    elif recover_from in ('ACYCLIC', 'ACYCLIC_INDIRECT', 'ACYCLIC_FOR_INDIRECT_RECOVERY'):
        if C <= 1:
            A_rec = np.zeros((n, n), dtype=dtype)
            if C == 1 and n > 0:
                A_rec[ii, ii] = Y[ii, colors]
            return A_rec

        Y_work = Y.astype(dtype, copy=True)

        if reconstruct_sparsely:
            diag_vals = Y[ii, colors].astype(dtype, copy=False)
            rows_off = []
            cols_off = []
            vals_off = []
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
                    rows_off.extend((x, y))
                    cols_off.extend((y, x))
                    vals_off.extend((a_xy, a_xy))
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
                cmin = np.minimum(col_u, col_v)
                cmax = np.maximum(col_u, col_v)
                valid = (cmin != cmax)
                if np.any(valid):
                    u_valid = u[valid]
                    v_valid = v[valid]
                    cmin_valid = cmin[valid]
                    cmax_valid = cmax[valid]
                    pair_code = cmin_valid * C + cmax_valid
                    order = np.argsort(pair_code, kind='mergesort')

                    start = 0
                    while start < order.size:
                        code = pair_code[order[start]]
                        end = start + 1
                        while end < order.size and pair_code[order[end]] == code:
                            end += 1
                        idx = order[start:end]
                        e_u = u_valid[idx]
                        e_v = v_valid[idx]
                        comp_edges = list(map(tuple, np.stack([e_u, e_v], axis=1)))
                        comp_vertices = np.unique(np.concatenate([e_u, e_v])).tolist()
                        c1 = int(code // C)
                        c2 = int(code % C)
                        peel_component(comp_vertices, comp_edges, c1, c2)
                        start = end

        if reconstruct_sparsely:
            if rows_off:
                rows = np.concatenate([ii, np.asarray(rows_off, dtype=np.intp)])
                cols = np.concatenate([ii, np.asarray(cols_off, dtype=np.intp)])
                vals = np.concatenate([diag_vals, np.asarray(vals_off, dtype=dtype)])
            else:
                rows = ii
                cols = ii
                vals = diag_vals
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

        A_rec = A_rec.astype(dtype, copy=False)
        return A_rec

    else:
        raise ValueError(f"Unknown recover_from: {recover_from}")
