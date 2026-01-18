###############################################################
### Multidimensional Scaling in Gini Pseudo Metric Spaces #####
###############################################################

import math
import numpy as np
import torch


def _as_device(t, ref):
    return t.to(device=ref.device, dtype=ref.dtype)

def _pairwise_diffs(X): # (n, n, d)
    return X[:, None, :] - X[None, :, :]

def _eye(n, like):
    return torch.eye(n, dtype=torch.float64, device=like.device)

def _ones(n, like):
    return torch.ones((n, n), dtype=torch.float64, device=like.device)

@torch.no_grad()
def rank_desc_average(diffs, atol: float = 0.0):
    """
    Descending ranks with average tie handling along the last dim
    """
    *batch, d = diffs.shape
    sorted_vals, order = torch.sort(diffs, dim=-1, descending=True)  # (..., d)
    pos = torch.arange(d, device=diffs.device, dtype=torch.int64)
    pos = pos.view(*(1,) * len(batch), d).expand(*batch, d)
    if atol > 0.0:
        eq_prev = torch.isclose(sorted_vals[..., 1:], sorted_vals[..., :-1], atol=atol, rtol=0.0)
    else:
        eq_prev = (sorted_vals[..., 1:] == sorted_vals[..., :-1])
    start_idx = torch.empty_like(pos)
    start_idx[..., 0] = 0
    for t in range(1, d):
        start_idx[..., t] = torch.where(eq_prev[..., t - 1], start_idx[..., t - 1], pos[..., t])
    end_idx = torch.empty_like(pos)
    end_idx[..., -1] = d - 1
    for t in range(d - 2, -1, -1):
        end_idx[..., t] = torch.where(eq_prev[..., t], end_idx[..., t + 1], pos[..., t])
    avg_rank_sorted = (start_idx + end_idx).to(diffs.dtype) / 2.0 + 1.0
    ranks = torch.empty_like(avg_rank_sorted)
    ranks.scatter_(-1, order, avg_rank_sorted)
    return ranks

def _double_center(D2):
    """Return centered Gram B and centering matrix J for D^2"""
    n = D2.shape[0]
    I = _eye(n, D2)
    J = I - _ones(n, D2) / n
    B = -0.5 * (J @ D2 @ J)
    return B, J

def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


class GiniMDS:
    """
    MDS with Gini pseudo-distance and GPU usage

    Parameters
    ----------
    n_components : output embedding dim
    nu           : Gini hyperparameter (must be > 1)
    gini_mode    : 'rank' 
    atol         : tie tolerance for 'rank' mode
    mds_method   : 'cmds' | 'smacof' | 'sammon' | 'huber'
    max_iter     : iterations for iterative MDS (ignored for 'cmds')
    tol          : tolerance on relative stress decrease
    device       : 'cpu' or 'cuda' (if available)
    dtype        : torch dtype to use internally (float64 recommended)
    """

    def __init__(
        self,
        n_components=2,
        nu=2.0,
        gini_mode='rank',
        atol=0.0,
        mds_method='cmds',
        max_iter=300,
        tol=1e-6,
        device=None,
        dtype=torch.float64,
        random_state=None,
        row_center=False, 
    ):
        if nu <= 1:
            raise ValueError("nu must be > 1")
        self.row_center = row_center
        self.n_components = int(n_components)
        self.nu = float(nu)
        self.gini_mode = gini_mode
        self.atol = float(atol)
        self.mds_method = mds_method
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.dtype = dtype
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.random_state = random_state

        # Fitted attributes
        self.embedding_ = None   # (n_train, m)
        self.train_D_ = None     # (n_train, n_train)
        self.V_ = None           # eigenvectors (cmds)
        self.Lam_ = None         # eigenvalues (cmds, >0)
        self.J_ = None           # centering matrix (cmds)
        self.D2_means_ = None    # (grand_mean, row_means) for out-of-sample
        self.train_X_ = None     # original features (if fit from X)
        self._rng = torch.Generator()
        if random_state is not None:
            self._rng.manual_seed(int(random_state))

    @torch.no_grad()
    def _gini_distance_matrix(self, X):
        """
        Compute pairwise Gini distances D (n×n) from X (n×p) using selected mode.
        """
        X = X.to(self.device, dtype=self.dtype)
        if self.row_center:
            X = X - X.mean(dim=1, keepdim=True) # if centering => Gini distance (no pseu-distance)
        n, d = X.shape
        diffs = _pairwise_diffs(X)  # (n, n, d)
        ranks = rank_desc_average(diffs, atol=self.atol)  # (n, n, d)
        Fbar = ranks / float(d)                  # (n, n, d)
        F0 = (d + 1.0) / (2.0 * d)               # scalar = mean rank
        pow_ = self.nu - 1.0

        inner = torch.sum(diffs * (Fbar.pow(pow_) - (F0 ** pow_)), dim=-1)  # (n, n)
        D = (-d) * inner
        D = 0.5 * (D + D.T)
        D.fill_diagonal_(0.0)
        D = torch.clamp(D, min=0.0)
        return D

    @torch.no_grad()
    def _classical_mds(self, D, out_dims=None):
        """
        Classical MDS (Torgerson). Returns (X, V, Lam, J, (grand_mean, row_means))
        where B = V diag(Lam) V^T and X = V Lam^{1/2}.
        Also stores means for out-of-sample Nyström.
        """
        if out_dims is None:
            out_dims = self.n_components
        D = D.to(self.device, dtype=torch.float64)
        n = D.shape[0]
        D2 = D ** 2
        # means for Nyström
        row_means = D2.mean(dim=1)                     
        grand_mean = float(D2.mean().item())
        B, J = _double_center(D2)
        evals, evecs = torch.linalg.eigh(B)            # ascending
        pos = evals > 1e-12
        if not torch.any(pos):
            X = torch.zeros((n, min(out_dims, max(1, n - 1))), dtype=torch.float64, device=self.device)
            return X.cpu().numpy(), evecs[:, :0], evals[:0], J, (grand_mean, row_means)

        evals = evals[pos]
        evecs = evecs[:, pos]
        idx = torch.argsort(evals, descending=True)
        evals = evals[idx]
        evecs = evecs[:, idx]

        m = int(min(out_dims, int((evals > 1e-12).sum().item()), max(1, n - 1)))
        Lam = evals[:m]
        V = evecs[:, :m]
        X = V * torch.sqrt(Lam).unsqueeze(0)
        return X.detach().cpu().numpy().astype(np.float64), V, Lam, J, (grand_mean, row_means)

    def _euclid_from_Z(self, Z):
        diff = Z[:, None, :] - Z[None, :, :]
        return torch.sqrt(torch.clamp((diff ** 2).sum(dim=-1), min=0.0))

    def _init_from_cmds(self, D):
        X_np, *_ = self._classical_mds(D, out_dims=self.n_components)
        return torch.tensor(X_np, device=self.device, dtype=self.dtype)

    def _stress1(self, D, Z, eps: float = 1e-12):
        Dhat = self._euclid_from_Z(Z)
        num = ((D - Dhat) ** 2).sum()
        den = (D ** 2).sum() + eps
        return torch.sqrt(torch.clamp(num / den, min=eps))

    def _sammon_loss(self, D, Z, eps: float = 1e-8):
        Dhat = self._euclid_from_Z(Z) + eps
        w = 1.0 / (D + eps)
        num = (w * (D - Dhat) ** 2).sum()
        den = w.sum() + eps
        return torch.clamp(num / den, min=eps)

    def _huber_stress(self, D, Z, delta: float = 1.0, eps: float = 1e-12):
        Dhat = self._euclid_from_Z(Z)
        R = D - Dhat
        absR = torch.abs(R)
        quad = 0.5 * (absR <= delta) * (R ** 2)
        lin  = (absR >  delta) * (delta * (absR - 0.5 * delta))
        num = (quad + lin).sum()
        den = (D ** 2).sum() + eps
        return torch.sqrt(torch.clamp(num / den, min=eps))

    def _iterative_mds(self, D, method: str):
        D = D.to(self.device, dtype=self.dtype)
        Z0 = self._init_from_cmds(D)
        with torch.no_grad():
            base = (self._stress1 if method == "smacof"
                    else self._sammon_loss if method == "sammon"
                    else self._huber_stress)
            best_loss = float(base(D, Z0).item())
            best_Z = Z0.detach().clone()

        with torch.enable_grad():
            Z = Z0.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([Z], lr=0.05)
            for _ in range(self.max_iter):
                opt.zero_grad(set_to_none=True)
                loss = (self._stress1(D, Z) if method == "smacof"
                        else self._sammon_loss(D, Z) if method == "sammon"
                        else self._huber_stress(D, Z))
                if not torch.isfinite(loss): break
                loss.backward()
                torch.nn.utils.clip_grad_norm_([Z], max_norm=1000.0)
                opt.step()
                val = float(loss.item())
                if val < best_loss and math.isfinite(val):
                    best_loss = val
                    best_Z = Z.detach().clone()
        return best_Z.detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def fit(self, X=None, D=None):
        """
        - Fit if X is provided: compute Gini distance then embed
        - Fit if D is provided: embed the distance matrix directly
        """
        if (X is None) == (D is None):
            raise ValueError("Provide exactly X or Distances")
        if X is not None:
            X_t = torch.as_tensor(X, device=self.device, dtype=self.dtype)
            self.train_X_ = X_t.clone()
            D_t = self._gini_distance_matrix(X_t)
        else:
            D_t = torch.as_tensor(D, device=self.device, dtype=self.dtype)
        self.train_D_ = D_t.detach().cpu().numpy().astype(np.float64)
        if self.mds_method == 'cmds':
            X_np, V, Lam, J, means = self._classical_mds(D_t, out_dims=self.n_components)
            self.embedding_ = X_np
            self.V_, self.Lam_, self.J_ = V, Lam, J
            self.D2_means_ = means
        else:
            self.embedding_ = self._iterative_mds(D_t, method=self.mds_method)
        return self

    @torch.no_grad()
    def fit_transform(self, X=None, D=None):
        return self.fit(X=X, D=D).embedding_

    @torch.no_grad()
    def fit_inverse_transform(
        self,
        D_test_train,
        D_train_train=None,
    ):
        """
        Given distances from test points to the training set, return test embeddings.
        If the model was fit with 'cmds', uses exact Nyström extension:
            x* = Λ^{-1/2} V^T b*,  where
            b* = -0.5 * J (d_*^2 - row_means - col_means + grand_mean).
        Here, 'col_means' for a single test point is its mean squared distance to training.

        If the model was fit with an iterative method, falls back to per-point stress
        minimization (keeps training embedding fixed), initialized with the Nyström guess
        computed from a one-shot CMDs on the joint (train+one test) distances if
        `D_train_train` is given; otherwise uses the train centroid.

        Inputs
        ------
        D_test_train : (n_test, n_train) distances between each test point and all training points
        D_train_train: (n_train, n_train), required only to get a stronger initialization for
                       iterative solvers when CMDs attributes are not available.
        """
        if self.embedding_ is None or self.train_D_ is None:
            raise RuntimeError("Call fit before fit_inverse_transform")

        Dt = torch.as_tensor(D_test_train, device=self.device, dtype=torch.float64)  # (t, n)
        t, n = Dt.shape
        if n != self.train_D_.shape[0]:
            raise ValueError("D_test_train must have n_train columns equal to the fitted training size")

        # Classical MDS: exact Nyström
        if self.mds_method == 'cmds' and self.V_ is not None and self.Lam_ is not None and self.J_ is not None and self.D2_means_ is not None:
            V = self.V_
            Lam = self.Lam_
            J = self.J_
            grand_mean, row_means = self.D2_means_
            D2_star = Dt ** 2                                 
            col_means = D2_star.mean(dim=1, keepdim=True)     
            row_means_vec = row_means.view(1, n)              
            centered = D2_star - row_means_vec - col_means + grand_mean
            b_star = -0.5 * (centered @ J)                    
            Lam_inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(Lam))
            X_star = (b_star @ V) @ Lam_inv_sqrt              
            return X_star.detach().cpu().numpy().astype(np.float64)

        # Iterative methods: optimize each test point against fixed training embedding
        Z_train = torch.as_tensor(self.embedding_, device=self.device, dtype=self.dtype)  # (n, m)
        m = Z_train.shape[1]
        X_out = torch.empty((t, m), dtype=self.dtype, device=self.device)
        init_centroid = Z_train.mean(dim=0, keepdim=True)     

        for i in range(t):
            d_i = Dt[i]                                      
            z = init_centroid.clone().requires_grad_(True)
            opt = torch.optim.LBFGS([z], max_iter=20, line_search_fn='strong_wolfe')
            def closure():
                opt.zero_grad(set_to_none=True)
                # distances from z to train
                diff = z - Z_train                            # (n, m)
                d_hat = torch.sqrt(torch.clamp((diff ** 2).sum(dim=1), min=0.0))  # (n,)
                if self.mds_method == 'sammon':
                    w = 1.0 / (d_i + 1e-12)
                    loss = ((w * (d_i - d_hat) ** 2).sum()) / (w.sum())
                elif self.mds_method == 'huber':
                    R = d_i - d_hat
                    absR = torch.abs(R)
                    delta = 1.0
                    quad = 0.5 * (absR <= delta) * (R ** 2)
                    lin = (absR > delta) * (delta * (absR - 0.5 * delta))
                    loss = (quad + lin).sum() / (torch.sum(d_i ** 2) + 1e-12)
                else:  
                    loss = torch.sqrt((((d_i - d_hat) ** 2).sum()) / (torch.sum(d_i ** 2) + 1e-12))
                loss.backward()
                return loss
            prev = float('inf')
            for _ in range(self.max_iter):
                loss = float(opt.step(closure).item())
                if (prev - loss) / (abs(prev) + 1e-12) < self.tol:
                    break
                prev = loss

            with torch.no_grad():
                X_out[i] = z.detach()

        return X_out.detach().cpu().numpy().astype(np.float64)


    @torch.no_grad()
    def gini_distances(self, X):
        X_t = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        D = self._gini_distance_matrix(X_t)
        return D.detach().cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def stress(self, D, Z, kind: str = 'stress1'):
        """Compute stress between distance matrix D and embedding Z"""
        D_t = torch.as_tensor(D, device=self.device, dtype=self.dtype)
        Z_t = torch.as_tensor(Z, device=self.device, dtype=self.dtype)
        if kind == 'stress1':
            val = self._stress1(D_t, Z_t)
        elif kind == 'sammon':
            val = self._sammon_loss(D_t, Z_t)
        elif kind == 'huber':
            val = self._huber_stress(D_t, Z_t)
        else:
            raise ValueError("kind must be 'stress1', 'sammon', or 'huber'")
        return float(val.item())
