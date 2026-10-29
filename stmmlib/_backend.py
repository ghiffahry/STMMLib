"""GPU/CPU backend abstraction with auto-detection.

Usage:
    backend = Backend(gpu="auto")  # auto-detect GPU
    backend = Backend(gpu=True)    # force GPU (error if unavailable)
    backend = Backend(gpu=False)   # force CPU

All array operations go through backend.xp (numpy or cupy).
"""

from __future__ import annotations

import warnings
from typing import Literal


class Backend:
    """Array backend that routes computation to CPU or GPU.

    Attributes:
        xp: Array namespace (numpy or cupy).
        sparse: Sparse matrix namespace (scipy.sparse or cupyx.scipy.sparse).
        device: "cpu" or "gpu".
    """

    def __init__(self, gpu: bool | Literal["auto"] = "auto") -> None:
        self.xp = None
        self.sparse = None
        self.device: Literal["cpu", "gpu"] = "cpu"

        use_gpu = self._resolve_gpu(gpu)

        if use_gpu:
            try:
                import cupy as cp  # type: ignore[import-untyped]
                import cupyx.scipy.sparse as csp  # type: ignore[import-untyped]

                cp.zeros(1, dtype=cp.float64)  # test allocation
                self.xp = cp  # type: ignore[assignment]
                self.sparse = csp  # type: ignore[assignment]
                self.device = "gpu"
                return
            except Exception as e:
                if gpu is True:
                    raise RuntimeError(f"GPU requested but unavailable: {e}") from e
                warnings.warn(
                    f"GPU unavailable ({e}), falling back to CPU.",
                    stacklevel=2,
                )

        import numpy as np
        import scipy.sparse as sp

        self.xp = np  # type: ignore[assignment]
        self.sparse = sp  # type: ignore[assignment]
        self.device = "cpu"

    @staticmethod
    def _resolve_gpu(gpu: bool | Literal["auto"]) -> bool:
        if isinstance(gpu, str):
            g = gpu.strip().lower()
            if g == "auto":
                try:
                    import cupy as cp  # type: ignore[import-untyped]

                    cp.zeros(1, dtype=cp.float64)
                    return True
                except Exception:
                    return False
            if g in ("no", "false", "0", "off", "cpu"):
                return False
            if g in ("yes", "true", "1", "on", "gpu"):
                return True
        if gpu == "auto":
            try:
                import cupy as cp  # type: ignore[import-untyped]

                cp.zeros(1, dtype=cp.float64)
                return True
            except Exception:
                return False
        return bool(gpu)

    def to_cpu(self, arr):
        """Move array to CPU. No-op if already on CPU."""
        if self.device == "gpu":
            return self.xp.asnumpy(arr)
        return arr

    def to_gpu(self, arr):
        """Move array to GPU. No-op if already on CPU."""
        if self.device == "gpu":
            return self.xp.asarray(arr)
        return arr

    def is_sparse(self, m) -> bool:
        """Check if matrix is a sparse matrix."""
        if self.device == "gpu":
            return self.sparse.issparse(m)
        import scipy.sparse as sp

        return sp.issparse(m)

    def to_sparse(self, m):
        """Convert dense array to sparse CSR."""
        if self.is_sparse(m):
            return m
        if self.device == "gpu":
            return self.sparse.csr_matrix(m)
        import scipy.sparse as sp

        return sp.csr_matrix(m)

    def sparse_kron(self, A, B, format="csr"):
        """Kronecker product of two (sparse) matrices."""
        return self.sparse.kron(A, B, format=format)

    def sparse_block_diag(self, mats):
        """Block-diagonal from list of matrices."""
        if self.device == "gpu":
            return self.sparse.block_diag(mats, format="csr")
        import scipy.sparse as sp

        return sp.block_diag(mats, format="csr")

    def sparse_eye(self, n, dtype=None):
        """Identity matrix (sparse)."""
        if dtype is None:
            dtype = self.xp.float64
        return self.sparse.eye(n, dtype=dtype, format="csr")

    def solve(self, A, b):
        """Solve linear system A @ x = b."""
        return self.xp.linalg.solve(A, b)

    def cholesky(self, A):
        """Cholesky decomposition. Returns lower-triangular L where A = L @ L.T."""
        return self.xp.linalg.cholesky(A)

    def slogdet(self, A):
        """Log-determinant via sign and log-abs-determinant."""
        return self.xp.linalg.slogdet(A)

    def eigh(self, A):
        """Eigenvalues and eigenvectors of symmetric matrix."""
        return self.xp.linalg.eigh(A)

    def pinv(self, A):
        """Moore-Penrose pseudo-inverse."""
        return self.xp.linalg.pinv(A)

    def __repr__(self) -> str:
        return f"Backend(device={self.device!r})"
