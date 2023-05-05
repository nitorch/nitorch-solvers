__all__ = ['MultiGrid', 'MultiGridCG']

import torch
from .pyramid import Prolong, Restrict
from .cg import ConjugateGradient, PrepCG


class MultiGridCG:
    """
    Multi-grid solver for linear systems of the form `(H+L)*x = b`, where
    `L` has a multi-resolution interpretation (e.g., spatial regularizer)
    """

    def __init__(self, nb_cycles=2, max_iter=2, tol=0,
                 prolong=None, restrict=None):
        """

        Parameters
        ----------
        nb_cycles : int
            Number of W cycles
        """
        self.nb_cycles = nb_cycles
        self.solver = ConjugateGradient(max_iter=max_iter, tol=tol)
        self.prolong = prolong
        self.restrict = restrict

    def __call__(self, *args, **kwargs):
        return self.solve(*args, **kwargs)

    def solve(self, target, forward, precond=None, init=None, inplace=False):
        """

        Parameters
        ----------
        target : list[tensor]
        forward : list[callable(tensor) -> tensor]
        precond : list[callable(tensor) -> tensor], optional
        init : tensor, optional
        inplace : bool, default=False

        Returns
        -------
        solution : tensor

        """
        solve_ = self.solver.solve_
        prolong = self.prolong or Prolong(target[0].ndim)
        restrict = self.restrict or Restrict(target[0].ndim)

        def residuals(A, x, g, r):
            return torch.sub(A(x), g, out=r)

        g = target
        A = forward
        P = precond or ([torch.clone] * len(g))
        x = (g[0].clone() if init is None else init.clone() if not inplace else init)
        N = len(target)

        x = [x] + list(map(torch.empty_like, g[1:]))
        for n in range(N-1):
            restrict(x[n], x[n+1])
        r = list(map(torch.empty_like, g))

        # initial solve at coarsest resolution
        solve_(x[-1], g[-1], A[-1], P[-1])

        for n_base in reversed(range(N-1)):
            prolong(x[n_base+1], x[n_base])

            for n_cycle in range(self.nb_cycles):
                for n in range(n_base, N-1):
                    solve_(x[n], g[n], A[n], P[n])
                    residuals(A[n], x[n], g[n], r[n])
                    restrict(r[n], g[n+1])
                    x[n+1].zero_()

                solve_(x[-1], g[-1], A[-1], P[-1])

                for n in reversed(range(n_base, N-1)):
                    prolong(x[n+1], r[n])
                    x[n] += r[n]
                    solve_(x[n], g[n], A[n], P[n])

        return x[0]

    def solve_(self, init, target, forward, precond=None):
        return self.solve(target, forward, precond, init, inplace=True)

    def advanced_solve_(self, init, target, forward, precond=None):
        """

        Parameters
        ----------
        init : tensor
        target : list[tensor]
        forward : list[callable(tensor) -> tensor]
        precond : list[callable(tensor) -> tensor], optional

        Returns
        -------
        solution : tensor

        """
        prolong = self.prolong or Prolong(target[0].ndim)
        restrict = self.restrict or Restrict(target[0].ndim)

        def residuals(A, x, g, r):
            return torch.sub(A(x), g, out=r)

        g = target
        A = forward
        P = precond or ([lambda x, out: out.copy_(x)] * len(g))
        x = init
        N = len(target)

        mx = self.solver.max_iter
        tol = self.solver.tol
        solve_ = [PrepCG(g1, A1, P1, mx, tol).advanced_solve_
                  for g1, A1, P1 in zip(g, A, P)]

        x = [x] + list(map(torch.empty_like, g[1:]))
        for n in range(N-1):
            restrict(x[n], x[n+1])
        r = list(map(torch.empty_like, g))

        # initial solve at coarsest resolution
        solve_[-1](x[-1])

        for n_base in reversed(range(N-1)):
            prolong(x[n_base+1], x[n_base])

            for n_cycle in range(self.nb_cycles):
                for n in range(n_base, N-1):
                    solve_[n](x[n])
                    residuals(A[n], x[n], g[n], r[n])
                    restrict(r[n], g[n+1])
                    x[n+1].zero_()

                solve_[-1](x[-1])

                for n in reversed(range(n_base, N-1)):
                    prolong(x[n+1], r[n])
                    x[n] += r[n]
                    solve_[n](x[n])

        return x[0]


class MultiGrid:
    """
    Multi-grid solver for linear systems of the form `(H+L)*x = b`, where
    `L` has a multi-resolution interpretation (e.g., spatial regularizer)
    """

    def __init__(self, nb_cycles=2, prolong=None, restrict=None):
        """

        Parameters
        ----------
        nb_cycles : int
            Number of W cycles
        """
        self.nb_cycles = nb_cycles
        self.prolong = prolong
        self.restrict = restrict

    def solve_(self, init, target, forward, solve_):
        """

        Parameters
        ----------
        init : tensor
        target : list[tensor]
        forward : list[callable(tensor) -> tensor]
        solve_ : list[callable(tensor) -> tensor]

        Returns
        -------
        solution : tensor

        """
        prolong = self.prolong or Prolong(target[0].ndim)
        restrict = self.restrict or Restrict(target[0].ndim)

        def residuals(A, x, g, r):
            return torch.sub(A(x), g, out=r)

        g = target
        A = forward
        x = init
        N = len(target)

        x = [x] + list(map(torch.empty_like, g[1:]))
        for n in range(N-1):
            restrict(x[n], x[n+1])
        r = list(map(torch.empty_like, g))
        for n in range(N):
            if x[0].is_cuda and not x[n].is_cuda:
                x[n] = x[n].pin_memory()
                r[n] = r[n].pin_memory()

        # initial solve at coarsest resolution
        solve_[-1](x[-1])

        for n_base in reversed(range(N-1)):
            prolong(x[n_base+1], x[n_base])

            for n_cycle in range(self.nb_cycles):
                for n in range(n_base, N-1):
                    solve_[n](x[n])
                    residuals(A[n], x[n], g[n], r[n])
                    restrict(r[n], g[n+1])
                    x[n+1].zero_()

                solve_[-1](x[-1])

                for n in reversed(range(n_base, N-1)):
                    prolong(x[n+1], r[n])
                    x[n] += r[n]
                    solve_[n](x[n])

        return x[0]
