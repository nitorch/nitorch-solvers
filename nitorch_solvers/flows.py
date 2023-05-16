__all__ = [
    'flow_solve_cg', 'flow_solve_cg_',
    'flow_solve_fmg', 'flow_solve_fmg_',
    'flow_solve_gs', 'flow_solve_gs_',
    'flow_forward', 'flow_penalty',
]
import torch
from torch import Tensor
from typing import Optional, Sequence, TypeVar, Union
from functools import wraps
from jitfields.regularisers import (
    flow_matvec, flow_precond,
    flow_forward, flow_relax_,
)
from jitfields.sym import sym_solve
from .core.utils import expanded_shape, make_vector, ensure_list
from .core.cg import ConjugateGradient
from .core.fmg import MultiGrid
from .core.pyramid import ProlongFlow, RestrictFlow, Restrict

T = TypeVar('T')
OneOrSeveral = Union[T, Sequence[T]]


# Alias for the regulariser's matrix-vector product
flow_penalty = wraps(flow_matvec)(flow_matvec)


def flow_solve_cg(
        hessian: Tensor,
        gradient: Tensor,
        init: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        absolute: float = 0,
        membrane: float = 0,
        bending: float = 0,
        shears: float = 0,
        div: float = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        precond: Optional[str] = 'H+diag(L)',
        max_iter: int = 32,
        tolerance: float = 1e-5,
) -> Tensor:
    r"""Solve a regularized linear system involving flows by conjugate gradient
    
    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.
    
    Parameters
    ----------
    hessian : (..., *spatial, DD) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        DD is one of {1, D, D*(D+1)//2, D*D}
    gradient :  (..., *spatial, D) tensor
        Point at which to solve the system, `g`.
    init : (..., *spatial, D) tensor, optional
        Initial value
    weight : (..., *spatial) tensor, optional
        Voxel-wise weight of the regularization
    absolute : float, optional
        Penalty on absolute values
    membrane : float, optional
        Penalty on first derivatives
    bending : float, optional
        Penalty on second derivatives
    shears : float, optional
        Penalty on shears (symmetric part of the Jacobian)
    div : float, optional
        Penalty on the divergence (trace of the Jacobian)
    voxel_size : [list of] float, optional
        Voxel size
    bound : [list of] {'dft', 'dct1', 'dct2', 'dst1', 'dst2', 'replicate', 'zero'}, optional
         Boundary conditions.
    precond : {'H+diag(L)', 'H', 'L', None}, optional
        Preconditioning.
    max_iter : int
        Maximum number of iterations
    tolerance : float
        Tolerance for early stopping

    Returns
    -------
    solution : (..., *spatial, D) tensor
        Solution of the linear system, `x`.

    """
    ndim = gradient.shape[-1]
    batch_shape, spatial_shape = _guess_shapes(hessian, gradient, weight)
    if init is None:
        init = gradient.new_zeros([*batch_shape, *spatial_shape, ndim])
    else:
        init = init.expand([*batch_shape, *spatial_shape, ndim]).clone()

    return flow_solve_cg_(
        init,
        hessian=hessian,
        gradient=gradient,
        weight=weight,
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        shears=shears,
        div=div,
        voxel_size=voxel_size,
        bound=bound,
        precond=precond,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def flow_solve_cg_(
        init: Tensor,
        hessian: Tensor,
        gradient: Tensor,
        weight: Optional[Tensor] = None,
        absolute: float = 0,
        membrane: float = 0,
        bending: float = 0,
        shears: float = 0,
        div: float = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        precond: Optional[str] = 'H+diag(L)',
        max_iter: int = 32,
        tolerance: float = 1e-5,
) -> Tensor:
    r"""Solve a regularized linear system involving flows by conjugate gradient

    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.

    This function solves the system in-place

    Parameters
    ----------
    init : (..., *spatial, D) tensor
        Initial value
    hessian : (..., *spatial, DD) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        DD is one of {1, D, D*(D+1)//2, D*D}
    gradient :  (..., *spatial, D) tensor
        Point at which to solve the system, `g`.
    weight : (..., *spatial) tensor, optional
        Voxel-wise weight of the regularization
    absolute : float, optional
        Penalty on absolute values
    membrane : float, optional
        Penalty on first derivatives
    bending : float, optional
        Penalty on second derivatives
    shears : float, optional
        Penalty on shears (symmetric part of the Jacobian)
    div : float, optional
        Penalty on the divergence (trace of the Jacobian)
    voxel_size : [list of] float, optional
        Voxel size
    bound : [list of] {'dft', 'dct1', 'dct2', 'dst1', 'dst2', 'replicate', 'zero'}, optional
         Boundary conditions.
    precond : {'H+diag(L)', 'H', 'L', None}, optional
        Preconditioning.
    max_iter : int
        Maximum number of iterations
    tolerance : float
        Tolerance for early stopping

    Returns
    -------
    solution : (..., *spatial, D) tensor
        Solution of the linear system, `x`.

    """
    ndim = gradient.shape[-1]

    def dot(x, y):
        return (x*y).sum(list(range(-ndim-1, 0)))

    forward = FlowForward(
        hessian,
        weight,
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        shears=shears,
        div=div,
        vx=voxel_size,
        bound=bound,
    )

    if precond == 'H + diag(L)' or precond is True:
        precond = FlowPrecondFull(hessian)
    elif precond == 'H':
        precond = FlowPrecondH(hessian)
    elif precond == 'L':
        # TODO
        precond = None
    else:
        precond = None

    ConjugateGradient(
        max_iter=max_iter,
        tol=tolerance,
        dot=dot,
    ).solve_(init, gradient, forward, precond)
    return init


def flow_solve_fmg(
        hessian: Tensor,
        gradient: Tensor,
        init: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        absolute: float = 0,
        membrane: float = 0,
        bending: float = 0,
        shears: float = 0,
        div: float = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        nb_cycles: int = 2,
        nb_iter: OneOrSeveral[int] = 2,
) -> Tensor:
    r"""Solve a regularized linear system involving flows by full multi-grid

    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.

    Parameters
    ----------
    hessian : (..., *spatial, DD) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        DD is one of {1, D, D*(D+1)//2, D*D}
    gradient :  (..., *spatial, D) tensor
        Point at which to solve the system, `g`.
    init : (..., *spatial, D) tensor, optional
        Initial value
    weight : (..., *spatial) tensor, optional
        Voxel-wise weight of the regularization
    absolute : float, optional
        Penalty on absolute values
    membrane : float, optional
        Penalty on first derivatives
    bending : float, optional
        Penalty on second derivatives
    shears : float, optional
        Penalty on shears (symmetric part of the Jacobian)
    div : float, optional
        Penalty on the divergence (trace of the Jacobian)
    voxel_size : [list of] float, optional
        Voxel size
    bound : [list of] {'dft', 'dct1', 'dct2', 'dst1', 'dst2', 'replicate', 'zero'}, optional
         Boundary conditions.
    nb_cycles : int
        Number of W cycles
    nb_iter : int
        Number of relaxation iterations

    Returns
    -------
    solution : (..., *spatial, D) tensor
        Solution of the linear system, `x`.

    """
    ndim = gradient.shape[-1]
    batch_shape, spatial_shape = _guess_shapes(hessian, gradient, weight)
    if init is None:
        init = gradient.new_zeros([*batch_shape, *spatial_shape, ndim])
    else:
        init = init.expand([*batch_shape, *spatial_shape, ndim]).clone()

    return flow_solve_fmg_(
        init,
        hessian=hessian,
        gradient=gradient,
        weight=weight,
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        shears=shears,
        div=div,
        voxel_size=voxel_size,
        bound=bound,
        nb_cycles=nb_cycles,
        nb_iter=nb_iter,
    )


def flow_solve_fmg_(
        init: Tensor,
        hessian: Tensor,
        gradient: Tensor,
        weight: Optional[Tensor] = None,
        absolute: float = 0,
        membrane: float = 0,
        bending: float = 0,
        shears: float = 0,
        div: float = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        nb_cycles: int = 2,
        nb_iter: OneOrSeveral[int] = 2,
) -> Tensor:
    r"""Solve a regularized linear system involving flows by full multi-grid

    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.

    This function solves the system in-place

    Parameters
    ----------
    init : (..., *spatial, D) tensor
        Initial value
    hessian : (..., *spatial, DD) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        DD is one of {1, D, D*(D+1)//2, D*D}
    gradient :  (..., *spatial, D) tensor
        Point at which to solve the system, `g`.
    weight : (..., *spatial) tensor, optional
        Voxel-wise weight of the regularization
    absolute : float, optional
        Penalty on absolute values
    membrane : float, optional
        Penalty on first derivatives
    bending : float, optional
        Penalty on second derivatives
    shears : float, optional
        Penalty on shears (symmetric part of the Jacobian)
    div : float, optional
        Penalty on the divergence (trace of the Jacobian)
    voxel_size : [list of] float, optional
        Voxel size
    bound : [list of] {'dft', 'dct1', 'dct2', 'dst1', 'dst2', 'replicate', 'zero'}, optional
         Boundary conditions.
    nb_cycles : int
        Number of W cycles
    nb_iter : int
        Number of relaxation iterations

    Returns
    -------
    solution : (..., *spatial, D) tensor
        Solution of the linear system, `x`.

    """
    ndim = gradient.shape[-1]
    voxel_size = make_vector(voxel_size, ndim, dtype=torch.float, device='cpu')

    prolong = ProlongFlow(ndim, order=2, bound=bound)
    restrict = RestrictFlow(ndim, order=1, bound=bound)
    restrict_weight = Restrict(ndim, order=1, bound=bound)

    prm = dict(
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        shears=shears,
        div=div,
        bound=bound,
        vx=voxel_size.tolist(),
    )

    # build pyramid
    nb_iter = ensure_list(nb_iter)
    gradients = [gradient]
    hessians = [hessian]
    weights = [weight]
    forwards = [FlowForward(hessians[0], weights[0], **prm)]
    solvers = [FlowRelax(hessians[0], gradients[0], weights[0],
                         nb_iter=nb_iter[0], **prm)]

    while gradients[-1].shape[-ndim-1:-1].numel() > 1:
        voxel_size = voxel_size * 2
        prm['vx'] = voxel_size.tolist()

        # restrict hessian & gradient
        gradients += [restrict(gradients[-1])]
        hessians += [restrict(hessians[-1])]
        if weights[-1] is None:
            weights += [None]
        else:
            weights += [restrict_weight(weights[-1])]

        # # if small, move to cpu
        # # NOTE: I tried to quickly benchmark and it was not saving much
        # #       so better to go for the simpler solution
        # if gradients[-1].numel() < 1024:
        #     gradients[-1] = gradients[-1].cpu()
        #     hessians[-1] = hessians[-1].cpu()
        #     if weights[-1] is not None:
        #         weights[-1] = weights[-1].cpu()

        # build objects
        nb_iter1 = nb_iter[min(len(nb_iter)-1, len(gradients)-1)]
        forwards += [FlowForward(hessians[-1], weights[-1], **prm)]
        solvers += [FlowRelax(hessians[-1], gradients[-1], weights[-1],
                              nb_iter=nb_iter1, **prm)]

    # solve
    MultiGrid(
        nb_cycles=nb_cycles,
        prolong=prolong,
        restrict=restrict,
    ).solve_(init, gradients, forwards, solvers)
    return init


def flow_solve_gs(
        hessian: Tensor,
        gradient: Tensor,
        init: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        absolute: float = 0,
        membrane: float = 0,
        bending: float = 0,
        shears: float = 0,
        div: float = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        max_iter: int = 32,
        tolerance: float = 1e-5,
) -> Tensor:
    r"""Solve a regularized linear system involving flows by Gauss-Seidel

    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.

    Parameters
    ----------
    hessian : (..., *spatial, DD) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        DD is one of {1, D, D*(D+1)//2, D*D}
    gradient :  (..., *spatial, D) tensor
        Point at which to solve the system, `g`.
    init : (..., *spatial, D) tensor, optional
        Initial value
    weight : (..., *spatial) tensor, optional
        Voxel-wise weight of the regularization
    absolute : float, optional
        Penalty on absolute values
    membrane : float, optional
        Penalty on first derivatives
    bending : float, optional
        Penalty on second derivatives
    shears : float, optional
        Penalty on shears (symmetric part of the Jacobian)
    div : float, optional
        Penalty on the divergence (trace of the Jacobian)
    voxel_size : [list of] float, optional
        Voxel size
    bound : [list of] {'dft', 'dct1', 'dct2', 'dst1', 'dst2', 'replicate', 'zero'}, optional
         Boundary conditions.
    max_iter : int
        Maximum number of iterations
    tolerance : float
        Tolerance for early stopping

    Returns
    -------
    solution : (..., *spatial, D) tensor
        Solution of the linear system, `x`.

    """
    ndim = gradient.shape[-1]
    batch_shape, spatial_shape = _guess_shapes(hessian, gradient, weight)
    if init is None:
        init = gradient.new_zeros([*batch_shape, *spatial_shape, ndim])
    else:
        init = init.expand([*batch_shape, *spatial_shape, ndim]).clone()

    return flow_solve_gs_(
        init,
        hessian=hessian,
        gradient=gradient,
        weight=weight,
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        shears=shears,
        div=div,
        voxel_size=voxel_size,
        bound=bound,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def flow_solve_gs_(
        init: Tensor,
        hessian: Tensor,
        gradient: Tensor,
        weight: Optional[Tensor] = None,
        absolute: float = 0,
        membrane: float = 0,
        bending: float = 0,
        shears: float = 0,
        div: float = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        max_iter: int = 32,
        tolerance: float = 1e-5,
) -> Tensor:
    r"""Solve a regularized linear system involving flows by Gauss-Seidel

    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.

    This function solves the system in-place

    Parameters
    ----------
    init : (..., *spatial, D) tensor
        Initial value
    hessian : (..., *spatial, DD) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        DD is one of {1, D, D*(D+1)//2, D*D}
    gradient :  (..., *spatial, D) tensor
        Point at which to solve the system, `g`.
    weight : (..., *spatial) tensor, optional
        Voxel-wise weight of the regularization
    absolute : float, optional
        Penalty on absolute values
    membrane : float, optional
        Penalty on first derivatives
    bending : float, optional
        Penalty on second derivatives
    shears : float, optional
        Penalty on shears (symmetric part of the Jacobian)
    div : float, optional
        Penalty on the divergence (trace of the Jacobian)
    voxel_size : [list of] float, optional
        Voxel size
    bound : [list of] {'dft', 'dct1', 'dct2', 'dst1', 'dst2', 'replicate', 'zero'}, optional
         Boundary conditions.
    max_iter : int
        Maximum number of iterations
    tolerance : float
        Tolerance for early stopping

    Returns
    -------
    solution : (..., *spatial, D) tensor
        Solution of the linear system, `x`.

    """
    prm = dict(
        weight=weight,
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        shears=shears,
        div=div,
        bound=bound,
        voxel_size=voxel_size,
    )

    if tolerance == 0:
        return flow_relax_(init, hessian, gradient, nb_iter=max_iter, **prm)

    # tolerance > 0: we must track the loss (much slower)
    buf = torch.empty_like(init)

    def get_loss(x):
        Ax = flow_forward(hessian, x, out=buf, **prm)
        return Ax.sub_(gradient).square().mean()

    def relax_(x):
        return flow_relax_(x, hessian, gradient, nb_iter=1, **prm)

    loss_prev = get_loss(init)
    for n_iter in range(max_iter):
        init = relax_(init)
        loss = get_loss(init)
        if loss_prev - loss < tolerance:
            break
        loss_prev = loss

    return init


class FlowForward:
    """Forward pass `(H + L) @ x`"""
    def __init__(self, h, w=None, vx=1, bound='dft',
                 absolute=0, membrane=0, bending=1, shears=0, div=0):
        self.h = h
        self.w = w
        self.vx = vx
        self.bound = bound
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.shears = shears
        self.div = div

    def __call__(self, x, out=None):
        return flow_forward(
            self.h, x, self.w,
            absolute=self.absolute,
            membrane=self.membrane,
            bending=self.bending,
            shears=self.shears,
            div=self.div,
            voxel_size=self.vx,
            bound=self.bound,
            out=out)


class FlowPrecondFull:
    """Block preconditioner `(H + diag(L)) \ x`"""
    def __init__(self, h, w=None, vx=1, bound='dft',
                 absolute=0, membrane=0, bending=1, shears=0, div=0):
        self.h = h
        self.w = w
        self.vx = vx
        self.bound = bound
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.shears = shears
        self.div = div

    def __call__(self, x, out=None):
        return flow_precond(
            self.h, x, self.w,
            absolute=self.absolute,
            membrane=self.membrane,
            bending=self.bending,
            shears=self.shears,
            div=self.div,
            voxel_size=self.vx,
            bound=self.bound,
            out=out)


class FlowPrecondH:
    """Simple block Preconditioner `H \ x`"""
    def __init__(self, h):
        self.h = h

    def __call__(self, x, out=None):
        return sym_solve(self.h, x, out=out)


class FlowRelax:
    """Gauss-Seidel solver"""
    def __init__(self, h, g, w=None, vx=1, bound='dft', nb_iter=4,
                 absolute=0, membrane=0, bending=1, shears=0, div=0):
        self.h = h
        self.g = g
        self.w = w
        self.vx = vx
        self.bound = bound
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.shears = shears
        self.div = div
        self.nb_iter = nb_iter

    def __call__(self, x):
        return flow_relax_(
            x, self.h, self.g, self.w,
            nb_iter=self.nb_iter,
            absolute=self.absolute,
            membrane=self.membrane,
            bending=self.bending,
            shears=self.shears,
            div=self.div,
            voxel_size=self.vx,
            bound=self.bound)


def _guess_shapes(hessian, gradient, weight):
    ndim = gradient.shape[-1]
    batch_shape = [hessian.shape[:-ndim-1], gradient.shape[:-ndim-1]]
    if weight is not None:
        batch_shape += [weight.shape[:-ndim]]
    batch_shape = expanded_shape(*batch_shape)
    spatial_shape = [hessian.shape[-ndim-1:-1], gradient.shape[-ndim-1:-1]]
    if weight is not None:
        spatial_shape += [weight.shape[-ndim:]]
    spatial_shape = expanded_shape(*spatial_shape)
    return batch_shape, spatial_shape