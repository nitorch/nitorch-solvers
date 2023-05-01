import torch
from torch import Tensor
from typing import Optional, Sequence, TypeVar
from jitfields.regularisers import field_forward, field_precond, field_relax_
from jitfields.sym import sym_solve
from .core.utils import expanded_shape, make_vector, ensure_list
from .core.cg import ConjugateGradient
from .core.fmg import MultiGrid
from .core.pyramid import Prolong,  Restrict

T = TypeVar('T')
OneOrSeveral = Sequence[T]


def field_solve_cg(
        ndim: int,
        hessian: Tensor,
        gradient: Tensor,
        init: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        absolute: OneOrSeveral[float] = 0,
        membrane: OneOrSeveral[float] = 0,
        bending: OneOrSeveral[float] = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        precond: Optional[str] = 'H+diag(L)',
        max_iter: int = 32,
        tolerance: float = 1e-5,
) -> Tensor:
    r"""Solve a regularized linear system involving vector fields by conjugate gradient

    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    hessian : (..., *spatial, CC) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        CC is one of {1, C, C*(C+1)//2, C*C}
    gradient :  (..., *spatial, C) tensor
        Point at which to solve the system, `g`.
    init : (..., *spatial, C) tensor, optional
        Initial value
    weight : (..., *spatial, 1|C) tensor, optional
        Voxel-wise and channel-wise weight of the regularization
    absolute : [list of] float, optional
        Penalty on absolute values
    membrane : [list of] float, optional
        Penalty on first derivatives
    bending : [list of] float, optional
        Penalty on second derivatives
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
    solution : (..., *spatial, C) tensor
        Solution of the linear system, `x`.

    """
    batch_shape, spatial_shape, nc = _guess_shapes(ndim, hessian, gradient, weight)
    if init is None:
        init = gradient.new_zeros([*batch_shape, *spatial_shape, nc])
    else:
        init = init.expand([*batch_shape, *spatial_shape, nc]).clone()

    return field_solve_cg_(
        ndim,
        init,
        hessian=hessian,
        gradient=gradient,
        weight=weight,
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        voxel_size=voxel_size,
        bound=bound,
        precond=precond,
        max_iter=max_iter,
        tolerance=tolerance,
    )


def field_solve_cg_(
        ndim: int,
        init: Tensor,
        hessian: Tensor,
        gradient: Tensor,
        weight: Optional[Tensor] = None,
        absolute: OneOrSeveral[float] = 0,
        membrane: OneOrSeveral[float] = 0,
        bending: OneOrSeveral[float] = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        precond: Optional[str] = 'H+diag(L)',
        max_iter: int = 32,
        tolerance: float = 1e-5,
) -> Tensor:
    r"""Solve a regularized linear system involving vector fields by conjugate gradient

    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.

    This function solves the system in-place

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    init : (..., *spatial, C) tensor
        Initial value
    hessian : (..., *spatial, CC) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        CC is one of {1, C, C*(C+1)//2, C*C}
    gradient :  (..., *spatial, C) tensor
        Point at which to solve the system, `g`.
    weight : (..., *spatial, 1|C) tensor, optional
        Voxel-wise and channel-wise weight of the regularization
    absolute : [list of] float, optional
        Penalty on absolute values
    membrane : [list of] float, optional
        Penalty on first derivatives
    bending : [list of] float, optional
        Penalty on second derivatives
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
    solution : (..., *spatial, C) tensor
        Solution of the linear system, `x`.

    """
    def dot(x, y):
        return (x * y).sum(list(range(-ndim - 1, 0)))

    forward = FieldForward(
        hessian,
        weight,
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        vx=voxel_size,
        bound=bound,
    )

    if precond == 'H + diag(L)':
        precond = FieldPrecondFull(hessian)
    elif precond == 'H':
        precond = FieldPrecondH(hessian)
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


def field_solve_fmg(
        ndim: int,
        hessian: Tensor,
        gradient: Tensor,
        init: Optional[Tensor] = None,
        weight: Optional[Tensor] = None,
        absolute: OneOrSeveral[float] = 0,
        membrane: OneOrSeveral[float] = 0,
        bending: OneOrSeveral[float] = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        nb_cycles: int = 2,
        nb_iter: OneOrSeveral[int] = 2,
) -> Tensor:
    r"""Solve a regularized linear system involving vector fields by conjugate gradient

    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    hessian : (..., *spatial, CC) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        DD is one of {1, C, C*(C+1)//2, C*c}
    gradient :  (..., *spatial, C) tensor
        Point at which to solve the system, `g`.
    init : (..., *spatial, C) tensor, optional
        Initial value
    weight : (..., *spatial, 1|C) tensor, optional
        Voxel-wise and channel-wise weight of the regularization
    absolute : [list of] float, optional
        Penalty on absolute values
    membrane : [list of] float, optional
        Penalty on first derivatives
    bending : [list of] float, optional
        Penalty on second derivatives
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
    solution : (..., *spatial, C) tensor
        Solution of the linear system, `x`.

    """
    batch_shape, spatial_shape, nc = _guess_shapes(ndim, hessian, gradient, weight)
    if init is None:
        init = gradient.new_zeros([*batch_shape, *spatial_shape, nc])
    else:
        init = init.expand([*batch_shape, *spatial_shape, nc]).clone()

    return field_solve_fmg_(
        ndim,
        init,
        hessian=hessian,
        gradient=gradient,
        weight=weight,
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        voxel_size=voxel_size,
        bound=bound,
        nb_cycles=nb_cycles,
        nb_iter=nb_iter,
    )


def field_solve_fmg_(
        ndim: int,
        init: Tensor,
        hessian: Tensor,
        gradient: Tensor,
        weight: Optional[Tensor] = None,
        absolute: OneOrSeveral[float] = 0,
        membrane: OneOrSeveral[float] = 0,
        bending: OneOrSeveral[float] = 0,
        voxel_size: OneOrSeveral[float] = 1,
        bound: OneOrSeveral[str] = 'dft',
        nb_cycles: int = 2,
        nb_iter: OneOrSeveral[int] = 2,
) -> Tensor:
    r"""Solve a regularized linear system involving vector fields by conjugate gradient

    Notes
    -----
    This function solves a system of the form `x = (H + L) \ g`,
    where `H` is block-diagonal and `L` encodes a smoothness penalty.

    This function solves the system in-place

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions
    init : (..., *spatial, C) tensor
        Initial value
    hessian : (..., *spatial, CC) tensor
        Block-diagonal matrix `H`, stored as a field of symmetric matrices.
        CC is one of {1, C, C*(C+1)//2, C*C}
    gradient :  (..., *spatial, C) tensor
        Point at which to solve the system, `g`.
    weight : (..., *spatial, 1|C) tensor, optional
        Voxel-wise and channel-wise weight of the regularization
    absolute : [list of] float, optional
        Penalty on absolute values
    membrane : [list of] float, optional
        Penalty on first derivatives
    bending : [list of] float, optional
        Penalty on second derivatives
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
    solution : (..., *spatial, C) tensor
        Solution of the linear system, `x`.

    """
    voxel_size = make_vector(voxel_size, ndim, dtype=torch.float, device='cpu')

    prolong = Prolong(ndim, order=2, bound=bound, channel_last=True)
    restrict = Restrict(ndim, order=1, bound=bound, channel_last=True)

    prm = dict(
        absolute=absolute,
        membrane=membrane,
        bending=bending,
        bound=bound,
        vx=voxel_size.tolist(),
    )

    # build pyramid
    nb_iter = ensure_list(nb_iter)
    gradients = [gradient]
    hessians = [hessian]
    weights = [weight]
    forwards = [FieldForward(hessians[0], weights[0], **prm)]
    solvers = [FieldRelax(hessians[0], gradients[0], weights[0],
                          nb_iter=nb_iter[0], **prm)]

    while gradients[-1].shape[-ndim - 1:-1].numel() > 1:
        voxel_size = voxel_size * 2
        prm['vx'] = voxel_size.tolist()

        # restrict hessian & gradient
        gradients += [restrict(gradients[-1])]
        hessians += [restrict(hessians[-1])]
        if weights[-1] is None:
            weights += [None]
        else:
            weights += [restrict(weights[-1])]

        # # if small, move to cpu
        # # NOTE: I tried to quickly benchmark and it was not saving much
        # #       so better to go for the simpler solution
        # if gradients[-1].numel() < 1024:
        #     gradients[-1] = gradients[-1].cpu()
        #     hessians[-1] = hessians[-1].cpu()
        #     if weights[-1] is not None:
        #         weights[-1] = weights[-1].cpu()

        # build objects
        nb_iter1 = nb_iter[min(len(nb_iter) - 1, len(gradients) - 1)]
        forwards += [FieldForward(hessians[-1], weights[-1], **prm)]
        solvers += [FieldRelax(hessians[-1], gradients[-1], weights[-1],
                               nb_iter=nb_iter1, **prm)]

    # solve
    MultiGrid(
        nb_cycles=nb_cycles,
        prolong=prolong,
        restrict=restrict,
    ).solve_(init, gradients, forwards, solvers)
    return init


class FieldForward:
    """Forward pass `(H + L) @ x`"""

    def __init__(self, h, w=None, vx=1, bound='dct2',
                 absolute=0, membrane=0, bending=1):
        self.h = h
        self.w = w
        self.vx = vx
        self.bound = bound
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending

    def __call__(self, x, out=None):
        return field_forward(
            self.h, x, self.w,
            absolute=self.absolute,
            membrane=self.membrane,
            bending=self.bending,
            voxel_size=self.vx,
            bound=self.bound,
            out=out)


class FieldPrecondFull:
    """Block preconditioner `(H + diag(L)) \ x`"""

    def __init__(self, h, w=None, vx=1, bound='dct2',
                 absolute=0, membrane=0, bending=1):
        self.h = h
        self.w = w
        self.vx = vx
        self.bound = bound
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending

    def __call__(self, x, out=None):
        return field_precond(
            self.h, x, self.w,
            absolute=self.absolute,
            membrane=self.membrane,
            bending=self.bending,
            voxel_size=self.vx,
            bound=self.bound,
            out=out)


class FieldPrecondH:
    """Simple block Preconditioner `H \ x`"""

    def __init__(self, h):
        self.h = h

    def __call__(self, x, out=None):
        return sym_solve(self.h, x, out=out)


class FieldRelax:
    """Gauss-Seidel solver"""

    def __init__(self, h, g, w=None, vx=1, bound='dct2', nb_iter=4,
                 absolute=0, membrane=0, bending=1):
        self.h = h
        self.g = g
        self.w = w
        self.vx = vx
        self.bound = bound
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.nb_iter = nb_iter

    def __call__(self, x):
        return field_relax_(
            x, self.h, self.g, self.w,
            nb_iter=self.nb_iter,
            absolute=self.absolute,
            membrane=self.membrane,
            bending=self.bending,
            voxel_size=self.vx,
            bound=self.bound)


def _guess_shapes(ndim, hessian, gradient, weight):
    nb_channels = gradient.shape[-1]
    batch_shape = [hessian.shape[:-ndim - 1], gradient.shape[:-ndim - 1]]
    if weight is not None:
        batch_shape += [weight.shape[:-ndim-1]]
    batch_shape = expanded_shape(*batch_shape)
    spatial_shape = [hessian.shape[-ndim - 1:-1], gradient.shape[-ndim - 1:-1]]
    if weight is not None:
        spatial_shape += [weight.shape[-ndim-1:-1]]
    spatial_shape = expanded_shape(*spatial_shape)
    return batch_shape, spatial_shape, nb_channels