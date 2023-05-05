# nitorch-solvers
Linear solvers for regularized problems involving dense scalar or vector fields

## Installation

This package relies on [`jitfields`](https://github.com/balbasty/jitfields)
and therefore on [`cppyy`](https://github.com/wlav/cppyy) and
[`cupy`](https://github.com/cupy/cupy). These dependencies -- and their
interaction with pytorch -- are much more stable when installed using conda
than pip. We advise first setting up the conda environment like this:
```shell
# cuda 10.2
conda install -c conda-forge -c pytorch -c nvidia cupy cppyy cudatoolkit=10.2 pytorch=1.8
# cuda 11.1
conda install -c conda-forge -c pytorch -c nvidia cupy cppyy cudatoolkit=11.1 pytorch=1.8
```
Then install `solvefields` using `pip`:
```shell
pip install "nitorch-solvers[cuda] @ https://github.com/balbasty/nitorch-solvers"
```

## API

### Scalar or vector fields (in arbitrary units)

```python
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
) -> Tensor: ...
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
) -> Tensor: ...
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
) -> Tensor: ...
r"""Solve a regularized linear system involving vector fields by full multi-grid

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
) -> Tensor: ...
r"""Solve a regularized linear system involving vector fields by full multi-grid

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
```

### Flow fields (in voxels)

```python

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
) -> Tensor: ...
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
) -> Tensor: ...
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
) -> Tensor: ...
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
nb_cycles : int
    Number of W cycles
nb_iter : int
    Number of relaxation iterations

Returns
-------
solution : (..., *spatial, D) tensor
    Solution of the linear system, `x`.
"""

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
) -> Tensor: ...
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
nb_cycles : int
    Number of W cycles
nb_iter : int
    Number of relaxation iterations

Returns
-------
solution : (..., *spatial, D) tensor
    Solution of the linear system, `x`.
"""
```
