import torch
from solvefields.core.solvers import ConjugateGradient, MultiGrid, MultiGridCG
from solvefields.core.pyramid import ProlongFlow, RestrictFlow, Restrict
from jitfields.regularisers import flow_forward, flow_precond, flow_relax_, flow_matvec
from jitfields import pull
from math import cos, sin, pi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

ndim = 2
shape = [192] * ndim
device = 'cuda'


def rotmat(theta):
    theta = theta*pi/180
    c, s = cos(theta), sin(theta)
    if ndim == 2:
        rot = [[c, -s], [s, c]]
    else:
        rot = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    mat = torch.as_tensor(rot, dtype=torch.float32, device=device)
    return mat


def identity(shape):
    grid = torch.meshgrid(*[torch.arange(s, dtype=torch.float32, device=device)
                            for s in shape])
    return torch.stack(grid, -1)


dot = lambda x, y: torch.dot(x.flatten(), y.flatten()) / x.numel()
norm = lambda x: dot(x, x)

id = identity(shape)

rot = id - torch.as_tensor(shape, device=device) / 2
rot = rotmat(30).matmul(rot.unsqueeze(-1)).squeeze(-1)
rot += torch.as_tensor(shape, device=device) / 2
rot -= id

irot = id - torch.as_tensor(shape, device=device) / 2
irot = rotmat(30).inverse().matmul(irot.unsqueeze(-1)).squeeze(-1)
irot += torch.as_tensor(shape, device=device) / 2
irot -= id

mask_square = torch.zeros(shape, dtype=torch.bool, device=device)
mask_square[48:-48, 48:-48] = 1
mask_circle = (id-192//2).square().sum(-1).sqrt() < 32

n = 1e8

g = torch.zeros([*shape, ndim], device=device)
g[64, 64, ..., :] = rot[64, 64, ..., :] * n
g[128, 64, ..., :] = rot[128, 64, ..., :] * n
g[128, 128, ..., :] = rot[128, 128, ..., :] * n
g[64, 128, ..., :] = rot[64, 128, ..., :] * n

h = torch.zeros([*shape, ndim*(ndim+1)//2], device=device)
h[64, 64, ..., :ndim] = n
h[64, 128, ..., :ndim] = n
h[128, 128, ..., :ndim] = n
h[128, 64, ..., :ndim] = n

w = torch.ones(shape, device=device)
w *= 1e-16
circle = (id-192//2).square().sum(-1).sqrt() < 148/2
w[circle] = 1
# w[32:-32, 32:-32] = 1


class Forward:
    def __init__(self, h, w=None, vx=1, absolute=0, membrane=0, bending=1,
                 shears=0, div=0):
        self.h = h
        self.w = w
        self.vx = vx
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.shears = shears
        self.div = div

    def __call__(self, x, out=None):
        return flow_forward(
            self.h, x, self.w, absolute=self.absolute, membrane=self.membrane, bending=self.bending,
            shears=self.shears, div=self.div, voxel_size=self.vx, out=out)


class Precond:
    def __init__(self, h, w=None, vx=1, absolute=0, membrane=0, bending=1,
                 shears=0, div=0):
        self.h = h
        self.w = w
        self.vx = vx
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.shears = shears
        self.div = div

    def __call__(self, x, out=None):
        return flow_precond(
            self.h, x, self.w, absolute=self.absolute, membrane=self.membrane, bending=self.bending,
            shears=self.shears, div=self.div, voxel_size=self.vx, out=out)


class Solver:

    def __init__(self, h, g, w=None, vx=1, absolute=0, membrane=0, bending=1,
                 shears=0, div=0, nb_iter=4):
        self.h = h
        self.g = g
        self.w = w
        self.vx = vx
        self.absolute = absolute
        self.membrane = membrane
        self.bending = bending
        self.shears = shears
        self.div = div
        self.nb_iter = nb_iter

    def __call__(self, x):
        return flow_relax_(
            x, self.h, self.g, self.w, nb_iter=self.nb_iter,
            absolute=self.absolute, membrane=self.membrane, bending=self.bending,
            shears=self.shears, div=self.div, voxel_size=self.vx)


prm = dict(
    absolute=0,
    membrane=0,
    bending=0,
    shears=1,
    div=1,
)
# w = None
BND = 'dct2'

A = Forward(h, w, **prm)
P = Precond(h, w, **prm)
x = torch.zeros_like(g)
# x = ConjugateGradient(max_iter=32, tol=0).advanced_solve_(x, g, A, P)

# for _ in range(32):
#     flow_relax_(x, h, g, w, **prm, bound=BND)
#     ll = norm(x - g) + dot(x, flow_matvec(x, w, **prm, bound=BND))
#     print(ll.item())
#     foo = 0
#
# for _ in range(128):
#     grad = flow_forward(h, x, w, **prm, bound=BND).sub_(g)
#     x.sub_(grad, alpha=1e-2 / n)
#     ll = norm(x - g) + dot(x, flow_matvec(x, w, **prm, bound=BND))
#     print(ll.item())
#     foo = 0

nbiter = 16
p = ProlongFlow(ndim, order=2, bound=BND)
r = RestrictFlow(ndim, order=1, bound=BND)
rw = Restrict(ndim, order=1, bound=BND)
g = [g]
h = [h]
w = [w]
A = [Forward(h[0], w[0], vx=1, **prm)]
P = [Precond(h[0], w[0], vx=1, **prm)]
S = [Solver(h[0], g[0], w[0], vx=1, nb_iter=nbiter, **prm)]
for k in range(1, 8):
    # nbiter *= 2
    g += [r(g[-1])]
    h += [r(h[-1])]
    if w[-1] is None:
        w += [None]
    else:
        w += [rw(w[-1])]
    A += [Forward(h[-1], w[-1], vx=2**k, **prm)]
    P += [Precond(h[-1], w[-1], vx=2**k, **prm)]
    S += [Solver(h[-1], g[-1], w[-1], vx=2**k, nb_iter=nbiter, **prm)]

x = torch.zeros_like(g[0])
# x = MultiGridCG(prolong=p, restrict=r, max_iter=2, nb_cycles=2).advanced_solve_(x, g, A, P)
tic = time.time()
x = MultiGrid(prolong=p, restrict=r, nb_cycles=2).solve_(x, g, A, S)
if x.is_cuda:
    torch.cuda.synchronize(x.device)
toc = time.time()
print(toc - tic)

print(norm(A[0](x) - g[0]))

plt.subplot(2, 2, 1)
plt.quiver(*reversed(rot[::4, ::4, :2].cpu().unbind(-1)), angles='xy', scale_units='xy', scale=4)

plt.subplot(2, 2, 2)
plt.quiver(*reversed(x[::4, ::4, :2].cpu().unbind(-1)), angles='xy', scale_units='xy', scale=4)

a = torch.zeros(shape, device=device)
a[...] = torch.linspace(1, 2, shape[0], device=device)[:, None]

a *= mask_square

c = pull(a[..., None], id+rot)[..., 0]
plt.subplot(2, 2, 3)
plt.imshow(c.cpu(), interpolation='nearest')
plt.colorbar()

c = pull(a[..., None], id+x)[..., 0]
plt.subplot(2, 2, 4)
plt.imshow(c.cpu(), interpolation='nearest')
plt.colorbar()

plt.show()

foo = 0
