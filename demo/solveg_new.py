import torch
from nitorch_solvers.flows import flow_solve_cg, flow_solve_fmg
from jitfields import pull
from jitfields.regularisers import flow_forward
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
# w = None


prm = dict(
    absolute=0,
    membrane=0,
    bending=0,
    shears=1,
    div=1,
    bound='dct2',
)

CG = False

tic = time.time()
if CG:
    x = flow_solve_cg(h, g, weight=w, **prm, max_iter=32)
else:
    x = flow_solve_fmg(h, g, weight=w, **prm, nb_iter=4)
if x.is_cuda:
    torch.cuda.synchronize(x.device)
toc = time.time()
print(toc - tic)

print((flow_forward(h, x, w, **prm) - g).square().mean().item())

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
