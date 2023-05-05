import torch
from nitorch_solvers.fields import field_solve_cg, field_solve_fmg, field_relax_
from jitfields.regularisers import field_forward
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time


def identity(shape):
    grid = torch.meshgrid(*[torch.arange(s, dtype=torch.float32, device=device)
                            for s in shape])
    return torch.stack(grid, -1)

ndim = 3
shape = [192] * ndim
device = 'cuda'


id = identity(shape)
g = torch.randn([*shape, 1], device=device) * 500
h = torch.ones([*shape, 1], device=device)

# nonstationary weighting
# w = torch.ones(shape, device=device)
# w *= 1e-16
# circle = (id-192//2).square().sum(-1).sqrt() < 148/2
# w[circle] = 1
# w = w.unsqueeze(-1)
w = None

prm = dict(
    absolute=0,
    membrane=0,
    bending=100,
    bound='dct2',
)

MODE = 'FMG'  # CG | RELAX | FMG
CG = True


print('init', g.square().mean().item())

tic = time.time()
if MODE == 'CG':
    x = field_solve_cg(ndim, h, g, weight=w, **prm, max_iter=256)
elif MODE == 'RELAX':
    x = torch.zeros_like(g)
    x = field_relax_(ndim, x, h, g, weight=w, **prm, nb_iter=256)
elif MODE == 'FMG':
    x = field_solve_fmg(ndim, h, g, weight=w, **prm, nb_iter=4)
else:
    assert False, MODE
if x.is_cuda:
    torch.cuda.synchronize(x.device)
toc = time.time()
print('time', toc - tic)

print('final', (field_forward(ndim, h, x, w, **prm) - g).square().mean().item())


def make2d(x):
    x = x.detach()
    x = x[..., 0].squeeze()
    while x.ndim > 2:
        x = x[..., x.shape[-1]//2]
    return x.cpu()


plt.subplot(1, 2, 1)
plt.imshow(make2d(g))
plt.axis('off')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow(make2d(x))
plt.axis('off')
plt.colorbar()
plt.show()


foo = 0
