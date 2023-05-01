__all__ = ['Prolong', 'Restrict', 'ProlongFlow', 'RestrictFlow']

import torch
from jitfields.resize import resize, restrict
from .utils import ensure_list


class Prolong:
    """Prolongation operator used in multi-grid solvers"""

    def __init__(self, ndim, factor=2, order=2, bound='dct2', anchor='e',
                 channel_last=False):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        factor : [list of] int
            Prolongation factor
        order : [list of] int
            Interpolation order
        bound : [list of] str
            Boundary conditions
        anchor : [list of] {'edge', 'center'}
            Anchor points
        channel_last : bool, default=False
            Whether the channel dimension is last
        """
        self.ndim = ndim
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor
        self.channel_last = channel_last

    def __call__(self, x, out=None):
        """
        Parameters
        ----------
        inp : (..., *spatial_in, [channel]) tensor
            Tensor to prolongate
        out : (..., *spatial_out, [channel]) tensor, optional
            Output placeholder

        Returns
        -------
        out : (..., *spatial_out, [channel]) tensor, optional
            Prolongated tensor
        """
        if self.channel_last:
            x = torch.movedim(x, -1, -self.ndim-1)
            if out is not None:
                out = torch.movedim(out, -1, -self.ndim-1)
        trueout = None
        if out is not None:
            prm = dict(shape=out.shape[-self.ndim:])
            if out.device != x.device:
                trueout, out = out, out.to(x)
        else:
            prm = dict(factor=self.factor)
        out = resize(x, **prm, ndim=self.ndim,
                     order=self.order, bound=self.bound, anchor=self.anchor,
                     prefilter=False, out=out)
        if trueout is not None:
            out = trueout.copy_(out)
        if self.channel_last:
            out = torch.movedim(out, -self.ndim - 1, -1)
        return out


class ProlongFlow:
    """Prolongation operator for displacement fields used in multi-grid solvers"""

    def __init__(self, ndim, factor=2, order=2, bound='dft', anchor='e'):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        factor : [list of] int
            Prolongation factor
        order : [list of] int
            Interpolation order
        bound : [list of] str
            Boundary conditions
        anchor : [list of] {'edge', 'center'}
            Anchor points
        """
        self.ndim = ndim
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor

    def get_scale(self, inshape, outshape):
        anchor = self.anchor[0].lower()
        factor = ensure_list(self.factor, self.ndim)
        if anchor == 'e':
            scale = [so / si for si, so in zip(inshape, outshape)]
        elif anchor == 'c':
            scale = [(so - 1) / (si - 1) for si, so in zip(inshape, outshape)]
        else:
            scale = factor
        return scale

    def __call__(self, x, out=None):
        """
        Parameters
        ----------
        inp : (..., *spatial_in, ndim) tensor
            Tensor to prolongate
        out : (..., *spatial_out, ndim) tensor, optional
            Output placeholder

        Returns
        -------
        out : (..., *spatial_out, ndim) tensor, optional
            Prolongated tensor
        """
        ndim = self.ndim
        trueout = None
        if out is not None:
            prm = dict(shape=out.shape[-ndim-1:-1])
            if out.device != x.device:
                trueout, out = out, out.to(x)
        else:
            prm = dict(factor=self.factor)
        x = torch.movedim(x, -1, -ndim-1)
        if out is not None:
            out = torch.movedim(out, -1, -ndim-1)
        out = resize(x, **prm, ndim=ndim,
                     order=self.order, bound=self.bound, anchor=self.anchor,
                     prefilter=False, out=out)
        scale = self.get_scale(x.shape[-ndim:], out.shape[-ndim:])
        out = torch.movedim(out, -ndim-1, -1)
        if out.shape[-1] == ndim:
            # Gradient
            for d, out1 in enumerate(out[..., :ndim].unbind(-1)):
                out1 *= scale[d]
        else:
            # if Hessian
            c = ndim
            for d in range(ndim):
                out[..., d] *= scale[d] * scale[d]
                for dd in range(d+1, ndim):
                    out[..., c] *= scale[d] * scale[dd]
                    c += 1
        if trueout is not None:
            out = trueout.copy_(out)
        return out


class Restrict:
    """Restriction operator used in multi-grid solvers"""

    def __init__(self, ndim, factor=2, order=1, bound='dct2', anchor='e',
                 channel_last=False):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        factor : [list of] int
            Prolongation factor
        order : [list of] int
            Interplation order
        bound : [list of] str
            Boundary conditions
        anchor : [list of] {'edge', 'center'}
            Anchor points
        channel_last : bool, default=False
            Whether the channel dimension is last
        """
        self.ndim = ndim
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor
        self.channel_last = channel_last

    def __call__(self, x, out=None):
        """
        Parameters
        ----------
        inp : (..., *spatial_in, [channel]) tensor
            Tensor to prolongate
        out : (..., *spatial_out, [channel]) tensor, optional
            Output placeholder

        Returns
        -------
        out : (..., *spatial_out, [channel]) tensor, optional
            Prolongated tensor
        """
        if self.channel_last:
            x = torch.movedim(x, -1, -self.ndim-1)
            if out is not None:
                out = torch.movedim(out, -1, -self.ndim-1)
        trueout = None
        if out is not None:
            prm = dict(shape=out.shape[-self.ndim:])
            if out.device != x.device:
                trueout, out = out, out.to(x)
        else:
            prm = dict(factor=self.factor)
        out = restrict(x, **prm, ndim=self.ndim,
                       order=self.order, bound=self.bound, anchor=self.anchor,
                       out=out)
        if trueout is not None:
            out = trueout.copy_(out)
        if self.channel_last:
            out = torch.movedim(out, -self.ndim - 1, -1)
        return out


class RestrictFlow:
    """Restriction operator for displacement fields used in multi-grid solvers"""

    def __init__(self, ndim, factor=2, order=1, bound='dft', anchor='e'):
        """
        Parameters
        ----------
        ndim : int
            Number of spatial dimensions
        factor : [list of] int
            Prolongation factor
        order : [list of] int
            Interplation order
        bound : [list of] str
            Boundary conditions
        anchor : [list of] {'edge', 'center'}
            Anchor points
        """
        self.ndim = ndim
        self.factor = factor
        self.order = order
        self.bound = bound
        self.anchor = anchor

    def get_scale(self, inshape, outshape):
        anchor = self.anchor[0].lower()
        factor = ensure_list(self.factor, self.ndim)
        if anchor == 'e':
            scale = [si / so for si, so in zip(inshape, outshape)]
        elif anchor == 'c':
            scale = [(si - 1) / (so - 1) for si, so in zip(inshape, outshape)]
        else:
            scale = factor
        return scale

    def __call__(self, x, out=None):
        """
        Parameters
        ----------
        inp : (..., *spatial_in, ndim) tensor
            Tensor to prolongate
        out : (..., *spatial_out, ndim) tensor, optional
            Output placeholder

        Returns
        -------
        out : (..., *spatial_out, ndim) tensor, optional
            Prolongated tensor
        """
        ndim = self.ndim
        trueout = None
        if out is not None:
            prm = dict(shape=out.shape[-ndim-1:-1])
            if out.device != x.device:
                trueout, out = out, out.to(x)
        else:
            prm = dict(factor=self.factor)
        x = torch.movedim(x, -1, -ndim-1)
        if out is not None:
            out = torch.movedim(out, -1, -ndim-1)
        out = restrict(x, **prm, ndim=ndim,
                       order=self.order, bound=self.bound, anchor=self.anchor,
                       out=out)
        scale = self.get_scale(x.shape[-ndim:], out.shape[-ndim:])
        out = torch.movedim(out, -ndim-1, -1)
        if out.shape[-1] == ndim:
            # Gradient
            for d, out1 in enumerate(out[..., :ndim].unbind(-1)):
                out1 *= scale[d]
        else:
            # if Hessian
            c = ndim
            for d in range(ndim):
                out[..., d] *= scale[d] * scale[d]
                for dd in range(d+1, ndim):
                    out[..., c] *= scale[d] * scale[dd]
                    c += 1
        if trueout is not None:
            out = trueout.copy_(out)
        return out
