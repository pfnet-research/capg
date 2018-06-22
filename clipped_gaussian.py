from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()

import chainer
from chainer import functions as F
import numpy as np

import chainerrl


def _wrap_by_variable(x):
    if isinstance(x, chainer.Variable):
        return x
    else:
        return chainer.Variable(x)


def _unwrap_variable(x):
    if isinstance(x, chainer.Variable):
        return x.data
    else:
        return x


def elementwise_gaussian_log_pdf(x, mean, var, ln_var):
    # log N(x|mean,var)
    #   = -0.5log(2pi) - 0.5log(var) - (x - mean)**2 / (2*var)
    return -0.5 * np.log(2 * np.pi) - \
        0.5 * ln_var - \
        ((x - mean) ** 2) / (2 * var)


NPY_SQRT1_2 = 1 / (2 ** 0.5)


def _ndtr(a):
    """CDF of the standard normal distribution.

    See https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c
    """
    if not isinstance(a, chainer.Variable):
        a = chainer.Variable(a)
    x = a * NPY_SQRT1_2
    z = abs(x)
    half_erfc_z = 0.5 * F.erfc(z)
    return F.where(
        z.data < NPY_SQRT1_2,
        0.5 + 0.5 * F.erf(x),
        F.where(
            x.data > 0,
            1.0 - half_erfc_z,
            half_erfc_z))


def _safe_log(x):
    """Logarithm function that won't backprop inf to input."""
    return F.log(F.where(x.data > 0, x, x.data))


def _log_ndtr(x):
    """Log CDF of the standard normal distribution.

    See https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c
    """
    if not isinstance(x, chainer.Variable):
        x = chainer.Variable(x)
    return F.where(
        x.data > 6,
        -_ndtr(-x),
        F.where(
            x.data > -14,
            _safe_log(_ndtr(x)),
            -0.5 * x * x - _safe_log(-x) - 0.5 * np.log(2 * np.pi)))


def _gaussian_log_cdf(x, mu, sigma):
    """Log CDF of a normal distribution."""
    if not isinstance(x, chainer.Variable):
        x = chainer.Variable(x)
    return _log_ndtr((x - mu) / sigma)


def _gaussian_log_sf(x, mu, sigma):
    """Log SF of a normal distribution."""
    if not isinstance(x, chainer.Variable):
        x = chainer.Variable(x)
    return _log_ndtr(-(x - mu) / sigma)


class ClippedGaussian(chainerrl.distribution.GaussianDistribution):
    """Clipped Gaussian distribution."""

    def __init__(self, mean, var, low, high):
        super().__init__(mean, var)
        self.low = F.broadcast_to(low, mean.shape)
        self.high = F.broadcast_to(high, mean.shape)
        assert isinstance(self.low, chainer.Variable)
        assert isinstance(self.high, chainer.Variable)

    def sample(self):
        unclipped = F.gaussian(self.mean, self.ln_var)
        return F.minimum(F.maximum(unclipped, self.low), self.high)

    def log_prob(self, x):
        unclipped_elementwise_log_prob = elementwise_gaussian_log_pdf(
            x, self.mean, self.var, self.ln_var)
        std = self.var ** 0.5
        low_log_prob = _gaussian_log_cdf(self.low, self.mean, std)
        high_log_prob = _gaussian_log_sf(self.high, self.mean, std)
        x_data = _unwrap_variable(x)
        elementwise_log_prob = F.where(
            (x_data <= self.low.data),
            low_log_prob,
            F.where(
                x_data >= self.high.data,
                high_log_prob,
                unclipped_elementwise_log_prob))
        return F.sum(elementwise_log_prob, axis=1)

    def prob(self, x):
        return F.exp(self.log_prob(x))

    def copy(self):
        return ClippedGaussian(_wrap_by_variable(self.mean.data),
                               _wrap_by_variable(self.var.data),
                               self.low,
                               self.high)
