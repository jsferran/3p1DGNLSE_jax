import time;
from math import pi, sqrt;

import jax;
import jax.numpy as jnp;
from types import SimpleNamespace;
from functools import partial;
from pathlib import Path;
import numpy as np;

from scipy.sparse.linalg import eigsh, LinearOperator, cg;

jax.config.update("jax_enable_x64", True);

################################################################################################################3
#### Make Spatial and Temporal Grids
################################################################################################################
def make_space(Lx, Nx, Ly, Ny):
    x = jnp.linspace(-Lx/2, Lx/2, Nx);
    y = jnp.linspace(-Ly/2, Ly/2, Ny);
    X, Y = jnp.meshgrid(x, y, indexing = 'ij');
    return X, Y;

###############################################################################################################
### Make refractive index profiles
###############################################################################################################
def make_polynomial_n(X, Y, n_core, n_clad, r_core, alpha = 2):
    a = r_core**2;
    delta = (n_core - n_clad)/n_core
    return jnp.where( X**2 + Y**2 < a, n_core*jnp.sqrt(1 - 2*delta*(jnp.sqrt(X**2 + Y**2)/r_core)**alpha), n_core*jnp.sqrt(1-2*delta));

def make_supergauss_index(X, Y, n_core, n_clad, r_core, m=20):
    r = jnp.sqrt(X**2 + Y**2);
    sg = jnp.exp(-(r/r_core)**(2*m));      # m~20–40 looks very "flat-top"
    return n_clad + (n_core - n_clad) * sg;