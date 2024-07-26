import numpy as np
import sympy
emc = float( sympy.S.EulerGamma.n(10) )
from tteVAMP.problem import Problem
from tteVAMP.simulations import sim_model
from tteVAMP.vamp import infere
from tteVAMP.utils import plot_metrics