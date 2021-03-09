"""
ITE: Information Theoretical Estimators Toolbox in Python
=========================================================

Subpackages
-----------
cost   - Estimators for entropy, mutual information, divergence,
         association measures, cross quantities, kernels on distributions
shared - Shared functions
      
"""

# automatically load submodules:
from . import cost

# explicitly tilt "from ite import *"; use "import ite".
__all__ = []
