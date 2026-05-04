"""Legacy compatibility module.

This file now forwards to the revised POMDP implementation.
Existing notebooks importing `CustomerSupportEnv` continue to work,
while new code should import `CustomerSupportPOMDP` from
`Simulation/pomdp_environment.py` or `src/simulation_core/env/pomdp_environment.py`.
"""

from src.simulation_core.config import ACTION_SPACE_7 as ACTION_SPACE
from src.simulation_core.env.pomdp_environment import CustomerSupportPOMDP


class CustomerSupportEnv(CustomerSupportPOMDP):
    pass
