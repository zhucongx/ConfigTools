import sys

sys.path.insert(0, "../..")
from configtools import *

kmc_simulation = kmc.KmcSimulation(config=cfg.read_config('start.cfg'),
                                   log_dump_step=100,
                                   config_dump_steps=10000,
                                   maximum_number=1000000000,
                                   type_set={"Al", "Mg", "Zn"},
                                   json_parameters_filename="kmc_parameters_bond.json")
kmc_simulation.simulate()
