from configtools import *

ansys.build_pd_file({"Al", "Mg", "Zn"},
                    "../all_data_neb_results/",
                    "./data/processed/all_compiled_data.pkl")
