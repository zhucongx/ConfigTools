from configtools import *

# ansys.build_pd_file({"Al", "Mg", "Zn"},
#                     "../all_data_neb_results/",
#                     "./data/processed/all_compiled_data.pkl")

ansys.build_pd_file({"Al", "Mg", "Zn", "Sn"},
                    "../all_data_neb_results_sn/",
                    "./data/processed/all_compiled_data_sn.pkl")
