from configtools import *

ansys.build_pd_file({"Al", "Mg", "Zn"},
                    "../all_data_neb_results/",
                    "./data/processed/AlMgZn_compiled.pkl")

# ansys.build_pd_file({"Al", "Mg", "Sn", "Zn"},
#                     "../all_data_neb_results/",
#                     "./data/processed/AlMgZn_compiled_q.pkl")

# ansys.build_pd_file({"Al", "Mg", "Zn", "Sn"},
#                     "../all_data_neb_results_sn/",
#                     "./data/processed/AlMgZnSn_compiled.pkl")
