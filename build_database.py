from configtools import *

ansys.build_pd_file({"Al", "Mg", "Zn"},
                    "./data/raw/all_data_neb_results/",
                    "./data/processed/AlMgZn_compiled.pkl")

# ansys.build_pd_file({"Al", "Mg", "Sn", "Zn"},
#                     "./data/raw/all_data_neb_results/",
#                     "./data/processed/AlMgZn_compiled_q.pkl")

# ansys.build_pd_file({"Al", "Mg", "Zn", "Sn"},
#                     "./data/raw/all_data_neb_results_sn/",
#                     "./data/processed/AlMgZnSn_compiled.pkl")
