from configtools import *

ansys.build_pd_file({"Al", "Mg", "Zn"},
                    "../data/raw/all_data_neb_AlMgZn/",
                    "../data/processed/AlMgZn_compiled.pkl")


ansys.build_pd_file({"Al", "Cu", "Mg", "Zn"},
                    "../data/raw/all_data_neb_AlCuMgZn/",
                    "./data/processed/AlCuMgZn_compiled.pkl")
ansys.build_pd_file({"Al", "Cu", "Mg", "Zn"},
                    "../data/raw/all_data_neb_AlMgZn/",
                    "../data/processed/AlMgZn_Cu_compiled.pkl")


ansys.build_pd_file({"Al", "Mg", "Sn", "Zn"},
                    "../data/raw/all_data_neb_AlMgSnZn/",
                    "./data/processed/AlMgSnZn_compiled.pkl")
ansys.build_pd_file({"Al", "Mg", "Sn", "Zn"},
                    "../data/raw/all_data_neb_AlMgZn/",
                    "../data/processed/AlMgZn_Sn_compiled.pkl")
