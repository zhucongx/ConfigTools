import unittest
from ansys.vac_jump import *


class TestVec(unittest.TestCase):
    def test_generate_one_hot_encode_dict_for_type(self):
        d1 = generate_one_hot_encode_dict_for_type({"Al", "Mg", "Zn"})
        self.assertEqual(d1, {"Al": [1.0, 0.0, 0.0],
                              "Mg": [0.0, 1.0, 0.0],
                              "Zn": [0.0, 0.0, 1.0],
                              "AlAl": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              "AlMg": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              "AlZn": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              "MgAl": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                              "MgMg": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                              "MgZn": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                              "ZnAl": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                              "ZnMg": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                              "ZnZn": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]})

    def test_cluster_type_key(self):
        c1 = Cluster(Atom(0, .0, "Al", .0, .0, .0))
        self.assertEqual(c1.type_key, "Al")

        c2 = Cluster(Atom(1, .0, "Al", .0, .0, .0), Atom(0, .0, "Mg", .0, .0, .0))
        self.assertEqual(c2.type_key, "MgAl")

        c3 = Cluster(Atom(0, .0, "Al", .0, .0, .0), Atom(1, .0, "Mg", .0, .0, .0))
        self.assertEqual(c3.type_key, "AlMg")

    def test_average_cluster_parameters(self):
        config = read_config("test_files/test.cfg")
        cluster_mapping = get_average_cluster_parameters_mapping(config)
        forward, backward = get_average_cluster_parameters_forward_and_backward_from_map(
            config, (18, 23), {"Al": 0, "Mg": 2, "Zn": -1}, cluster_mapping)
        test_list = forward + backward
        result_list = [1, 1, -1, -0.5, -0.5, 2, 1, -0.5, -1, -0.5, 2, 1, -1, -0.5, -0.5, -0.25, 1, 0, 0, 0, 0, 0, 0,
                       -1, -0.5, -1, 2, 0.5, -2, -1, 0.5, -1, -1, 0.5, 4, 2, -1, -2, -1, 4, -1, -0.5, 2, 0.5, -1,
                       0.5, -2, -1, 0.5, -1, -0.5, 0.25, 4, 2, -2, -1, -1, -0.5, 2, -1, -0.5, 0, 0.5, -1, 0, 0.25,
                       -0.5, 0, -1, 0, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0.5, 1, 0.5, 4, 4, 2,
                       2, 2, 0.25, 0.5, 1, 0.5, 4, 2, 0, 2, 2, 0.5, 0, 0.25, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                       1, 0, -0.5, -0.5, -0.25, 2, 1, -1, -0.5, -1, -0.5, 2, 1, -1, -0.5, -0.5, 1, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, -0.5, -1, -0.5, 2, 0, 0, 0, 0.25, -1, 0.5, -1, -1, 0.5, -0.5, -0.5,
                       0.25, 4, 2, -2, -1, -2, -1, 4, -1, -0.5, 2, 1, -2, 1, 0.5, -1, 0.5, -2, -1, 0.5, -1, -0.5, 4,
                       2, -2, -1, -1, 2, -1, 0.5, -1, -0.5, -1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0.5, 0.25, 0.5,
                       0.25, 4, 4, 2, 2, 2, 0.5, 0.5, 0.5, 1, 0.5, 4, 2, 2, 2, 0.5, 2]
        self.assertEqual(len(test_list), len(result_list))
        for test, result in zip(test_list, result_list):
            self.assertEqual(test, result)

    def test_one_hot_encode(self):
        config = read_config("test_files/test.cfg")
        cluster_mapping = get_average_cluster_parameters_mapping(config)
        forward, backward = get_one_hot_encoding_list_forward_and_backward_from_map(
            config, (18, 23), {"Al", "Mg", "Zn"}, cluster_mapping)
        self.assertEqual(len(forward), len(backward))
        self.assertEqual(len(forward), 21 * 3 + (115 - 21) * 9)


if __name__ == "__main__":
    unittest.main()
