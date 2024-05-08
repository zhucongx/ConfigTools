import unittest
from configtools import *


class TestVec(unittest.TestCase):

    def test1(self):
        config = cfg.read_config("./test_files/test.cfg")
        vacancy_id = cfg.get_vacancy_index(config)
        self.assertEqual(vacancy_id, 18)


if __name__ == "__main__":
    unittest.main()
