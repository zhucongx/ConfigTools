import typing
import numpy as np

BOLTZMANN_CONSTANT = 8.617333262145e-5
TEMPERATURE = 300
BOLTZMANN_CONSTANT_TEMPERATURE_INV = 1 / TEMPERATURE / BOLTZMANN_CONSTANT
PREFACTOR = 1e14


class KmcEvent(object):
    def __init__(self,
                 jump_pair: typing.Tuple[int, int],
                 barrier_and_diff: typing.Tuple[float, float]):
        self._jump_pair = jump_pair
        self._barrier = barrier_and_diff[0]
        self._rate = (np.exp(-barrier_and_diff[0] * BOLTZMANN_CONSTANT_TEMPERATURE_INV))
        self._energy_change = barrier_and_diff[1]
        self._probability = 0
        self.cumulative_probability = 0

    @property
    def jump_pair(self) -> typing.Tuple[int, int]:
        return self._jump_pair

    @property
    def forward_barrier(self) -> float:
        return self._barrier

    @property
    def forward_rate(self) -> float:
        return self._rate

    @property
    def energy_change(self) -> float:
        return self._energy_change

    @property
    def probability(self) -> float:
        return self._probability

    def calculate_probability(self, total_rate: float):
        self._probability = self._rate / total_rate

