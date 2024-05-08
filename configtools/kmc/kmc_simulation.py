from configtools import cfg
from configtools.kmc.kmc_event import KmcEvent, PREFACTOR
from configtools.kmc.predictor import Predictor
import typing
import random
import math
import numpy as np


class KmcSimulation(object):
    def __init__(self,
                 config: cfg.Config,
                 log_dump_step: int,
                 config_dump_steps: int,
                 maximum_number: int,
                 type_set: typing.Set[str],
                 json_parameters_filename: str):
        self._config = config
        self._log_dump_step = log_dump_step
        self._config_dump_steps = config_dump_steps
        self._maximum_number = maximum_number
        self._vacancy_index_ = cfg.get_vacancy_index(config)
        self._predictor = Predictor(json_parameters_filename, type_set)
        self._step = 0
        self._time = 0
        self._energy = 0

    def _build_event_list(self):
        self._event_list = []
        self._total_rate = 0
        for neighbor_index in self._config.atom_list[self._vacancy_index_].first_nearest_neighbor_list:
            jump_pair = (self._vacancy_index_, neighbor_index)
            event = KmcEvent(jump_pair,
                             self._predictor.get_barrier_and_diff(self._config, jump_pair))
            self._total_rate += event.forward_rate
            self._event_list.append(event)
        cumulative_probability = 0.0
        for event in self._event_list:
            event.calculate_probability(self._total_rate)
            cumulative_probability += event.probability
            event.cumulative_probability = cumulative_probability

    def _select_event(self):
        random_number = random.random()

        for index, event in enumerate(self._event_list):
            if event.cumulative_probability >= random_number:
                return index
        return 11

    def simulate(self):
        f = open("kmc_log.txt", "a")
        f.write("steps\ttime\tenergy\n")
        print("steps\ttime\tenergy")
        f.flush()
        while self._step < self._maximum_number:
            if self._step % self._log_dump_step == 0:
                f.write(f"{self._step}\t{self._time}\t{self._energy}\n")
                f.flush()
                print(f"{self._step}\t{self._time}\t{self._energy}")
            if self._step % self._config_dump_steps == 0:
                cfg.write_config(self._config, f'{self._step}.cfg', False)
            self._build_event_list()
            event_index = self._select_event()
            executed_invent = self._event_list[event_index]
            cfg.config.atoms_jump(self._config, executed_invent.jump_pair)
            self._time += -math.log(random.random()) / self._total_rate / PREFACTOR
            self._energy += executed_invent.energy_change
            self._step += 1
