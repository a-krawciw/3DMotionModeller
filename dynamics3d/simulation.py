import numpy as np


class Simulated:
    def __init__(self, step_time, v_initial=None, p_initial=None, w_initial=None, initial_orientation=None):
        if initial_orientation is None:
            initial_orientation = [1, 0, 0]
        if p_initial is None:
            p_initial = [0, 0, 0]
        if v_initial is None:
            v_initial = [0, 0, 0]
        if w_initial is None:
            w_initial = [0, 0, 0]
        self.dT = step_time
        self.accel_history = []
        self.pos_history = [np.array(p_initial)]
        self.vel_history = [np.array(v_initial)]
        self.alpha_history = []
        self.omega_history = [np.array(w_initial)]
        self.theta_history = [np.array(initial_orientation)]
        self.n = 0

    def simulation_complete(self):
        return self.n > 0

    def run_for_n_steps(self, n: int):
        for i in range(0, n):
            self.step(i * self.dT)
        self.n = n

    def run_for_time(self, time):
        self.run_for_n_steps(int(time/self.dT))

    def step(self, t):
        pass

    @property
    def time(self):
        return np.arange(0, self.n+1)*self.dT