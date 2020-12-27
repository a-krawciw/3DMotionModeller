import numpy as np

from dynamics3d.inertialvectors import rotation_quaternion
from dynamics3d.rigidbodies import Body

class MotionStore3D:
    def __init__(self, values: list):
        self.data = values

    def alongIndex(self, i):
        return [v[i] for v in self.data]

    @property
    def x(self):
        return self.alongIndex(0)

    @property
    def y(self):
        return self.alongIndex(1)

    @property
    def z(self):
        return self.alongIndex(2)

    @property
    def magnitude(self):
        return [np.linalg.norm(v) for v in self.data]

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
        self._accel_history = []
        self._pos_history = [np.array(p_initial)]
        self._vel_history = [np.array(v_initial)]
        self._alpha_history = []
        self._omega_history = [np.array(w_initial)]
        self._theta_history = [np.array(initial_orientation)]
        self.n = 0

    def simulation_complete(self):
        return self.n > 0

    def run_for_n_steps(self, n: int):
        for i in range(0, n):
            self.step(i * self.dT)
        self.n = n

    def run_for_time(self, time):
        self.run_for_n_steps(int(time / self.dT))

    def step(self, t):
        pass

    @property
    def time(self):
        return np.arange(0, self.n + 1) * self.dT

    @property
    def position_history(self):
        return MotionStore3D(self._pos_history)

    @property
    def velocity_history(self):
        return MotionStore3D(self._vel_history)

    @property
    def acceleration_history(self):
        return MotionStore3D(self._accel_history)

    @property
    def theta_history(self):
        return MotionStore3D(self._theta_history)

    @property
    def omega_history(self):
        return MotionStore3D(self._omega_history)

    @property
    def alpha_history(self):
        return MotionStore3D(self._alpha_history)


class SimulatedBody(Body, Simulated):

    def __init__(self, mass, cg, moments_of_inertia, step_time=0.01, v_initial=None, p_initial=None, w_initial=None,
                 initial_orientation=None):
        Body.__init__(self, mass, cg, moments_of_inertia)
        Simulated.__init__(self, step_time, v_initial=v_initial, p_initial=p_initial, w_initial=w_initial,
                           initial_orientation=initial_orientation)

    def step(self, t):
        a = self.acceleration
        self._accel_history.append(a)
        v1 = self._vel_history[-1] + a * self.dT
        p1 = self._pos_history[-1] + self._vel_history[-1] * self.dT + 0.5 * a * self.dT ** 2

        alpha = self.angular_acceleration
        w1 = self._omega_history[-1] + alpha * self.dT
        delta_theta = self._omega_history[-1] * self.dT + 0.5 * alpha * self.dT ** 2
        q = rotation_quaternion(np.linalg.norm(delta_theta), delta_theta)
        self.direction = self.direction * q

        self.update_forces(t, p1, v1, a, w1, alpha)

        self._alpha_history.append(alpha)
        self._omega_history.append(w1)
        self._theta_history.append(self._theta_history[-1] + delta_theta)
        self._vel_history.append(v1)
        self._pos_history.append(p1)

    def update_forces(self, t, p, v, a, w, alpha):
        pass
