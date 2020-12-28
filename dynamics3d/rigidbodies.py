import functools

import numpy as np
from dynamics3d import Force3D
import quaternion


class Body:
    """Represent the instantaneous state of a rigid body with a collection of forces
        acting upon it"""

    def __init__(self, mass: float, cg: list, moments_of_inertia: list or np.ndarray):
        self.forces = []
        self.cg = np.array(cg)
        self.moments_of_inertia = np.array(moments_of_inertia)
        self.mass = mass
        self.direction = np.quaternion(1, 0, 0, 0)

    @property
    def rotation_matrix(self):
        return quaternion.as_rotation_matrix(self.direction)

    @property
    def net_force(self) -> np.ndarray:
        return functools.reduce(lambda a, b: a + b.in_frame(self.rotation_matrix), self.forces, np.array([0, 0, 0]))

    def net_moment(self, point=None) -> np.ndarray:
        if point is None:
            point = self.cg
        return functools.reduce(lambda a, b: a + b.moment_around(point), self.forces, np.array([0, 0, 0]))

    @property
    def equivalent_force(self, point=None) -> Force3D:
        eq_moment = self.net_moment(point)
        eq_pos = -np.cross(eq_moment, self.net_force) / np.dot(self.net_force, self.net_force)
        return Force3D(self.net_force, eq_pos)

    def add_force(self, force, location=None):
        if issubclass(force.__class__, Force3D):
            self.forces.append(force)
        else:
            if location is None:
                location = [0, 0, 0]
            self.forces.append(Force3D(force, location))

    @property
    def acceleration(self) -> np.ndarray:
        return self.net_force / self.mass

    def moment_of_inertia(self, direction: np.ndarray) -> float:
        direction = direction[np.newaxis]
        inter = direction @ self.moments_of_inertia @ direction.transpose()
        return float(inter) / np.linalg.norm(direction) ** 2

    @property
    def angular_acceleration(self):
        moment = self.net_moment()
        if np.array_equal([0, 0, 0], moment):
            return np.array([0, 0, 0])
        return moment / self.moment_of_inertia(moment)
