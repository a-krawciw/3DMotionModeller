import numpy as np
from numpy.ma import cos, sin


class Force3D:
    def __init__(self, force, position=None, orientation_dependent=True):
        if position is None:
            position = [0, 0, 0]
        self.vect = np.array(force)
        self.location = np.array(position)
        self.orientation_dependent = orientation_dependent

    @property
    def magnitude(self) -> float:
        return np.linalg.norm(self.vect)

    def moment_around(self, r_vect):
        return np.cross((self.location - r_vect), self.vect)

    def in_frame(self, rotation_matrix: np.ndarray):
        if self.orientation_dependent:
            return (rotation_matrix @ self.vect[np.newaxis].T).reshape((3,))
        else:
            return self.vect

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        return np.array_equal(other.vect, self.vect) and np.array_equal(other.location, self.location)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"Force {self.vect}, Position {self.location}"


class TimeVaryingForce(Force3D):

    def __init__(self, func, position=None, orientation_dependent=None):
        super().__init__(func(0), position, orientation_dependent)
        self.function = func

    def update(self, time):
        self.vect = self.function(time)

    def __repr__(self):
        return f"Time Varying force currently {self.vect} at {self.location}"


def rotation_quaternion(angle: float, axis: np.ndarray):
    if not np.array_equal([0, 0, 0], axis):
        axis = axis / np.linalg.norm(axis)
    a = sin(angle / 2)
    return np.quaternion(cos(angle / 2), a * axis[0], a * axis[1], a * axis[2])
