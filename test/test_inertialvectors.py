import numpy as np

from dynamics3d.inertialvectors import Force3D, rotation_quaternion, TimeVaryingForce
from testutils import ArrayTestCase


class TestForce3D(ArrayTestCase):

    def test_magnitude(self):
        self.force = Force3D([10, 0, 0])
        self.assertAlmostEqual(10, self.force.magnitude, delta=0.001)

        self.force = Force3D([10, 0, 0], [4, 0, 0])
        self.assertAlmostEqual(10, self.force.magnitude, delta=0.001, msg="Check with a non zero location")

        self.force = Force3D([-10, 0, 10])
        self.assertAlmostEqual(14.142, self.force.magnitude, delta=0.001)

    def test_moment_around(self):
        self.force = Force3D([10, 0, 0])
        self.assertArrayEqual([0, 0, 0], self.force.moment_around([0, 0, 0]))

        self.force = Force3D([0, 0, 10])
        self.assertArrayEqual([-10, 0, 0], self.force.moment_around([0, 1, 0]))

        self.force = Force3D([0, 0, 10], [0, 1, 0])
        self.assertArrayEqual([10, 0, 0], self.force.moment_around([0, 0, 0]))

    def test_equality(self):
        self.force = Force3D([10, 0, 0])
        self.assertEqual(self.force, Force3D([10, 0, 0]))

    def test_equality_bad_input(self):
        self.assertNotEqual(Force3D([10, 0, 0]), None)

    def test_force_orientation_independent(self):
        self.force = Force3D([10, 0, 0], orientation_dependent=False)
        self.assertArrayEqual([10, 0, 0], self.force.in_frame(np.array([[1, 0, 0], [0, -1, 0], [0, -1, 0]])))

    def test_force_orientation_dependent(self):
        self.force = Force3D([10, 0, 0], orientation_dependent=True)
        self.assertArrayEqual([10, 0, 0], self.force.in_frame(np.array([[1, 0, 0], [0, -1, 0], [0, -1, 0]])))

    def test_force_orientation_dependent_changes(self):
        self.force = Force3D([10, 0, 0], orientation_dependent=True)
        self.assertArrayEqual([0, 10, 0], self.force.in_frame(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])))

    def test_rotation_quaternion_valid(self):
        self.assertQuaternionAlmostEqual(np.quaternion(0, 0, 1, 0), rotation_quaternion(np.pi, np.array([0, 1, 0])))


class TestTimeVaryingForce(ArrayTestCase):

    def test_time_evaluation(self):
        self.force = TimeVaryingForce(lambda t: [t+2, 0, 0])
        self.assertArrayEqual([2, 0, 0], self.force.vect)

    def test_force_on_update(self):
        self.force = TimeVaryingForce(lambda t: [t+2, 0, 0])
        self.force.update(10)
        self.assertArrayEqual([12, 0, 0], self.force.vect)

    def test_force_on_update_np(self):
        self.force = TimeVaryingForce(lambda t: np.array([t+2, 0, t]))
        self.force.update(10)
        self.assertArrayEqual([12, 0, 10], self.force.vect)
