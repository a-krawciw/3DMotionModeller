import numpy as np

from dynamics3d import Force3D
from dynamics3d import Body
from .testutils import ArrayTestCase


class TestBody(ArrayTestCase):

    def setUp(self) -> None:
        self.body = Body(10, [0, 0, 0], [[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    def test_net_force_with_no_forces(self):
        self.assertArrayEqual([0, 0, 0], self.body.net_force)

    def test_net_force_with_net_force(self):
        self.body.add_force([10, 0, 0])
        self.body.add_force([0, 10, 0], [1, 0, 1])
        self.body.add_force([0, 0, 10])
        self.assertArrayEqual([10, 10, 10], self.body.net_force)
        self.assertEqual((3,), self.body.net_force.shape)

    def test_equivalent_force_at_origin(self):
        self.body.add_force([10, 0, 0])
        self.assertEqual(Force3D([10, 0, 0]), self.body.equivalent_force)

    def test_equivalent_force_not_at_origin(self):
        self.body.add_force([10, 0, 0], location=[0, -1, 0])
        self.assertEqual(Force3D([10, 0, 0], position=[0, -1, 0]), self.body.equivalent_force)

    def test_equivalent_forces_not_at_origin(self):
        self.body.add_force([10, 0, 0], location=[0, -1, 0])
        self.body.add_force([-20, 0, 0])
        self.assertEqual(Force3D([-10, 0, 0], position=[0, 1, 0]), self.body.equivalent_force)

    def test_specific_moment_of_inertia(self):
        self.assertEqual(1, self.body.moment_of_inertia(np.array([2, 0, 0])))
        self.assertEqual(2, self.body.moment_of_inertia(np.array([0, 2, 0])))
        self.assertEqual(3, self.body.moment_of_inertia(np.array([0, 0, 3])))

    def test_zero_vector_moment_of_inertia(self):
        self.assertIsNan(self.body.moment_of_inertia(np.array([0, 0, 0])))

    def test_net_moment(self):
        self.body.add_force(Force3D([10, 0, 0], position=[0, 1, 0]))
        self.assertArrayEqual([0, 0, -10], self.body.net_moment())
        self.assertEqual((3,), self.body.net_moment().shape)

    def test_acceleration(self):
        self.body.add_force([10, 0, 0])
        self.assertArrayEqual([1, 0, 0], self.body.acceleration)

    def test_orientation_matrix(self):
        self.assertArrayEqual(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), self.body.rotation_matrix)
