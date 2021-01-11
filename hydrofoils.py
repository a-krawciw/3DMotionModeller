from warnings import warn

import numpy as np

from dynamics3d import SimulatedBody, Force3D, g, Moment
import dynamics3d.plotting as dplt
from dynamics3d.simulation import InvalidSimulationException


class Hydrofoil(Force3D):

    def __init__(self, pos=None, **kwargs):
        super(Hydrofoil, self).__init__([0, 0, 0], position=pos, orientation_dependent=True)
        self._aoa = 0
        self.cL = 0
        self.cD = 0
        self.cM = 0

        self.cL_over_aoa = kwargs.get("cL_over_aoa", 1)
        self.cL_0 = kwargs.get("cL_0", 0)
        self.cD_over_aoa = kwargs.get("cD_over_aoa", 1)
        self.cD_0 = kwargs.get("cD_0", 0)
        self.cM_over_aoa = kwargs.get("cM_over_aoa", 1)
        self.cM_0 = kwargs.get("cM_0", 0)

        self.rho = kwargs.get("density", 1000)  # kg/m^3
        self.vel_mag = 0
        self.area = kwargs.get("area", 1)  # m^2
        self.chord = kwargs.get("chord", 1)  # m

    @property
    def angle_of_attack(self):
        return self._aoa

    def set_aoa(self, angle_of_attack):
        if abs(angle_of_attack) > 15:
            warn("Angle of attack is too big. Foil has stalled")
            self.cL = 0

        self._aoa = angle_of_attack
        self.cL = angle_of_attack * self.cL_over_aoa
        self.cD = angle_of_attack * self.cD_over_aoa
        self.cM = angle_of_attack * self.cM_over_aoa

    def set_velocity(self, velocity):
        self.vel_mag = np.linalg.norm(velocity)

        self.vect = np.array([-0.5 * self.cD * self.rho * self.vel_mag ** 2 * self.area,
                              0,
                              0.5 * self.cL * self.rho * self.vel_mag ** 2 * self.area])

    def moment_around(self, r_vect):
        return Force3D.moment_around(self, r_vect) + \
               np.array([0, 0.5 * self.cM * self.rho * self.vel_mag ** 2 * self.area * self.chord, 0])


class HydrofoilBoat(SimulatedBody):

    def __init__(self, step_time=0.01):
        # Assuming boat is a rectangular prism
        mass = 200
        w = 1  # m
        h = 0.5  # m
        l = 3  # m
        I = mass / 12 * np.array([[(h ** 2 + l ** 2), 0, 0], [0, (w ** 2 + l ** 2), 0], [0, 0, (w ** 2 + h ** 2)]])
        super(HydrofoilBoat, self).__init__(mass, step_time=step_time, cg=[0, 0, 0], moments_of_inertia=I)
        self.left_thrust = Force3D([0, 0, 0], position=[-0.2, -0.5, -0.9])
        self.left_static = Hydrofoil(cL_over_aoa=0.08, area=0.165, chord=0.2, cD_over_aoa=0.005, cM_over_aoa=0.01,
                                     pos=[-0.2, -0.5, -0.9])
        self.left_static.set_aoa(3)

        self.right_thrust = Force3D([0, 0, 0], [-0.2, 0.5, -0.9])
        self.right_static = Hydrofoil(cL_over_aoa=0.08, area=0.165, chord=0.2, cD_over_aoa=0.005, cM_over_aoa=0.01,
                                      pos=[-0.2, 0.5, -0.9])
        self.right_static.set_aoa(3)

        self.right_dynamic = Hydrofoil(cL_over_aoa=0.08, area=0.165, chord=0.2, cD_over_aoa=0.005, cM_over_aoa=0.01,
                                       pos=[0.8, 0.5, -0.9])
        self.left_dynamic = Hydrofoil(cL_over_aoa=0.08, area=0.165, chord=0.2, cD_over_aoa=0.005, cM_over_aoa=0.01,
                                      pos=[0.8, -0.5, -0.9])

        self.left_dynamic.set_aoa(10)
        self.right_dynamic.set_aoa(10)

        self.pitch_target = 5

        self.weight = Force3D([0, 0, -self.mass * g], orientation_dependent=False)
        self.buoyancy = Force3D([0, 0, self.mass * g], orientation_dependent=False)
        self.buoyancyMoment = Moment([0, 0, 0], orientation_dependent=False)

        self.add_force(self.left_thrust)
        self.add_force(self.right_thrust)
        self.add_force(self.left_static)
        self.add_force(self.right_static)
        self.add_force(self.left_dynamic)
        self.add_force(self.right_dynamic)
        self.add_force(self.weight)
        self.add_force(self.buoyancy)
        self.add_force(self.buoyancyMoment)

    def update_forces(self, t, p, v, a, w, alpha, theta):
        target_v = np.array([5, 0, 0])
        self.left_thrust.vect = 200 * (target_v - v * np.array([1, 0, 0]))
        self.right_thrust.vect = 200 * (target_v - v * np.array([1, 0, 0]))

        self.left_static.set_aoa(-np.rad2deg(theta[1]) + 5)
        self.left_static.set_velocity(v)

        self.right_static.set_aoa(-np.rad2deg(theta[1]) + 5)
        self.right_static.set_velocity(v)

        pitch_error = self.pitch_target + np.rad2deg(theta[1])
        pitch_gain = 5

        pitch_v_error = self.pitch_target + np.rad2deg(w[1])
        pitch_v_gain = 2

        pitch_factor = pitch_error * pitch_gain + pitch_v_gain * pitch_v_error

        self.left_dynamic.set_aoa(-np.rad2deg(theta[1]) + pitch_factor)
        self.left_dynamic.set_velocity(v)
        self.right_dynamic.set_aoa(-np.rad2deg(theta[1]) + pitch_factor)
        self.right_dynamic.set_velocity(v)

        self.buoyancy.vect = np.array([0, 0, 0])
        # self.buoyancyMoment.moment = np.array([0, 0, 0])

        self.buoyancy.vect = self.net_force * np.array([0, 0, -1])
        if self.buoyancy.vect[2] > 0 and p[2] <= 0:
            # self.buoyancyMoment.moment = self.net_moment() * np.array([-1, -1, 0])
            print(t)
            print(self.buoyancy.vect)
            print(self.buoyancyMoment.moment)
            self.pitch_target = 5
        else:
            self.buoyancy.vect = np.array([0, 0, 0])
            self.pitch_target = 0

        if abs(self.net_moment()[1]) > 10000:
            raise InvalidSimulationException()


if __name__ == '__main__':
    boat = HydrofoilBoat(step_time=0.005)
    try:
        boat.run_for_time(10)
    except InvalidSimulationException as e:
        print(e)

    dplt.plot_against_time(boat.theta_history, y_label='Theta (rad)')
    dplt.plot_against_time(boat.omega_history, y_label='Omega (rad/s)')
    dplt.plot_against_time(boat.alpha_history, y_label='Alpha (rad/s^2)')
    dplt.plot_against_time(boat.position_history, y_label='Position (m)')
    dplt.plot_against_time(boat.velocity_history, y_label='Velocity (m/s)')
    dplt.plot_against_time(boat.acceleration_history, y_label='Acceleration (m/s^2)')
    dplt.trace_plot_2d(boat.position_history, 'x', 'z')
    dplt.show()
