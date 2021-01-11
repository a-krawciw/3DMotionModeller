import numpy as np

from dynamics3d import g, rho_air, G
from dynamics3d import Force3D, TimeVaryingForce
from dynamics3d import SimulatedBody


class Ball(SimulatedBody):

    def __init__(self, mass, radius, step_time=0.01):
        I = 2 / 5 * mass * radius ** 2
        SimulatedBody.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]], step_time=step_time)
        self.air_drag = Force3D([0, 0, 0], orientation_dependent=False)
        self.add_force(Force3D([0, 0, -mass * g], orientation_dependent=False))
        self.add_force(self.air_drag)
        self.cd = 0.5
        self.area = np.pi * radius ** 2

    def update_forces(self, t, p, v, a, w, alpha, theta):
        self.air_drag.vect = -v * np.linalg.norm(v) * 0.5 * self.cd * rho_air * self.area


class ProjectileBall(SimulatedBody):

    def __init__(self, mass, radius, step_time=0.01):
        I = 2 / 5 * mass * radius ** 2
        SimulatedBody.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]],
                               step_time=step_time, v_initial=[10, 0, 0])
        self.add_force(Force3D([0, 0, -mass * g], orientation_dependent=False))


class SpinningBall(SimulatedBody):

    def __init__(self, mass, radius, step_time=0.01):
        I = 2 / 5 * mass * radius ** 2
        SimulatedBody.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]],
                               step_time=step_time, v_initial=[10, 0, 0])
        self.air_drag = Force3D([0, 0, 0], orientation_dependent=False)
        self.add_force(Force3D([0, 0, -mass*g], orientation_dependent=False))
        self.left_booster = Force3D([0, 1, 0], position=[radius, 0, 0])
        self.add_force(self.air_drag)
        self.add_force(self.left_booster)
        self.cd = 0.5
        self.area = np.pi * radius ** 2

    def update_forces(self, t, p, v, a, w, alpha, theta):
        self.air_drag.vect = -v * np.linalg.norm(v) * 0.5 * self.cd * rho_air * self.area


class BlockOnSpringForced(SimulatedBody):

    def __init__(self, mass, c, k, width=1, initial_displacement=0, step_time=0.01):
        I = mass * width ** 2 / 6
        SimulatedBody.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]],
                               step_time=step_time, p_initial=[initial_displacement, 0, 0])

        self.k = k
        self.c = c

        self.spring_force = Force3D([-k * initial_displacement, 0, 0])
        self.damper_force = Force3D([0, 0, 0])
        self.external_force = TimeVaryingForce(lambda t: np.array([np.sin(10 * t), 0, 0]))

        self.add_force(self.spring_force)
        self.add_force(self.damper_force)
        self.add_force(self.external_force)

    def update_forces(self, t, p, v, a, w, alpha, theta):
        self.spring_force.vect = -self.k * p
        self.damper_force.vect = -self.c * v
        self.external_force.update(t)


class HoveringRocket(SimulatedBody):

    def __init__(self, mass, radius, step_time=0.01):
        I = 2 / 5 * mass * radius ** 2
        SimulatedBody.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]],
                               step_time=step_time)

        self.air_drag = Force3D([0, 0, 0], orientation_dependent=False)
        self.thruster = Force3D([0, 0, 0])

        self.add_force(Force3D([0, 0, -mass * g], orientation_dependent=False))
        self.add_force(self.air_drag)
        self.add_force(self.thruster)

        self.error = np.array([0, 0, 0.0])
        self.cd = 0.5
        self.area = np.pi * radius ** 2

    def update_forces(self, t, p, v, a, w, alpha, theta):
        target = np.array([0, 0, 10.0])

        self.error += target - p

        self.air_drag.vect = -v * np.linalg.norm(v) * 0.5 * self.cd * rho_air * self.area
        self.thruster.vect = 15 * (target - p) - 8 * v + 0.05 * self.error


class Moon(SimulatedBody):

    def __init__(self, v_initial, step_time=0.01):
        mass = 7.34767309e22
        radius = 1737100
        self.dist = 384400000
        I = 2/5*mass*radius**2
        SimulatedBody.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]], step_time=step_time, p_initial=[self.dist, 0, 0], v_initial=[0, v_initial, 0])

        self.gravity = Force3D([0, 0, 0], orientation_dependent=False)
        self.add_force(self.gravity)

    def update_forces(self, t, p, v, a, w, alpha, theta):
        mass_earth = 5.972e24
        self.gravity.vect = -G*mass_earth*self.mass/np.linalg.norm(p)**3*p
