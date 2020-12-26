import numpy as np
import quaternion

from dynamics3d import g, rho_air
from dynamics3d.inertialvectors import Force3D, rotation_quaternion, TimeVaryingForce
from dynamics3d.rigidbodies import Body
from dynamics3d.simulation import Simulated


class Ball(Body, Simulated):

    def __init__(self, mass, radius, step_time=0.01):
        I = 2 / 5 * mass * radius ** 2
        Body.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]])
        Simulated.__init__(self, step_time)
        self.air_drag = Force3D([0, 0, 0], orientation_dependent=False)
        self.add_force(Force3D([0, 0, -mass * g], orientation_dependent=False))
        self.add_force(self.air_drag)
        self.cd = 0.5
        self.area = np.pi * radius ** 2

    def step(self, t):
        a = self.acceleration
        self.accel_history.append(a)
        v1 = self.vel_history[-1] + a * self.dT
        p1 = self.pos_history[-1] + self.vel_history[-1] * self.dT + 0.5 * a * self.dT ** 2

        self.air_drag.vect = -v1 * np.linalg.norm(v1) * 0.5 * self.cd * rho_air * self.area

        self.vel_history.append(v1)
        self.pos_history.append(p1)


class ProjectileBall(Body, Simulated):

    def __init__(self, mass, radius, step_time=0.01):
        I = 2 / 5 * mass * radius ** 2
        Body.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]])
        Simulated.__init__(self, step_time, v_initial=[10, 0, 0])
        self.add_force(Force3D([0, 0, -mass * g], orientation_dependent=False))
        self.cd = 0.5
        self.area = np.pi * radius ** 2

    def step(self, t):
        a = self.acceleration
        self.accel_history.append(a)
        v1 = self.vel_history[-1] + a * self.dT
        p1 = self.pos_history[-1] + self.vel_history[-1] * self.dT + 0.5 * a * self.dT ** 2

        self.vel_history.append(v1)
        self.pos_history.append(p1)


class SpinningBall(Body, Simulated):

    def __init__(self, mass, radius, step_time=0.01):
        I = 2 / 5 * mass * radius ** 2
        Body.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]])
        Simulated.__init__(self, step_time)
        self.air_drag = Force3D([0, 0, 0], orientation_dependent=False)
        # self.add_force(Force3D([0, 0, -mass*g], orientation_dependent=False))
        self.left_booster = Force3D([0, 1, 0], position=[radius, 0, 0])
        # self.add_force(self.air_drag)
        # self.add_force(self.left_booster)
        self.cd = 0.5
        self.area = np.pi * radius ** 2
        self.vel_history = [np.array([10, 0, 0])]

    def step(self, t):
        a = self.acceleration
        self.accel_history.append(a)
        v1 = self.vel_history[-1] + a * self.dT
        p1 = self.pos_history[-1] + self.vel_history[-1] * self.dT + 0.5 * a * self.dT ** 2

        alpha = self.angular_acceleration
        w1 = self.omega_history[-1] + alpha * self.dT
        delta_theta = self.omega_history[-1] * self.dT + 0.5 * alpha * self.dT ** 2
        q = rotation_quaternion(np.linalg.norm(delta_theta), delta_theta)
        self.direction = self.direction * q

        self.air_drag.vect = -v1 * np.linalg.norm(v1) * 0.5 * self.cd * rho_air * self.area

        self.alpha_history.append(alpha)
        self.omega_history.append(w1)
        self.theta_history.append(self.theta_history[-1] + delta_theta)
        self.vel_history.append(v1)
        self.pos_history.append(p1)


class SpiralBall(Body, Simulated):

    def __init__(self, mass, radius, step_time=0.01):
        I = 2 / 5 * mass * radius ** 2
        Body.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]])
        Simulated.__init__(self, step_time)
        self.booster = Force3D([0, 100, 0])
        self.add_force(self.booster)
        self.cd = 0.5
        self.area = np.pi * radius ** 2

    def step(self, t):
        a = self.acceleration
        self.accel_history.append(a)
        v1 = self.vel_history[-1] + a * self.dT
        p1 = self.pos_history[-1] + self.vel_history[-1] * self.dT + 0.5 * a * self.dT ** 2

        alpha = self.angular_acceleration
        w1 = self.omega_history[-1] + alpha * self.dT
        delta_theta = self.omega_history[-1] * self.dT + 0.5 * alpha * self.dT ** 2
        q = rotation_quaternion(np.linalg.norm(delta_theta), delta_theta)
        self.direction = self.direction * q

        self.booster.vect = 100 * np.array([np.sin(t), np.cos(t), 0])

        self.alpha_history.append(alpha)
        self.omega_history.append(w1)
        self.theta_history.append(self.theta_history[-1] + delta_theta)
        self.vel_history.append(v1)
        self.pos_history.append(p1)


class BlockOnSpringForced(Body, Simulated):

    def __init__(self, mass, c, k, width=1, initial_displacement=0, step_time=0.01):
        I = mass * width ** 2 / 6
        Body.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]])
        Simulated.__init__(self, step_time, p_initial=[initial_displacement, 0, 0])

        self.k = k
        self.c = c

        self.spring_force = Force3D([-k * initial_displacement, 0, 0])
        self.damper_force = Force3D([0, 0, 0])
        self.external_force = TimeVaryingForce(lambda t: np.array([np.sin(10 * t), 0, 0]))

        self.add_force(self.spring_force)
        self.add_force(self.damper_force)
        self.add_force(self.external_force)

    def step(self, t):
        a = self.acceleration
        self.accel_history.append(a)
        v1 = self.vel_history[-1] + a * self.dT
        p1 = self.pos_history[-1] + self.vel_history[-1] * self.dT + 0.5 * a * self.dT ** 2

        self.spring_force.vect = -self.k * p1
        self.damper_force.vect = -self.c * v1
        self.external_force.update(t)

        self.vel_history.append(v1)
        self.pos_history.append(p1)


class HoveringRocket(Body, Simulated):

    def __init__(self, mass, radius, step_time=0.01):
        I = 2 / 5 * mass * radius ** 2
        Body.__init__(self, mass, [0, 0, 0], [[I, 0, 0], [0, I, 0], [0, 0, I]])
        Simulated.__init__(self, step_time)

        self.air_drag = Force3D([0, 0, 0], orientation_dependent=False)
        self.thruster = Force3D([0, 0, 0])

        self.add_force(Force3D([0, 0, -mass * g], orientation_dependent=False))
        self.add_force(self.air_drag)
        self.add_force(self.thruster)

        self.error = np.array([0, 0, 0.0])
        self.cd = 0.5
        self.area = np.pi * radius ** 2

    def step(self, t):
        a = self.acceleration
        self.accel_history.append(a)
        v1 = self.vel_history[-1] + a * self.dT
        p1 = self.pos_history[-1] + self.vel_history[-1] * self.dT + 0.5 * a * self.dT ** 2

        target = np.array([0, 0, 10.0])

        self.error += target - p1

        self.air_drag.vect = -v1 * np.linalg.norm(v1) * 0.5 * self.cd * rho_air * self.area
        self.thruster.vect = 15 * (np.array([0, 0, 10]) - p1) - 8 * v1 + 0.05 * self.error

        self.vel_history.append(v1)
        self.pos_history.append(p1)
