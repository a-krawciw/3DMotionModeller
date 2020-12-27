import numpy as np

from bodies import Ball, BlockOnSpringForced, HoveringRocket, SpinningBall, ProjectileBall, SpiralBall, Moon
import matplotlib.pyplot as plt
import matplotlib

from dynamics3d import g

matplotlib.use('Qt5Agg')

moon = Moon(1020, step_time=200)
moon.run_for_time(2500000)

f = plt.figure()
ax = f.add_subplot(111, projection='3d')

moon_pos = moon.position_history
ax.plot(moon_pos.x, moon_pos.y, moon_pos.z, label="Numerical Sim")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.plot(moon.time, moon.velocity_history.magnitude)
plt.show()
