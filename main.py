import numpy as np

from bodies import Ball, BlockOnSpringForced, HoveringRocket, SpinningBall, ProjectileBall, SpiralBall
import matplotlib.pyplot as plt
import matplotlib

from dynamics3d import g

matplotlib.use('Qt5Agg')



spring = BlockOnSpringForced(10, 5, 100, initial_displacement=0.1)
spring.run_for_time(100)
plt.figure(4)
ax4 = plt.axes(label = 3)
ax4.plot(spring.time, [p[0] for p in spring.pos_history])


b = SpiralBall(mass=10, radius=0.1, step_time=0.01)

b.run_for_time(100)

f = plt.figure()
ax = f.add_subplot(111, projection='3d')
ax.plot([p[0] for p in b.pos_history], [p[1] for p in b.pos_history], [p[2] for p in b.pos_history], label="Numerical Sim")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')



plt.figure(2)
ax2 = plt.axes(label=2)
for i in range(0, 3):
    ax2.plot(b.time[:-1], [p[i] for p in b.accel_history], label=f"Acceleration {i}")
plt.legend()


plt.figure(3)
ax3 = plt.axes(label=3)
for i in range(0, 3):
    ax3.plot(b.time, [p[i] for p in b.vel_history], label=f"Velocity {i}")
plt.legend()






plt.show()


