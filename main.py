from bodies import Moon
import matplotlib.pyplot as plt
import matplotlib
import dynamics3d.plotting as dyn_plot


moon = Moon(1104, step_time=400)
moon.run_for_time(2000000)

moon_pos = moon.position_history
f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.plot(moon.time, moon_pos.magnitude)

dyn_plot.plot_against_time(moon.velocity_history, z=False, y_label="Velocity (m/s)")
dyn_plot.trace_plot_2d(moon_pos, 'x', 'y')
dyn_plot.trace_plot_3d(moon_pos)

plt.show()
