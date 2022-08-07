import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots()
ax1.plot(speed.data["time"], speed.data["current_output"], label="output")
ax1.plot(speed.data["time"], speed.data["current_speed"], label="speed")
ax1.set_xlabel("time")
ax1.set_ylabel("out")
ax1.legend()
ax1.show()

fig2, ax2 = plt.subplots()
ax2.plot(direction.data["time"], direction.data["current_output"], label="output")
ax2.plot(direction.data["time"], direction.data["current_direction"], label="direction")
ax2.set_xlabel("time")
ax2.set_ylabel("out")
ax2.legend()
ax2.show()

