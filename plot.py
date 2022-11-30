import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(speed.data["time"], speed.data["current_output"], label="output")
ax.plot(speed.data["time"], speed.data["current_speed"], label="speed")
ax.set_xlabel("time")
ax.set_ylabel("out")
ax.legend()
ax.show()

