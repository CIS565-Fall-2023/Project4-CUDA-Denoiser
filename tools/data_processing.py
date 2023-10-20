import matplotlib.pyplot as plt
import numpy as np
filter_size = ["10", "30",  "50",  "70", "90"]
# Drawing time with different filter size
milliseconds = [1.013, 3.09, 4.78, 5.05, 5.16]

plt.xlabel("filter size")
plt.ylabel("filter time (ms)")
plt.yticks(np.arange(0, 5.5, 0.5))
plt.bar(filter_size, milliseconds, )
plt.bar_label(plt.bar(filter_size, milliseconds ))
# plt.title("Filter time with different filter size")
plt.savefig("../img/filter_time_with_different_filter_size.png")

plt.clf()


if __name__ == "__main__":
    pass