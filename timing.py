from scipy import interpolate
from slice import CviSlice, CviNode
from matplotlib import pyplot as plt
import numpy as np
import time

n_trials = 1000

start = time.perf_counter()
for _ in range(n_trials):
    slc = CviSlice(
        atm_var=0.2 * 0.2,
        skew=2.0,
        nodes=[
            CviNode(-2.0, 0.0),
            CviNode(-1.0, 1.0),
            CviNode(0.0, 2.0),
            CviNode(1.0, 1.0),
            CviNode(2.0, 0.0),
        ],
        ref_fwd=100.0
    )

    log_mns = np.array([-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5])
    # log_mns = np.linspace(-3.5, 2.0)
    strikes = 100 * np.exp(log_mns)
    vols = slc.vol_deriv1_deriv2_z(log_mns)
end = time.perf_counter()
print(f"{end - start:.5f}")

# start = time.perf_counter()
# for _ in range(n_trials):
#     slc = interpolate.CubicSpline(log_mns, vols[0])
#     vols2 = slc(log_mns)
# end = time.perf_counter()
# print(f"{end - start:.5f}")

fig, ax = plt.subplots(3, 1, sharex="all")
ax[0].plot(log_mns, vols[0])
ax[1].plot(log_mns, vols[1])
ax[2].plot(log_mns, vols[2])
plt.show()

x = 1
