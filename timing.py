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
        skew=5.0,
        nodes=[
            CviNode(-6.0, 0.0),
            CviNode(-4.0, 0.03),
            CviNode(-2.0, 0.06),
            CviNode(0.0, 0.12),
            CviNode(2.0, 0.03),
            CviNode(4.0, 0.015),
            CviNode(6.0, 0.0),
        ],
        ref_fwd=100.0
    )

    log_mns = np.linspace(-8, 8.0, num=3000)
    log_mns_coarse = np.array([-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0])
    strikes = 100 * np.exp(log_mns)
    strikes_coarse = 100 * np.exp(log_mns)
    vols = slc.var_deriv1_deriv2_z(log_mns)
    vols_coarse = slc.var_deriv1_deriv2_z(log_mns_coarse)
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
ax[0].plot(log_mns_coarse[3], vols_coarse[0][3], 'ro')
ax[0].grid()
ax[1].plot(log_mns, vols[1])
ax[1].plot(log_mns_coarse[3], vols_coarse[1][3], 'ro')
ax[1].grid()
ax[2].plot(log_mns, vols[2])
ax[2].plot(log_mns_coarse, vols_coarse[2], 'ro')
ax[2].grid()
plt.show()

x = 1
