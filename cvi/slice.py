import numpy as np
import numpy.typing as npt
from math import sqrt


class CviNode:
    loc: float
    crv: float

    def __init__(self, loc: float, crv: float):
        self.loc = loc
        self.crv = crv

    def is_zero(self):
        return self.loc == 0.0


class CviSlice:
    _atm_var: float
    # d variance / d log(mns)
    _skew: float
    _nodes: list[CviNode]
    _ref_fwd: float
    # time to expiry
    _t_e: float
    _atm_anchor_var: float
    # denominator when calculating z coordinate
    _z_denom: float
    _zero_idx: int
    _crvs: npt.NDArray[np.float64]
    _locs: npt.NDArray[np.float64]

    def __init__(
        self,
        atm_var: float,
        skew: float,
        nodes: list[CviNode],
        ref_fwd: float,
        t_e: float,
        atm_anchor_var: float | None = None,
    ):
        self._atm_var = atm_var
        self._skew = -skew / 100.0
        self._nodes = sorted(nodes, key=lambda n: n.loc)
        self._ref_fwd = ref_fwd
        self._t_e = t_e
        self._atm_anchor_var = atm_anchor_var or atm_var
        self._z_denom = sqrt(atm_anchor_var * t_e)

        try:
            self._zero_idx = next(
                i for i, node in enumerate(self._nodes) if node.is_zero()
            )
        except StopIteration:
            raise ValueError("Nodes must contain a node at z = 0")

        if self._nodes[0].crv != 0.0:
            raise ValueError("First node must have a curvature of 0")
        if self._nodes[-1].crv != 0.0:
            raise ValueError("Last node must have a curvature of 0")

        self._crvs = np.array([n.crv for n in self._nodes])
        self._locs = np.array([n.loc for n in self._nodes])

    def __call__(self, z: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
        return np.sqrt(self._cubic_spline(z)[0])

    def var_deriv1_deriv2_z(
        self, z: npt.NDArray[np.float64] | float
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        return self._cubic_spline(z)

    def vol_deriv1_deriv2_z(
        self, z: npt.NDArray[np.float64] | float
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        vals = self._cubic_spline(z)
        vols = np.sqrt(vals[0])
        deriv1_vols_z = vals[1] / (2.0 * vols)
        return (vols, deriv1_vols_z, (0.5 * vals[2] - deriv1_vols_z**2.0) / vols)

    def vol_deriv1_deriv2_k(
        self, k: npt.NDArray[np.float64] | float
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        z = np.log(k / self._ref_fwd) / self._z_denom
        vals = self._cubic_spline(z)
        vols = np.sqrt(vals[0])
        deriv1_vols_z = vals[1] / (2.0 * vols)
        deriv2_vols_z = (0.5 * vals[2] - deriv1_vols_z**2.0) / vols

        return (vols, deriv1_vols_z / k, deriv2_vols_z / k - deriv1_vols_z / (k**2.0))
