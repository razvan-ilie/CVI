import numpy as np
import numpy.typing as npt
from math import sqrt
from scipy.interpolate import BSpline, splev
from scipy.linalg import solve
from typing import cast


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
    _skew: float  # d variance / d log(mns)
    _nodes: list[CviNode]
    _ref_fwd: float
    # time to expiry
    _t_e: float
    _atm_anchor_var: float
    _z_denom: float  # denominator when calculating z coordinate
    _zero_idx: int
    _crvs: list[float]
    _locs: list[float]
    _bspline: BSpline
    _deriv_left: float
    _deriv_right: float

    def __init__(
        self,
        atm_var: float,
        skew: float,
        nodes: list[CviNode],
        ref_fwd: float,
        t_e: float,
        atm_anchor_var: float | None = None,
    ):
        self._nodes = sorted(nodes, key=lambda n: n.loc)

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

        self._atm_var = atm_var
        self._skew = -skew / 100.0
        self._ref_fwd = ref_fwd
        self._t_e = t_e
        self._atm_anchor_var = atm_anchor_var or atm_var
        self._z_denom = sqrt(self._atm_anchor_var * t_e)
        self._crvs = [n.crv for n in self._nodes]
        self._locs = [n.loc for n in self._nodes]

        self._init_bspline()

    def _init_bspline(self):
        knots = np.array([self._locs[0]] * 3 + self._locs + [self._locs[-1]] * 3)
        n = len(self._locs) + 2

        c = [0.0] * n
        mat = np.zeros((n, n))
        for i in range(n):
            c[i] = 1.0
            if i != 0:
                c[i - 1] = 0.0
            mat[: n - 2, i] = splev(self._locs, (knots, c, 3), der=2, ext=1)
            mat[n - 2, i] = splev(0.0, (knots, c, 3), der=1, ext=1)
            mat[n - 1, i] = splev(0.0, (knots, c, 3), der=0, ext=1)

        b = self._crvs + [self._skew, self._atm_var]
        c = solve(mat, b)

        self._bspline_tck = (knots, c, 3)

        derivs: np.ndarray = splev(
            [self._locs[0], self._locs[-1]], self._bspline_tck, der=1, ext=1
        )
        self._deriv_left = cast(float, derivs[0])
        self._deriv_right = cast(float, derivs[1])

    def __call__(self, z: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
        return np.sqrt(self._bspline(z))

    def var_deriv1_deriv2_z(
        self, z: npt.NDArray[np.float64] | float
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        right_extrap = np.where(z <= self._locs[-1], 0.0, z - self._locs[-1])
        left_extrap = np.where(z >= self._locs[0], 0.0, z - self._locs[0])
        return (
            splev(z, self._bspline_tck, ext=3)
            + right_extrap * self._deriv_right
            + left_extrap * self._deriv_left,
            splev(z, self._bspline_tck, der=1, ext=1)
            + (z < self._locs[0]) * self._deriv_left
            + (z > self._locs[-1]) * self._deriv_right,
            splev(z, self._bspline_tck, der=2, ext=1),
        )

    def vol_deriv1_deriv2_z(
        self, z: npt.NDArray[np.float64] | float
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        vals = self.var_deriv1_deriv2_z(z)
        vols = np.sqrt(vals[0])
        deriv1_vols_z = vals[1] / (2.0 * vols)
        return vols, deriv1_vols_z, (0.5 * vals[2] - deriv1_vols_z**2.0) / vols

    def vol_deriv1_deriv2_k(
        self, k: npt.NDArray[np.float64] | float
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        z = np.log(k / self._ref_fwd) / self._z_denom
        vals = self._bspline(z)
        vols = np.sqrt(vals[0])
        deriv1_vols_z = vals[1] / (2.0 * vols)
        deriv2_vols_z = (0.5 * vals[2] - deriv1_vols_z**2.0) / vols

        return vols, deriv1_vols_z / k, deriv2_vols_z / k - deriv1_vols_z / (k**2.0)
