import numpy as np
import numpy.typing as npt
from math import sqrt
from scipy.interpolate import BSpline
from scipy.linalg import solve


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
        self._crvs = np.array([n.crv for n in self._nodes])
        self._locs = np.array([n.loc for n in self._nodes])

        self._init_bspline()

    def _init_bspline(self):
        knots = np.concatenate(
            [
                [self._locs[0] - 4e-10 + (i + 1) * 1e-10 for i in range(3)],
                self._locs,
                [self._locs[-1] + (i + 1) * 1e-10 for i in range(3)],
            ]
        )
        degree = 3
        basis_funcs = [
            BSpline.basis_element(knots[i : i + degree + 2], extrapolate=False)
            for i in range(len(knots) - degree - 1)
        ]
        n = len(self._locs) + 2
        mat = np.zeros((n, n))
        for i, basis_func in enumerate(basis_funcs):
            mat[: n - 2, i] = np.nan_to_num(basis_func(self._locs, nu=2))
            mat[n - 2, i] = np.nan_to_num(basis_func(0.0, nu=1))
            mat[n - 1, i] = np.nan_to_num(basis_func(0.0))
        b = np.concatenate([self._crvs, [self._skew, self._atm_var]])
        c = solve(mat, b)
        self._bspline = BSpline(knots, c, 3, extrapolate=False)
        self._deriv_left = self._bspline(self._locs[0], nu=1)
        self._deriv_right = self._bspline(self._locs[-1], nu=1)

    def __call__(self, z: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64]:
        return np.sqrt(self._bspline(z))

    def var_deriv1_deriv2_z(
        self, z: npt.NDArray[np.float64] | float
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        right_extrap = np.where(z <= self._locs[-1], 0.0, z - self._locs[-1])
        left_extrap = np.where(z >= self._locs[0], 0.0, z - self._locs[0])
        middle = np.where(
            z < self._locs[0],
            self._locs[0],
            np.where(z > self._locs[-1], self._locs[-1], z),
        )
        return (
            self._bspline(middle)
            + right_extrap * self._deriv_right
            + left_extrap * self._deriv_left,
            self._bspline(middle, nu=1),
            self._bspline(middle, nu=2),
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
