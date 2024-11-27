import numba
import numpy as np
import numpy.typing as npt


class CviNode:
    loc: float
    crv: float

    def __init__(self, loc: float, crv: float):
        self.loc = loc
        self.crv = crv

    def is_zero(self):
        return self.loc == 0.0


class CubicSpline:
    _locs: npt.NDArray[np.float64]
    _params: npt.NDArray[tuple[np.float64, np.float64, np.float64, np.float64]]

    def __init__(self, locs: npt.NDArray[np.float64], params: npt.NDArray[np.float64]):
        self._locs = locs
        self._params = np.array(
            [(x[0], x[1], x[2], x[3]) for x in np.reshape(params, (-1, 4))]
        )

    def __call__(
        self, z: npt.NDArray[np.float64] | float
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        idx = np.searchsorted(self._locs, z)
        params = np.transpose(self._params[np.array(idx)])
        return (
            params[0] * z**3.0 + params[1] * z**2.0 + params[2] * z + params[3],
            3.0 * params[0] * z**2.0 + 2.0 * params[1] * z + params[2],
            6.0 * params[0] * z + 2.0 * params[1],
        )


class CviSlice:
    _atm_var: float
    # d variance / d log(mns)
    _skew: float
    _nodes: list[CviNode]
    _ref_fwd: float
    _zero_idx: int
    _cubic_spline: CubicSpline
    _crvs: npt.NDArray[np.float64]
    _locs: npt.NDArray[np.float64]

    def __init__(
        self, atm_var: float, skew: float, nodes: list[CviNode], ref_fwd: float
    ):
        self._atm_var = atm_var
        self._skew = -skew / 100.0
        self._nodes = sorted(nodes, key=lambda n: n.loc)
        self._ref_fwd = ref_fwd

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

        self._calc_spline_params()

    @staticmethod
    @numba.jit
    def _calc_spline_params_numba(
        atm_var: float,
        skew: float,
        locs: npt.NDArray[np.float64],
        crvs: npt.NDArray[np.float64],
        zero_idx: int,
    ):
        num_polys = len(locs) + 1
        params = np.empty(shape=(num_polys, 4), dtype=float)

        params[1 : num_polys - 1, 0] = (
            (crvs[: num_polys - 2] - crvs[1 : num_polys - 1])
            / (locs[: num_polys - 2] - locs[1 : num_polys - 1])
            / 6.0
        )
        params[1 : num_polys - 1, 1] = (
            0.5 * crvs[1 : num_polys - 1]
            - 3.0 * params[1 : num_polys - 1, 0] * locs[1 : num_polys - 1]
        )
        params[0, 0] = params[-1, 0] = 0.0
        params[0, 1] = 0.5 * crvs[0]
        params[-1, 1] = 0.5 * crvs[-1]

        params[zero_idx, 3] = params[zero_idx + 1, 3] = atm_var
        params[zero_idx, 2] = params[zero_idx + 1, 2] = skew

        for i in range(zero_idx - 1, -1, -1):
            loc_sq = locs[i] * locs[i]
            loc_cb = loc_sq * locs[i]
            params[i, 2] = (
                3.0 * params[i + 1, 0] * loc_sq
                + 2.0 * params[i + 1, 1] * locs[i]
                + params[i + 1, 2]
            ) - (3.0 * params[i, 0] * loc_sq + 2.0 * params[i, 1] * locs[i])
            params[i, 3] = (
                params[i + 1, 0] * loc_cb
                + params[i + 1, 1] * loc_sq
                + params[i + 1, 2] * locs[i]
                + params[i + 1, 3]
            ) - (params[i, 0] * loc_cb + params[i, 1] * loc_sq + params[i, 2] * locs[i])

        for i in range(zero_idx + 1, num_polys - 1):
            loc_sq = locs[i] * locs[i]
            loc_cb = loc_sq * locs[i]
            params[i + 1, 2] = (
                3.0 * params[i, 0] * loc_sq
                + 2.0 * params[i, 1] * locs[i]
                + params[i, 2]
            ) - (3.0 * params[i + 1, 0] * loc_sq + 2.0 * params[i + 1, 1] * locs[i])
            params[i + 1, 3] = (
                params[i, 0] * loc_cb
                + params[i, 1] * loc_sq
                + params[i, 2] * locs[i]
                + params[i, 3]
            ) - (
                params[i + 1, 0] * loc_cb
                + params[i + 1, 1] * loc_sq
                + params[i + 1, 2] * locs[i]
            )

        return params

    def _calc_spline_params(self):
        params = self._calc_spline_params_numba(
            self._atm_var, self._skew, self._locs, self._crvs, self._zero_idx
        )
        self._cubic_spline = CubicSpline(self._locs, params)

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
        z = np.log(k / self._ref_fwd)
        vals = self._cubic_spline(z)
        vols = np.sqrt(vals[0])
        deriv1_vols_z = vals[1] / (2.0 * vols)
        deriv2_vols_z = (0.5 * vals[2] - deriv1_vols_z**2.0) / vols

        return (vols, deriv1_vols_z / k, deriv2_vols_z / k - deriv1_vols_z / (k**2.0))
