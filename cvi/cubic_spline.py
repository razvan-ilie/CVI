import numpy as np
import numpy.typing as npt


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

    @staticmethod
    def cvi_spline_params(
        atm_var: float,
        skew: float,
        locs: npt.NDArray[np.float64],
        crvs: npt.NDArray[np.float64],
        zero_idx: int | None = None,
    ):
        zero_idx = zero_idx or next(i for i, loc in enumerate(locs) if loc == 0.0)
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

        return CubicSpline(locs, params)
