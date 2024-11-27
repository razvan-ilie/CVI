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

    def __init__(
        self,
        atm_var: float,
        skew: float,
        nodes: list[CviNode],
        ref_fwd: float
    ):
        self._atm_var = atm_var
        self._skew = -skew / 100.0
        self._nodes = sorted(nodes, key=lambda n: n.loc)
        self._ref_fwd = ref_fwd

        try:
            self._zero_idx = next(
                i for i, node in enumerate(self._nodes) if node.is_zero()
            )
        except:
            raise ValueError("Nodes must contain a node at z = 0")

        if self._nodes[0].crv != 0.0:
            raise ValueError("First node must have a curvature of 0")
        if self._nodes[-1].crv != 0.0:
            raise ValueError("Last node must have a curvature of 0")

        self.calc_spline_params()

    def calc_spline_params(self):
        num_params = 4 * len(self._nodes) + 4
        num_nodes = len(self._nodes)
        A = np.zeros((num_params, num_params))
        c = np.zeros(num_params)

        # Second derivative must match convexities
        row = 0
        for j in range(num_nodes):
            A[row, j * 4] = 6.0 * self._nodes[j].loc
            A[row + 1, (j + 1) * 4] = 6.0 * self._nodes[j].loc

            A[row, j * 4 + 1] = 2.0
            A[row + 1, (j + 1) * 4 + 1] = 2.0

            c[row] = self._nodes[j].crv
            c[row + 1] = self._nodes[j].crv

            row += 2
        # row = 2 * num_nodes

        # For z = 0, value must match ATM var, derivative must match ATM skew
        for j in range(2):
            A[row, (self._zero_idx + j) * 4] = self._nodes[self._zero_idx].loc ** 3.0
            A[row + 1, (self._zero_idx + j) * 4] = (
                3.0 * self._nodes[self._zero_idx].loc ** 2.0
            )

            A[row, (self._zero_idx + j) * 4 + 1] = (
                self._nodes[self._zero_idx].loc ** 2.0
            )
            A[row + 1, (self._zero_idx + j) * 4 + 1] = (
                2.0 * self._nodes[self._zero_idx].loc
            )

            A[row, (self._zero_idx + j) * 4 + 2] = self._nodes[self._zero_idx].loc
            A[row + 1, (self._zero_idx + j) * 4 + 2] = 1.0

            A[row, (self._zero_idx + j) * 4 + 3] = 1.0

            c[row] = self._atm_var
            c[row + 1] = self._skew
            row += 2
        # row = 2 * num_nodes + 4

        # Cubic term is 0 for first and last polynomial
        A[row, 0] = 1.0
        row += 1
        A[row, -4] = 1.0
        row += 1
        # row = 2 * num_nodes + 6

        # Value and derivative is continuous
        for j in range(num_nodes):
            if j == self._zero_idx:
                continue

            A[row, j * 4] = self._nodes[j].loc ** 3.0
            A[row + 1, j * 4] = 3.0 * self._nodes[j].loc ** 2.0

            A[row, j * 4 + 1] = self._nodes[j].loc ** 2.0
            A[row + 1, j * 4 + 1] = 2.0 * self._nodes[j].loc

            A[row, j * 4 + 2] = self._nodes[j].loc
            A[row + 1, j * 4 + 2] = 1.0

            A[row, j * 4 + 3] = 1.0

            A[row, j * 4 + 4] = -1.0 * self._nodes[j].loc ** 3.0
            A[row + 1, j * 4 + 4] = -3.0 * self._nodes[j].loc ** 2.0

            A[row, j * 4 + 5] = -1.0 * self._nodes[j].loc ** 2.0
            A[row + 1, j * 4 + 5] = -2.0 * self._nodes[j].loc

            A[row, j * 4 + 6] = -1.0 * self._nodes[j].loc
            A[row + 1, j * 4 + 6] = -1.0

            A[row, j * 4 + 7] = -1.0

            row += 2
        # row = 4 * (num_nodes + 1)

        params = np.linalg.solve(A, c)
        self._cubic_spline = CubicSpline(np.array([n.loc for n in self._nodes]), params)

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
        return (
            vols,
            deriv1_vols_z,
            (0.5 * vals[2] - deriv1_vols_z**2.0) / vols
        )

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

        return (
            vols,
            deriv1_vols_z / k,
            deriv2_vols_z / k - deriv1_vols_z / (k**2.0)
        )
