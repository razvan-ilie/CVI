import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev
from scipy.linalg import solve
from typing import cast, Self

from .node import CviNode


class CviRealParams:
    atm_var: float
    skew: float  # d variance / d z
    nodes: list[CviNode]
    crvs: list[float]
    locs: list[float]

    def __init__(
        self,
        atm_var: float,
        skew: float,
        nodes: list[CviNode],
    ):
        self.nodes = sorted(nodes, key=lambda n: n.loc)

        if not any(node.is_zero for node in self.nodes):
            raise ValueError("Nodes must contain a node at z = 0")
        if self.nodes[0].crv != 0.0:
            raise ValueError("First node must have a curvature of 0")
        if self.nodes[-1].crv != 0.0:
            raise ValueError("Last node must have a curvature of 0")

        self.atm_var = atm_var
        self.skew = -skew / 100.0

        self.crvs = [n.crv for n in self.nodes]
        self.locs = [n.loc for n in self.nodes]

    @classmethod
    def from_spline_params(cls, p: "CviCubicBSplineParams") -> Self:
        locs = p.knots[3:-3]

        atm_var = cast(float, splev(0.0, p.tck))
        skew = cast(float, splev(0.0, p.tck, der=1))
        crvs = splev(locs, p.tck, der=2)

        nodes = [CviNode(loc, crv) for loc, crv in zip(locs, crvs)]

        return cls(atm_var, -100.0 * skew, nodes)


class CviCubicBSplineParams:
    knots: npt.NDArray[np.float64]
    coeffs: npt.NDArray[np.float64]
    deriv_left: float
    deriv_right: float

    def __init__(
        self,
        knots: npt.NDArray[np.float64] | list[float],
        coeffs: npt.NDArray[np.float64] | list[float],
    ):
        if isinstance(knots, list):
            knots = np.array(knots)
        if isinstance(coeffs, list):
            coeffs = np.array(coeffs)

        self.knots = knots
        self.coeffs = coeffs

        derivs: np.ndarray = splev(
            [self.knots[0], self.knots[-1]], self.tck, der=1, ext=1
        )
        self.deriv_left = cast(float, derivs[0])
        self.deriv_right = cast(float, derivs[1])

    @property
    def tck(self):
        return self.knots, self.coeffs, 3

    @classmethod
    def from_real_params(cls, p: "CviRealParams") -> Self:
        knots = np.array([p.locs[0]] * 3 + p.locs + [p.locs[-1]] * 3)
        n = len(p.locs) + 2

        c = [0.0] * n
        mat = np.zeros((n, n))
        for i in range(n):
            c[i] = 1.0
            if i != 0:
                c[i - 1] = 0.0
            mat[: n - 2, i] = splev(p.locs, (knots, c, 3), der=2, ext=1)
            mat[n - 2, i] = splev(0.0, (knots, c, 3), der=1, ext=1)
            mat[n - 1, i] = splev(0.0, (knots, c, 3), der=0, ext=1)

        b = p.crvs + [p.skew, p.atm_var]
        coeffs = solve(mat, b)

        return cls(knots, coeffs)

    def val_basis_funcs(
        self, x: npt.NDArray[np.float64], der: int
    ) -> npt.NDArray[np.float64]:
        """Returns a matrix of floats where each row corresponds to the
           value (or nth derivative) of each basis function for a fixed x.
           Each column would then correspond to value of one fixed basis
           function for all x.

        Args:
            x (npt.NDArray[np.float64]): inputs to evaluate basis functions at.
                size: n x 1
            der (int): nth derivative to take

        Returns:
            npt.NDArray[np.float64]: matrix of values (or nth derivatives)
                of each basis function at each input value.
                size: n x m, where n is the length of the input x and
                m is the number of coefficients
        """
        n = len(x)
        m = len(self.knots) - 4
        res = np.zeros((n, m))
        t, _, k = self.tck
        c = np.zeros(m)
        for i in range(m):
            c[i] = 1.0
            if i != 0:
                c[i - 1] = 0
            res[:, i] = splev(x, (t, c, k), der=der, ext=1)

        return res
