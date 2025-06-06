from math import sqrt
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.interpolate import splev

from .params import CviCubicBSplineParams, CviRealParams


class CviSliceInitError(ValueError):
    def __init__(self):
        super().__init__(
            "CviSlice objects should be created using 'from_real_params' or 'from_spline_params'!"
        )


class CviSlice:
    real_params: CviRealParams
    spline_params: CviCubicBSplineParams
    _ref_fwd: float
    _t_e: float  # time to expiry
    _atm_anchor_var: float
    _z_denom: float  # denominator when calculating z coordinate
    _deriv_left: float
    _deriv_right: float

    _create_key = object()

    def __init__(
        self,
        key: object,
        real_params: CviRealParams,
        spline_params: CviCubicBSplineParams,
        ref_fwd: float,
        t_e: float,
        atm_anchor_var: float | None = None,
    ):
        if key != CviSlice._create_key:
            raise CviSliceInitError()

        self.real_params = real_params
        self.spline_params = spline_params
        self._ref_fwd = ref_fwd
        self._t_e = t_e
        self._atm_anchor_var = atm_anchor_var or self.real_params.atm_var
        self._z_denom = sqrt(self._atm_anchor_var * t_e)

    @classmethod
    def from_real_params(
        cls,
        real_params: CviRealParams,
        ref_fwd: float,
        t_e: float,
        atm_anchor_vol: float | None = None,
    ) -> Self:
        spline_params = CviCubicBSplineParams.from_real_params(real_params)
        atm_anchor_var = atm_anchor_vol**2 if atm_anchor_vol is not None else None
        return cls(cls._create_key, real_params, spline_params, ref_fwd, t_e, atm_anchor_var)

    @classmethod
    def from_spline_params(
        cls,
        spline_params: CviCubicBSplineParams,
        ref_fwd: float,
        t_e: float,
        atm_anchor_var: float | None = None,
    ) -> Self:
        real_params = CviRealParams.from_spline_params(spline_params)
        return cls(cls._create_key, real_params, spline_params, ref_fwd, t_e, atm_anchor_var)

    @property
    def atm_anchor_var(self):
        return self._atm_anchor_var

    @property
    def atm_anchor_vol(self):
        return sqrt(self._atm_anchor_var)

    def var_deriv1_deriv2_z(
        self, z: np.ndarray | float
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        right_extrap = np.where(z <= self.real_params.locs[-1], 0.0, z - self.real_params.locs[-1])
        left_extrap = np.where(z >= self.real_params.locs[0], 0.0, z - self.real_params.locs[0])
        return (
            splev(z, self.spline_params.tck, ext=3)
            + right_extrap * self.spline_params.deriv_right
            + left_extrap * self.spline_params.deriv_left,
            splev(z, self.spline_params.tck, der=1, ext=1)  # type: ignore
            + (z > self.real_params.locs[-1]) * self.spline_params.deriv_right  # type: ignore
            + (z < self.real_params.locs[0]) * self.spline_params.deriv_left,
            splev(z, self.spline_params.tck, der=2, ext=1),  # type: ignore
        )

    def vol_deriv1_deriv2_z(
        self, z: np.ndarray | float
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        vals = self.var_deriv1_deriv2_z(z)

        vols = np.sqrt(vals[0])
        deriv1_vols_z = vals[1] / (2.0 * vols)
        deriv2_vols_z = (0.5 * vals[2] - deriv1_vols_z**2) / vols

        return vols, deriv1_vols_z, deriv2_vols_z

    def vol_deriv1_deriv2_k(
        self, k: npt.NDArray | float
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        z = self.k_to_z(k)

        vals = self.vol_deriv1_deriv2_z(z)

        deriv1_vols_k = vals[1] / k / self._z_denom

        deriv2_vols_k = vals[2] / (k**2) / (self._z_denom**2) - vals[1] / (k**2.0) / self._z_denom
        return vals[0], deriv1_vols_k, deriv2_vols_k

    def k_to_z(self, k: npt.NDArray | float) -> npt.NDArray:
        return np.log(k / self._ref_fwd) / self._z_denom

    def z_to_k(self, z: npt.NDArray | float) -> npt.NDArray:
        return self._ref_fwd * np.exp(z * self._z_denom)
