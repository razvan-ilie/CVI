import typing

import clarabel as cb
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from scipy.sparse import block_array, csc_matrix

from cvi.option_chain import OptionChain
from cvi.slice import CviCubicBSplineParams, CviNode, CviRealParams, CviSlice

from .fitter_options import CviVolFitterOptions


class CviVolFitter:
    _slices: dict[pd.Timestamp, CviSlice]

    def __init__(self, fitter_options: CviVolFitterOptions | None = None):
        self._slices = dict()
        self.fitter_options: CviVolFitterOptions = fitter_options or CviVolFitterOptions()

    def fit(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ) -> dict[pd.Timestamp, CviSlice]:
        chain_annotated = self._annotate_chain(chain, node_locs)
        weights_least_sq_mat = self._weights_least_sq(chain_annotated, self.fitter_options)
        mid_var = np.array(chain_annotated["iv_mid"] ** 2)

        basis_value_matrix = self._basis_value_matrix(chain_annotated)

        # TODO: Add below bid and above ask penalties
        # For the above ask and below bid penalties, we need to add a slack variable >= 0 for
        # each (strike, expiry) pair that has a bid (ask) but no ask (bid) and enforce the
        # constraint that the slack variable be >= (CVI variance - ask variance) or
        # >= (bid variance - CVI variance). We minimize the weighted sum of the square of
        # these slack variables.

        p_mat = self.p_matrix(basis_value_matrix, weights_least_sq_mat)
        q_vec = self.q_vector(basis_value_matrix, weights_least_sq_mat, mid_var)
        num_mids = q_vec.shape[0]
        a_mat, b_vec, cones = self.constraints(chain_annotated, node_locs)

        settings = cb.DefaultSettings()
        solver = cb.DefaultSolver(p_mat, q_vec, a_mat, b_vec, cones, settings)
        sol = solver.solve()

        expiries = list()
        for exp in chain_annotated["expiry"]:
            if exp not in expiries:
                expiries.append(exp)

        for i, exp in enumerate(expiries):
            original_slice = self._slices[exp]
            num_params = len(original_slice.spline_params.coeffs)

            self._slices[exp] = CviSlice.from_spline_params(
                CviCubicBSplineParams(
                    knots=original_slice.spline_params.knots,
                    coeffs=sol.x[i * num_params : (i + 1) * num_params],
                ),
                original_slice._ref_fwd,
                original_slice._t_e,
                atm_anchor_var=original_slice._atm_anchor_var,
            )

        return self._slices

    def _weights_least_sq(
        self,
        chain: DataFrame[OptionChain],
        fitter_options: CviVolFitterOptions,
    ) -> np.ndarray:
        """The weight to apply to each error term in the least squares penalty."""
        if fitter_options.weighting_least_sq == "var_spread":
            inv_sq_var_spreads = 1.0 / ((chain["iv_ask"] ** 2 - chain["iv_bid"] ** 2) ** 2)
            return np.diag(inv_sq_var_spreads / chain["num_mids_at_expiry"])
        elif fitter_options.weighting_least_sq == "none":
            return np.eye(chain.shape[0])
        else:
            raise NotImplementedError(
                f"{fitter_options.weighting_least_sq} weighting not implemented"
            )

    def _weights_outside_bidask(
        self,
        chain: DataFrame[OptionChain],
        fitter_options: CviVolFitterOptions,
        quote_type: typing.Literal["bid", "ask"],
    ) -> np.ndarray:
        """The weight to apply to each error term in the above ask penalty."""
        setting = (
            fitter_options.weighting_above_ask
            if quote_type == "ask"
            else fitter_options.weighting_below_bid
        )
        if setting == "vega_normalized":
            q_j = chain.groupby("expiry")[[f"iv_{quote_type}", f"iv_{quote_type}"]].apply(
                lambda df_exp: (
                    1.0 / ((df_exp[f"iv_{quote_type}"] ** 2 - df_exp[f"iv_{quote_type}"] ** 2) ** 2)
                ).sum()
            )
            sum_q_vegas = chain.groupby("expiry")[f"vega_{quote_type}"].sum()
            weights = chain[["expiry", f"vega_{quote_type}"]].apply(
                lambda r: q_j[r["expiry"]] * r[f"vega_{quote_type}"] / sum_q_vegas[r["expiry"]],
                axis=1,
            )
            return np.diag(weights / chain[f"num_{quote_type}s_at_expiry"])
        else:
            raise NotImplementedError(
                f"{fitter_options.weighting_least_sq} weighting not implemented"
            )

    def _basis_value_matrix(self, chain: DataFrame[OptionChain]) -> csc_matrix:
        """The basis value matrix for the spline basis functions."""
        mats = chain.groupby("expiry")[["expiry", "z"]].apply(
            lambda df_exp: self._slices[df_exp["expiry"].iloc[0]].spline_params.val_basis_funcs(  # type: ignore
                df_exp["z"],  # type: ignore
                der=0,
            ),
            include_groups=False,
        )

        blocks = [[None] * i + [mat] + [None] * (len(mats) - i - 1) for i, mat in enumerate(mats)]

        return csc_matrix(block_array(blocks))

    def p_matrix(
        self,
        basis_value_matrix: csc_matrix,
        weights_least_sq_mat: np.ndarray,
    ) -> csc_matrix:
        """The P matrix in the least squares problem."""
        return csc_matrix(basis_value_matrix.T @ weights_least_sq_mat @ basis_value_matrix)

    def q_vector(
        self,
        basis_value_matrix: csc_matrix,
        weights_least_sq_mat: np.ndarray,
        mid_var: np.ndarray,
    ) -> np.ndarray:
        """The q vector in the least squares problem."""
        return -basis_value_matrix.T @ weights_least_sq_mat @ mid_var

    def constraints(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ) -> tuple[csc_matrix, np.ndarray, list[cb.ZeroConeT | cb.NonnegativeConeT]]:  # type: ignore
        """The A constraint matrix."""
        positive_var_constraint = self._positive_variance_constraint(chain, node_locs)
        positive_var_vec = np.zeros(positive_var_constraint.shape[0])  # type: ignore
        positive_var_cone = cb.NonnegativeConeT(positive_var_constraint.shape[0])  # type: ignore

        linear_extrapolation_constraint = self._linear_extrapolation_constraint(chain, node_locs)
        linear_extrapolation_vec = np.zeros(linear_extrapolation_constraint.shape[0])  # type: ignore
        linear_extrapolation_cone = cb.ZeroConeT(linear_extrapolation_vec.shape[0])  # type: ignore

        upward_sloping_constraint = self._upward_sloping_constraint(chain, node_locs)
        upward_sloping_vec = np.zeros(upward_sloping_constraint.shape[0])  # type: ignore
        upward_sloping_cone = cb.NonnegativeConeT(upward_sloping_vec.shape[0])  # type: ignore

        tail_bounds_constraint = self._tail_bounds_constraint(chain, node_locs)
        tail_bounds_vec = np.ones(tail_bounds_constraint.shape[0])  # type: ignore
        tail_bounds_cone = cb.NonnegativeConeT(tail_bounds_vec.shape[0])  # type: ignore

        return (
            csc_matrix(
                block_array(
                    [
                        [positive_var_constraint],
                        [linear_extrapolation_constraint],
                        [upward_sloping_constraint],
                        [tail_bounds_constraint],
                    ]
                )
            ),
            np.concatenate(
                (positive_var_vec, linear_extrapolation_vec, upward_sloping_vec, tail_bounds_vec)
            ),
            [
                positive_var_cone,
                linear_extrapolation_cone,
                upward_sloping_cone,
                tail_bounds_cone,
            ],
        )

    def b_vector(
        self,
        basis_value_matrix: csc_matrix,
        weights_least_sq_mat: np.ndarray,
        mid_var: np.ndarray,
    ) -> np.ndarray:
        """The q vector in the least squares problem."""
        return -basis_value_matrix.T @ weights_least_sq_mat @ mid_var

    def _positive_variance_constraint(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ) -> csc_matrix:
        """The constraint that the variance is positive."""
        num_pts = self.fitter_options.num_positive_variance_points
        points = np.linspace(node_locs[0], node_locs[-1], num_pts)
        first_expiry = chain["expiry"].iloc[0]

        mats = chain.groupby("expiry")[["expiry", "z"]].apply(
            lambda df_exp: self._slices[df_exp["expiry"].iloc[0]].spline_params.val_basis_funcs(  # type: ignore
                points,  # type: ignore
                der=0,
            )
            if df_exp["expiry"].iloc[0] == first_expiry  # type: ignore
            else None,
            include_groups=False,
        )

        mat = csc_matrix(-1 * mats.iloc[0])
        mat.resize((num_pts, mat.shape[1] * len(mats)))  # type: ignore

        return mat

    def _linear_extrapolation_constraint(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ) -> csc_matrix:
        """The constraint that the variance is linear in the wings."""
        points = np.array([node_locs[0], node_locs[-1]])

        mats = chain.groupby("expiry")[["expiry", "z"]].apply(
            lambda df_exp: self._slices[df_exp["expiry"].iloc[0]].spline_params.val_basis_funcs(  # type: ignore
                points,
                der=2,
            ),
            include_groups=False,
        )

        blocks = [[None] * i + [mat] + [None] * (len(mats) - i - 1) for i, mat in enumerate(mats)]

        return csc_matrix(block_array(blocks))  # type: ignore

    def _upward_sloping_constraint(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ) -> csc_matrix:
        """The constraint that the variance is upward sloping in the wings."""

        mats_put = chain.groupby("expiry")[["expiry", "z"]].apply(
            lambda df_exp: self._slices[df_exp["expiry"].iloc[0]].spline_params.val_basis_funcs(  # type: ignore
                node_locs[0],
                der=1,
            ),
            include_groups=False,
        )
        mats_call = chain.groupby("expiry")[["expiry", "z"]].apply(
            lambda df_exp: self._slices[df_exp["expiry"].iloc[0]].spline_params.val_basis_funcs(  # type: ignore
                node_locs[-1],
                der=1,
            )
            * -1.0,
            include_groups=False,
        )

        blocks_put = [
            [None] * i + [mat] + [None] * (len(mats_put) - i - 1) for i, mat in enumerate(mats_put)
        ]
        blocks_call = [
            [None] * i + [mat] + [None] * (len(mats_call) - i - 1)
            for i, mat in enumerate(mats_call)
        ]

        return csc_matrix(block_array(blocks_put + blocks_call))

    def _tail_bounds_constraint(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ) -> csc_matrix:
        """The constraint that the variance is within the Lee bounds in the tails."""
        mats_put = chain.groupby("expiry")[
            [
                "expiry",
                "z",
                "t_e",
            ]
        ].apply(
            lambda df_exp: self._slices[df_exp["expiry"].iloc[0]].spline_params.val_basis_funcs(  # type: ignore
                node_locs[0],
                der=1,
            )
            * (df_exp["t_e"].iloc[0] ** 0.5)
            * -0.5
            / self._slices[df_exp["expiry"].iloc[0]].atm_anchor_vol,  # type: ignore
            include_groups=False,
        )
        mats_call = chain.groupby("expiry")[
            [
                "expiry",
                "z",
                "t_e",
            ]
        ].apply(
            lambda df_exp: self._slices[df_exp["expiry"].iloc[0]].spline_params.val_basis_funcs(  # type: ignore
                node_locs[-1],
                der=1,
            )
            * (df_exp["t_e"].iloc[0] ** 0.5)
            * 0.5
            / self._slices[df_exp["expiry"].iloc[0]].atm_anchor_vol,  # type: ignore
            include_groups=False,
        )

        blocks_put = [
            [None] * i + [mat] + [None] * (len(mats_put) - i - 1) for i, mat in enumerate(mats_put)
        ]
        blocks_call = [
            [None] * i + [mat] + [None] * (len(mats_call) - i - 1)
            for i, mat in enumerate(mats_call)
        ]

        return csc_matrix(block_array(blocks_put + blocks_call))

    def _annotate_chain(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ) -> DataFrame[OptionChain]:
        for exp in chain["expiry"].unique():
            df = chain[chain["expiry"] == exp]

            fwd = df["fwd_mid"].iloc[0]
            t_e = df["t_e"].iloc[0]
            # take vol nearest to the forward as the anchor
            atm_anchor_vol = df.iloc[(df["strike"] - fwd).abs().argsort().iloc[0]]["iv_mid"]

            self._slices[exp] = CviSlice.from_real_params(
                CviRealParams(
                    atm_var=atm_anchor_vol,
                    skew=0.0,
                    nodes=[CviNode(loc, 0.0) for loc in node_locs],
                ),
                fwd,
                t_e,
                atm_anchor_vol,
            )

        chain["z"] = chain.apply(lambda r: self._slices[r["expiry"]].k_to_z(r["strike"]), axis=1)

        annotated_chain = chain[chain["iv_mid"].notna()]
        annotated_chain = annotated_chain[annotated_chain["z"].between(node_locs[0], node_locs[-1])]

        for q_type in ("bid", "ask", "mid"):
            num_qs_by_exp = annotated_chain.groupby("expiry").count()[f"iv_{q_type}"]
            annotated_chain[f"num_{q_type}s_at_expiry"] = annotated_chain["expiry"].apply(
                lambda exp, num_qs_by_exp=num_qs_by_exp: num_qs_by_exp[exp]
            )

        return annotated_chain
