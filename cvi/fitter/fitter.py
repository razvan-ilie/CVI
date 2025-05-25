import clarabel as cb
import numpy as np
import pandas as pd
from pandera.typing import DataFrame, Series
from scipy.sparse import block_array, csc_matrix

from cvi.option_chain import OptionChain
from cvi.slice import CviCubicBSplineParams, CviNode, CviRealParams, CviSlice

from .fitter_options import CviVolFitterOptions


class CviVolFitter:
    _slices: dict[pd.Timestamp, CviSlice]
    _mid_chain: DataFrame[OptionChain]
    _mid_chain_num_mids: Series[int]

    def __init__(self, fitter_options: CviVolFitterOptions | None = None):
        self._slices = dict()
        self.fitter_options: CviVolFitterOptions = fitter_options or CviVolFitterOptions()

    def fit(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ) -> dict[pd.Timestamp, CviSlice]:
        self._initialize(chain, node_locs)
        weights_least_sq_mat = self.weights_least_sq(self.fitter_options)
        mid_var = self._mid_chain["iv_mid"] ** 2
        basis_val_mat = self.basis_val_matrix()

        P_mat = csc_matrix(basis_val_mat.T @ weights_least_sq_mat @ basis_val_mat)
        q_vec = -basis_val_mat.T @ weights_least_sq_mat @ mid_var
        num_mids = q_vec.shape[0]
        A_mat = csc_matrix(-1.0 * np.eye(num_mids))
        b_vec = np.zeros(num_mids)

        cones = [cb.NonnegativeConeT(num_mids)]

        settings = cb.DefaultSettings()
        solver = cb.DefaultSolver(P_mat, q_vec, A_mat, b_vec, cones, settings)
        sol = solver.solve()

        expiries = list()
        for exp in self._mid_chain["expiry"]:
            if exp not in expiries:
                expiries.append(exp)

        for i, exp in enumerate(expiries):
            original_slice = self._slices[exp]
            num_params = len(original_slice.spline_params.coeffs)

            self._slices[exp] = CviSlice.from_spline_params(
                CviCubicBSplineParams(
                    knots=original_slice.spline_params.knots,
                    coeffs=sol.x[i * num_params + 1 : (i + 1) * num_params - 1],
                ),
                original_slice._ref_fwd,
                original_slice._t_e,
                atm_anchor_var=original_slice._atm_anchor_var,
            )

        return self._slices

    def weights_least_sq(self, fitter_options: CviVolFitterOptions) -> np.ndarray:
        """The weight to apply to each error term in the least squares penalty."""
        if fitter_options.weighting_least_sq == "var_spread":
            var_spreads = self._mid_chain["iv_ask"] ** 2 - self._mid_chain["iv_bid"] ** 2
            return np.diag(var_spreads / self._mid_chain_num_mids)
        elif fitter_options.weighting_least_sq == "none":
            return np.eye(self._mid_chain.shape[0])
        else:
            raise NotImplementedError(
                f"{fitter_options.weighting_least_sq} weighting not implemented"
            )

    def basis_val_matrix(self) -> csc_matrix:
        # Create a mapping from expiry to a unique index, preserving first-seen order
        expiry_order = []
        expiry_to_idx = {}
        for exp in self._mid_chain["expiry"]:
            if exp not in expiry_to_idx:
                expiry_to_idx[exp] = len(expiry_order)
                expiry_order.append(exp)

        # Return a Series of (basis_func_result, expiry_index) tuples
        mats_and_indexes = self._mid_chain.apply(
            lambda row: (
                self._slices[row["expiry"]].spline_params.val_basis_funcs(
                    row["z"],  # type: ignore
                    der=0,
                ),
                expiry_to_idx[row["expiry"]],
            ),
            axis=1,
        )

        blocks = [
            [None] * idx + [mat] + [None] * (len(expiry_order) - idx - 1)
            for mat, idx in mats_and_indexes.values
        ]

        return block_array(blocks).toarray()  # type: ignore

    def _initialize(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ):
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

        self._mid_chain = chain[chain["iv_mid"].notna()]
        self._mid_chain = self._mid_chain[self._mid_chain["z"].between(node_locs[0], node_locs[-1])]

        num_mids_by_exp = self._mid_chain.groupby("expiry").count()["iv_mid"]
        self._mid_chain_num_mids = self._mid_chain["expiry"].apply(lambda exp: num_mids_by_exp[exp])
