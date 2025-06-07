import typing

import clarabel as cb
import numpy as np
import pandas as pd
from pandera.typing import DataFrame
from scipy.sparse import block_array, csc_matrix

from cvi.option_chain import EnrichedOptionChain, OptionChain
from cvi.slice import CviCubicBSplineParams, CviNode, CviRealParams, CviSlice

from .fitter_options import CviVolFitterOptions


class CviVolFitter:
    _slices: dict[pd.Timestamp, CviSlice]

    def __init__(self, fitter_options: CviVolFitterOptions):
        self._slices = dict()
        self.fitter_options: CviVolFitterOptions = fitter_options

    def fit(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
        *,
        verbose: bool = False,
    ) -> tuple[dict[pd.Timestamp, CviSlice], DataFrame[EnrichedOptionChain]]:
        self._init_slices(chain, node_locs)
        chain_enriched = self._enrich_chain(chain)

        chain_mids = self._chain_with_mids(chain_enriched, node_locs)
        # NOTE: consider changing below to asks_only and bids_only
        chain_bids = self._chain_with_asks(chain_enriched)
        chain_asks = self._chain_with_bids(chain_enriched, node_locs)

        weights_least_sq_mat = self._weights_least_sq(chain_mids, self.fitter_options)
        weights_above_ask_mat = np.zeros((len(chain_asks), len(chain_asks)))
        weights_below_bid_mat = np.zeros((len(chain_bids), len(chain_bids)))
        # weights_above_ask_mat = self._weights_outside_bidask(chain_asks, self.fitter_options, "ask")
        # weights_below_bid_mat = self._weights_outside_bidask(chain_bids, self.fitter_options, "bid")
        basis_value_matrix = self._basis_value_matrix(chain_mids)

        p_mat = self.p_matrix(
            basis_value_matrix,
            weights_least_sq_mat,
            weights_above_ask_mat,
            weights_below_bid_mat,
        )
        q_vec = self.q_vector(
            basis_value_matrix,
            chain_bids,
            chain_mids,
            chain_asks,
            weights_least_sq_mat,
        )
        a_mat, b_vec, cones = self.constraints(chain_bids, chain_mids, chain_asks, node_locs)

        settings = cb.DefaultSettings()
        settings.verbose = verbose
        solver = cb.DefaultSolver(p_mat, q_vec, a_mat, b_vec, cones, settings)
        sol = solver.solve()

        expiries = list()
        for exp in chain_enriched["expiry"]:
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

        return self._slices, chain_enriched

    def _weights_least_sq(
        self,
        chain: DataFrame[EnrichedOptionChain],
        fitter_options: CviVolFitterOptions,
    ) -> np.ndarray:
        """The weight to apply to each error term in the least squares penalty."""
        if fitter_options.weighting_least_sq == "var_spread":
            inv_sq_var_spreads = 1.0 / ((chain["var_ask"] - chain["var_bid"]) ** 2)
            num_mids_by_exp = chain.groupby("expiry")["iv_mid"].count()
            num_mids = chain["expiry"].apply(lambda exp: num_mids_by_exp[exp])
            return np.diag(inv_sq_var_spreads / num_mids)
        elif fitter_options.weighting_least_sq == "none":
            return np.eye(chain.shape[0])
        else:
            raise NotImplementedError(
                f"{fitter_options.weighting_least_sq} weighting not implemented"
            )

    def _weights_outside_bidask(
        self,
        chain: DataFrame[EnrichedOptionChain],
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
            q_j = chain.groupby("expiry")[["var_ask", "var_bid"]].apply(
                lambda df_exp: (1.0 / ((df_exp["var_ask"] - df_exp["var_bid"]) ** 2)).sum()
            )
            sum_q_vegas = chain.groupby("expiry")[f"vega_{quote_type}"].sum()
            num_quotes_at_exp = chain.groupby("expiry")[f"iv_{quote_type}"].count()
            weights = chain[["expiry", f"vega_{quote_type}"]].apply(
                lambda r: q_j[r["expiry"]]
                * r[f"vega_{quote_type}"]
                / sum_q_vegas[r["expiry"]]
                / num_quotes_at_exp[r["expiry"]],
                axis=1,
            )
            return np.diag(weights)
        else:
            raise NotImplementedError(
                f"{fitter_options.weighting_least_sq} weighting not implemented"
            )

    def _basis_value_matrix(self, chain: DataFrame[EnrichedOptionChain]) -> csc_matrix:
        """The basis value matrix for the spline basis functions."""
        matrices = chain.groupby("expiry")[["expiry", "z"]].apply(
            lambda df_exp: self._slices[df_exp["expiry"].iloc[0]].spline_params.val_basis_funcs(  # type: ignore
                df_exp["z"],  # type: ignore
                der=0,
            ),
            include_groups=False,
        )

        blocks = [
            [None] * i + [mat] + [None] * (len(matrices) - i - 1) for i, mat in enumerate(matrices)
        ]

        return csc_matrix(block_array(blocks))

    def p_matrix(
        self,
        basis_value_matrix: csc_matrix,
        weights_least_sq_mat: np.ndarray,
        weights_above_ask_mat: np.ndarray,
        weights_below_bid_mat: np.ndarray,
    ) -> csc_matrix:
        """The P matrix in the quadratic problem."""
        least_sq_mat = csc_matrix(basis_value_matrix.T @ weights_least_sq_mat @ basis_value_matrix)
        above_ask_mat = csc_matrix(np.eye(weights_above_ask_mat.shape[1]))
        below_bid_mat = csc_matrix(np.eye(weights_below_bid_mat.shape[1]))

        return csc_matrix(
            block_array(
                [
                    [least_sq_mat, None, None],
                    [None, above_ask_mat, None],
                    [None, None, below_bid_mat],
                ]
            )
        )

    def q_vector(
        self,
        basis_value_matrix: csc_matrix,
        chain_bid: DataFrame[EnrichedOptionChain],
        chain_mid: DataFrame[EnrichedOptionChain],
        chain_ask: DataFrame[EnrichedOptionChain],
        weights_least_sq_mat: np.ndarray,
    ) -> np.ndarray:
        """The q vector in the least squares problem."""
        mid_var = chain_mid["var_mid"].to_numpy()
        least_sq_vec = -basis_value_matrix.T @ weights_least_sq_mat @ mid_var
        above_ask_vec = np.zeros(len(chain_ask))
        below_bid_vec = np.zeros(len(chain_bid))
        return np.concat([least_sq_vec, above_ask_vec, below_bid_vec])

    def constraints(
        self,
        chain_bid: DataFrame[EnrichedOptionChain],
        chain_mid: DataFrame[EnrichedOptionChain],
        chain_ask: DataFrame[EnrichedOptionChain],
        node_locs: list[float],
    ) -> tuple[csc_matrix, np.ndarray, list[cb.ZeroConeT | cb.NonnegativeConeT]]:  # type: ignore
        """The A constraint matrix, b constraint values and the cones"""

        above_ask_below_bid_constraint_left, above_ask_below_bid_constraint_right = (
            self._above_ask_below_bid_constraint_left_right(chain_bid, chain_ask)
        )
        above_ask_below_bid_vec = self._above_ask_below_bid_constraint_vec(chain_bid, chain_ask)
        above_ask_below_bid_cone = cb.NonnegativeConeT(above_ask_below_bid_vec.shape[0])

        positive_var_constraint = self._positive_variance_constraint(chain_mid, node_locs)
        positive_var_vec = np.zeros(positive_var_constraint.shape[0])  # type: ignore
        positive_var_cone = cb.NonnegativeConeT(positive_var_constraint.shape[0])  # type: ignore

        linear_extrapolation_constraint = self._linear_extrapolation_constraint(
            chain_mid, node_locs
        )
        linear_extrapolation_vec = np.zeros(linear_extrapolation_constraint.shape[0])  # type: ignore
        linear_extrapolation_cone = cb.ZeroConeT(linear_extrapolation_vec.shape[0])  # type: ignore

        upward_sloping_constraint = self._upward_sloping_constraint(chain_mid, node_locs)
        upward_sloping_vec = np.zeros(upward_sloping_constraint.shape[0])  # type: ignore
        upward_sloping_cone = cb.NonnegativeConeT(upward_sloping_vec.shape[0])  # type: ignore

        tail_bounds_constraint = self._tail_bounds_constraint(chain_mid, node_locs)
        tail_bounds_vec = np.ones(tail_bounds_constraint.shape[0])  # type: ignore
        tail_bounds_cone = cb.NonnegativeConeT(tail_bounds_vec.shape[0])  # type: ignore

        A_mat = csc_matrix(
            block_array(
                [
                    [above_ask_below_bid_constraint_left, above_ask_below_bid_constraint_right],
                    [positive_var_constraint, None],
                    [linear_extrapolation_constraint, None],
                    [upward_sloping_constraint, None],
                    [tail_bounds_constraint, None],
                ]
            )
        )

        b_vec = np.concatenate(
            (
                # above_ask_below_bid_vec,
                positive_var_vec,
                linear_extrapolation_vec,
                upward_sloping_vec,
                tail_bounds_vec,
            )
        )

        cones = [
            # above_ask_below_bid_cone,
            positive_var_cone,
            linear_extrapolation_cone,
            upward_sloping_cone,
            tail_bounds_cone,
        ]

        return (
            csc_matrix(A_mat.toarray()[above_ask_below_bid_constraint_left.shape[0] :]),
            b_vec,
            cones,
        )

    def _above_ask_below_bid_constraint_left_right(
        self,
        chain_bid: DataFrame[EnrichedOptionChain],
        chain_ask: DataFrame[EnrichedOptionChain],
    ) -> tuple[csc_matrix, csc_matrix]:
        basis_val_mat_bid = self._basis_value_matrix(chain_bid)
        basis_val_mat_ask = self._basis_value_matrix(chain_ask)
        num_bids = len(chain_bid["var_bid"].to_numpy())
        num_asks = len(chain_ask["var_ask"].to_numpy())
        ident_bid = np.identity(num_bids)
        ident_ask = np.identity(num_asks)
        return (
            csc_matrix(
                block_array(
                    [
                        [basis_val_mat_ask],
                        [np.zeros((num_asks, basis_val_mat_ask.shape[1]))],  # type: ignore
                        [-basis_val_mat_bid],
                        [np.zeros((num_bids, basis_val_mat_bid.shape[1]))],  # type: ignore
                    ]
                )
            ),
            csc_matrix(
                block_array(
                    [
                        [-ident_ask, None],
                        [-ident_ask, None],
                        [None, -ident_bid],
                        [None, -ident_bid],
                    ]
                )
            ),
        )

        # return csc_matrix(
        #     block_array(
        #         [
        #             [basis_val_mat_ask, -ident_ask, None],
        #             [None, -ident_ask, None],
        #             [-basis_val_mat_bid, None, -ident_bid],
        #             [None, None, -ident_bid],
        #         ]
        #     )
        # )

    def _above_ask_below_bid_constraint_vec(
        self,
        chain_bid: DataFrame[EnrichedOptionChain],
        chain_ask: DataFrame[EnrichedOptionChain],
    ) -> np.ndarray:
        num_asks = len(chain_ask)
        num_bids = len(chain_bid)
        return np.concat(
            [
                chain_ask["var_ask"].to_numpy(),
                np.zeros(num_asks),
                -chain_bid["var_bid"].to_numpy(),
                np.zeros(num_bids),
            ]
        )

    def _positive_variance_constraint(
        self,
        chain: DataFrame[EnrichedOptionChain],
        node_locs: list[float],
    ) -> csc_matrix:
        """The constraint that the variance is positive."""
        num_pts = self.fitter_options.num_positive_variance_points
        points = np.linspace(node_locs[0], node_locs[-1], num_pts)

        mats = [
            -1
            * self._slices[exp].spline_params.val_basis_funcs(  # type: ignore
                points,  # type: ignore
                der=0,
            )
            if i == 0
            else None
            for i, exp in enumerate(chain["expiry"].unique())
        ]

        # mats = chain.groupby("expiry")["expiry"].apply(
        #     lambda exp: self._slices[first_expiry].spline_params.val_basis_funcs(  # type: ignore
        #         points,  # type: ignore
        #         der=0,
        #     )
        #     if exp == first_expiry  # type: ignore
        #     else None,
        #     include_groups=False,
        # )

        mat = csc_matrix(mats[0])
        mat.resize((num_pts, mat.shape[1] * len(mats)))  # type: ignore

        return mat

    def _linear_extrapolation_constraint(
        self,
        chain: DataFrame[EnrichedOptionChain],
        node_locs: list[float],
    ) -> csc_matrix:
        """The constraint that the variance is linear in the wings."""
        points = np.array([node_locs[0], node_locs[-1]])

        mats = [
            self._slices[exp].spline_params.val_basis_funcs(points, der=2)
            for exp in chain["expiry"].unique()
        ]

        # mats = chain.groupby("expiry")["expiry"].apply(
        #     lambda exp: self._slices[exp].spline_params.val_basis_funcs(  # type: ignore
        #         points,
        #         der=2,
        #     ),
        #     include_groups=False,
        # )

        blocks = [[None] * i + [mat] + [None] * (len(mats) - i - 1) for i, mat in enumerate(mats)]

        return csc_matrix(block_array(blocks))  # type: ignore

    def _upward_sloping_constraint(
        self,
        chain: DataFrame[EnrichedOptionChain],
        node_locs: list[float],
    ) -> csc_matrix:
        """The constraint that the variance is upward sloping in the wings."""
        expiries = chain["expiry"].unique()

        mats_put = [
            self._slices[exp].spline_params.val_basis_funcs(  # type: ignore
                node_locs[0],
                der=1,
            )
            for exp in expiries
        ]

        mats_call = [
            -1
            * self._slices[exp].spline_params.val_basis_funcs(  # type: ignore
                node_locs[-1],
                der=1,
            )
            for exp in expiries
        ]

        # mats_put = chain.groupby("expiry")["expiry"].apply(
        #     lambda exp: self._slices[exp].spline_params.val_basis_funcs(  # type: ignore
        #         node_locs[0],
        #         der=1,
        #     ),
        #     include_groups=False,
        # )
        # mats_call = chain.groupby("expiry")["expiry"].apply(
        #     lambda exp: self._slices[exp].spline_params.val_basis_funcs(  # type: ignore
        #         node_locs[-1],
        #         der=1,
        #     )
        #     * -1.0,
        #     include_groups=False,
        # )

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
        chain: DataFrame[EnrichedOptionChain],
        node_locs: list[float],
    ) -> csc_matrix:
        """The constraint that the variance is within the Lee bounds in the tails."""
        mats_put = chain.groupby("expiry")[["expiry", "t_e"]].apply(
            lambda df_exp: self._slices[df_exp["expiry"].iloc[0]].spline_params.val_basis_funcs(  # type: ignore
                node_locs[0],
                der=1,
            )
            * (df_exp["t_e"].iloc[0] ** 0.5)
            * -0.5
            / self._slices[df_exp["expiry"].iloc[0]].atm_anchor_vol,  # type: ignore
            include_groups=False,
        )
        mats_call = chain.groupby("expiry")[["expiry", "t_e"]].apply(
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

    def _init_slices(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ) -> None:
        """Initialize the slices for each expiry in the chain."""
        for exp in chain["expiry"].unique():
            df = chain[chain["expiry"] == exp]

            fwd = df["fwd_mid"].iloc[0]
            t_e = df["t_e"].iloc[0]
            # Take vol nearest to the forward as the anchor (sigma star)
            atm_anchor_vol = df.iloc[(df["strike"] - fwd).abs().argsort().iloc[0]]["iv_mid"]

            # Set up dummy slices with the initial
            self._slices[exp] = CviSlice.from_real_params(
                CviRealParams(
                    atm_var=atm_anchor_vol**2,
                    skew=0.0,
                    nodes=[CviNode(loc, 0.0) for loc in node_locs],
                ),
                fwd,
                t_e,
                atm_anchor_vol,
            )

    def _enrich_chain(
        self,
        chain: DataFrame[OptionChain],
    ) -> DataFrame[EnrichedOptionChain]:
        enriched_chain = DataFrame[EnrichedOptionChain](chain.copy(deep=True))

        # Add z-values
        enriched_chain["z"] = enriched_chain.apply(
            lambda r: self._slices[r["expiry"]].k_to_z(r["strike"]), axis=1
        )

        # Add the variances
        for q_type in ("bid", "ask", "mid"):
            enriched_chain[f"var_{q_type}"] = enriched_chain[f"iv_{q_type}"] ** 2

        return enriched_chain

    def _chain_with_mids(
        self,
        chain: DataFrame[EnrichedOptionChain],
        node_locs: list[float],
    ) -> DataFrame[EnrichedOptionChain]:
        """Return a chain with only strikes that have a mid implied vol and that are within the node locations."""
        return chain[chain["iv_mid"].notna() & (chain["z"].between(node_locs[0], node_locs[-1]))]

    def _chain_with_bids(
        self,
        chain: DataFrame[EnrichedOptionChain],
        node_locs: list[float],
    ) -> DataFrame[EnrichedOptionChain]:
        """Return a chain with all strikes that have a bid implied vol.
        We don't bother checking if we are below bid on strikes outside
        of the node locations that we use to fit so we exclude them."""
        return chain[(chain["iv_bid"].notna()) & (chain["z"].between(node_locs[0], node_locs[-1]))]

    def _chain_with_bids_only(
        self,
        chain: DataFrame[EnrichedOptionChain],
        node_locs: list[float],
    ) -> DataFrame[EnrichedOptionChain]:
        """Return a chain with strikes that have only a bid implied vol and no ask/mid.
        We don't bother checking if we are below bid on strikes outside
        of the node locations that we use to fit so we exclude them."""
        chain_with_bids = self._chain_with_bids(chain, node_locs)
        return chain_with_bids[chain_with_bids["iv_mid"].isna()]

    def _chain_with_asks(
        self,
        chain: DataFrame[EnrichedOptionChain],
    ) -> DataFrame[EnrichedOptionChain]:
        """Return a chain with all strikes that have an ask vol."""
        return chain[chain["iv_ask"].notna()]

    def _chain_with_asks_only(
        self,
        chain: DataFrame[EnrichedOptionChain],
    ) -> DataFrame[EnrichedOptionChain]:
        """Return a chain with only strikes that have an ask implied vol.
        Ask vols are useful even outside of the strikes that we are using to fit.
        Often times you see strikes that have an ask vol but no bid vol in the wings.
        This is useful to help keep our wings from exploding. This could be useful for
        variance swaps and exotics."""
        chain_with_asks = self._chain_with_asks(chain)
        return chain_with_asks[chain_with_asks["iv_mid"].isna()]
