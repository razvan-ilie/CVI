import pandas as pd
import numpy as np
import numpy.typing as npt
from pandera.typing import DataFrame, Series

from cvi.option_chain import OptionChain
from cvi.slice import CviRealParams, CviNode, CviSlice

from .fitter_options import CviVolFitterOptions


class CviVolFitter:
    _slices: dict[pd.Timestamp, CviSlice]
    _mid_chain: DataFrame[OptionChain]
    _mid_chain_num_mids: Series[int]

    def __init__(self):
        self._slices = dict()

    def fit(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
        fitter_options: CviVolFitterOptions,
    ):
        self._initialize(chain, node_locs)
        # weights_mat = self.weights(fitter_options)
        # mid_var = self._mid_chain["iv_mid"] ** 2
        self._mid_chain["z"] = self._mid_chain.apply(
            lambda r: self._slices[r["expiry"]].k_to_z(r["strike"]), axis=1
        )

    def weights(self, fitter_options: CviVolFitterOptions) -> npt.NDArray[np.float64]:
        if fitter_options.weighting_least_sq == "vol_spread":
            inverse_vol_spreads = 1.0 / (
                self._mid_chain["iv_ask"] - self._mid_chain["iv_bid"]
            )
            sum_inverse_vol_spreads = inverse_vol_spreads.sum()
            sum_inverse_var_spreads = (
                1.0 / (self._mid_chain["iv_ask"] ** 2 - self._mid_chain["iv_bid"] ** 2)
            ).sum()

            return np.diag(
                inverse_vol_spreads
                / sum_inverse_vol_spreads
                * sum_inverse_var_spreads
                / self._mid_chain_num_mids
            )

        return np.eye(self._mid_chain.shape[0])

    def basis_val_matrix(self):
        ####
        self._mid_chain["z"] = self._mid_chain.apply(
            lambda r: self._slices[r["expiry"]].k_to_z(r["strike"]), axis=1
        )
        ####

        mats = self._mid_chain.groupby("expiry")[["expiry", "z"]].apply(
            lambda df_exp: self._slices[
                df_exp["expiry"].iloc[0]
            ].spline_params.val_basis_funcs(df_exp["z"], der=0),
            include_groups=False,
        )
        return mats

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
            atm_anchor_vol = df.iloc[(df["strike"] - fwd).abs().argsort().iloc[0]][
                "iv_mid"
            ]

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

        self._mid_chain = chain[chain["iv_mid"].notna()]

        num_mids_by_exp = self._mid_chain.groupby("expiry").count()["iv_mid"]
        self._mid_chain_num_mids = self._mid_chain["expiry"].apply(
            lambda exp: num_mids_by_exp[exp]
        )
