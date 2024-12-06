import pandas as pd
import numpy as np
from pandera.typing import DataFrame

from cvi.option_chain import OptionChain
from cvi.slice import CviRealParams, CviNode, CviSlice

from .fitter_options import CviVolFitterOptions


class CviVolFitter:
    _slices: dict[pd.Timestamp, CviSlice]
    _mid_chain: DataFrame[OptionChain]

    def __init__(self):
        self.reset()

    def fit(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
        fitter_options: CviVolFitterOptions,
    ):
        self._reset()
        self._init(chain, node_locs)

    def weights(self, fitter_options: CviVolFitterOptions):
        if fitter_options.weighting_least_sq == "vol_spread":
            inverse_vol_spreads = 1.0 / (
                self._mid_chain["iv_ask"] - self._mid_chain["iv_bid"]
            )
            sum_inverse_vol_spreads = inverse_vol_spreads.sum()
            sum_inverse_var_spreads = (
                1.0 / (self._mid_chain["iv_ask"] ** 2 - self._mid_chain["iv_bid"] ** 2)
            ).sum()
            return (
                inverse_vol_spreads / sum_inverse_vol_spreads * sum_inverse_var_spreads
            )
        return np.eye(self._mid_chain.shape[0])

    def _reset(self):
        self._slices = dict()
        self._mid_chain = dict()

    def _init(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ):
        for exp in chain["expiry"]:
            df = chain[chain["expiry"] == exp]

            fwd = df["fwd_mid"].iloc[0]
            t_e = df["t_e"].iloc[0]
            # take vol nearest to the forward as the anchor
            atm_anchor_vol = df.iloc[(df["strike"] - fwd).abs().argsort()[0]]["iv_mid"]

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
