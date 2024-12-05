import pandas as pd
from pandera.typing import DataFrame

from cvi.option_chain import OptionChain
from cvi.slice import CviRealParams, CviNode, CviSlice

from .fitter_options import CviVolFitterOptions


class CviVolFitter:
    _slices: dict[pd.Timestamp, CviSlice]
    _chains: dict[pd.Timestamp, DataFrame[OptionChain]]

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

    def _reset(self):
        self._slices = dict()
        self._chains = dict()

    def _init(
        self,
        chain: DataFrame[OptionChain],
        node_locs: list[float],
    ):
        for exp in chain["expiry"]:
            df = chain[chain["expiry"] == exp]
            self._chains[exp] = df

            fwd = df["fwd_mid"].iloc[0]
            t_e = df["t_e"].iloc[0]
            atm_anchor_vol = df.iloc[(df["strike"] - fwd).abs().argsort()[0]]["iv_mid"]

            self._slices[exp] = CviSlice.from_real_params(
                CviRealParams(
                    atm_var=0.2,
                    skew=0.0,
                    nodes=[CviNode(loc, 0.0) for loc in node_locs],
                ),
                fwd,
                t_e,
                atm_anchor_vol,
            )
