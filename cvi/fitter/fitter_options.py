from typing import Literal

from pydantic import BaseModel

LeastSquaresWeighting = Literal["vol_spread", "var_spread", "none"]
OutsideBidAskWeighting = Literal["vega_normalized", "none"]


class CviVolFitterOptions(BaseModel):
    weighting_least_sq: LeastSquaresWeighting = "var_spread"
    weighting_above_ask: OutsideBidAskWeighting = "vega_normalized"
    weighting_below_bid: OutsideBidAskWeighting = "vega_normalized"
    num_positive_variance_points: int = 10
    strike_regularization_factor: float = 0.05
