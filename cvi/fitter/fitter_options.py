from typing import Literal
import pydantic


class CviVolFitterOptions(pydantic.BaseModel):
    weighting_least_sq: Literal["vol_spread"] | None
    weighting_above_ask: Literal["vega"] | None
    weighting_below_bid: Literal["vega"] | None
    strike_regularization_factor: float | None
