from typing import Literal

from pydantic import BaseModel, Field


class CviVolFitterOptions(BaseModel):
    weighting_least_sq: Literal["vol_spread"] | None = Field(default="vol_spread")
    weighting_above_ask: Literal["vega"] | None = Field(default="vega")
    weighting_below_bid: Literal["vega"] | None = Field(default="vega")
    strike_regularization_factor: float | None = Field(default=0.05)
