import pandera as pa
from pandera.typing import Series


class OptionChain(pa.DataFrameModel):
    expiry: Series[pa.DateTime] = pa.Field(coerce=True)
    num_mids_at_expiry: Series[int] = pa.Field(nullable=True)
    num_bids_at_expiry: Series[int] = pa.Field(nullable=True)
    num_asks_at_expiry: Series[int] = pa.Field(nullable=True)
    strike: Series[float] = pa.Field()
    t_e: Series[float] = pa.Field()
    c_bid: Series[float] = pa.Field(nullable=True)
    c_ask: Series[float] = pa.Field(nullable=True)
    c_mid: Series[float] = pa.Field(nullable=True)
    p_bid: Series[float] = pa.Field(nullable=True)
    p_ask: Series[float] = pa.Field(nullable=True)
    p_mid: Series[float] = pa.Field(nullable=True)
    fwd_bid: Series[float] = pa.Field(nullable=True)
    fwd_ask: Series[float] = pa.Field(nullable=True)
    fwd_mid: Series[float] = pa.Field()
    disc_bid: Series[float] = pa.Field(nullable=True)
    disc_ask: Series[float] = pa.Field(nullable=True)
    disc_mid: Series[float] = pa.Field()
    iv_c_bid: Series[float] = pa.Field(nullable=True)
    iv_c_ask: Series[float] = pa.Field(nullable=True)
    iv_c_mid: Series[float] = pa.Field(nullable=True)
    iv_p_bid: Series[float] = pa.Field(nullable=True)
    iv_p_ask: Series[float] = pa.Field(nullable=True)
    iv_p_mid: Series[float] = pa.Field(nullable=True)
    iv_bid: Series[float] = pa.Field(nullable=True)
    iv_ask: Series[float] = pa.Field(nullable=True)
    iv_mid: Series[float] = pa.Field(nullable=True)
    vega_bid: Series[float] = pa.Field(nullable=True)
    vega_ask: Series[float] = pa.Field(nullable=True)
    vega_mid: Series[float] = pa.Field(nullable=True)


class EnrichedOptionChain(OptionChain):
    z: Series[float] = pa.Field(nullable=True)
    var_bid: Series[float] = pa.Field(nullable=True)
    var_mid: Series[float] = pa.Field(nullable=True)
    var_ask: Series[float] = pa.Field(nullable=True)
