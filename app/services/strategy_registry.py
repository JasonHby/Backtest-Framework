from __future__ import annotations

from app.schemas.strategy import StrategyFieldSpec, StrategySpecResponse


def list_strategy_specs() -> list[StrategySpecResponse]:
    return [
        StrategySpecResponse(
            name="cta",
            engine_type="event-driven",
            description="EMA crossover CTA with ATR stops, partial exits, and giveback logic.",
            fields=[
                StrategyFieldSpec(name="symbol", type="str", default="BTCUSDT", description="Instrument symbol."),
                StrategyFieldSpec(name="short", type="int", default=50, description="Fast EMA length."),
                StrategyFieldSpec(name="long", type="int", default=336, description="Slow EMA length."),
                StrategyFieldSpec(name="stop_atr", type="float", default=2.5, description="ATR stop multiple."),
                StrategyFieldSpec(name="atr_len", type="int", default=50, description="ATR lookback window."),
                StrategyFieldSpec(name="allow_long", type="bool", default=True, description="Allow long entries."),
                StrategyFieldSpec(name="allow_short", type="bool", default=True, description="Allow short entries."),
                StrategyFieldSpec(name="take_profit_r1", type="float", default=0.3, description="First partial take-profit in R."),
                StrategyFieldSpec(name="take_profit_frac1", type="float", default=0.5, description="Fraction of position to close on first take-profit."),
                StrategyFieldSpec(name="take_profit_r", type="float", default=3.0, description="Full take-profit in R."),
                StrategyFieldSpec(name="breakeven_r", type="float", default=0.5, description="Breakeven arm threshold in R."),
                StrategyFieldSpec(name="giveback_k", type="float", default=4.0, description="Giveback ATR multiple."),
                StrategyFieldSpec(name="prefer_giveback", type="bool", default=False, description="Check giveback before fixed take-profit."),
            ],
        )
    ]


def get_strategy_spec(name: str) -> StrategySpecResponse | None:
    for spec in list_strategy_specs():
        if spec.name == name:
            return spec
    return None
