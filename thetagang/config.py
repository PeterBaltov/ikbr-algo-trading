import math
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, model_validator
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from typing_extensions import Self

from thetagang.fmt import dfmt, ffmt, pfmt

error_console = Console(stderr=True, style="bold red")


class DisplayMixin:
    def add_to_table(self, table: Table, section: str = "") -> None:
        raise NotImplementedError


class AccountConfig(BaseModel, DisplayMixin):
    number: str = Field(...)
    margin_usage: float = Field(..., ge=0.0)
    cancel_orders: bool = Field(default=True)
    market_data_type: int = Field(default=1, ge=1, le=4)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_row("[spring_green1]Account details")
        table.add_row("", "Account number", "=", self.number)
        table.add_row("", "Cancel existing orders", "=", f"{self.cancel_orders}")
        table.add_row(
            "",
            "Margin usage",
            "=",
            f"{self.margin_usage} ({pfmt(self.margin_usage, 0)})",
        )
        table.add_row("", "Market data type", "=", f"{self.market_data_type}")


class ConstantsConfig(BaseModel, DisplayMixin):
    class WriteThreshold(BaseModel):
        write_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        write_threshold_sigma: Optional[float] = Field(default=None, ge=0.0)

    write_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    write_threshold_sigma: Optional[float] = Field(default=None, ge=0.0)
    daily_stddev_window: str = Field(default="30 D")
    calls: Optional["ConstantsConfig.WriteThreshold"] = None
    puts: Optional["ConstantsConfig.WriteThreshold"] = None

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Constants")
        table.add_row("", "Daily stddev window", "=", self.daily_stddev_window)

        c_write_thresh = (
            f"{ffmt(self.calls.write_threshold_sigma)}σ"
            if self.calls and self.calls.write_threshold_sigma
            else pfmt(self.calls.write_threshold if self.calls else None)
        )
        p_write_thresh = (
            f"{ffmt(self.puts.write_threshold_sigma)}σ"
            if self.puts and self.puts.write_threshold_sigma
            else pfmt(self.puts.write_threshold if self.puts else None)
        )

        table.add_row("", "Write threshold for puts", "=", p_write_thresh)
        table.add_row("", "Write threshold for calls", "=", c_write_thresh)


class OptionChainsConfig(BaseModel):
    expirations: int = Field(..., ge=1)
    strikes: int = Field(..., ge=1)


class AlgoSettingsConfig(BaseModel):
    strategy: str = Field("Adaptive")
    params: List[List[str]] = Field(
        default_factory=lambda: [["adaptivePriority", "Patient"]],
        min_length=0,
        max_length=1,
    )


class OrdersConfig(BaseModel, DisplayMixin):
    minimum_credit: float = Field(default=0.0, ge=0.0)
    exchange: str = Field(default="SMART")
    algo: AlgoSettingsConfig = Field(
        default=AlgoSettingsConfig(
            strategy="Adaptive", params=[["adaptivePriority", "Patient"]]
        )
    )
    price_update_delay: List[int] = Field(
        default_factory=lambda: [30, 60], min_length=2, max_length=2
    )

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Order settings")
        table.add_row("", "Exchange", "=", self.exchange)
        table.add_row("", "Params", "=", f"{self.algo.params}")
        table.add_row("", "Price update delay", "=", f"{self.price_update_delay}")
        table.add_row("", "Minimum credit", "=", f"{dfmt(self.minimum_credit)}")


class IBAsyncConfig(BaseModel):
    api_response_wait_time: int = Field(default=60, ge=0)
    logfile: Optional[str] = Field(default="/Users/petarbaltov/Documents/trading/ikbr-algo-trading/logs/ib_async.log")


class IBCConfig(BaseModel):
    tradingMode: Literal["live", "paper"] = Field(default="paper")
    password: Optional[str] = None
    userid: Optional[str] = None
    gateway: bool = Field(default=True)
    RaiseRequestErrors: bool = Field(default=False)
    ibcPath: str = Field(default="/opt/ibc")
    ibcIni: str = Field(default="/etc/thetagang/config.ini")
    twsPath: Optional[str] = None
    twsSettingsPath: Optional[str] = None
    javaPath: str = Field(default="/opt/java/openjdk/bin")
    fixuserid: Optional[str] = None
    fixpassword: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tradingMode": self.tradingMode,
            "password": self.password,
            "userid": self.userid,
            "gateway": self.gateway,
            "ibcPath": self.ibcPath,
            "ibcIni": self.ibcIni,
            "twsPath": self.twsPath,
            "twsSettingsPath": self.twsSettingsPath,
            "javaPath": self.javaPath,
            "fixuserid": self.fixuserid,
            "fixpassword": self.fixpassword,
        }


class WatchdogConfig(BaseModel):
    class ProbeContract(BaseModel):
        currency: str = Field(default="USD")
        exchange: str = Field(default="SMART")
        secType: str = Field(default="STK")
        symbol: str = Field(default="SPY")

    appStartupTime: int = Field(default=30)
    appTimeout: int = Field(default=20)
    clientId: int = Field(default=1)
    connectTimeout: int = Field(default=2)
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=7497)
    probeTimeout: int = Field(default=4)
    readonly: bool = Field(default=False)
    retryDelay: int = Field(default=2)
    probeContract: "WatchdogConfig.ProbeContract" = Field(
        default_factory=lambda: WatchdogConfig.ProbeContract()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "appStartupTime": self.appStartupTime,
            "appTimeout": self.appTimeout,
            "clientId": self.clientId,
            "connectTimeout": self.connectTimeout,
            "host": self.host,
            "port": self.port,
            "probeTimeout": self.probeTimeout,
            "readonly": self.readonly,
            "retryDelay": self.retryDelay,
        }


class CashManagementConfig(BaseModel, DisplayMixin):
    class Orders(BaseModel):
        exchange: str = Field(default="SMART")
        algo: AlgoSettingsConfig = Field(
            default_factory=lambda: AlgoSettingsConfig(strategy="Vwap", params=[])
        )

    enabled: bool = Field(default=False)
    cash_fund: str = Field(default="SGOV")
    target_cash_balance: int = Field(default=0, ge=0)
    buy_threshold: int = Field(default=10000, ge=0)
    sell_threshold: int = Field(default=10000, ge=0)
    primary_exchange: str = Field(default="")
    orders: "CashManagementConfig.Orders" = Field(
        default_factory=lambda: CashManagementConfig.Orders()
    )

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Cash management")
        table.add_row("", "Enabled", "=", f"{self.enabled}")
        table.add_row("", "Cash fund", "=", f"{self.cash_fund}")
        table.add_row("", "Target cash", "=", f"{dfmt(self.target_cash_balance)}")
        table.add_row("", "Buy threshold", "=", f"{dfmt(self.buy_threshold)}")
        table.add_row("", "Sell threshold", "=", f"{dfmt(self.sell_threshold)}")


class VIXCallHedgeConfig(BaseModel, DisplayMixin):
    class Allocation(BaseModel):
        weight: float = Field(..., ge=0.0)
        lower_bound: Optional[float] = Field(default=None, ge=0.0)
        upper_bound: Optional[float] = Field(default=None, ge=0.0)

    enabled: bool = Field(default=False)
    delta: float = Field(default=0.3, ge=0.0, le=1.0)
    target_dte: int = Field(default=30, gt=0)
    ignore_dte: int = Field(default=0, ge=0)
    max_dte: Optional[int] = Field(default=None, ge=1)
    close_hedges_when_vix_exceeds: Optional[float] = None
    allocation: List["VIXCallHedgeConfig.Allocation"] = Field(
        default_factory=lambda: [
            VIXCallHedgeConfig.Allocation(
                lower_bound=None, upper_bound=15.0, weight=0.0
            ),
            VIXCallHedgeConfig.Allocation(
                lower_bound=15.0, upper_bound=30.0, weight=0.01
            ),
            VIXCallHedgeConfig.Allocation(
                lower_bound=30.0, upper_bound=50.0, weight=0.005
            ),
            VIXCallHedgeConfig.Allocation(
                lower_bound=50.0, upper_bound=None, weight=0.0
            ),
        ]
    )

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Hedging with VIX calls")
        table.add_row("", "Enabled", "=", f"{self.enabled}")
        table.add_row("", "Target delta", "<=", f"{self.delta}")
        table.add_row("", "Target DTE", ">=", f"{self.target_dte}")
        table.add_row("", "Ignore DTE", "<=", f"{self.ignore_dte}")
        if self.close_hedges_when_vix_exceeds:
            table.add_row(
                "",
                "Close hedges when VIX",
                ">=",
                f"{self.close_hedges_when_vix_exceeds}",
            )

        for alloc in self.allocation:
            if alloc.lower_bound or alloc.upper_bound:
                table.add_row()
                if alloc.lower_bound:
                    table.add_row(
                        "",
                        f"Allocate {pfmt(alloc.weight)} when VIXMO",
                        ">=",
                        f"{alloc.lower_bound}",
                    )
                if alloc.upper_bound:
                    table.add_row(
                        "",
                        f"Allocate {pfmt(alloc.weight)} when VIXMO",
                        "<=",
                        f"{alloc.upper_bound}",
                    )


class WriteWhenConfig(BaseModel, DisplayMixin):
    class Puts(BaseModel):
        green: bool = Field(default=False)
        red: bool = Field(default=True)

    class Calls(BaseModel):
        green: bool = Field(default=True)
        red: bool = Field(default=False)
        cap_factor: float = Field(default=1.0, ge=0.0, le=1.0)
        cap_target_floor: float = Field(default=0.0, ge=0.0, le=1.0)
        excess_only: bool = Field(default=False)

    calculate_net_contracts: bool = Field(default=False)
    calls: "WriteWhenConfig.Calls" = Field(
        default_factory=lambda: WriteWhenConfig.Calls()
    )
    puts: "WriteWhenConfig.Puts" = Field(default_factory=lambda: WriteWhenConfig.Puts())

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]When writing new contracts")
        table.add_row(
            "",
            "Calculate net contract positions",
            "=",
            f"{self.calculate_net_contracts}",
        )
        table.add_row("", "Puts, write when red", "=", f"{self.puts.red}")
        table.add_row("", "Puts, write when green", "=", f"{self.puts.green}")
        table.add_row("", "Calls, write when green", "=", f"{self.calls.green}")
        table.add_row("", "Calls, write when red", "=", f"{self.calls.red}")
        table.add_row("", "Call cap factor", "=", f"{pfmt(self.calls.cap_factor)}")
        table.add_row(
            "", "Call cap target floor", "=", f"{pfmt(self.calls.cap_target_floor)}"
        )
        table.add_row("", "Excess only", "=", f"{self.calls.excess_only}")


class RollWhenConfig(BaseModel, DisplayMixin):
    class Calls(BaseModel):
        itm: bool = Field(default=True)
        always_when_itm: bool = Field(default=False)
        credit_only: bool = Field(default=False)
        has_excess: bool = Field(default=True)
        maintain_high_water_mark: bool = Field(default=False)

    class Puts(BaseModel):
        itm: bool = Field(default=False)
        always_when_itm: bool = Field(default=False)
        credit_only: bool = Field(default=False)
        has_excess: bool = Field(default=True)

    dte: int = Field(..., ge=0)
    pnl: float = Field(default=0.0, ge=0.0, le=1.0)
    min_pnl: float = Field(default=0.0)
    close_at_pnl: float = Field(default=1.0)
    close_if_unable_to_roll: bool = Field(default=False)
    max_dte: Optional[int] = Field(default=None, ge=1)
    calls: "RollWhenConfig.Calls" = Field(
        default_factory=lambda: RollWhenConfig.Calls()
    )
    puts: "RollWhenConfig.Puts" = Field(default_factory=lambda: RollWhenConfig.Puts())

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Close option positions")
        table.add_row("", "When P&L", ">=", f"{pfmt(self.close_at_pnl, 0)}")
        table.add_row(
            "", "Close if unable to roll", "=", f"{self.close_if_unable_to_roll}"
        )

        table.add_section()
        table.add_row("[spring_green1]Roll options when either condition is true")
        table.add_row(
            "",
            "Days to expiry",
            "<=",
            f"{self.dte} and P&L >= {self.min_pnl} ({pfmt(self.min_pnl, 0)})",
        )

        if self.max_dte:
            table.add_row(
                "",
                "P&L",
                ">=",
                f"{self.pnl} ({pfmt(self.pnl, 0)}) and DTE <= {self.max_dte}",
            )
        else:
            table.add_row("", "P&L", ">=", f"{self.pnl} ({pfmt(self.pnl, 0)})")

        table.add_row("", "Puts: credit only", "=", f"{self.puts.credit_only}")
        table.add_row("", "Puts: roll excess", "=", f"{self.puts.has_excess}")
        table.add_row("", "Calls: credit only", "=", f"{self.calls.credit_only}")
        table.add_row("", "Calls: roll excess", "=", f"{self.calls.has_excess}")
        table.add_row(
            "",
            "Calls: maintain high water mark",
            "=",
            f"{self.calls.maintain_high_water_mark}",
        )

        table.add_section()
        table.add_row("[spring_green1]When contracts are ITM")
        table.add_row(
            "",
            "Roll puts",
            "=",
            f"{self.puts.itm}",
        )
        table.add_row(
            "",
            "Roll puts always",
            "=",
            f"{self.puts.always_when_itm}",
        )
        table.add_row(
            "",
            "Roll calls",
            "=",
            f"{self.calls.itm}",
        )
        table.add_row(
            "",
            "Roll calls always",
            "=",
            f"{self.calls.always_when_itm}",
        )


class TargetConfig(BaseModel, DisplayMixin):
    class Puts(BaseModel):
        delta: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    class Calls(BaseModel):
        delta: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    dte: int = Field(..., ge=0)
    minimum_open_interest: int = Field(..., ge=0)
    maximum_new_contracts_percent: float = Field(0.05, ge=0.0, le=1.0)
    delta: float = Field(default=0.3, ge=0.0, le=1.0)
    max_dte: Optional[int] = Field(default=None, ge=1)
    maximum_new_contracts: Optional[int] = Field(default=None, ge=1)
    calls: Optional["TargetConfig.Calls"] = None
    puts: Optional["TargetConfig.Puts"] = None

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Write options with targets of")
        table.add_row("", "Days to expiry", ">=", f"{self.dte}")
        if self.max_dte:
            table.add_row("", "Days to expiry", "<=", f"{self.max_dte}")
        table.add_row("", "Default delta", "<=", f"{self.delta}")
        if self.puts and self.puts.delta:
            table.add_row("", "Delta for puts", "<=", f"{self.puts.delta}")
        if self.calls and self.calls.delta:
            table.add_row("", "Delta for calls", "<=", f"{self.calls.delta}")
        table.add_row(
            "",
            "Maximum new contracts",
            "=",
            f"{pfmt(self.maximum_new_contracts_percent, 0)} of buying power",
        )
        table.add_row("", "Minimum open interest", "=", f"{self.minimum_open_interest}")


class SymbolConfig(BaseModel):
    class WriteWhen(BaseModel):
        green: Optional[bool] = None
        red: Optional[bool] = None

    class Calls(BaseModel):
        cap_factor: Optional[float] = Field(default=None, ge=0, le=1)
        cap_target_floor: Optional[float] = Field(default=None, ge=0, le=1)
        excess_only: Optional[bool] = None
        delta: Optional[float] = Field(default=None, ge=0, le=1)
        write_threshold: Optional[float] = Field(default=None, ge=0, le=1)
        write_threshold_sigma: Optional[float] = Field(default=None, gt=0)
        strike_limit: Optional[float] = Field(default=None, gt=0)
        maintain_high_water_mark: Optional[bool] = None
        write_when: Optional["SymbolConfig.WriteWhen"] = Field(
            default_factory=lambda: SymbolConfig.WriteWhen()
        )

    class Puts(BaseModel):
        delta: Optional[float] = Field(default=None, ge=0, le=1)
        write_threshold: Optional[float] = Field(default=None, ge=0, le=1)
        write_threshold_sigma: Optional[float] = Field(default=None, gt=0)
        strike_limit: Optional[float] = Field(default=None, gt=0)
        write_when: Optional["SymbolConfig.WriteWhen"] = Field(
            default_factory=lambda: SymbolConfig.WriteWhen()
        )

    weight: float = Field(..., ge=0, le=1)
    primary_exchange: str = Field(default="", min_length=1)
    delta: Optional[float] = Field(default=None, ge=0, le=1)
    write_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    write_threshold_sigma: Optional[float] = Field(default=None, gt=0)
    max_dte: Optional[int] = Field(default=None, ge=1)
    dte: Optional[int] = Field(default=None, ge=0)
    close_if_unable_to_roll: Optional[bool] = None
    calls: Optional["SymbolConfig.Calls"] = None
    puts: Optional["SymbolConfig.Puts"] = None
    adjust_price_after_delay: bool = Field(default=False)
    no_trading: Optional[bool] = None
    buy_only_rebalancing: Optional[bool] = None
    buy_only_min_threshold_shares: Optional[int] = Field(default=None, ge=1)
    buy_only_min_threshold_amount: Optional[float] = Field(default=None, ge=0.0)


class StockStrategyConfig(BaseModel):
    """Configuration for stock trading strategies."""

    strategy: str = Field(default="bx_trender")
    shares: int = Field(default=0, ge=0)


class ActionWhenClosedEnum(str, Enum):
    wait = "wait"
    exit = "exit"
    continue_ = "continue"


class ExchangeHoursConfig(BaseModel, DisplayMixin):
    exchange: str = Field(default="XNYS")
    action_when_closed: ActionWhenClosedEnum = Field(default=ActionWhenClosedEnum.exit)
    delay_after_open: int = Field(default=1800, ge=0)
    delay_before_close: int = Field(default=1800, ge=0)
    max_wait_until_open: int = Field(default=3600, ge=0)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_row("[spring_green1]Exchange hours")
        table.add_row("", "Exchange", "=", self.exchange)
        table.add_row("", "Action when closed", "=", self.action_when_closed)
        table.add_row("", "Delay after open", "=", f"{self.delay_after_open}s")
        table.add_row("", "Delay before close", "=", f"{self.delay_before_close}s")
        table.add_row("", "Max wait until open", "=", f"{self.max_wait_until_open}s")


# Phase 5: Advanced Configuration Models
# ====================================

class StrategyParametersConfig(BaseModel):
    """Base class for strategy-specific parameters"""
    pass


class WheelStrategyParametersConfig(StrategyParametersConfig):
    """Configuration for Wheel strategy parameters"""
    min_premium: float = Field(default=0.01, ge=0.0, le=1.0, description="Minimum premium to collect")
    target_dte: int = Field(default=30, ge=1, le=365, description="Target days to expiration")
    delta_threshold: float = Field(default=0.30, ge=0.0, le=1.0, description="Maximum delta for options")


class MomentumScalperParametersConfig(StrategyParametersConfig):
    """Configuration for momentum scalping strategy parameters"""
    rsi_period: int = Field(default=14, ge=2, le=100)
    rsi_oversold: float = Field(default=30, ge=0, le=100)
    rsi_overbought: float = Field(default=70, ge=0, le=100)
    macd_fast: int = Field(default=12, ge=1, le=100)
    macd_slow: int = Field(default=26, ge=1, le=200)
    macd_signal: int = Field(default=9, ge=1, le=50)
    ema_period: int = Field(default=20, ge=1, le=200)
    position_size: float = Field(default=0.02, ge=0.001, le=1.0)
    stop_loss: float = Field(default=0.015, ge=0.001, le=1.0)
    take_profit: float = Field(default=0.025, ge=0.001, le=1.0)


class VixHedgeParametersConfig(StrategyParametersConfig):
    """Configuration for VIX hedging strategy parameters"""
    vix_trigger: float = Field(default=25.0, ge=0, le=100)
    hedge_ratio: float = Field(default=0.05, ge=0, le=1.0)
    sma_period: int = Field(default=20, ge=1, le=200)
    rebalance_frequency: str = Field(default="weekly", pattern="^(daily|weekly|monthly)$")


class MeanReversionParametersConfig(StrategyParametersConfig):
    """Configuration for mean reversion strategy parameters"""
    bb_period: int = Field(default=20, ge=2, le=100)
    bb_std_dev: float = Field(default=2.0, ge=0.1, le=5.0)
    rsi_period: int = Field(default=14, ge=2, le=100)
    volume_sma_period: int = Field(default=20, ge=1, le=200)
    entry_threshold: float = Field(default=0.95, ge=0, le=1.0)
    exit_threshold: float = Field(default=0.50, ge=0, le=1.0)
    position_size: float = Field(default=0.03, ge=0.001, le=1.0)


class StrategyConfig(BaseModel, DisplayMixin):
    """Configuration for individual trading strategies"""
    enabled: bool = Field(default=True, description="Enable/disable strategy")
    type: Literal["options", "stocks", "mixed"] = Field(..., description="Strategy type")
    timeframes: List[str] = Field(default_factory=list, description="Strategy timeframes")
    indicators: List[str] = Field(default_factory=list, description="Required indicators")
    description: str = Field(default="", description="Strategy description")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Strategy parameters")

    def add_to_table(self, table: Table, section: str = "") -> None:
        if section:
            table.add_section()
            table.add_row(f"[spring_green1]{section}")
        
        table.add_row("", "Enabled", "=", f"{self.enabled}")
        table.add_row("", "Type", "=", self.type)
        table.add_row("", "Timeframes", "=", ", ".join(self.timeframes))
        table.add_row("", "Indicators", "=", ", ".join(self.indicators))
        if self.description:
            table.add_row("", "Description", "=", self.description)


class BacktestExecutionConfig(BaseModel):
    """Backtesting execution configuration"""
    commission: float = Field(default=0.001, ge=0.0, le=0.1, description="Commission per trade")
    slippage: float = Field(default=0.0005, ge=0.0, le=0.1, description="Slippage factor")
    market_impact: float = Field(default=0.0002, ge=0.0, le=0.1, description="Market impact factor")
    enable_partial_fills: bool = Field(default=True, description="Enable partial fills")
    liquidity_constraint: bool = Field(default=True, description="Apply liquidity constraints")


class BacktestRiskConfig(BaseModel):
    """Backtesting risk management configuration"""
    max_drawdown: float = Field(default=0.20, ge=0.0, le=1.0, description="Maximum drawdown limit")
    position_size_limit: float = Field(default=0.10, ge=0.0, le=1.0, description="Maximum position size")
    daily_loss_limit: float = Field(default=0.05, ge=0.0, le=1.0, description="Daily loss limit")
    margin_requirement: float = Field(default=1.0, ge=0.1, le=10.0, description="Margin requirement")


class BacktestDataConfig(BaseModel):
    """Backtesting data configuration"""
    primary_source: str = Field(default="ibkr", description="Primary data source")
    secondary_source: str = Field(default="yahoo", description="Secondary data source")
    validate_data: bool = Field(default=True, description="Enable data validation")
    handle_missing_data: str = Field(default="forward_fill", pattern="^(forward_fill|interpolate|drop)$")
    survivorship_bias_correction: bool = Field(default=True, description="Correct survivorship bias")


class BacktestAnalyticsConfig(BaseModel):
    """Backtesting analytics configuration"""
    enable_detailed_analytics: bool = Field(default=True, description="Enable detailed analytics")
    benchmark_symbol: str = Field(default="SPY", description="Benchmark symbol")
    risk_free_rate: float = Field(default=0.02, ge=0.0, le=1.0, description="Risk-free rate")
    calculate_var: bool = Field(default=True, description="Calculate Value at Risk")
    var_confidence_level: float = Field(default=0.95, ge=0.5, le=0.99, description="VaR confidence level")
    monte_carlo_iterations: int = Field(default=1000, ge=100, le=10000, description="Monte Carlo iterations")


class BacktestReportingConfig(BaseModel):
    """Backtesting reporting configuration"""
    auto_generate_reports: bool = Field(default=True, description="Auto-generate reports")
    report_format: str = Field(default="html", pattern="^(html|pdf|json|csv)$", description="Report format")
    include_charts: bool = Field(default=True, description="Include charts in reports")
    export_trades: bool = Field(default=True, description="Export trade details")
    detailed_breakdown: bool = Field(default=True, description="Include detailed breakdown")


class BacktestConfig(BaseModel, DisplayMixin):
    """Comprehensive backtesting configuration"""
    enabled: bool = Field(default=False, description="Enable backtesting mode")
    start_date: str = Field(..., description="Backtest start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="Backtest end date (YYYY-MM-DD)")
    initial_capital: float = Field(default=100000.0, ge=1000.0, description="Initial capital")
    
    execution: BacktestExecutionConfig = Field(default_factory=BacktestExecutionConfig)
    risk: BacktestRiskConfig = Field(default_factory=BacktestRiskConfig)
    data: BacktestDataConfig = Field(default_factory=BacktestDataConfig)
    analytics: BacktestAnalyticsConfig = Field(default_factory=BacktestAnalyticsConfig)
    reporting: BacktestReportingConfig = Field(default_factory=BacktestReportingConfig)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Backtesting Configuration")
        table.add_row("", "Enabled", "=", f"{self.enabled}")
        table.add_row("", "Date Range", "=", f"{self.start_date} to {self.end_date}")
        table.add_row("", "Initial Capital", "=", f"${self.initial_capital:,.2f}")
        table.add_row("", "Commission", "=", f"{self.execution.commission:.3%}")
        table.add_row("", "Slippage", "=", f"{self.execution.slippage:.3%}")
        table.add_row("", "Max Drawdown", "=", f"{self.risk.max_drawdown:.1%}")


class TrendIndicatorConfig(BaseModel):
    """Trend indicator configuration"""
    sma_period: int = Field(default=20, ge=1, le=200)
    ema_period: int = Field(default=20, ge=1, le=200)
    wma_period: int = Field(default=20, ge=1, le=200)


class MomentumIndicatorConfig(BaseModel):
    """Momentum indicator configuration"""
    rsi_period: int = Field(default=14, ge=2, le=100)
    macd_fast: int = Field(default=12, ge=1, le=100)
    macd_slow: int = Field(default=26, ge=1, le=200)
    macd_signal: int = Field(default=9, ge=1, le=50)
    stochastic_k: int = Field(default=14, ge=1, le=100)
    stochastic_d: int = Field(default=3, ge=1, le=50)
    williams_r_period: int = Field(default=14, ge=1, le=100)


class VolatilityIndicatorConfig(BaseModel):
    """Volatility indicator configuration"""
    bollinger_period: int = Field(default=20, ge=2, le=100)
    bollinger_std_dev: float = Field(default=2.0, ge=0.1, le=5.0)
    atr_period: int = Field(default=14, ge=1, le=100)


class VolumeIndicatorConfig(BaseModel):
    """Volume indicator configuration"""
    volume_sma_period: int = Field(default=20, ge=1, le=200)
    obv_enabled: bool = Field(default=True)
    vwap_enabled: bool = Field(default=True)


class IndicatorConfig(BaseModel, DisplayMixin):
    """Technical indicator configuration"""
    trend: TrendIndicatorConfig = Field(default_factory=TrendIndicatorConfig)
    momentum: MomentumIndicatorConfig = Field(default_factory=MomentumIndicatorConfig)
    volatility: VolatilityIndicatorConfig = Field(default_factory=VolatilityIndicatorConfig)
    volume: VolumeIndicatorConfig = Field(default_factory=VolumeIndicatorConfig)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Indicator Configuration")
        table.add_row("", "SMA Period", "=", f"{self.trend.sma_period}")
        table.add_row("", "RSI Period", "=", f"{self.momentum.rsi_period}")
        table.add_row("", "Bollinger Period", "=", f"{self.volatility.bollinger_period}")
        table.add_row("", "Volume SMA Period", "=", f"{self.volume.volume_sma_period}")


class TimeframeSynchronizationConfig(BaseModel):
    """Timeframe synchronization configuration"""
    method: str = Field(default="forward_fill", pattern="^(forward_fill|interpolate|nearest)$")
    alignment: str = Field(default="market_open", pattern="^(market_open|market_close|custom)$")
    timezone: str = Field(default="US/Eastern")
    handle_gaps: bool = Field(default=True)


class TimeframePerformanceConfig(BaseModel):
    """Timeframe performance configuration"""
    max_cache_size: int = Field(default=1000, ge=100, le=10000)
    cleanup_frequency: str = Field(default="daily", pattern="^(daily|weekly|monthly)$")
    lazy_loading: bool = Field(default=True)
    parallel_processing: bool = Field(default=True)


class TimeframeConfig(BaseModel, DisplayMixin):
    """Multi-timeframe configuration"""
    primary: List[str] = Field(default_factory=lambda: ["1D"], description="Primary timeframes")
    secondary: List[str] = Field(default_factory=lambda: ["1H", "4H"], description="Secondary timeframes")
    high_frequency: List[str] = Field(default_factory=lambda: ["5M", "15M"], description="High-frequency timeframes")
    
    synchronization: TimeframeSynchronizationConfig = Field(default_factory=TimeframeSynchronizationConfig)
    performance: TimeframePerformanceConfig = Field(default_factory=TimeframePerformanceConfig)

    def add_to_table(self, table: Table, section: str = "") -> None:
        table.add_section()
        table.add_row("[spring_green1]Timeframe Configuration")
        table.add_row("", "Primary", "=", ", ".join(self.primary))
        table.add_row("", "Secondary", "=", ", ".join(self.secondary))
        table.add_row("", "High Frequency", "=", ", ".join(self.high_frequency))
        table.add_row("", "Sync Method", "=", self.synchronization.method)
        table.add_row("", "Cache Size", "=", f"{self.performance.max_cache_size}")


class Config(BaseModel, DisplayMixin):
    account: AccountConfig
    option_chains: OptionChainsConfig
    roll_when: RollWhenConfig
    target: TargetConfig
    exchange_hours: ExchangeHoursConfig = Field(default_factory=ExchangeHoursConfig)

    orders: OrdersConfig = Field(default_factory=OrdersConfig)
    ib_async: IBAsyncConfig = Field(default_factory=IBAsyncConfig)
    ibc: IBCConfig = Field(default_factory=IBCConfig)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)
    cash_management: CashManagementConfig = Field(default_factory=CashManagementConfig)
    vix_call_hedge: VIXCallHedgeConfig = Field(default_factory=VIXCallHedgeConfig)
    write_when: WriteWhenConfig = Field(default_factory=WriteWhenConfig)
    symbols: Dict[str, SymbolConfig] = Field(default_factory=dict)
    stocks: Dict[str, StockStrategyConfig] = Field(default_factory=dict)
    constants: ConstantsConfig = Field(default_factory=ConstantsConfig)
    
    # Phase 5: Enhanced Configuration
    strategies: Dict[str, StrategyConfig] = Field(default_factory=dict, description="Strategy configurations")
    backtesting: Optional[BacktestConfig] = Field(default=None, description="Backtesting configuration")
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig, description="Indicator configuration")
    timeframes: TimeframeConfig = Field(default_factory=TimeframeConfig, description="Timeframe configuration")

    def trading_is_allowed(self, symbol: str) -> bool:
        symbol_config = self.symbols.get(symbol)
        return not symbol_config or not symbol_config.no_trading

    def is_buy_only_rebalancing(self, symbol: str) -> bool:
        symbol_config = self.symbols.get(symbol)
        return symbol_config is not None and symbol_config.buy_only_rebalancing is True

    def symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        return self.symbols.get(symbol)

    @model_validator(mode="after")
    def check_symbols(self) -> Self:
        if not self.symbols:
            raise ValueError("At least one symbol must be specified")
        return self

    @model_validator(mode="after")
    def check_symbol_weights(self) -> Self:
        if not math.isclose(
            1, sum([s.weight or 0.0 for s in self.symbols.values()]), rel_tol=1e-5
        ):
            raise ValueError("Symbol weights must sum to 1.0")
        return self

    def get_target_delta(self, symbol: str, right: str) -> float:
        p_or_c = "calls" if right.upper().startswith("C") else "puts"
        symbol_config = self.symbols.get(symbol)

        if symbol_config:
            option_config = getattr(symbol_config, p_or_c, None)
            if option_config and option_config.delta is not None:
                return option_config.delta
            if symbol_config.delta is not None:
                return symbol_config.delta

        target_option = getattr(self.target, p_or_c, None)
        if target_option and target_option.delta is not None:
            return target_option.delta

        return self.target.delta

    def maintain_high_water_mark(self, symbol: str) -> bool:
        symbol_config = self.symbols.get(symbol)
        if (
            symbol_config
            and symbol_config.calls
            and symbol_config.calls.maintain_high_water_mark is not None
        ):
            return symbol_config.calls.maintain_high_water_mark
        return self.roll_when.calls.maintain_high_water_mark

    # Phase 5: Enhanced Configuration Helper Methods
    
    def get_strategy_config(self, strategy_name: str) -> Optional[StrategyConfig]:
        """Get configuration for a specific strategy"""
        return self.strategies.get(strategy_name)
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled"""
        strategy = self.get_strategy_config(strategy_name)
        return strategy is not None and strategy.enabled
    
    def get_enabled_strategies(self) -> Dict[str, StrategyConfig]:
        """Get all enabled strategies"""
        return {name: config for name, config in self.strategies.items() if config.enabled}
    
    def get_strategies_by_type(self, strategy_type: str) -> Dict[str, StrategyConfig]:
        """Get strategies filtered by type (options, stocks, mixed)"""
        return {name: config for name, config in self.strategies.items() 
                if config.type == strategy_type and config.enabled}
    
    def get_strategies_by_timeframe(self, timeframe: str) -> Dict[str, StrategyConfig]:
        """Get strategies that use a specific timeframe"""
        return {name: config for name, config in self.strategies.items() 
                if timeframe in config.timeframes and config.enabled}
    
    def is_backtesting_enabled(self) -> bool:
        """Check if backtesting mode is enabled"""
        return self.backtesting is not None and self.backtesting.enabled
    
    def get_indicator_config(self, indicator_type: str, indicator_name: str) -> Any:
        """Get configuration for a specific indicator"""
        indicator_category = getattr(self.indicators, indicator_type, None)
        if indicator_category:
            return getattr(indicator_category, indicator_name, None)
        return None
    
    def get_all_required_timeframes(self) -> List[str]:
        """Get all timeframes required by enabled strategies"""
        timeframes = set()
        for strategy in self.get_enabled_strategies().values():
            timeframes.update(strategy.timeframes)
        
        # Add timeframes from timeframe config
        timeframes.update(self.timeframes.primary)
        timeframes.update(self.timeframes.secondary)
        timeframes.update(self.timeframes.high_frequency)
        
        return sorted(list(timeframes))
    
    def get_all_required_indicators(self) -> List[str]:
        """Get all indicators required by enabled strategies"""
        indicators = set()
        for strategy in self.get_enabled_strategies().values():
            indicators.update(strategy.indicators)
        return sorted(list(indicators))

    def get_write_threshold_sigma(
        self,
        symbol: str,
        right: str,
    ) -> Optional[float]:
        p_or_c = "calls" if right.upper().startswith("C") else "puts"
        symbol_config = self.symbols.get(symbol)

        if symbol_config:
            option_config = getattr(symbol_config, p_or_c, None)
            if option_config:
                if option_config.write_threshold_sigma is not None:
                    return option_config.write_threshold_sigma
                if option_config.write_threshold is not None:
                    return None

            if symbol_config.write_threshold_sigma is not None:
                return symbol_config.write_threshold_sigma
            if symbol_config.write_threshold is not None:
                return None

        option_constants = getattr(self.constants, p_or_c, None)
        if option_constants and option_constants.write_threshold_sigma is not None:
            return option_constants.write_threshold_sigma
        if self.constants.write_threshold_sigma is not None:
            return self.constants.write_threshold_sigma

        return None

    def get_write_threshold_perc(
        self,
        symbol: str,
        right: str,
    ) -> float:
        p_or_c = "calls" if right.upper().startswith("C") else "puts"
        symbol_config = self.symbols.get(symbol)

        if symbol_config:
            option_config = getattr(symbol_config, p_or_c, None)
            if option_config and option_config.write_threshold is not None:
                return option_config.write_threshold
            if symbol_config.write_threshold is not None:
                return symbol_config.write_threshold

        option_constants = getattr(self.constants, p_or_c, None)
        if option_constants and option_constants.write_threshold is not None:
            return option_constants.write_threshold
        if self.constants.write_threshold is not None:
            return self.constants.write_threshold

        return 0.0

    def create_symbols_table(self) -> Table:
        table = Table(
            title="Configured symbols and target weights",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
        )
        table.add_column("Symbol")
        table.add_column("Weight", justify="right")
        table.add_column("Buy-only", justify="center")
        table.add_column("Call delta", justify="right")
        table.add_column("Call strike limit", justify="right")
        table.add_column("Call threshold", justify="right")
        table.add_column("HWM", justify="right")
        table.add_column("Put delta", justify="right")
        table.add_column("Put strike limit", justify="right")
        table.add_column("Put threshold", justify="right")

        for symbol, sconfig in self.symbols.items():
            call_thresh = (
                f"{ffmt(self.get_write_threshold_sigma(symbol, 'C'))}σ"
                if self.get_write_threshold_sigma(symbol, "C")
                else pfmt(self.get_write_threshold_perc(symbol, "C"))
            )
            put_thresh = (
                f"{ffmt(self.get_write_threshold_sigma(symbol, 'P'))}σ"
                if self.get_write_threshold_sigma(symbol, "P")
                else pfmt(self.get_write_threshold_perc(symbol, "P"))
            )

            table.add_row(
                symbol,
                pfmt(sconfig.weight or 0.0),
                "✓" if sconfig.buy_only_rebalancing else "",
                ffmt(self.get_target_delta(symbol, "C")),
                dfmt(sconfig.calls.strike_limit if sconfig.calls else None),
                call_thresh,
                str(self.maintain_high_water_mark(symbol)),
                ffmt(self.get_target_delta(symbol, "P")),
                dfmt(sconfig.puts.strike_limit if sconfig.puts else None),
                put_thresh,
            )
        return table

    def display(self, config_path: str) -> None:
        console = Console()
        config_table = Table(box=box.SIMPLE_HEAVY)
        config_table.add_column("Section")
        config_table.add_column("Setting")
        config_table.add_column("")
        config_table.add_column("Value")

        # Add all component tables
        self.account.add_to_table(config_table)
        self.exchange_hours.add_to_table(config_table)
        if self.constants:
            self.constants.add_to_table(config_table)
        self.orders.add_to_table(config_table)
        self.roll_when.add_to_table(config_table)
        self.write_when.add_to_table(config_table)
        self.target.add_to_table(config_table)
        self.cash_management.add_to_table(config_table)
        self.vix_call_hedge.add_to_table(config_table)

        # Create tree and add tables
        tree = Tree(":control_knobs:")
        tree.add(Group(f":file_cabinet: Loaded from {config_path}", config_table))
        tree.add(Group(":yin_yang: Symbology", self.create_symbols_table()))

        console.print(Panel(tree, title="Config"))

    def get_target_dte(self, symbol: str) -> int:
        symbol_config = self.symbols.get(symbol)
        return (
            symbol_config.dte
            if symbol_config and symbol_config.dte is not None
            else self.target.dte
        )

    def get_cap_factor(self, symbol: str) -> float:
        symbol_config = self.symbols.get(symbol)
        if (
            symbol_config is not None
            and symbol_config.calls is not None
            and symbol_config.calls.cap_factor is not None
        ):
            return symbol_config.calls.cap_factor
        return self.write_when.calls.cap_factor

    def get_cap_target_floor(self, symbol: str) -> float:
        symbol_config = self.symbols.get(symbol)
        if (
            symbol_config is not None
            and symbol_config.calls is not None
            and symbol_config.calls.cap_target_floor is not None
        ):
            return symbol_config.calls.cap_target_floor
        return self.write_when.calls.cap_target_floor

    def get_strike_limit(self, symbol: str, right: str) -> Optional[float]:
        p_or_c = "calls" if right.upper().startswith("C") else "puts"
        symbol_config = self.symbols.get(symbol)
        option_config = getattr(symbol_config, p_or_c, None) if symbol_config else None
        return option_config.strike_limit if option_config else None

    def write_excess_calls_only(self, symbol: str) -> bool:
        symbol_config = self.symbols.get(symbol)
        if (
            symbol_config is not None
            and symbol_config.calls is not None
            and symbol_config.calls.excess_only is not None
        ):
            return symbol_config.calls.excess_only
        return self.write_when.calls.excess_only

    def get_max_dte_for(self, symbol: str) -> Optional[int]:
        if symbol == "VIX" and self.vix_call_hedge.max_dte is not None:
            return self.vix_call_hedge.max_dte
        symbol_config = self.symbols.get(symbol)
        if symbol_config is not None and symbol_config.max_dte is not None:
            return symbol_config.max_dte
        return self.target.max_dte

    def can_write_when(self, symbol: str, right: str) -> Tuple[bool, bool]:
        symbol_config = self.symbols.get(symbol)
        p_or_c = "calls" if right.upper().startswith("C") else "puts"
        option_config = (
            getattr(symbol_config, p_or_c, None) if symbol_config is not None else None
        )
        default_config = getattr(self.write_when, p_or_c)
        can_write_when_green = (
            option_config.write_when.green
            if option_config is not None
            and option_config.write_when is not None
            and option_config.write_when.green is not None
            else default_config.green
        )
        can_write_when_red = (
            option_config.write_when.red
            if option_config is not None
            and option_config.write_when is not None
            and option_config.write_when.red is not None
            else default_config.red
        )
        return (can_write_when_green, can_write_when_red)

    def close_if_unable_to_roll(self, symbol: str) -> bool:
        symbol_config = self.symbols.get(symbol)
        return (
            symbol_config.close_if_unable_to_roll
            if symbol_config is not None
            and symbol_config.close_if_unable_to_roll is not None
            else self.roll_when.close_if_unable_to_roll
        )


def normalize_config(config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # Do any pre-processing necessary to the config here, such as handling
    # defaults, deprecated values, config changes, etc.
    if "minimum_cushion" in config["account"]:
        raise RuntimeError(
            "Config error: minimum_cushion is deprecated and replaced with margin_usage. See sample config for details."
        )

    if "ib_insync" in config:
        error_console.print(
            "WARNING: config param `ib_insync` is deprecated, please rename it to the equivalent `ib_async`.",
        )

        if "ib_async" not in config:
            # swap the old ib_insync key to the new ib_async key
            config["ib_async"] = config["ib_insync"]
        del config["ib_insync"]

    if "twsVersion" in config["ibc"]:
        error_console.print(
            "WARNING: config param ibc.twsVersion is deprecated, please remove it from your config.",
        )

        # TWS version is pinned to latest stable, delete any existing config if it's present
        del config["ibc"]["twsVersion"]

    if "maximum_new_contracts" in config["target"]:
        error_console.print(
            "WARNING: config param target.maximum_new_contracts is deprecated, please remove it from your config.",
        )

        del config["target"]["maximum_new_contracts"]

    # xor: should have weight OR parts, but not both
    if any(["weight" in s for s in config["symbols"].values()]) == any(
        ["parts" in s for s in config["symbols"].values()]
    ):
        raise RuntimeError(
            "ERROR: all symbols should have either a weight or parts specified, but parts and weights cannot be mixed."
        )

    if "parts" in list(config["symbols"].values())[0]:
        # If using "parts" instead of "weight", convert parts into weights
        total_parts = float(sum([s["parts"] for s in config["symbols"].values()]))
        for k in config["symbols"].keys():
            config["symbols"][k]["weight"] = config["symbols"][k]["parts"] / total_parts
        for s in config["symbols"].values():
            del s["parts"]

    if (
        "close_at_pnl" in config["roll_when"]
        and config["roll_when"]["close_at_pnl"]
        and config["roll_when"]["close_at_pnl"] <= config["roll_when"]["min_pnl"]
    ):
        raise RuntimeError(
            "ERROR: roll_when.close_at_pnl needs to be greater than roll_when.min_pnl."
        )

    return config
