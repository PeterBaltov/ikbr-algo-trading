-- moneytrailz Dashboard Database Initialization
-- Phase 1: Basic schema setup

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Portfolio snapshots table (time-series)
CREATE TABLE portfolio_snapshots (
    time TIMESTAMPTZ NOT NULL,
    total_value DECIMAL(15,2),
    day_pnl DECIMAL(15,2),
    total_pnl DECIMAL(15,2),
    cash_balance DECIMAL(15,2),
    margin_used DECIMAL(15,2),
    buying_power DECIMAL(15,2),
    day_pnl_percent DECIMAL(8,4),
    total_pnl_percent DECIMAL(8,4),
    win_rate DECIMAL(5,2),
    active_strategies INTEGER
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('portfolio_snapshots', 'time');

-- Strategy performance table (time-series)
CREATE TABLE strategy_performance (
    time TIMESTAMPTZ NOT NULL,
    strategy_name VARCHAR(50) NOT NULL,
    strategy_type VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL,
    allocation DECIMAL(5,2),
    daily_pnl DECIMAL(15,2),
    total_pnl DECIMAL(15,2),
    pnl_percentage DECIMAL(8,4),
    win_rate DECIMAL(5,2),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,4),
    total_trades INTEGER
);

-- Convert to hypertable
SELECT create_hypertable('strategy_performance', 'time');

-- Trade executions table (time-series)
CREATE TABLE trade_executions (
    time TIMESTAMPTZ NOT NULL,
    trade_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_name VARCHAR(50),
    symbol VARCHAR(20),
    action VARCHAR(20),
    quantity INTEGER,
    price DECIMAL(12,4),
    commission DECIMAL(8,2),
    pnl DECIMAL(15,2),
    order_type VARCHAR(20),
    status VARCHAR(20)
);

-- Convert to hypertable
SELECT create_hypertable('trade_executions', 'time');

-- Positions table (current state)
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    quantity INTEGER NOT NULL,
    avg_price DECIMAL(12,4),
    current_price DECIMAL(12,4),
    market_value DECIMAL(15,2),
    unrealized_pnl DECIMAL(15,2),
    unrealized_pnl_percent DECIMAL(8,4),
    strategy VARCHAR(50),
    position_type VARCHAR(20),
    side VARCHAR(10),
    open_date TIMESTAMPTZ,
    last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- Market data table (time-series)
CREATE TABLE market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(12,4),
    change_amount DECIMAL(12,4),
    change_percent DECIMAL(8,4),
    volume BIGINT,
    bid DECIMAL(12,4),
    ask DECIMAL(12,4),
    high DECIMAL(12,4),
    low DECIMAL(12,4),
    open_price DECIMAL(12,4),
    implied_volatility DECIMAL(8,4),
    delta DECIMAL(8,4),
    gamma DECIMAL(8,4),
    theta DECIMAL(8,4),
    vega DECIMAL(8,4)
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'time');

-- Alerts table
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_type VARCHAR(20) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT,
    strategy VARCHAR(50),
    symbol VARCHAR(20),
    acknowledged BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    acknowledged_at TIMESTAMPTZ
);

-- Create indexes for better performance
CREATE INDEX idx_portfolio_snapshots_time ON portfolio_snapshots (time DESC);
CREATE INDEX idx_strategy_performance_time_name ON strategy_performance (time DESC, strategy_name);
CREATE INDEX idx_trade_executions_time_strategy ON trade_executions (time DESC, strategy_name);
CREATE INDEX idx_trade_executions_symbol ON trade_executions (symbol, time DESC);
CREATE INDEX idx_positions_strategy ON positions (strategy);
CREATE INDEX idx_positions_symbol ON positions (symbol);
CREATE INDEX idx_market_data_time_symbol ON market_data (time DESC, symbol);
CREATE INDEX idx_alerts_created_at ON alerts (created_at DESC);
CREATE INDEX idx_alerts_acknowledged ON alerts (acknowledged, created_at DESC);

-- Create continuous aggregates for performance (TimescaleDB feature)
-- Daily portfolio summary
CREATE MATERIALIZED VIEW daily_portfolio_summary
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    AVG(total_value) as avg_total_value,
    MAX(total_value) as max_total_value,
    MIN(total_value) as min_total_value,
    SUM(day_pnl) as total_day_pnl,
    AVG(win_rate) as avg_win_rate
FROM portfolio_snapshots
GROUP BY day;

-- Hourly strategy performance
CREATE MATERIALIZED VIEW hourly_strategy_performance
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    strategy_name,
    AVG(daily_pnl) as avg_hourly_pnl,
    SUM(total_trades) as total_trades,
    AVG(win_rate) as avg_win_rate,
    AVG(sharpe_ratio) as avg_sharpe_ratio
FROM strategy_performance
GROUP BY hour, strategy_name;

-- Daily trade volume by strategy
CREATE MATERIALIZED VIEW daily_trade_volume
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    strategy_name,
    COUNT(*) as trade_count,
    SUM(quantity * price) as total_volume,
    SUM(pnl) as total_pnl
FROM trade_executions
GROUP BY day, strategy_name;

-- Enable refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('daily_portfolio_summary',
    start_offset => INTERVAL '1 month',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('hourly_strategy_performance',
    start_offset => INTERVAL '1 week',
    end_offset => INTERVAL '15 minutes',
    schedule_interval => INTERVAL '15 minutes');

SELECT add_continuous_aggregate_policy('daily_trade_volume',
    start_offset => INTERVAL '1 month',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- Insert some sample data for testing
INSERT INTO portfolio_snapshots (time, total_value, day_pnl, total_pnl, cash_balance, margin_used, buying_power, day_pnl_percent, total_pnl_percent, win_rate, active_strategies)
VALUES 
(NOW(), 125450.23, 1250.75, 12340.50, 25000.00, 15000.00, 85450.23, 1.01, 10.87, 73.2, 5),
(NOW() - INTERVAL '1 hour', 124199.48, 950.00, 11089.75, 25000.00, 15000.00, 84199.48, 0.77, 9.81, 72.8, 5),
(NOW() - INTERVAL '2 hours', 123249.48, 500.25, 10139.75, 25000.00, 15000.00, 83249.48, 0.41, 8.95, 72.5, 5);

COMMIT; 
