#!/usr/bin/env python3
"""
Paper Trading Validation Module for Q-FAIRS System
Comprehensive paper trading environment with realistic market simulation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import sqlite3
import time
import json
from datetime import datetime, timedelta
import asyncio
from collections import deque, defaultdict

logger = logging.getLogger('Q-FAIRS-Paper')

# ==================== PAPER TRADING CONFIGURATION ====================

class PaperTradingMode(Enum):
    VALIDATION = "validation"  # Short-term validation mode
    BACKTEST = "backtest"      # Historical backtesting
    SIMULATION = "simulation"  # Long-term simulation

@dataclass
class PaperTradingConfig:
    """Paper trading configuration"""
    mode: PaperTradingMode = PaperTradingMode.VALIDATION
    initial_capital: float = 100000  # $100k starting capital
    validation_duration_minutes: int = 5  # Short validation period
    transaction_cost_percentage: float = 0.001  # 0.1% transaction cost
    slippage_percentage: float = 0.0005  # 0.05% slippage
    market_impact_factor: float = 0.0001  # Market impact per trade
    realistic_order_execution: bool = True
    enable_latency_simulation: bool = True
    
# ==================== PAPER TRADING POSITION ====================

@dataclass
class PaperPosition:
    """Paper trading position"""
    symbol: str
    side: str
    amount: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    trade_count: int = 0
    
    def update_price(self, new_price: float):
        """Update position with new price"""
        self.current_price = new_price
        if self.side == 'buy':
            self.unrealized_pnl = (new_price - self.entry_price) * self.amount
        else:
            self.unrealized_pnl = (self.entry_price - new_price) * self.amount
    
    def close_position(self, exit_price: float, exit_amount: float = None) -> Dict:
        """Close position and calculate final P&L"""
        if exit_amount is None:
            exit_amount = self.amount
        
        exit_amount = min(exit_amount, self.amount)
        
        # Calculate realized P&L
        if self.side == 'buy':
            realized_pnl = (exit_price - self.entry_price) * exit_amount
        else:
            realized_pnl = (self.entry_price - exit_price) * exit_amount
        
        # Update position
        self.amount -= exit_amount
        self.realized_pnl += realized_pnl
        
        return {
            'realized_pnl': realized_pnl,
            'exit_price': exit_price,
            'exit_amount': exit_amount,
            'remaining_amount': self.amount
        }

# ==================== PAPER TRADING PORTFOLIO ====================

class PaperPortfolio:
    """Paper trading portfolio management"""
    
    def __init__(self, config: PaperTradingConfig):
        self.config = config
        self.cash = config.initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        self.start_time = datetime.now()
        
        # Performance tracking
        self.portfolio_values = deque(maxlen=1000)
        self.daily_returns = deque(maxlen=252)  # 1 year of daily returns
        self.update_portfolio_value()
        
    def update_portfolio_value(self):
        """Update total portfolio value"""
        position_values = sum(
            pos.amount * pos.current_price for pos in self.positions.values()
        )
        total_value = self.cash + position_values
        self.portfolio_values.append({
            'timestamp': datetime.now(),
            'total_value': total_value,
            'cash': self.cash,
            'positions_value': position_values
        })
        return total_value
    
    def execute_paper_trade(self, trade_request: Dict) -> Dict:
        """Execute paper trade with realistic simulation"""
        try:
            symbol = trade_request['symbol']
            side = trade_request['side']
            requested_amount = trade_request['amount']
            market_price = trade_request['market_price']
            
            # Simulate realistic execution
            execution_result = self._simulate_realistic_execution(
                symbol, side, requested_amount, market_price
            )
            
            if not execution_result['success']:
                return execution_result
            
            executed_price = execution_result['executed_price']
            executed_amount = execution_result['executed_amount']
            total_cost = execution_result['total_cost']
            fees = execution_result['fees']
            
            # Check if we have enough cash for buy orders
            if side == 'buy' and total_cost > self.cash:
                return {
                    'success': False,
                    'error': 'Insufficient cash',
                    'required_cash': total_cost,
                    'available_cash': self.cash
                }
            
            # Execute the trade
            if side == 'buy':
                self.cash -= total_cost
                self._add_or_update_position(symbol, side, executed_amount, executed_price)
            else:  # sell
                self.cash += total_cost
                self._reduce_or_close_position(symbol, executed_amount, executed_price)
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'requested_amount': requested_amount,
                'executed_amount': executed_amount,
                'requested_price': market_price,
                'executed_price': executed_price,
                'fees': fees,
                'total_cost': total_cost,
                'cash_after': self.cash,
                'portfolio_value': self.update_portfolio_value()
            }
            
            self.trade_history.append(trade_record)
            
            return {
                'success': True,
                'trade_record': trade_record,
                'execution_details': execution_result
            }
            
        except Exception as e:
            logger.error(f"❌ Paper trade execution failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _simulate_realistic_execution(self, symbol: str, side: str, 
                                    amount: float, market_price: float) -> Dict:
        """Simulate realistic order execution"""
        
        # Base execution (assume 95% fill rate)
        fill_probability = 0.95
        if np.random.random() > fill_probability:
            return {
                'success': False,
                'error': 'Order not filled (market conditions)',
                'fill_probability': fill_probability
            }
        
        # Simulate partial fills for large orders
        if amount > 10:  # Large order threshold
            executed_amount = amount * np.random.uniform(0.8, 1.0)
        else:
            executed_amount = amount
        
        # Simulate price slippage
        if self.config.realistic_order_execution:
            slippage = np.random.normal(0, self.config.slippage_percentage)
            if side == 'buy':
                executed_price = market_price * (1 + abs(slippage))
            else:
                executed_price = market_price * (1 - abs(slippage))
        else:
            executed_price = market_price
        
        # Simulate market impact
        market_impact = executed_amount * self.config.market_impact_factor
        if side == 'buy':
            executed_price *= (1 + market_impact)
        else:
            executed_price *= (1 - market_impact)
        
        # Calculate fees
        trade_value = executed_amount * executed_price
        fees = trade_value * self.config.transaction_cost_percentage
        total_cost = trade_value + fees
        
        # Simulate latency if enabled
        if self.config.enable_latency_simulation:
            execution_latency = np.random.uniform(0.1, 2.0)  # 0.1-2 seconds
            time.sleep(execution_latency / 1000)  # Convert to seconds
        
        return {
            'success': True,
            'executed_amount': executed_amount,
            'executed_price': executed_price,
            'fees': fees,
            'total_cost': total_cost,
            'slippage': slippage if self.config.realistic_order_execution else 0,
            'market_impact': market_impact,
            'execution_latency': execution_latency if self.config.enable_latency_simulation else 0
        }
    
    def _add_or_update_position(self, symbol: str, side: str, amount: float, price: float):
        """Add new position or update existing one"""
        if symbol not in self.positions:
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                side=side,
                amount=amount,
                entry_price=price,
                entry_time=datetime.now(),
                current_price=price
            )
        else:
            # Update existing position (averaging)
            existing_pos = self.positions[symbol]
            total_amount = existing_pos.amount + amount
            weighted_price = ((existing_pos.amount * existing_pos.entry_price) + 
                            (amount * price)) / total_amount
            
            existing_pos.amount = total_amount
            existing_pos.entry_price = weighted_price
            existing_pos.current_price = price
    
    def _reduce_or_close_position(self, symbol: str, amount: float, price: float):
        """Reduce or close existing position"""
        if symbol in self.positions:
            position = self.positions[symbol]
            close_result = position.close_position(price, amount)
            
            # Remove position if fully closed
            if position.amount <= 0:
                del self.positions[symbol]
    
    def update_market_prices(self, price_updates: Dict[str, float]):
        """Update all positions with new market prices"""
        for symbol, new_price in price_updates.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(new_price)
        
        # Update portfolio value
        self.update_portfolio_value()
    
    def calculate_performance_statistics(self) -> Dict:
        """Calculate comprehensive performance statistics"""
        if len(self.portfolio_values) < 2:
            return self._create_default_performance_stats()
        
        # Calculate returns
        values = [pv['total_value'] for pv in self.portfolio_values]
        returns = np.diff(values) / values[:-1]
        
        # Basic statistics
        total_return = (values[-1] - values[0]) / values[0]
        annualized_return = (1 + total_return) ** (365.25 / len(returns)) - 1 if returns else 0
        volatility = np.std(returns) * np.sqrt(252) if returns else 0
        
        # Risk metrics
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
        max_drawdown = self._calculate_max_drawdown(values)
        
        # Trading metrics
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()
        avg_trade_return = np.mean(returns) if returns else 0
        
        # Risk-adjusted metrics
        var_95 = np.percentile(returns, 5) if returns else 0
        expected_shortfall = np.mean([r for r in returns if r <= var_95]) if returns else 0
        
        performance_stats = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade_return': avg_trade_return,
            'var_95': abs(var_95),
            'expected_shortfall': abs(expected_shortfall),
            'total_trades': len(self.trade_history),
            'current_portfolio_value': values[-1],
            'peak_portfolio_value': max(values),
            'trading_period_days': (datetime.now() - self.start_time).days
        }
        
        return performance_stats
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.0
        
        # Calculate P&L for each trade
        trade_pnl = []
        for i, trade in enumerate(self.trade_history):
            if i == 0:
                continue
            
            prev_value = self.trade_history[i-1]['portfolio_value']
            curr_value = trade['portfolio_value']
            pnl = (curr_value - prev_value) / prev_value
            trade_pnl.append(pnl)
        
        if not trade_pnl:
            return 0.0
        
        winning_trades = sum(1 for pnl in trade_pnl if pnl > 0)
        return winning_trades / len(trade_pnl)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not self.trade_history:
            return 0.0
        
        gross_profit = 0
        gross_loss = 0
        
        trade_pnl = []
        for i, trade in enumerate(self.trade_history):
            if i == 0:
                continue
            
            prev_value = self.trade_history[i-1]['portfolio_value']
            curr_value = trade['portfolio_value']
            pnl = curr_value - prev_value
            
            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        
        return gross_profit / gross_loss
    
    def _create_default_performance_stats(self) -> Dict:
        """Create default performance statistics"""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade_return': 0.0,
            'var_95': 0.0,
            'expected_shortfall': 0.0,
            'total_trades': 0,
            'current_portfolio_value': self.config.initial_capital,
            'peak_portfolio_value': self.config.initial_capital,
            'trading_period_days': 0
        }

# ==================== PAPER TRADING VALIDATION ====================

class PaperTradingValidator:
    """Paper trading validation and performance assessment"""
    
    def __init__(self, config: PaperTradingConfig):
        self.config = config
        self.portfolio = PaperPortfolio(config)
        self.validation_results = {}
        self.is_validation_active = False
        
    async def start_validation(self, trading_system, duration_minutes: int = None):
        """Start paper trading validation"""
        if duration_minutes is None:
            duration_minutes = self.config.validation_duration_minutes
        
        self.is_validation_active = True
        logger.info(f"🧪 Starting paper trading validation for {duration_minutes} minutes")
        
        validation_start_time = datetime.now()
        end_time = validation_start_time + timedelta(minutes=duration_minutes)
        
        # Validation loop
        while datetime.now() < end_time and self.is_validation_active:
            try:
                # Get trading signals from the system
                trading_signals = await self._get_trading_signals(trading_system)
                
                # Execute paper trades
                for signal in trading_signals:
                    if self._should_execute_trade(signal):
                        await self._execute_paper_trade(signal)
                
                # Update market prices
                await self._update_market_prices(trading_system)
                
                # Log progress
                self._log_validation_progress()
                
                # Wait before next cycle
                await asyncio.sleep(10)  # 10 second cycles
                
            except Exception as e:
                logger.error(f"❌ Validation loop error: {e}")
                await asyncio.sleep(5)
                continue
        
        # Complete validation
        await self._complete_validation()
        
    async def _get_trading_signals(self, trading_system) -> List[Dict]:
        """Get trading signals from the trading system"""
        try:
            # This would interface with the main trading system
            # For now, simulate some trading signals
            signals = []
            
            # Simulate market analysis
            if np.random.random() > 0.7:  # 30% chance of signal
                symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
                symbol = np.random.choice(symbols)
                side = np.random.choice(['buy', 'sell'])
                
                signals.append({
                    'symbol': symbol,
                    'side': side,
                    'amount': np.random.uniform(0.01, 0.1),
                    'confidence': np.random.uniform(0.5, 0.9),
                    'market_price': np.random.uniform(40000, 60000) if 'BTC' in symbol 
                                  else np.random.uniform(2000, 4000)
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Failed to get trading signals: {e}")
            return []
    
    def _should_execute_trade(self, signal: Dict) -> bool:
        """Determine if trade should be executed based on signal quality"""
        # Minimum confidence threshold
        if signal['confidence'] < 0.5:
            return False
        
        # Risk management check
        position_size = signal['amount'] * signal['market_price']
        portfolio_value = self.portfolio.update_portfolio_value()
        
        if position_size / portfolio_value > 0.05:  # 5% position limit
            return False
        
        return True
    
    async def _execute_paper_trade(self, signal: Dict):
        """Execute paper trade based on signal"""
        try:
            trade_request = {
                'symbol': signal['symbol'],
                'side': signal['side'],
                'amount': signal['amount'],
                'market_price': signal['market_price']
            }
            
            result = self.portfolio.execute_paper_trade(trade_request)
            
            if result['success']:
                logger.info(f"📊 Paper trade executed: {signal['side']} "
                           f"{signal['amount']:.4f} {signal['symbol']} @ "
                           f"${signal['market_price']:,.2f}")
            else:
                logger.warning(f"⚠️ Paper trade failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Paper trade execution error: {e}")
    
    async def _update_market_prices(self, trading_system):
        """Update portfolio with current market prices"""
        try:
            # Get current market prices from trading system
            # For now, simulate price updates
            price_updates = {}
            
            for symbol in self.portfolio.positions.keys():
                if 'BTC' in symbol:
                    new_price = np.random.uniform(40000, 60000)
                elif 'ETH' in symbol:
                    new_price = np.random.uniform(2000, 4000)
                else:
                    new_price = np.random.uniform(100, 500)
                
                price_updates[symbol] = new_price
            
            # Update portfolio prices
            self.portfolio.update_market_prices(price_updates)
            
        except Exception as e:
            logger.error(f"❌ Price update error: {e}")
    
    def _log_validation_progress(self):
        """Log validation progress"""
        current_value = self.portfolio.update_portfolio_value()
        performance_stats = self.portfolio.calculate_performance_statistics()
        
        elapsed_time = datetime.now() - self.start_time
        
        logger.info(f"🧪 Validation Progress - Time: {elapsed_time}, "
                   f"Portfolio: ${current_value:,.2f}, "
                   f"Return: {performance_stats['total_return']:.2%}, "
                   f"Sharpe: {performance_stats['sharpe_ratio']:.2f}, "
                   f"Trades: {performance_stats['total_trades']}")
    
    async def _complete_validation(self):
        """Complete validation and generate results"""
        self.is_validation_active = False
        
        # Calculate final performance statistics
        performance_stats = self.portfolio.calculate_performance_statistics()
        
        # Generate validation report
        validation_report = self._generate_validation_report(performance_stats)
        
        # Store results
        self.validation_results = validation_report
        
        # Log results
        logger.info("🧪 Paper Trading Validation Completed")
        logger.info(f"📊 Final Results:")
        logger.info(f"   Total Return: {performance_stats['total_return']:.2%}")
        logger.info(f"   Sharpe Ratio: {performance_stats['sharpe_ratio']:.2f}")
        logger.info(f"   Max Drawdown: {performance_stats['max_drawdown']:.2%}")
        logger.info(f"   Win Rate: {performance_stats['win_rate']:.2%}")
        logger.info(f"   Total Trades: {performance_stats['total_trades']}")
        logger.info(f"   Validation Status: {validation_report['status']}")
        
    def _generate_validation_report(self, performance_stats: Dict) -> Dict:
        """Generate comprehensive validation report"""
        
        # Define validation criteria
        validation_criteria = {
            'min_sharpe_ratio': 1.0,
            'max_drawdown_limit': 0.10,  # 10%
            'min_win_rate': 0.50,  # 50%
            'max_var_95': 0.05,  # 5%
            'min_total_trades': 5
        }
        
        # Check validation criteria
        criteria_results = {
            'sharpe_ratio_pass': performance_stats['sharpe_ratio'] >= validation_criteria['min_sharpe_ratio'],
            'drawdown_pass': performance_stats['max_drawdown'] <= validation_criteria['max_drawdown_limit'],
            'win_rate_pass': performance_stats['win_rate'] >= validation_criteria['min_win_rate'],
            'var_pass': performance_stats['var_95'] <= validation_criteria['max_var_95'],
            'trades_pass': performance_stats['total_trades'] >= validation_criteria['min_total_trades']
        }
        
        # Overall validation status
        all_passed = all(criteria_results.values())
        
        status = "PASSED" if all_passed else "FAILED"
        
        return {
            'status': status,
            'performance_stats': performance_stats,
            'validation_criteria': validation_criteria,
            'criteria_results': criteria_results,
            'validation_duration_minutes': self.config.validation_duration_minutes,
            'validation_start_time': self.start_time,
            'validation_end_time': datetime.now(),
            'recommendations': self._generate_recommendations(performance_stats, criteria_results)
        }
    
    def _generate_recommendations(self, performance_stats: Dict, criteria_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not criteria_results.get('sharpe_ratio_pass', False):
            recommendations.append("Consider adjusting strategy parameters to improve risk-adjusted returns")
        
        if not criteria_results.get('drawdown_pass', False):
            recommendations.append("Implement stricter risk management controls to limit drawdowns")
        
        if not criteria_results.get('win_rate_pass', False):
            recommendations.append("Review signal generation logic to improve prediction accuracy")
        
        if not criteria_results.get('var_pass', False):
            recommendations.append("Reduce position sizes or implement tighter stop losses")
        
        if not criteria_results.get('trades_pass', False):
            recommendations.append("Increase trading frequency or reduce signal thresholds")
        
        if not recommendations:
            recommendations.append("System performance meets validation criteria - ready for live trading")
        
        return recommendations
    
    def get_validation_results(self) -> Dict:
        """Get validation results"""
        return self.validation_results
    
    def stop_validation(self):
        """Stop validation early"""
        self.is_validation_active = False
        logger.info("🧪 Paper trading validation stopped by user")

# ==================== PAPER TRADING DATABASE ====================

class PaperTradingDatabase:
    """Database for paper trading results"""
    
    def __init__(self, db_path: str = 'qfairs_paper_trading.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize paper trading database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Paper trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    requested_amount REAL NOT NULL,
                    executed_amount REAL NOT NULL,
                    requested_price REAL NOT NULL,
                    executed_price REAL NOT NULL,
                    fees REAL NOT NULL,
                    total_cost REAL NOT NULL,
                    cash_after REAL NOT NULL,
                    portfolio_value REAL NOT NULL,
                    execution_latency_ms REAL,
                    slippage_percentage REAL,
                    market_impact_percentage REAL
                )
            ''')
            
            # Paper portfolio values table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_portfolio_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL
                )
            ''')
            
            # Validation results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    validation_status TEXT NOT NULL,
                    total_return REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    total_trades INTEGER NOT NULL,
                    validation_duration_minutes INTEGER NOT NULL,
                    criteria_results TEXT NOT NULL,
                    recommendations TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            logger.info("✅ Paper trading database initialized")

# ==================== MAIN PAPER TRADING ====================

class PaperTradingSystem:
    """Main paper trading system"""
    
    def __init__(self):
        self.config = PaperTradingConfig()
        self.database = PaperTradingDatabase()
        self.validator = PaperTradingValidator(self.config)
        
    async def run_paper_trading_validation(self, trading_system) -> Dict:
        """Run complete paper trading validation"""
        try:
            logger.info("🚀 Starting Q-FAIRS Paper Trading Validation")
            
            # Start validation
            await self.validator.start_validation(trading_system)
            
            # Get results
            results = self.validator.get_validation_results()
            
            logger.info("✅ Paper trading validation completed")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Paper trading validation failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e)
            }
    
    def stop_validation(self):
        """Stop paper trading validation"""
        self.validator.stop_validation()

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Test paper trading system
    async def test_paper_trading():
        paper_system = PaperTradingSystem()
        
        # Mock trading system
        class MockTradingSystem:
            pass
        
        mock_system = MockTradingSystem()
        
        # Run validation
        results = await paper_system.run_paper_trading_validation(mock_system)
        
        print(f"Paper trading validation results: {results}")
        
        return results
    
    # Run test
    results = asyncio.run(test_paper_trading())
    print("✅ Paper trading system test completed")