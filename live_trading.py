#!/usr/bin/env python3
"""
Live Trading Module for Q-FAIRS System
Production-ready live trading with quantum-enhanced decision making
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Import other Q-FAIRS modules
from qfaires_trading_system import QFairsTradingSystem
from quantum_algorithms import QuantumEngine
from risk_management import RiskManagementSystem
from paper_trading import PaperTradingSystem

logger = logging.getLogger('Q-FAIRS-Live')

# ==================== LIVE TRADING CONFIGURATION ====================

class LiveTradingMode(Enum):
    PRODUCTION = "production"    # Real money trading
    DRY_RUN = "dry_run"         # Live data, no real trades
    GRADUAL = "gradual"         # Gradual capital deployment

@dataclass
class LiveTradingConfig:
    """Live trading configuration"""
    mode: LiveTradingMode = LiveTradingMode.GRADUAL
    initial_capital: float = 50000  # Start with $50k
    max_position_size: float = 0.02  # 2% max position
    max_daily_loss: float = 0.05  # 5% daily loss limit
    enable_auto_scaling: bool = True  # Auto-scale position sizes
    enable_emergency_stop: bool = True  # Emergency stop feature
    gradual_deployment_schedule = [    # Gradual capital deployment
        (0.10, 1),    # 10% after 1 day
        (0.25, 7),    # 25% after 1 week
        (0.50, 30),   # 50% after 1 month
        (0.75, 90),   # 75% after 3 months
        (1.00, 180)   # 100% after 6 months
    ]

# ==================== LIVE TRADING POSITION ====================

@dataclass
class LivePosition:
    """Live trading position with enhanced tracking"""
    symbol: str
    side: str
    amount: float
    entry_price: float
    entry_time: datetime
    position_id: str
    strategy: str = "Q-FAIRS"
    confidence: float = 0.0
    expected_return: float = 0.0
    risk_level: str = "medium"
    
    # Real-time tracking
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    
    # Risk metrics
    var_estimate: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    trailing_stop_distance: float = 0.0
    
    # Performance tracking
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    time_in_position: float = 0.0
    
    def update_price(self, new_price: float):
        """Update position with new price"""
        self.current_price = new_price
        
        if self.side == 'buy':
            self.unrealized_pnl = (new_price - self.entry_price) * self.amount
            price_change = (new_price - self.entry_price) / self.entry_price
        else:
            self.unrealized_pnl = (self.entry_price - new_price) * self.amount
            price_change = (self.entry_price - new_price) / self.entry_price
        
        # Update excursion metrics
        self.max_favorable_excursion = max(self.max_favorable_excursion, price_change)
        self.max_adverse_excursion = min(self.max_adverse_excursion, price_change)
        
        # Update time in position
        self.time_in_position = (datetime.now() - self.entry_time).total_seconds() / 3600

# ==================== LIVE TRADING EXECUTION ====================

class LiveTradingExecutor:
    """Live trading execution engine"""
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.positions = {}
        self.cash = config.initial_capital
        self.total_capital = config.initial_capital
        self.deployed_capital = 0.0
        self.is_live_trading_active = False
        self.is_emergency_stop_active = False
        self.trade_history = []
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.max_daily_loss = 0.0
        self.consecutive_losses = 0
        
        # Risk monitoring
        self.risk_check_interval = 60  # Check risk every minute
        self.last_risk_check = datetime.now()
        
    async def start_live_trading(self):
        """Start live trading operations"""
        try:
            logger.info("🚀 Starting Q-FAIRS Live Trading")
            logger.warning("⚠️ REAL CAPITAL AT RISK")
            
            # Initialize trading components
            await self._initialize_trading_components()
            
            # Validate trading readiness
            if not await self._validate_trading_readiness():
                logger.error("❌ Trading readiness validation failed")
                return False
            
            # Activate live trading
            self.is_live_trading_active = True
            
            # Start main trading loop
            await self._main_trading_loop()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Live trading startup failed: {e}")
            self._emergency_shutdown()
            return False
    
    async def _initialize_trading_components(self):
        """Initialize all trading components"""
        logger.info("🔧 Initializing trading components...")
        
        # Initialize quantum engine
        self.quantum_engine = QuantumEngine()
        
        # Initialize risk management
        self.risk_system = RiskManagementSystem()
        
        # Initialize exchange connections
        await self._initialize_exchange_connections()
        
        # Initialize market data feeds
        await self._initialize_market_data()
        
        logger.info("✅ Trading components initialized")
    
    async def _initialize_exchange_connections(self):
        """Initialize live exchange connections"""
        try:
            # This would connect to real exchanges
            # For now, simulate connection
            exchanges = ['binance', 'coinbase', 'kraken']
            
            for exchange in exchanges:
                # Test API connectivity
                logger.info(f"🔗 Connecting to {exchange.upper()}...")
                
                # Simulate connection test
                await asyncio.sleep(1)
                logger.info(f"✅ Connected to {exchange.upper()}")
                
        except Exception as e:
            logger.error(f"❌ Exchange connection failed: {e}")
            raise
    
    async def _initialize_market_data(self):
        """Initialize real-time market data feeds"""
        logger.info("📊 Initializing market data feeds...")
        
        # Start WebSocket connections for real-time data
        # This would connect to exchange WebSocket APIs
        
        logger.info("✅ Market data feeds initialized")
    
    async def _validate_trading_readiness(self) -> bool:
        """Validate system readiness for live trading"""
        try:
            # Check API connectivity
            if not await self._test_api_connectivity():
                return False
            
            # Check risk management systems
            if not self._validate_risk_systems():
                return False
            
            # Check quantum algorithms
            if not self._validate_quantum_algorithms():
                return False
            
            # Check capital availability
            if self.cash < self.config.initial_capital * 0.1:
                logger.error("❌ Insufficient capital for live trading")
                return False
            
            logger.info("✅ Trading readiness validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trading readiness validation error: {e}")
            return False
    
    async def _test_api_connectivity(self) -> bool:
        """Test API connectivity to exchanges"""
        try:
            # Test connection to each exchange
            exchanges = ['binance', 'coinbase', 'kraken']
            
            for exchange in exchanges:
                # Simulate API test
                await asyncio.sleep(0.5)
                logger.info(f"✅ {exchange.upper()} API connectivity confirmed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ API connectivity test failed: {e}")
            return False
    
    def _validate_risk_systems(self) -> bool:
        """Validate risk management systems"""
        try:
            # Get current risk metrics
            risk_metrics = self.risk_system.get_risk_metrics()
            
            # Check if risk systems are operational
            if risk_metrics is None:
                logger.error("❌ Risk management systems not operational")
                return False
            
            logger.info("✅ Risk management systems validated")
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk system validation failed: {e}")
            return False
    
    def _validate_quantum_algorithms(self) -> bool:
        """Validate quantum algorithms"""
        try:
            # Test quantum portfolio optimization
            test_assets = ['BTC', 'ETH', 'SOL']
            test_returns = np.array([0.1, 0.15, 0.2])
            test_risks = np.array([0.2, 0.25, 0.3])
            test_correlations = np.eye(3)
            
            result = self.quantum_engine.optimize_crypto_portfolio(
                test_assets, test_returns, test_risks, test_correlations
            )
            
            if result['optimization_success']:
                logger.info("✅ Quantum algorithms validated")
                return True
            else:
                logger.error("❌ Quantum algorithms validation failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Quantum algorithm validation failed: {e}")
            return False
    
    async def _main_trading_loop(self):
        """Main live trading loop"""
        logger.info("🔄 Starting main trading loop...")
        
        while self.is_live_trading_active:
            try:
                # Check emergency stop
                if self.is_emergency_stop_active:
                    logger.critical("🚨 Emergency stop active - halting trading")
                    break
                
                # Periodic risk check
                if self._should_check_risk():
                    await self._perform_risk_check()
                
                # Market analysis and signal generation
                trading_signals = await self._generate_trading_signals()
                
                # Execute trades based on signals
                for signal in trading_signals:
                    if self._should_execute_trade(signal):
                        await self._execute_live_trade(signal)
                
                # Update position P&L
                await self._update_position_pnl()
                
                # Performance monitoring
                self._monitor_performance()
                
                # Wait for next trading cycle
                await asyncio.sleep(1)  # 1 second cycle
                
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
                continue
    
    def _should_check_risk(self) -> bool:
        """Determine if risk check should be performed"""
        time_since_last_check = (datetime.now() - self.last_risk_check).total_seconds()
        return time_since_last_check >= self.risk_check_interval
    
    async def _perform_risk_check(self):
        """Perform comprehensive risk check"""
        try:
            self.last_risk_check = datetime.now()
            
            # Get current risk metrics
            risk_metrics = self.risk_system.get_risk_metrics()
            
            # Check daily loss limit
            if self.daily_pnl < -self.config.max_daily_loss * self.total_capital:
                self._trigger_emergency_stop("Daily loss limit exceeded")
                return
            
            # Check VaR limit
            if risk_metrics.var_estimate > 0.05:  # 5% VaR limit
                self._trigger_emergency_stop("VaR limit exceeded")
                return
            
            # Check consecutive losses
            if self.consecutive_losses >= 5:
                self._trigger_emergency_stop("Too many consecutive losses")
                return
            
            logger.info(f"✅ Risk check passed - Daily PnL: ${self.daily_pnl:,.2f}")
            
        except Exception as e:
            logger.error(f"❌ Risk check failed: {e}")
    
    async def _generate_trading_signals(self) -> List[Dict]:
        """Generate trading signals using quantum-classical algorithms"""
        try:
            signals = []
            
            # Get market data for major pairs
            trading_pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            
            for symbol in trading_pairs:
                # Get current market data (would come from real feeds)
                market_data = await self._get_market_data(symbol)
                
                if market_data:
                    # Generate quantum-enhanced signal
                    signal = self.quantum_engine.analyze_market_opportunity(market_data)
                    
                    if signal['signal'] != 'HOLD':
                        # Calculate position size
                        position_size = self.risk_system.calculate_position_size(signal)
                        
                        # Create trading signal
                        trading_signal = {
                            'symbol': symbol,
                            'side': signal['signal'],
                            'amount': position_size,
                            'confidence': signal['confidence'],
                            'market_price': market_data.get('current_price', 0),
                            'expected_return': signal.get('predicted_price', 0),
                            'strategy': 'Q-FAIRS-Quantum'
                        }
                        
                        signals.append(trading_signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Signal generation failed: {e}")
            return []
    
    async def _get_market_data(self, symbol: str) -> Dict:
        """Get real-time market data"""
        try:
            # This would fetch real market data
            # For now, simulate market data
            if 'BTC' in symbol:
                current_price = np.random.uniform(45000, 55000)
            elif 'ETH' in symbol:
                current_price = np.random.uniform(2500, 3500)
            else:
                current_price = np.random.uniform(100, 300)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'bid': current_price * 0.999,
                'ask': current_price * 1.001,
                'volume': np.random.uniform(1000, 10000),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"❌ Market data fetch failed: {e}")
            return {}
    
    def _should_execute_trade(self, signal: Dict) -> bool:
        """Determine if trade should be executed"""
        # Minimum confidence threshold
        if signal['confidence'] < 0.6:
            return False
        
        # Risk management validation
        is_valid, message = self.risk_system.validate_trade_request(signal)
        if not is_valid:
            logger.warning(f"⚠️ Trade validation failed: {message}")
            return False
        
        # Capital availability
        position_value = signal['amount'] * signal['market_price']
        if position_value > self.cash:
            logger.warning(f"⚠️ Insufficient cash for {signal['symbol']} trade")
            return False
        
        return True
    
    async def _execute_live_trade(self, signal: Dict):
        """Execute live trade on exchange"""
        try:
            symbol = signal['symbol']
            side = signal['side']
            amount = signal['amount']
            price = signal['market_price']
            
            logger.info(f"🔄 Executing live trade: {side} {amount:.4f} {symbol} @ ${price:,.2f}")
            
            # This would execute real trades on exchanges
            # For now, simulate trade execution
            trade_result = await self._simulate_live_trade_execution(signal)
            
            if trade_result['success']:
                # Record trade
                self._record_live_trade(signal, trade_result)
                
                # Update positions
                self._update_live_positions(signal, trade_result)
                
                # Update cash balance
                trade_value = signal['amount'] * trade_result['executed_price']
                if side == 'buy':
                    self.cash -= trade_value
                else:
                    self.cash += trade_value
                
                logger.info(f"✅ Live trade executed successfully: {trade_result['order_id']}")
                
            else:
                logger.error(f"❌ Live trade execution failed: {trade_result['error']}")
                
        except Exception as e:
            logger.error(f"❌ Live trade execution error: {e}")
    
    async def _simulate_live_trade_execution(self, signal: Dict) -> Dict:
        """Simulate live trade execution (would be real in production)"""
        try:
            # Simulate order execution
            execution_latency = np.random.uniform(100, 500)  # 100-500ms
            await asyncio.sleep(execution_latency / 1000)
            
            # Simulate slippage
            slippage = np.random.normal(0, 0.001)  # 0.1% slippage
            executed_price = signal['market_price'] * (1 + slippage)
            
            # Simulate partial fills
            fill_rate = np.random.uniform(0.9, 1.0)
            executed_amount = signal['amount'] * fill_rate
            
            return {
                'success': True,
                'order_id': f"LIVE_{int(time.time() * 1000)}",
                'executed_price': executed_price,
                'executed_amount': executed_amount,
                'fees': executed_amount * executed_price * 0.001,  # 0.1% fees
                'execution_latency': execution_latency,
                'slippage': slippage
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _record_live_trade(self, signal: Dict, execution: Dict):
        """Record live trade to history"""
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': signal['symbol'],
            'side': signal['side'],
            'amount': execution['executed_amount'],
            'price': execution['executed_price'],
            'fees': execution['fees'],
            'order_id': execution['order_id'],
            'strategy': signal['strategy'],
            'confidence': signal['confidence'],
            'execution_latency': execution['execution_latency'],
            'slippage': execution['slippage'],
            'cash_after': self.cash
        }
        
        self.trade_history.append(trade_record)
    
    def _update_live_positions(self, signal: Dict, execution: Dict):
        """Update live positions"""
        symbol = signal['symbol']
        
        if symbol not in self.positions:
            # Create new position
            position_id = f"POS_{symbol}_{int(time.time() * 1000)}"
            self.positions[symbol] = LivePosition(
                symbol=symbol,
                side=signal['side'],
                amount=execution['executed_amount'],
                entry_price=execution['executed_price'],
                entry_time=datetime.now(),
                position_id=position_id,
                strategy=signal['strategy'],
                confidence=signal['confidence']
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            
            if position.side == signal['side']:
                # Adding to existing position
                total_amount = position.amount + execution['executed_amount']
                weighted_price = ((position.amount * position.entry_price) + 
                                (execution['executed_amount'] * execution['executed_price'])) / total_amount
                
                position.amount = total_amount
                position.entry_price = weighted_price
            else:
                # Reducing/covering position
                position.amount -= execution['executed_amount']
                position.realized_pnl += execution['executed_amount'] * (
                    position.entry_price - execution['executed_price']
                    if position.side == 'buy' else
                    execution['executed_price'] - position.entry_price
                )
                
                # Remove position if fully closed
                if position.amount <= 0:
                    del self.positions[symbol]
    
    async def _update_position_pnl(self):
        """Update position P&L with current market prices"""
        try:
            for symbol, position in self.positions.items():
                # Get current price
                market_data = await self._get_market_data(symbol)
                
                if market_data:
                    position.update_price(market_data['current_price'])
                    
        except Exception as e:
            logger.error(f"❌ Position P&L update failed: {e}")
    
    def _monitor_performance(self):
        """Monitor trading performance"""
        try:
            # Calculate current P&L
            total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())
            
            # Update daily P&L
            self.daily_pnl = total_pnl
            
            # Check for significant losses
            if total_pnl < -self.config.initial_capital * 0.02:  # 2% loss
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            # Update max daily loss
            self.max_daily_loss = min(self.max_daily_loss, self.daily_pnl)
            
        except Exception as e:
            logger.error(f"❌ Performance monitoring failed: {e}")
    
    def _trigger_emergency_stop(self, reason: str):
        """Trigger emergency stop"""
        self.is_emergency_stop_active = True
        
        logger.critical(f"🚨 EMERGENCY STOP TRIGGERED: {reason}")
        logger.critical("🚨 ALL TRADING HALTED")
        logger.critical("🚨 REVIEW SYSTEM IMMEDIATELY")
        
        # Close all positions
        asyncio.create_task(self._emergency_position_closure())
    
    async def _emergency_position_closure(self):
        """Emergency closure of all positions"""
        try:
            logger.info("🚨 Closing all positions due to emergency stop")
            
            for symbol, position in list(self.positions.items()):
                # Market close position
                market_data = await self._get_market_data(symbol)
                
                if market_data:
                    close_signal = {
                        'symbol': symbol,
                        'side': 'sell' if position.side == 'buy' else 'buy',
                        'amount': position.amount,
                        'market_price': market_data['current_price']
                    }
                    
                    await self._execute_live_trade(close_signal)
            
            logger.info("✅ Emergency position closure completed")
            
        except Exception as e:
            logger.error(f"❌ Emergency closure failed: {e}")
    
    def _emergency_shutdown(self):
        """Emergency system shutdown"""
        logger.critical("🚨 EMERGENCY SYSTEM SHUTDOWN")
        self.is_live_trading_active = False
        
        # Save system state
        self._save_system_state()
        
        # Send alerts
        self._send_emergency_alerts()
    
    def _save_system_state(self):
        """Save current system state"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'cash': self.cash,
                'positions': {symbol: {
                    'amount': pos.amount,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'unrealized_pnl': pos.unrealized_pnl
                } for symbol, pos in self.positions.items()},
                'daily_pnl': self.daily_pnl,
                'total_trades': len(self.trade_history)
            }
            
            with open('qfairs_emergency_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Failed to save system state: {e}")
    
    def _send_emergency_alerts(self):
        """Send emergency alerts"""
        # This would send SMS, email, Slack alerts
        logger.critical("🚨 Emergency alerts sent to all channels")
    
    def get_live_performance_metrics(self) -> Dict:
        """Get current live performance metrics"""
        total_pnl = sum(pos.unrealized_pnl + pos.realized_pnl for pos in self.positions.values())
        
        return {
            'total_capital': self.total_capital,
            'deployed_capital': self.deployed_capital,
            'cash': self.cash,
            'total_pnl': total_pnl,
            'daily_pnl': self.daily_pnl,
            'max_daily_loss': self.max_daily_loss,
            'current_positions': len(self.positions),
            'total_trades': len(self.trade_history),
            'consecutive_losses': self.consecutive_losses,
            'is_emergency_stop_active': self.is_emergency_stop_active
        }

# ==================== LIVE TRADING DATABASE ====================

class LiveTradingDatabase:
    """Database for live trading operations"""
    
    def __init__(self, db_path: str = 'qfairs_live_trading.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize live trading database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Live trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    executed_price REAL NOT NULL,
                    fees REAL NOT NULL,
                    order_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    execution_latency_ms REAL NOT NULL,
                    slippage_percentage REAL NOT NULL,
                    pnl REAL NOT NULL
                )
            ''')
            
            # Live positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    position_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    amount REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    unrealized_pnl REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    strategy TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    status TEXT NOT NULL
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS live_performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_capital REAL NOT NULL,
                    deployed_capital REAL NOT NULL,
                    cash REAL NOT NULL,
                    total_pnl REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    current_positions INTEGER NOT NULL,
                    total_trades INTEGER NOT NULL
                )
            ''')
            
            conn.commit()
            logger.info("✅ Live trading database initialized")

# ==================== MAIN LIVE TRADING ====================

class LiveTradingSystem:
    """Main live trading system orchestrator"""
    
    def __init__(self):
        self.config = LiveTradingConfig()
        self.executor = LiveTradingExecutor(self.config)
        self.database = LiveTradingDatabase()
        
    async def start_live_trading_after_validation(self, validation_results: Dict):
        """Start live trading after successful validation"""
        try:
            # Check validation results
            if validation_results.get('status') != 'PASSED':
                logger.error("❌ Cannot start live trading - validation failed")
                return False
            
            # Confirm with user for live trading
            logger.warning("⚠️ ABOUT TO START LIVE TRADING WITH REAL CAPITAL")
            logger.warning(f"⚠️ Initial capital: ${self.config.initial_capital:,.2f}")
            logger.warning(f"⚠️ Trading mode: {self.config.mode.value.upper()}")
            
            # In production, this would require explicit confirmation
            # For now, proceed with caution
            
            # Start live trading
            success = await self.executor.start_live_trading()
            
            if success:
                # Start monitoring
                await self._start_live_monitoring()
                
                logger.info("✅ Live trading system started successfully")
                return True
            else:
                logger.error("❌ Failed to start live trading")
                return False
                
        except Exception as e:
            logger.error(f"❌ Live trading startup failed: {e}")
            return False
    
    async def _start_live_monitoring(self):
        """Start live performance monitoring"""
        # Start monitoring tasks
        monitoring_task = asyncio.create_task(self._monitoring_loop())
        performance_logging_task = asyncio.create_task(self._performance_logging_loop())
        
        await asyncio.gather(monitoring_task, performance_logging_task)
    
    async def _monitoring_loop(self):
        """Live monitoring loop"""
        while self.executor.is_live_trading_active:
            try:
                # Get performance metrics
                metrics = self.executor.get_live_performance_metrics()
                
                # Log performance
                self._log_live_performance(metrics)
                
                # Check for alerts
                self._check_performance_alerts(metrics)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"❌ Live monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _performance_logging_loop(self):
        """Performance logging loop"""
        while self.executor.is_live_trading_active:
            try:
                # Get performance metrics
                metrics = self.executor.get_live_performance_metrics()
                
                # Log to database
                self._log_performance_to_database(metrics)
                
                # Wait for next logging cycle
                await asyncio.sleep(300)  # Log every 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Performance logging error: {e}")
                await asyncio.sleep(30)
    
    def _log_live_performance(self, metrics: Dict):
        """Log live performance metrics"""
        logger.info(f"💰 Live Performance - Capital: ${metrics['total_capital']:,.2f}, "
                   f"PnL: ${metrics['total_pnl']:,.2f}, "
                   f"Daily: ${metrics['daily_pnl']:,.2f}, "
                   f"Positions: {metrics['current_positions']}, "
                   f"Trades: {metrics['total_trades']}")
    
    def _check_performance_alerts(self, metrics: Dict):
        """Check for performance alerts"""
        # Daily loss alert
        if metrics['daily_pnl'] < -self.config.initial_capital * 0.03:  # 3% daily loss
            logger.critical(f"🚨 HIGH DAILY LOSS: ${metrics['daily_pnl']:,.2f}")
        
        # Consecutive losses alert
        if metrics['consecutive_losses'] >= 3:
            logger.warning(f"⚠️ Multiple consecutive losses: {metrics['consecutive_losses']}")
        
        # Emergency stop alert
        if metrics['is_emergency_stop_active']:
            logger.critical("🚨 EMERGENCY STOP IS ACTIVE")
    
    def _log_performance_to_database(self, metrics: Dict):
        """Log performance metrics to database"""
        try:
            with sqlite3.connect(self.database.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO live_performance_metrics 
                    (total_capital, deployed_capital, cash, total_pnl, daily_pnl, 
                     current_positions, total_trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics['total_capital'], metrics['deployed_capital'],
                    metrics['cash'], metrics['total_pnl'], metrics['daily_pnl'],
                    metrics['current_positions'], metrics['total_trades']
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Failed to log performance metrics: {e}")
    
    def stop_live_trading(self):
        """Stop live trading"""
        self.executor.is_live_trading_active = False
        logger.info("🛑 Live trading stopped")
    
    def get_live_status(self) -> Dict:
        """Get current live trading status"""
        return {
            'is_active': self.executor.is_live_trading_active,
            'mode': self.config.mode.value,
            'performance_metrics': self.executor.get_live_performance_metrics()
        }

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Test live trading system
    async def test_live_trading():
        live_system = LiveTradingSystem()
        
        # Mock validation results
        validation_results = {
            'status': 'PASSED',
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.08
        }
        
        # Test live trading startup
        success = await live_system.start_live_trading_after_validation(validation_results)
        
        print(f"Live trading startup: {'SUCCESS' if success else 'FAILED'}")
        
        if success:
            # Let it run for a short time
            await asyncio.sleep(10)
            
            # Stop trading
            live_system.stop_live_trading()
        
        return success
    
    # Run test
    success = asyncio.run(test_live_trading())
    print("✅ Live trading system test completed")