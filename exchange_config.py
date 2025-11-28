#!/usr/bin/env python3
"""
Exchange Configuration for Q-FAIRS Trading System
Production-ready exchange integration with proper error handling
"""

import os
from typing import Dict, Any

# ==================== EXCHANGE CONFIGURATION ====================

class ExchangeConfig:
    """Production exchange configuration with security best practices"""
    
    def __init__(self):
        # Load API keys from environment variables (NEVER hardcode in production)
        self.api_keys = {
            'binance': {
                'apiKey': os.getenv('BINANCE_API_KEY', 'demo_key'),
                'secret': os.getenv('BINANCE_SECRET', 'demo_secret'),
                'sandbox': False,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            },
            'coinbase': {
                'apiKey': os.getenv('COINBASE_API_KEY', 'demo_key'),
                'secret': os.getenv('COINBASE_SECRET', 'demo_secret'),
                'sandbox': False,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            },
            'kraken': {
                'apiKey': os.getenv('KRAKEN_API_KEY', 'demo_key'),
                'secret': os.getenv('KRAKEN_SECRET', 'demo_secret'),
                'sandbox': False,
                'enableRateLimit': True
            }
        }
        
        # Trading configuration
        self.trading_pairs = {
            'primary': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
            'secondary': ['ADA/USDT', 'DOT/USDT', 'LINK/USDT', 'AVAX/USDT'],
            'test': ['BTC/USDT']  # For testing
        }
        
        # Rate limiting configuration
        self.rate_limits = {
            'binance': {
                'requests_per_second': 10,
                'order_placements_per_second': 1,
                'burst_allowance': 5
            },
            'coinbase': {
                'requests_per_second': 5,
                'order_placements_per_second': 1,
                'burst_allowance': 3
            },
            'kraken': {
                'requests_per_second': 1,
                'order_placements_per_second': 1,
                'burst_allowance': 2
            }
        }
        
        # WebSocket configuration
        self.websocket_config = {
            'reconnect_interval': 5,  # seconds
            'heartbeat_interval': 30,  # seconds
            'max_reconnect_attempts': 10,
            'connection_timeout': 10  # seconds
        }

# ==================== RISK MANAGEMENT CONFIG ====================

class RiskManagementConfig:
    """Risk management parameters"""
    
    def __init__(self):
        # Position sizing
        self.max_position_size = 0.02  # 2% of portfolio per position
        self.max_portfolio_risk = 0.10  # 10% total portfolio risk
        
        # Stop loss and take profit
        self.stop_loss_percentage = 0.03  # 3% stop loss
        self.take_profit_percentage = 0.06  # 6% take profit
        self.trailing_stop_percentage = 0.02  # 2% trailing stop
        
        # Risk metrics
        self.max_daily_loss = 0.05  # 5% maximum daily loss
        self.max_consecutive_losses = 3
        self.max_drawdown_threshold = 0.15  # 15% maximum drawdown
        
        # Circuit breakers
        self.circuit_breaker_threshold = 0.10  # 10% circuit breaker
        self.cooldown_period_minutes = 60  # 1 hour cooldown

# ==================== PERFORMANCE TARGETS ====================

class PerformanceTargets:
    """System performance targets"""
    
    def __init__(self):
        # Latency targets
        self.order_execution_latency_ms = 200  # Max 200ms order execution
        self.data_feed_latency_ms = 100  # Max 100ms data feed latency
        self.analysis_latency_ms = 50  # Max 50ms analysis latency
        
        # Accuracy targets
        self.prediction_accuracy = 0.65  # 65% prediction accuracy minimum
        self.signal_confidence_threshold = 0.5  # 50% minimum confidence
        
        # Trading performance
        self.target_sharpe_ratio = 1.5
        self.target_win_rate = 0.60  # 60% win rate
        self.max_acceptable_drawdown = 0.10  # 10% maximum drawdown
        
        # System reliability
        self.target_uptime_percentage = 99.9
        self.max_system_downtime_minutes = 60  # per month

# ==================== QUANTUM CONFIGURATION ====================

class QuantumConfig:
    """Quantum algorithm configuration"""
    
    def __init__(self):
        # Quantum simulation parameters
        self.num_qubits = 20  # Number of qubits for simulation
        self.simulation_backend = 'qiskit_aer'  # or 'cirq', 'pennylane'
        
        # Quantum algorithm parameters
        self.variational_circuit_depth = 3
        self.optimization_iterations = 100
        
        # Quantum-inspired parameters
        self.annealing_temperature = 1.0
        self.annealing_cooling_rate = 0.95
        self.quantum_feature_map_degree = 2

# ==================== MONITORING & ALERTING ====================

class MonitoringConfig:
    """System monitoring and alerting configuration"""
    
    def __init__(self):
        # Alert thresholds
        self.latency_alert_threshold_ms = 300
        self.drawdown_alert_threshold = 0.08  # 8% drawdown
        self.sharpe_ratio_alert_threshold = 1.0
        
        # Monitoring intervals
        self.metrics_collection_interval_seconds = 60
        self.health_check_interval_seconds = 30
        self.performance_report_interval_hours = 1
        
        # Alert channels (would be configured in production)
        self.alert_channels = {
            'email': True,
            'sms': True,
            'slack': True,
            'webhook': True
        }

# ==================== SECURITY CONFIGURATION ====================

class SecurityConfig:
    """Security and compliance configuration"""
    
    def __init__(self):
        # API key security
        self.encrypt_api_keys = True
        self.rotate_api_keys_monthly = True
        
        # Data security
        self.encrypt_trading_data = True
        self.secure_log_storage = True
        
        # Compliance
        self.enable_audit_logging = True
        self.compliance_reporting = True
        
        # Network security
        self.use_vpn_connections = True
        self.enable_ddos_protection = True

# ==================== GLOBAL CONFIGURATION ====================

class QFairsConfig:
    """Main system configuration"""
    
    def __init__(self):
        self.exchange = ExchangeConfig()
        self.risk = RiskManagementConfig()
        self.performance = PerformanceTargets()
        self.quantum = QuantumConfig()
        self.monitoring = MonitoringConfig()
        self.security = SecurityConfig()
        
        # System-wide settings
        self.paper_trading_duration_minutes = 5
        self.max_simultaneous_positions = 10
        self.enable_automatic_trading = True
        self.enable_real_time_monitoring = True
        
        # Logging configuration
        self.log_level = 'INFO'
        self.log_to_file = True
        self.log_to_console = True
        self.log_retention_days = 30

# ==================== VALIDATION FUNCTIONS ====================

def validate_configuration(config: QFairsConfig) -> bool:
    """Validate system configuration"""
    
    # Check API keys
    for exchange, keys in config.exchange.api_keys.items():
        if keys['apiKey'] == 'demo_key':
            print(f"⚠️  WARNING: {exchange.upper()} using demo API keys")
            return False
    
    # Check risk parameters
    if config.risk.max_position_size > 0.05:
        print("❌ ERROR: Position size too large")
        return False
    
    # Check performance targets
    if config.performance.target_sharpe_ratio < 1.0:
        print("❌ ERROR: Unrealistic Sharpe ratio target")
        return False
    
    print("✅ Configuration validation passed")
    return True

def load_environment_variables():
    """Load configuration from environment variables"""
    
    # Exchange API keys (set these in your environment)
    exchanges = ['BINANCE', 'COINBASE', 'KRAKEN']
    
    for exchange in exchanges:
        api_key = os.getenv(f'{exchange}_API_KEY')
        secret = os.getenv(f'{exchange}_SECRET')
        
        if not api_key or not secret:
            print(f"⚠️  WARNING: {exchange} API credentials not found in environment")
            print(f"   Set {exchange}_API_KEY and {exchange}_SECRET environment variables")
    
    # Optional: Load from .env file in development
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

# ==================== MAIN CONFIGURATION ====================

# Load environment variables
load_environment_variables()

# Create global configuration instance
config = QFairsConfig()

# Validate configuration
if not validate_configuration(config):
    print("❌ Configuration validation failed - check settings")
    exit(1)

print("🚀 Q-FAIRS Configuration loaded successfully")
print(f"   Exchanges: {', '.join(config.exchange.api_keys.keys())}")
print(f"   Trading pairs: {len(config.exchange.trading_pairs['primary'])} primary, "
      f"{len(config.exchange.trading_pairs['secondary'])} secondary")
print(f"   Risk limits: {config.risk.max_position_size:.1%} position size, "
      f"{config.risk.max_daily_loss:.1%} daily loss")
print(f"   Performance targets: Sharpe {config.performance.target_sharpe_ratio}, "
      f"{config.performance.prediction_accuracy:.1%} accuracy")