#!/usr/bin/env python3
"""
Q-FAIRS Backend API Server
Flask-based REST API for quantum trading system
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import Q-FAIRS modules
from quantum_algorithms import QuantumEngine
from risk_management import RiskManagementSystem
from market_data import MarketDataProvider
from database import Database

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('Q-FAIRS-API')

# Initialize components
quantum_engine = QuantumEngine()
risk_system = RiskManagementSystem()
market_data = MarketDataProvider()
db = Database()

# ==================== HEALTH & STATUS ====================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'service': 'Q-FAIRS Backend',
        'version': '9.0',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'quantum_engine': 'operational',
            'risk_system': 'operational',
            'market_data': 'connected'
        }
    })

@app.route('/api/status', methods=['GET'])
def system_status():
    """Get detailed system status"""
    try:
        risk_metrics = risk_system.get_risk_metrics()
        
        return jsonify({
            'success': True,
            'system': {
                'uptime': time.time() - app.start_time,
                'circuit_breaker': not risk_system.is_circuit_breaker_active,
                'quantum_core': 'operational',
                'trading_active': True
            },
            'risk': {
                'portfolio_var': risk_metrics.get('var_estimate', 0),
                'max_drawdown': risk_metrics.get('max_drawdown', 0),
                'current_exposure': risk_metrics.get('total_exposure', 0)
            }
        })
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== MARKET DATA ====================

@app.route('/api/market-data', methods=['GET'])
def get_market_data():
    """Get real-time market data for all trading pairs"""
    try:
        symbols = request.args.get('symbols', 'BTC/USDT,ETH/USDT,SOL/USDT').split(',')
        
        market_data_response = {}
        for symbol in symbols:
            data = market_data.get_ticker(symbol.strip())
            if data:
                market_data_response[symbol.strip()] = data
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data': market_data_response
        })
    except Exception as e:
        logger.error(f"Market data fetch failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/orderbook/<symbol>', methods=['GET'])
def get_orderbook(symbol):
    """Get order book for specific symbol"""
    try:
        depth = int(request.args.get('depth', 20))
        orderbook = market_data.get_orderbook(symbol, depth)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'orderbook': orderbook,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Orderbook fetch failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== TRADING ====================

@app.route('/api/execute-trade', methods=['POST'])
def execute_trade():
    """Execute trade order"""
    try:
        trade_data = request.json
        
        # Validate request
        required_fields = ['symbol', 'side', 'amount']
        if not all(field in trade_data for field in required_fields):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        # Risk validation
        is_valid, message = risk_system.validate_trade_request(trade_data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': f'Risk validation failed: {message}'
            }), 403
        
        # Execute trade (paper trading for now)
        result = {
            'success': True,
            'order_id': f"ORD_{int(time.time() * 1000)}",
            'status': 'executed',
            'symbol': trade_data['symbol'],
            'side': trade_data['side'],
            'amount': trade_data['amount'],
            'executed_price': trade_data.get('price', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Log to database
        db.log_trade(result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Trade execution failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get current open positions"""
    try:
        positions = db.get_open_positions()
        
        return jsonify({
            'success': True,
            'positions': positions,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/close-position', methods=['POST'])
def close_position():
    """Close open position"""
    try:
        data = request.json
        position_id = data.get('position_id')
        
        if not position_id:
            return jsonify({
                'success': False,
                'error': 'position_id required'
            }), 400
        
        # Close position
        result = db.close_position(position_id)
        
        return jsonify({
            'success': True,
            'message': 'Position closed',
            'position_id': position_id,
            'pnl': result.get('pnl', 0)
        })
        
    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== QUANTUM ANALYSIS ====================

@app.route('/api/quantum-analysis', methods=['POST'])
def quantum_analysis():
    """Run quantum portfolio optimization"""
    try:
        data = request.json
        
        # Validate inputs
        required = ['assets', 'returns', 'risks']
        if not all(field in data for field in required):
            return jsonify({
                'success': False,
                'error': 'Missing required fields'
            }), 400
        
        # Run quantum optimization
        import numpy as np
        
        assets = data['assets']
        returns = np.array(data['returns'])
        risks = np.array(data['risks'])
        correlations = np.eye(len(assets))  # Simplified
        
        result = quantum_engine.optimize_crypto_portfolio(
            assets, returns, risks, correlations
        )
        
        return jsonify({
            'success': True,
            'quantum_result': {
                'optimal_weights': result['weights'].tolist(),
                'expected_return': float(result['expected_return']),
                'portfolio_risk': float(result['portfolio_risk']),
                'sharpe_ratio': float(result['sharpe_ratio']),
                'quantum_advantage': result.get('quantum_advantage', {})
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Quantum analysis failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/quantum-predict', methods=['POST'])
def quantum_predict():
    """Quantum ML market prediction"""
    try:
        data = request.json
        market_features = data.get('features', [])
        
        regime = quantum_engine.predict_market_regime(
            np.array(market_features)
        )
        
        return jsonify({
            'success': True,
            'prediction': {
                'market_regime': regime,
                'confidence': 0.85,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Quantum prediction failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== RISK MANAGEMENT ====================

@app.route('/api/risk-metrics', methods=['GET'])
def get_risk_metrics():
    """Get current risk metrics"""
    try:
        metrics = risk_system.get_risk_metrics()
        
        return jsonify({
            'success': True,
            'risk_metrics': {
                'var_estimate': float(metrics.get('var_estimate', 0)),
                'max_drawdown': float(metrics.get('max_drawdown', 0)),
                'sharpe_ratio': float(metrics.get('sharpe_ratio', 0)),
                'portfolio_volatility': float(metrics.get('volatility', 0)),
                'exposure_percentage': float(metrics.get('total_exposure', 0))
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Risk metrics fetch failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/circuit-breaker', methods=['POST'])
def circuit_breaker_control():
    """Control circuit breaker"""
    try:
        data = request.json
        action = data.get('action')  # 'activate' or 'deactivate'
        
        if action == 'activate':
            risk_system.activate_circuit_breaker("Manual activation")
            message = "Circuit breaker activated"
        elif action == 'deactivate':
            risk_system.is_circuit_breaker_active = False
            message = "Circuit breaker deactivated"
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid action'
            }), 400
        
        return jsonify({
            'success': True,
            'message': message,
            'circuit_breaker_active': risk_system.is_circuit_breaker_active
        })
        
    except Exception as e:
        logger.error(f"Circuit breaker control failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== ANALYTICS ====================

@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get trading performance metrics"""
    try:
        timeframe = request.args.get('timeframe', '24h')
        
        performance = db.get_performance_metrics(timeframe)
        
        return jsonify({
            'success': True,
            'performance': performance,
            'timeframe': timeframe,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Performance fetch failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/trade-history', methods=['GET'])
def get_trade_history():
    """Get trade history"""
    try:
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        
        trades = db.get_trade_history(limit, offset)
        
        return jsonify({
            'success': True,
            'trades': trades,
            'count': len(trades),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Trade history fetch failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ==================== MAIN ====================

if __name__ == '__main__':
    # Store start time
    app.start_time = time.time()
    
    # Get port from environment
    port = int(os.environ.get('PORT', 5000))
    
    # Run app
    logger.info(f"Starting Q-FAIRS Backend on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
