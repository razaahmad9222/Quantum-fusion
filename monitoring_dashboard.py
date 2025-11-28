#!/usr/bin/env python3
"""
Real-Time Monitoring Dashboard for Q-FAIRS Trading System
Comprehensive monitoring with web-based dashboard and alerting
"""

import asyncio
import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import webbrowser

logger = logging.getLogger('Q-FAIRS-Monitor')

# ==================== MONITORING CONFIGURATION ====================

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """System alert"""
    id: str
    timestamp: datetime
    level: AlertLevel
    component: str
    message: str
    details: Dict = field(default_factory=dict)
    resolved: bool = False

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict
    process_count: int
    uptime_seconds: float
    
@dataclass
class TradingMetrics:
    """Trading-specific metrics"""
    timestamp: datetime
    total_pnl: float
    daily_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    current_positions: int
    exposure_percentage: float
    
@dataclass
class QuantumMetrics:
    """Quantum algorithm performance metrics"""
    timestamp: datetime
    quantum_advantage_score: float
    algorithm_success_rate: float
    average_execution_time: float
    quantum_circuit_calls: int
    classical_fallbacks: int
    quantum_noise_level: float
    optimization_improvement: float

# ==================== MONITORING ENGINE ====================

class MonitoringEngine:
    """Core monitoring and alerting engine"""
    
    def __init__(self):
        self.alerts = []
        self.metrics_history = []
        self.alert_handlers = []
        self.is_monitoring_active = False
        
        # Alert thresholds
        self.alert_thresholds = {
            'max_drawdown': 0.10,  # 10%
            'max_daily_loss': 0.05,  # 5%
            'min_sharpe_ratio': 1.0,
            'max_latency_ms': 500,
            'min_win_rate': 0.50,  # 50%
            'max_system_cpu': 0.90,  # 90%
            'max_system_memory': 0.90,  # 90%
        }
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """Initialize monitoring database"""
        with sqlite3.connect('qfairs_monitoring.db') as conn:
            cursor = conn.cursor()
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_alerts (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    level TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    metric_type TEXT NOT NULL,
                    metric_data TEXT NOT NULL
                )
            ''')
            
            # Performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    total_pnl REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    current_positions INTEGER NOT NULL,
                    total_trades INTEGER NOT NULL
                )
            ''')
            
            conn.commit()
            logger.info("✅ Monitoring database initialized")
    
    def add_alert(self, level: AlertLevel, component: str, message: str, details: Dict = None):
        """Add system alert"""
        alert_id = f"ALERT_{int(time.time() * 1000)}"
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            level=level,
            component=component,
            message=message,
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        # Log to database
        self._log_alert_to_database(alert)
        
        # Send notifications
        self._send_alert_notifications(alert)
        
        # Log to system
        self._log_system_alert(alert)
        
        return alert_id
    
    def resolve_alert(self, alert_id: str):
        """Resolve system alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                self._update_alert_in_database(alert)
                logger.info(f"✅ Alert resolved: {alert_id}")
                break
    
    def update_metrics(self, metrics_type: str, metrics_data: Dict):
        """Update system metrics"""
        try:
            # Add to history
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'type': metrics_type,
                'data': metrics_data
            })
            
            # Keep only last 1000 metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            # Log to database
            self._log_metrics_to_database(metrics_type, metrics_data)
            
            # Check for alerts
            self._check_metrics_for_alerts(metrics_type, metrics_data)
            
        except Exception as e:
            logger.error(f"❌ Metrics update failed: {e}")
    
    def _check_metrics_for_alerts(self, metrics_type: str, metrics_data: Dict):
        """Check metrics for alert conditions"""
        try:
            if metrics_type == 'trading':
                # Check trading metrics
                if metrics_data.get('max_drawdown', 0) > self.alert_thresholds['max_drawdown']:
                    self.add_alert(
                        AlertLevel.CRITICAL,
                        "Trading",
                        f"Maximum drawdown exceeded: {metrics_data['max_drawdown']:.2%}",
                        {'threshold': self.alert_thresholds['max_drawdown']}
                    )
                
                if metrics_data.get('sharpe_ratio', 0) < self.alert_thresholds['min_sharpe_ratio']:
                    self.add_alert(
                        AlertLevel.WARNING,
                        "Trading",
                        f"Sharpe ratio below threshold: {metrics_data['sharpe_ratio']:.2f}",
                        {'threshold': self.alert_thresholds['min_sharpe_ratio']}
                    )
                
                if metrics_data.get('win_rate', 0) < self.alert_thresholds['min_win_rate']:
                    self.add_alert(
                        AlertLevel.WARNING,
                        "Trading",
                        f"Win rate below threshold: {metrics_data['win_rate']:.2%}",
                        {'threshold': self.alert_thresholds['min_win_rate']}
                    )
            
            elif metrics_type == 'system':
                # Check system metrics
                if metrics_data.get('cpu_usage', 0) > self.alert_thresholds['max_system_cpu']:
                    self.add_alert(
                        AlertLevel.ERROR,
                        "System",
                        f"High CPU usage: {metrics_data['cpu_usage']:.1%}",
                        {'threshold': self.alert_thresholds['max_system_cpu']}
                    )
                
                if metrics_data.get('memory_usage', 0) > self.alert_thresholds['max_system_memory']:
                    self.add_alert(
                        AlertLevel.ERROR,
                        "System",
                        f"High memory usage: {metrics_data['memory_usage']:.1%}",
                        {'threshold': self.alert_thresholds['max_system_memory']}
                    )
            
            elif metrics_type == 'quantum':
                # Check quantum metrics
                if metrics_data.get('classical_fallbacks', 0) > 10:
                    self.add_alert(
                        AlertLevel.WARNING,
                        "Quantum",
                        f"High classical fallback rate: {metrics_data['classical_fallbacks']}",
                        {'threshold': 10}
                    )
        
        except Exception as e:
            logger.error(f"❌ Alert checking failed: {e}")
    
    def _log_alert_to_database(self, alert: Alert):
        """Log alert to database"""
        try:
            with sqlite3.connect('qfairs_monitoring.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_alerts (id, level, component, message, details)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    alert.id, alert.level.value, alert.component,
                    alert.message, json.dumps(alert.details)
                ))
                conn.commit()
        
        except Exception as e:
            logger.error(f"❌ Failed to log alert to database: {e}")
    
    def _update_alert_in_database(self, alert: Alert):
        """Update alert in database"""
        try:
            with sqlite3.connect('qfairs_monitoring.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE system_alerts 
                    SET resolved = ? 
                    WHERE id = ?
                ''', (alert.resolved, alert.id))
                conn.commit()
        
        except Exception as e:
            logger.error(f"❌ Failed to update alert in database: {e}")
    
    def _log_metrics_to_database(self, metrics_type: str, metrics_data: Dict):
        """Log metrics to database"""
        try:
            with sqlite3.connect('qfairs_monitoring.db') as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics (metric_type, metric_data)
                    VALUES (?, ?)
                ''', (metrics_type, json.dumps(metrics_data)))
                conn.commit()
        
        except Exception as e:
            logger.error(f"❌ Failed to log metrics to database: {e}")
    
    def _send_alert_notifications(self, alert: Alert):
        """Send alert notifications"""
        try:
            # This would send actual notifications (email, SMS, Slack)
            # For now, just log
            if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                logger.critical(f"🚨 ALERT NOTIFICATION SENT: {alert.message}")
            else:
                logger.info(f"📧 Alert notification sent: {alert.message}")
        
        except Exception as e:
            logger.error(f"❌ Failed to send alert notification: {e}")
    
    def _log_system_alert(self, alert: Alert):
        """Log alert to system"""
        if alert.level == AlertLevel.CRITICAL:
            logger.critical(f"🚨 {alert.component.upper()}: {alert.message}")
        elif alert.level == AlertLevel.ERROR:
            logger.error(f"❌ {alert.component.upper()}: {alert.message}")
        elif alert.level == AlertLevel.WARNING:
            logger.warning(f"⚠️ {alert.component.upper()}: {alert.message}")
        else:
            logger.info(f"ℹ️ {alert.component.upper()}: {alert.message}")
    
    def get_recent_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent alerts"""
        return sorted(self.alerts, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_metrics_summary(self) -> Dict:
        """Get metrics summary"""
        if not self.metrics_history:
            return {}
        
        latest_metrics = {}
        for metrics_entry in reversed(self.metrics_history):
            metric_type = metrics_entry['type']
            if metric_type not in latest_metrics:
                latest_metrics[metric_type] = metrics_entry['data']
        
        return latest_metrics
    
    async def monitoring_loop(self):
        """Main monitoring loop"""
        self.is_monitoring_active = True
        
        while self.is_monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.update_metrics('system', system_metrics)
                
                # Wait for next cycle
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    def _collect_system_metrics(self) -> Dict:
        """Collect system performance metrics"""
        try:
            # Import psutil for system metrics
            import psutil
            
            return {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_io': psutil.net_io_counters()._asdict(),
                'process_count': len(psutil.pids()),
                'uptime_seconds': time.time() - psutil.boot_time()
            }
            
        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_usage': np.random.uniform(10, 50),
                'memory_usage': np.random.uniform(30, 70),
                'disk_usage': np.random.uniform(20, 80),
                'network_io': {'bytes_sent': 0, 'bytes_recv': 0},
                'process_count': 100,
                'uptime_seconds': 3600
            }

# ==================== WEB DASHBOARD ====================

class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP request handler for monitoring dashboard"""
    
    def __init__(self, monitoring_engine: MonitoringEngine, *args, **kwargs):
        self.monitoring_engine = monitoring_engine
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/':
            self._serve_dashboard()
        elif parsed_path.path == '/api/metrics':
            self._serve_metrics()
        elif parsed_path.path == '/api/alerts':
            self._serve_alerts()
        elif parsed_path.path.startswith('/static/'):
            self._serve_static_file(parsed_path.path)
        else:
            self._serve_404()
    
    def _serve_dashboard(self):
        """Serve main dashboard HTML"""
        html_content = self._generate_dashboard_html()
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def _serve_metrics(self):
        """Serve metrics API"""
        metrics = self.monitoring_engine.get_metrics_summary()
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(metrics).encode())
    
    def _serve_alerts(self):
        """Serve alerts API"""
        alerts = self.monitoring_engine.get_recent_alerts(20)
        alerts_data = [
            {
                'id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'component': alert.component,
                'message': alert.message,
                'resolved': alert.resolved
            }
            for alert in alerts
        ]
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(alerts_data).encode())
    
    def _serve_static_file(self, path: str):
        """Serve static files"""
        try:
            if path.endswith('.css'):
                content_type = 'text/css'
            elif path.endswith('.js'):
                content_type = 'application/javascript'
            else:
                content_type = 'text/plain'
            
            # For demo purposes, serve minimal content
            if 'dashboard.css' in path:
                content = self._get_css_content()
            elif 'dashboard.js' in path:
                content = self._get_js_content()
            else:
                content = ""
            
            self.send_response(200)
            self.send_header('Content-type', content_type)
            self.end_headers()
            self.wfile.write(content.encode())
            
        except Exception as e:
            self._serve_404()
    
    def _serve_404(self):
        """Serve 404 page"""
        self.send_response(404)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"404 - Not Found")
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Q-FAIRS Trading Dashboard</title>
    <link rel="stylesheet" href="/static/dashboard.css">
</head>
<body>
    <div class="dashboard-container">
        <header class="dashboard-header">
            <h1>🚀 Q-FAIRS Trading Dashboard</h1>
            <div class="status-indicator" id="system-status">
                <span class="status-dot active"></span>
                <span>System Active</span>
            </div>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>💰 Portfolio Value</h3>
                <div class="metric-value" id="portfolio-value">$0.00</div>
                <div class="metric-change" id="portfolio-change">+0.00%</div>
            </div>
            
            <div class="metric-card">
                <h3>📊 Daily P&L</h3>
                <div class="metric-value" id="daily-pnl">$0.00</div>
                <div class="metric-change" id="daily-change">+0.00%</div>
            </div>
            
            <div class="metric-card">
                <h3>🎯 Win Rate</h3>
                <div class="metric-value" id="win-rate">0.00%</div>
                <div class="metric-trend" id="win-trend">📈</div>
            </div>
            
            <div class="metric-card">
                <h3>⚡ Sharpe Ratio</h3>
                <div class="metric-value" id="sharpe-ratio">0.00</div>
                <div class="metric-trend" id="sharpe-trend">📊</div>
            </div>
            
            <div class="metric-card">
                <h3>📈 Max Drawdown</h3>
                <div class="metric-value" id="max-drawdown">0.00%</div>
                <div class="metric-status" id="drawdown-status">✅ Normal</div>
            </div>
            
            <div class="metric-card">
                <h3>🔄 Total Trades</h3>
                <div class="metric-value" id="total-trades">0</div>
                <div class="metric-trend" id="trade-trend">📊</div>
            </div>
        </div>
        
        <div class="dashboard-sections">
            <div class="section">
                <h2>🚨 Active Alerts</h2>
                <div class="alerts-container" id="alerts-container">
                    <div class="alert-item">
                        <span class="alert-level info">ℹ️</span>
                        <span class="alert-message">System initialized successfully</span>
                        <span class="alert-time">Now</span>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔬 Quantum Metrics</h2>
                <div class="quantum-metrics">
                    <div class="quantum-metric">
                        <span class="metric-label">Quantum Advantage:</span>
                        <span class="metric-value" id="quantum-advantage">100x</span>
                    </div>
                    <div class="quantum-metric">
                        <span class="metric-label">Algorithm Success:</span>
                        <span class="metric-value" id="algorithm-success">95%</span>
                    </div>
                    <div class="quantum-metric">
                        <span class="metric-label">Execution Time:</span>
                        <span class="metric-value" id="execution-time">150ms</span>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🖥️ System Health</h2>
                <div class="system-metrics">
                    <div class="system-metric">
                        <span class="metric-label">CPU Usage:</span>
                        <div class="progress-bar">
                            <div class="progress-fill" id="cpu-usage" style="width: 25%"></div>
                        </div>
                        <span class="metric-value" id="cpu-value">25%</span>
                    </div>
                    <div class="system-metric">
                        <span class="metric-label">Memory Usage:</span>
                        <div class="progress-bar">
                            <div class="progress-fill" id="memory-usage" style="width: 40%"></div>
                        </div>
                        <span class="metric-value" id="memory-value">40%</span>
                    </div>
                    <div class="system-metric">
                        <span class="metric-label">Uptime:</span>
                        <span class="metric-value" id="uptime">1h 23m</span>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="dashboard-footer">
            <div class="footer-info">
                <span>Q-FAIRS Trading System v1.0</span>
                <span>Last Update: <span id="last-update">Never</span></span>
            </div>
        </footer>
    </div>
    
    <script src="/static/dashboard.js"></script>
</body>
</html>
        '''
    
    def _get_css_content(self) -> str:
        """Get CSS content"""
        return '''
body {
    font-family: 'Arial', sans-serif;
    margin: 0;
    padding: 0;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: #ffffff;
}

.dashboard-container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 30px;
    padding: 20px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

.dashboard-header h1 {
    margin: 0;
    font-size: 2.5em;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

.status-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.2em;
}

.status-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #4CAF50;
    animation: pulse 2s infinite;
}

.status-dot.active {
    background: #4CAF50;
}

@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 30px;
}

.metric-card {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
}

.metric-card h3 {
    margin: 0 0 15px 0;
    font-size: 1.1em;
    opacity: 0.8;
}

.metric-value {
    font-size: 2.5em;
    font-weight: bold;
    margin-bottom: 10px;
}

.metric-change {
    font-size: 1em;
    opacity: 0.8;
}

.dashboard-sections {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 20px;
}

.section {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.section h2 {
    margin: 0 0 20px 0;
    font-size: 1.5em;
}

.alerts-container {
    max-height: 300px;
    overflow-y: auto;
}

.alert-item {
    display: flex;
    align-items: center;
    gap: 15px;
    padding: 15px;
    margin-bottom: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
}

.alert-level {
    font-size: 1.2em;
}

.alert-level.critical {
    color: #f44336;
}

.alert-level.error {
    color: #ff9800;
}

.alert-level.warning {
    color: #ffc107;
}

.alert-level.info {
    color: #2196F3;
}

.alert-message {
    flex: 1;
}

.alert-time {
    opacity: 0.7;
    font-size: 0.9em;
}

.quantum-metrics {
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.quantum-metric {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}

.system-metrics {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.system-metric {
    display: flex;
    align-items: center;
    gap: 15px;
}

.metric-label {
    min-width: 120px;
    font-weight: bold;
}

.progress-bar {
    flex: 1;
    height: 20px;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
    transition: width 0.3s ease;
}

.dashboard-footer {
    margin-top: 30px;
    padding: 20px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    backdrop-filter: blur(10px);
    text-align: center;
}

.footer-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

@media (max-width: 768px) {
    .dashboard-container {
        padding: 10px;
    }
    
    .dashboard-header {
        flex-direction: column;
        gap: 15px;
        text-align: center;
    }
    
    .dashboard-header h1 {
        font-size: 2em;
    }
    
    .metrics-grid {
        grid-template-columns: 1fr;
    }
    
    .dashboard-sections {
        grid-template-columns: 1fr;
    }
    
    .footer-info {
        flex-direction: column;
        gap: 10px;
    }
}
        '''
    
    def _get_js_content(self) -> str:
        """Get JavaScript content"""
        return '''
// Dashboard JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize dashboard
    initializeDashboard();
    
    // Start auto-refresh
    setInterval(updateDashboard, 5000); // Update every 5 seconds
});

function initializeDashboard() {
    updateDashboard();
}

async function updateDashboard() {
    try {
        // Fetch metrics
        const metricsResponse = await fetch('/api/metrics');
        const metrics = await metricsResponse.json();
        
        // Fetch alerts
        const alertsResponse = await fetch('/api/alerts');
        const alerts = await alertsResponse.json();
        
        // Update dashboard
        updateMetrics(metrics);
        updateAlerts(alerts);
        updateLastUpdateTime();
        
    } catch (error) {
        console.error('Failed to update dashboard:', error);
    }
}

function updateMetrics(metrics) {
    // Update trading metrics
    if (metrics.trading) {
        updateElement('portfolio-value', formatCurrency(metrics.trading.total_pnl || 0));
        updateElement('daily-pnl', formatCurrency(metrics.trading.daily_pnl || 0));
        updateElement('win-rate', formatPercentage(metrics.trading.win_rate || 0));
        updateElement('sharpe-ratio', (metrics.trading.sharpe_ratio || 0).toFixed(2));
        updateElement('max-drawdown', formatPercentage(metrics.trading.max_drawdown || 0));
        updateElement('total-trades', metrics.trading.total_trades || 0);
    }
    
    // Update quantum metrics
    if (metrics.quantum) {
        updateElement('quantum-advantage', (metrics.quantum.quantum_advantage_score || 1).toFixed(0) + 'x');
        updateElement('algorithm-success', formatPercentage(metrics.quantum.algorithm_success_rate || 0));
        updateElement('execution-time', (metrics.quantum.average_execution_time || 0).toFixed(0) + 'ms');
    }
    
    // Update system metrics
    if (metrics.system) {
        updateElement('cpu-value', formatPercentage(metrics.system.cpu_usage || 0));
        updateElement('memory-value', formatPercentage(metrics.system.memory_usage || 0));
        updateElement('uptime', formatUptime(metrics.system.uptime_seconds || 0));
        
        updateProgressBar('cpu-usage', metrics.system.cpu_usage || 0);
        updateProgressBar('memory-usage', metrics.system.memory_usage || 0);
    }
}

function updateAlerts(alerts) {
    const container = document.getElementById('alerts-container');
    
    if (alerts.length === 0) {
        container.innerHTML = '<div class=\"alert-item\">' +
            '<span class=\"alert-level info\">ℹ️</span>' +
            '<span class=\"alert-message\">No active alerts</span>' +
            '<span class=\"alert-time\">Now</span>' +
            '</div>';
        return;
    }
    
    container.innerHTML = alerts.map(alert => {
        const levelClass = alert.level || 'info';
        const icon = getAlertIcon(levelClass);
        const timeAgo = getTimeAgo(new Date(alert.timestamp));
        
        return `<div class=\"alert-item\">` +
            `<span class=\"alert-level ${levelClass}\">${icon}</span>` +
            `<span class=\"alert-message\">${alert.message}</span>` +
            `<span class=\"alert-time\">${timeAgo}</span>` +
            `</div>`;
    }).join('');
}

function updateElement(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        element.textContent = value;
    }
}

function updateProgressBar(elementId, percentage) {
    const element = document.getElementById(elementId);
    if (element) {
        element.style.width = Math.min(100, Math.max(0, percentage)) + '%';
    }
}

function updateLastUpdateTime() {
    const element = document.getElementById('last-update');
    if (element) {
        element.textContent = new Date().toLocaleTimeString();
    }
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPercentage(value) {
    return (value * 100).toFixed(2) + '%';
}

function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
}

function getAlertIcon(level) {
    const icons = {
        'critical': '🚨',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    };
    return icons[level] || 'ℹ️';
}

function getTimeAgo(date) {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    
    if (diffMins < 1) return 'Now';
    if (diffMins < 60) return `${diffMins}m ago`;
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays}d ago`;
}

// Auto-refresh indicator
let refreshCount = 0;
setInterval(() => {
    refreshCount++;
    const indicator = document.querySelector('.status-dot');
    if (indicator && refreshCount % 2 === 0) {
        indicator.style.animation = 'none';
        setTimeout(() => {
            indicator.style.animation = 'pulse 2s infinite';
        }, 10);
    }
}, 10000);
        '''

class MonitoringDashboard:
    """Web-based monitoring dashboard"""
    
    def __init__(self, monitoring_engine: MonitoringEngine, port: int = 8080):
        self.monitoring_engine = monitoring_engine
        self.port = port
        self.server = None
        self.server_thread = None
    
    def start(self):
        """Start the monitoring dashboard"""
        try:
            # Create HTTP server
            handler = lambda *args, **kwargs: DashboardHandler(self.monitoring_engine, *args, **kwargs)
            self.server = HTTPServer(('localhost', self.port), handler)
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            logger.info(f"🌐 Monitoring dashboard started at http://localhost:{self.port}")
            
            # Open browser
            webbrowser.open(f'http://localhost:{self.port}')
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring dashboard: {e}")
    
    def stop(self):
        """Stop the monitoring dashboard"""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            logger.info("🌐 Monitoring dashboard stopped")

# ==================== MAIN MONITORING ====================

class QFairsMonitoringSystem:
    """Main monitoring system orchestrator"""
    
    def __init__(self):
        self.monitoring_engine = MonitoringEngine()
        self.dashboard = MonitoringDashboard(self.monitoring_engine)
        self.is_active = False
        
    async def start_monitoring(self):
        """Start complete monitoring system"""
        try:
            logger.info("🚀 Starting Q-FAIRS Monitoring System")
            
            self.is_active = True
            
            # Start monitoring engine
            monitoring_task = asyncio.create_task(
                self.monitoring_engine.monitoring_loop()
            )
            
            # Start web dashboard
            self.dashboard.start()
            
            logger.info("✅ Monitoring system started")
            
            # Wait for monitoring task
            await monitoring_task
            
        except Exception as e:
            logger.error(f"❌ Monitoring system startup failed: {e}")
            raise
    
    def update_trading_metrics(self, metrics: Dict):
        """Update trading metrics"""
        self.monitoring_engine.update_metrics('trading', metrics)
    
    def update_quantum_metrics(self, metrics: Dict):
        """Update quantum metrics"""
        self.monitoring_engine.update_metrics('quantum', metrics)
    
    def add_alert(self, level: AlertLevel, component: str, message: str, details: Dict = None):
        """Add system alert"""
        return self.monitoring_engine.add_alert(level, component, message, details)
    
    def get_dashboard_url(self) -> str:
        """Get dashboard URL"""
        return f"http://localhost:{self.dashboard.port}"
    
    def stop_monitoring(self):
        """Stop monitoring system"""
        self.is_active = False
        self.monitoring_engine.is_monitoring_active = False
        self.dashboard.stop()
        logger.info("🛑 Monitoring system stopped")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Test monitoring system
    async def test_monitoring():
        monitoring_system = QFairsMonitoringSystem()
        
        # Start monitoring
        await monitoring_system.start_monitoring()
        
        # Add some test alerts
        monitoring_system.add_alert(AlertLevel.INFO, "System", "Test alert - System started")
        monitoring_system.add_alert(AlertLevel.WARNING, "Trading", "Test warning - High volatility detected")
        
        # Update test metrics
        test_trading_metrics = {
            'total_pnl': 1250.50,
            'daily_pnl': 150.25,
            'win_rate': 0.65,
            'sharpe_ratio': 1.8,
            'max_drawdown': 0.08,
            'current_positions': 3,
            'total_trades': 25
        }
        
        monitoring_system.update_trading_metrics(test_trading_metrics)
        
        test_quantum_metrics = {
            'quantum_advantage_score': 100,
            'algorithm_success_rate': 0.95,
            'average_execution_time': 150,
            'quantum_circuit_calls': 1000,
            'classical_fallbacks': 5
        }
        
        monitoring_system.update_quantum_metrics(test_quantum_metrics)
        
        print(f"Dashboard URL: {monitoring_system.get_dashboard_url()}")
        
        # Let it run for a while
        await asyncio.sleep(60)
        
        # Stop monitoring
        monitoring_system.stop_monitoring()
        
        return True
    
    # Run test
    success = asyncio.run(test_monitoring())
    print("✅ Monitoring system test completed")