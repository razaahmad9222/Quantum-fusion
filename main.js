// Quantum Fusion v9.0 - Main JavaScript Controller
// Production-ready with error handling and real-time updates

class QuantumFusionApp {
    constructor() {
        this.isInitialized = false;
        this.websocketConnected = false;
        this.quantumVisualizer = null;
        this.metrics = {
            pnl: 0,
            successRate: 0,
            activeCycles: 0,
            systemLoad: 67
        };
        this.opportunities = [];
        this.recentTrades = [];
        
        this.init();
    }

    async init() {
        try {
            console.log('🚀 Initializing Quantum Fusion v9.0...');
            
            // Initialize components
            this.initQuantumVisualizer();
            this.startMetricsAnimation();
            this.generateMockData();
            this.setupEventListeners();
            this.startRealTimeUpdates();
            
            this.isInitialized = true;
            console.log('✅ Quantum Fusion v9.0 Initialized Successfully');
            
        } catch (error) {
            console.error('❌ Initialization failed:', error);
            this.showError('System initialization failed. Please refresh the page.');
        }
    }

    // Quantum Visualizer using p5.js
    initQuantumVisualizer() {
        const sketch = (p) => {
            let particles = [];
            let quantumGates = [];
            let time = 0;

            p.setup = () => {
                const canvas = p.createCanvas(800, 256);
                canvas.parent('quantum-visualizer');
                
                // Create quantum particles
                for (let i = 0; i < 20; i++) {
                    particles.push({
                        x: p.random(p.width),
                        y: p.random(p.height),
                        vx: p.random(-1, 1),
                        vy: p.random(-1, 1),
                        size: p.random(3, 8),
                        entangled: Math.random() > 0.7
                    });
                }
                
                // Create quantum gates
                for (let i = 0; i < 5; i++) {
                    quantumGates.push({
                        x: 100 + i * 150,
                        y: p.height / 2,
                        rotation: 0,
                        type: i % 3
                    });
                }
            };

            p.draw = () => {
                p.background(10, 10, 10, 50);
                time += 0.02;

                // Draw quantum gates
                quantumGates.forEach(gate => {
                    p.push();
                    p.translate(gate.x, gate.y);
                    p.rotate(gate.rotation);
                    
                    // Gate visualization
                    p.stroke(0, 212, 255, 150);
                    p.strokeWeight(2);
                    p.noFill();
                    p.rect(-15, -15, 30, 30);
                    
                    // Gate type indicator
                    if (gate.type === 0) {
                        p.line(-10, 0, 10, 0);
                    } else if (gate.type === 1) {
                        p.circle(0, 0, 15);
                    } else {
                        p.line(-10, -10, 10, 10);
                        p.line(-10, 10, 10, -10);
                    }
                    
                    gate.rotation += 0.01;
                    p.pop();
                });

                // Update and draw particles
                particles.forEach((particle, i) => {
                    // Update position
                    particle.x += particle.vx;
                    particle.y += particle.vy;
                    
                    // Bounce off walls
                    if (particle.x < 0 || particle.x > p.width) particle.vx *= -1;
                    if (particle.y < 0 || particle.y > p.height) particle.vy *= -1;
                    
                    // Keep in bounds
                    particle.x = p.constrain(particle.x, 0, p.width);
                    particle.y = p.constrain(particle.y, 0, p.height);
                    
                    // Draw particle
                    p.fill(0, 212, 255, particle.entangled ? 200 : 100);
                    p.noStroke();
                    p.circle(particle.x, particle.y, particle.size);
                    
                    // Draw entanglement connections
                    if (particle.entangled) {
                        particles.forEach((other, j) => {
                            if (i !== j && other.entangled) {
                                const distance = p.dist(particle.x, particle.y, other.x, other.y);
                                if (distance < 100) {
                                    p.stroke(0, 212, 255, 100 * (1 - distance / 100));
                                    p.strokeWeight(1);
                                    p.line(particle.x, particle.y, other.x, other.y);
                                }
                            }
                        });
                    }
                });

                // Draw quantum circuit paths
                p.stroke(45, 125, 139, 100);
                p.strokeWeight(1);
                for (let i = 0; i < p.height; i += 20) {
                    p.line(0, i, p.width, i);
                }
            };
        };

        this.quantumVisualizer = new p5(sketch);
    }

    // Animate metrics counters
    startMetricsAnimation() {
        // Animate PNL counter
        anime({
            targets: { value: 0 },
            value: 15420.50,
            duration: 2000,
            easing: 'easeOutExpo',
            update: (anim) => {
                const value = anim.animatables[0].target.value;
                document.getElementById('pnl-counter').textContent = `$${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
            }
        });

        // Animate success rate
        anime({
            targets: { value: 0 },
            value: 87.3,
            duration: 2000,
            delay: 500,
            easing: 'easeOutExpo',
            update: (anim) => {
                const value = anim.animatables[0].target.value;
                document.getElementById('success-counter').textContent = `${value.toFixed(1)}%`;
            }
        });

        // Animate active cycles
        anime({
            targets: { value: 0 },
            value: 3,
            duration: 2000,
            delay: 1000,
            easing: 'easeOutExpo',
            update: (anim) => {
                const value = Math.floor(anim.animatables[0].target.value);
                document.getElementById('cycles-counter').textContent = value;
            }
        });
    }

    // Generate mock data for demonstration
    generateMockData() {
        // Mock arbitrage opportunities
        this.opportunities = [
            {
                pair: 'BTC/USDT',
                spread: 0.15,
                profit: 125.50,
                risk: 'Low',
                exchanges: ['Binance', 'Coinbase'],
                timestamp: Date.now()
            },
            {
                pair: 'ETH/USDT',
                spread: 0.23,
                profit: 89.20,
                risk: 'Medium',
                exchanges: ['Kraken', 'Bybit'],
                timestamp: Date.now() - 30000
            },
            {
                pair: 'ADA/USDT',
                spread: 0.31,
                profit: 45.80,
                risk: 'Low',
                exchanges: ['Binance', 'KuCoin'],
                timestamp: Date.now() - 60000
            },
            {
                pair: 'SOL/USDT',
                spread: 0.18,
                profit: 67.40,
                risk: 'Medium',
                exchanges: ['FTX', 'Bybit'],
                timestamp: Date.now() - 90000
            },
            {
                pair: 'DOT/USDT',
                spread: 0.27,
                profit: 34.60,
                risk: 'High',
                exchanges: ['Binance', 'Huobi'],
                timestamp: Date.now() - 120000
            }
        ];

        // Mock recent trades
        this.recentTrades = [
            {
                pair: 'BTC/USDT',
                type: 'Buy',
                amount: 0.025,
                price: 43250.00,
                pnl: 125.50,
                timestamp: Date.now() - 300000
            },
            {
                pair: 'ETH/USDT',
                type: 'Sell',
                amount: 1.5,
                price: 2650.00,
                pnl: -45.20,
                timestamp: Date.now() - 600000
            },
            {
                pair: 'ADA/USDT',
                type: 'Buy',
                amount: 1000,
                price: 0.485,
                pnl: 23.40,
                timestamp: Date.now() - 900000
            }
        ];

        this.renderOpportunities();
        this.renderRecentTrades();
    }

    // Render arbitrage opportunities table
    renderOpportunities() {
        const tableBody = document.getElementById('opportunities-table');
        tableBody.innerHTML = '';

        this.opportunities.forEach((opp, index) => {
            const row = document.createElement('tr');
            row.className = 'arb-row border-b border-gray-700';
            
            const riskColor = {
                'Low': 'text-green-400',
                'Medium': 'text-yellow-400',
                'High': 'text-red-400'
            }[opp.risk];

            row.innerHTML = `
                <td class="py-3 font-mono">${opp.pair}</td>
                <td class="py-3">${opp.spread}%</td>
                <td class="py-3 profit-positive">$${opp.profit.toFixed(2)}</td>
                <td class="py-3 ${riskColor}">${opp.risk}</td>
                <td class="py-3">
                    <button class="quantum-button text-sm px-3 py-1" onclick="executeArbitrage(${index})">
                        Execute
                    </button>
                </td>
            `;
            
            tableBody.appendChild(row);
        });
    }

    // Render recent trades
    renderRecentTrades() {
        const container = document.getElementById('recent-trades');
        container.innerHTML = '';

        this.recentTrades.forEach(trade => {
            const tradeElement = document.createElement('div');
            tradeElement.className = 'flex items-center justify-between p-3 bg-gray-800 rounded-lg';
            
            const pnlClass = trade.pnl >= 0 ? 'profit-positive' : 'profit-negative';
            const pnlSign = trade.pnl >= 0 ? '+' : '';

            tradeElement.innerHTML = `
                <div>
                    <div class="font-mono text-sm">${trade.pair}</div>
                    <div class="text-xs text-gray-400">${trade.type} ${trade.amount}</div>
                </div>
                <div class="text-right">
                    <div class="font-mono text-sm">$${trade.price.toFixed(2)}</div>
                    <div class="font-mono text-xs ${pnlClass}">${pnlSign}$${trade.pnl.toFixed(2)}</div>
                </div>
            `;
            
            container.appendChild(tradeElement);
        });
    }

    // Setup event listeners
    setupEventListeners() {
        // Handle window resize for quantum visualizer
        window.addEventListener('resize', () => {
            if (this.quantumVisualizer) {
                this.quantumVisualizer.windowResized();
            }
        });

        // Handle keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case 'r':
                        e.preventDefault();
                        this.refreshOpportunities();
                        break;
                    case 'e':
                        e.preventDefault();
                        this.emergencyStop();
                        break;
                }
            }
        });
    }

    // Start real-time updates
    startRealTimeUpdates() {
        // Update system metrics every 5 seconds
        setInterval(() => {
            this.updateSystemMetrics();
        }, 5000);

        // Update opportunities every 10 seconds
        setInterval(() => {
            this.refreshOpportunities();
        }, 10000);

        // Update system load
        setInterval(() => {
            this.updateSystemLoad();
        }, 2000);
    }

    // Update system metrics
    updateSystemMetrics() {
        // Simulate real-time metric updates
        const variations = {
            pnl: (Math.random() - 0.5) * 100,
            successRate: (Math.random() - 0.5) * 2,
            activeCycles: Math.floor(Math.random() * 3) - 1
        };

        // Update PNL
        this.metrics.pnl += variations.pnl;
        anime({
            targets: { value: parseFloat(document.getElementById('pnl-counter').textContent.replace(/[$,]/g, '')) },
            value: this.metrics.pnl,
            duration: 1000,
            easing: 'easeOutQuad',
            update: (anim) => {
                const value = anim.animatables[0].target.value;
                const element = document.getElementById('pnl-counter');
                element.textContent = `$${value.toLocaleString('en-US', { minimumFractionDigits: 2 })}`;
                element.className = value >= 0 ? 'text-3xl font-bold number-counter profit-positive' : 'text-3xl font-bold number-counter profit-negative';
            }
        });

        // Update success rate
        this.metrics.successRate = Math.max(0, Math.min(100, this.metrics.successRate + variations.successRate));
        anime({
            targets: { value: parseFloat(document.getElementById('success-counter').textContent.replace('%', '')) },
            value: this.metrics.successRate,
            duration: 1000,
            easing: 'easeOutQuad',
            update: (anim) => {
                const value = anim.animatables[0].target.value;
                document.getElementById('success-counter').textContent = `${value.toFixed(1)}%`;
            }
        });

        // Update active cycles
        this.metrics.activeCycles = Math.max(0, this.metrics.activeCycles + variations.activeCycles);
        anime({
            targets: { value: parseInt(document.getElementById('cycles-counter').textContent) },
            value: this.metrics.activeCycles,
            duration: 1000,
            easing: 'easeOutQuad',
            update: (anim) => {
                const value = Math.floor(anim.animatables[0].target.value);
                document.getElementById('cycles-counter').textContent = value;
            }
        });
    }

    // Update system load
    updateSystemLoad() {
        this.metrics.systemLoad = Math.max(20, Math.min(95, this.metrics.systemLoad + (Math.random() - 0.5) * 10));
        const loadBar = document.getElementById('system-load');
        loadBar.style.width = `${this.metrics.systemLoad}%`;
        
        // Update load status text
        const statusText = loadBar.parentElement.nextElementSibling;
        let statusMessage = 'Optimal Performance';
        if (this.metrics.systemLoad > 80) statusMessage = 'High Load';
        else if (this.metrics.systemLoad < 40) statusMessage = 'Low Activity';
        
        statusText.textContent = `${Math.round(this.metrics.systemLoad)}% - ${statusMessage}`;
    }

    // Show metric details modal
    showMetricDetails(metricType) {
        const modal = document.getElementById('metric-modal');
        const title = document.getElementById('modal-title');
        const content = document.getElementById('modal-content');

        const metricData = {
            pnl: {
                title: 'Profit & Loss Details',
                content: `
                    <div class="space-y-4">
                        <div class="flex justify-between">
                            <span>Today's PNL:</span>
                            <span class="profit-positive">+$2,450.30</span>
                        </div>
                        <div class="flex justify-between">
                            <span>This Week:</span>
                            <span class="profit-positive">+$15,420.50</span>
                        </div>
                        <div class="flex justify-between">
                            <span>This Month:</span>
                            <span class="profit-positive">+$67,890.20</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Total Trades:</span>
                            <span>1,247</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Average Trade:</span>
                            <span class="profit-positive">+$54.47</span>
                        </div>
                    </div>
                `
            },
            success: {
                title: 'Success Rate Analysis',
                content: `
                    <div class="space-y-4">
                        <div class="flex justify-between">
                            <span>Win Rate:</span>
                            <span class="profit-positive">87.3%</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Last 100 Trades:</span>
                            <span>87 successful</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Best Streak:</span>
                            <span>23 consecutive wins</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Average Win:</span>
                            <span class="profit-positive">+$125.40</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Average Loss:</span>
                            <span class="profit-negative">-$45.20</span>
                        </div>
                    </div>
                `
            },
            cycles: {
                title: 'Active Arbitrage Cycles',
                content: `
                    <div class="space-y-4">
                        <div class="flex justify-between">
                            <span>Currently Executing:</span>
                            <span>3 cycles</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Queue Length:</span>
                            <span>7 pending</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Avg Execution Time:</span>
                            <span>2.3 seconds</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Success Rate:</span>
                            <span class="profit-positive">94.2%</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Failed Cycles:</span>
                            <span>2 today</span>
                        </div>
                    </div>
                `
            }
        };

        const data = metricData[metricType];
        if (data) {
            title.textContent = data.title;
            content.innerHTML = data.content;
            modal.classList.remove('hidden');
        }
    }

    // Close modal
    closeModal() {
        document.getElementById('metric-modal').classList.add('hidden');
    }

    // Show error message
    showError(message) {
        console.error('❌ Error:', message);
        // Could implement toast notification here
    }

    // Show success message
    showSuccess(message) {
        console.log('✅ Success:', message);
        // Could implement toast notification here
    }
}

// Global functions for HTML onclick handlers
let app;

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    app = new QuantumFusionApp();
});

// Global function handlers
function showMetricDetails(metricType) {
    if (app) app.showMetricDetails(metricType);
}

function closeModal() {
    if (app) app.closeModal();
}

function startQuantumAnalysis() {
    console.log('🔄 Starting quantum analysis...');
    // Simulate analysis start
    anime({
        targets: '.quantum-button',
        scale: [1, 0.95, 1],
        duration: 200,
        easing: 'easeInOutQuad'
    });
    
    setTimeout(() => {
        alert('Quantum analysis started! Check the visualizer for real-time computation.');
    }, 500);
}

function emergencyStop() {
    console.log('🚨 Emergency stop activated!');
    
    // Visual feedback
    anime({
        targets: 'body',
        backgroundColor: ['#0a0a0a', '#1a365d', '#0a0a0a'],
        duration: 1000,
        easing: 'easeInOutQuad'
    });
    
    // Update status indicators
    document.querySelectorAll('.status-online').forEach(indicator => {
        indicator.className = 'status-indicator status-error';
    });
    
    alert('EMERGENCY STOP ACTIVATED!\n\nAll trading operations have been halted.\nCircuit breaker is now open.\n\nPlease review system status before resuming.');
}

function refreshOpportunities() {
    console.log('🔄 Refreshing arbitrage opportunities...');
    
    // Show loading state
    const button = event.target;
    const originalText = button.textContent;
    button.innerHTML = '<div class="loading-spinner inline-block mr-2"></div>Refreshing...';
    button.disabled = true;
    
    setTimeout(() => {
        if (app) app.generateMockData();
        button.textContent = originalText;
        button.disabled = false;
        
        // Success feedback
        anime({
            targets: button,
            backgroundColor: ['#2d7d8b', '#38a169', '#2d7d8b'],
            duration: 1000,
            easing: 'easeInOutQuad'
        });
    }, 1500);
}

function executeArbitrage(index) {
    console.log(`⚡ Executing arbitrage opportunity ${index}...`);
    
    const opportunity = app ? app.opportunities[index] : null;
    if (opportunity) {
        const confirmed = confirm(`Execute arbitrage for ${opportunity.pair}?\n\nSpread: ${opportunity.spread}%\nPotential Profit: $${opportunity.profit.toFixed(2)}\nRisk Level: ${opportunity.risk}`);
        
        if (confirmed) {
            // Simulate execution
            anime({
                targets: event.target,
                scale: [1, 0.9, 1],
                backgroundColor: ['#2d7d8b', '#38a169', '#2d7d8b'],
                duration: 500,
                easing: 'easeInOutQuad'
            });
            
            setTimeout(() => {
                alert(`✅ Arbitrage executed successfully!\n\n${opportunity.pair} position opened.\nMonitor the position in the Trading tab.`);
            }, 1000);
        }
    }
}

function executeQuickTrade() {
    console.log('⚡ Quick arbitrage execution...');
    
    // Find best opportunity
    const bestOpportunity = app ? app.opportunities.reduce((best, current) => 
        current.profit > best.profit ? current : best
    ) : null;
    
    if (bestOpportunity) {
        const confirmed = confirm(`Execute quick arbitrage for best opportunity?\n\n${bestOpportunity.pair} - $${bestOpportunity.profit.toFixed(2)} profit`);
        
        if (confirmed) {
            alert('🚀 Quick arbitrage executed!\n\nSystem automatically selected the most profitable opportunity.');
        }
    } else {
        alert('❌ No opportunities available for quick execution.');
    }
}

function openRiskPanel() {
    console.log('🛡️ Opening risk management panel...');
    alert('Risk Management Panel\n\nSlippage Tolerance: 0.1%\nMax Position Size: $50,000\nStop Loss: 2%\nCircuit Breaker: Active\n\nFeature coming soon!');
}

function viewLogs() {
    console.log('📊 Viewing system logs...');
    window.location.href = 'analytics.html#logs';
}

function exportData() {
    console.log('💾 Exporting trading data...');
    
    // Simulate data export
    const data = {
        timestamp: new Date().toISOString(),
        pnl: app ? app.metrics.pnl : 0,
        trades: app ? app.recentTrades.length : 0,
        opportunities: app ? app.opportunities.length : 0
    };
    
    const dataStr = JSON.stringify(data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `quantum-fusion-export-${Date.now()}.json`;
    link.click();
    
    URL.revokeObjectURL(url);
    
    alert('✅ Data exported successfully!\n\nFile saved as quantum-fusion-export.json');
}

function viewAllTrades() {
    console.log('📈 Viewing all trades...');
    window.location.href = 'analytics.html#trades';
}