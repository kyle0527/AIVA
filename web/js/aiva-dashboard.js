/**
 * AIVA Security Platform - Dashboard JavaScript
 * 
 * 提供 Web 管理界面的所有交互功能，包括認證、
 * 掃描管理、結果展示等。
 */

class AIVADashboard {
    constructor() {
        this.apiBase = 'http://localhost:8000';
        this.accessToken = localStorage.getItem('aiva_token');
        this.currentScanType = null;
        this.refreshInterval = null;
        
        this.init();
    }
    
    async init() {
        // 檢查登入狀態
        if (this.accessToken) {
            try {
                await this.verifyToken();
                this.showMainContent();
                this.startAutoRefresh();
            } catch (error) {
                this.showLogin();
            }
        } else {
            this.showLogin();
        }
    }
    
    showLogin() {
        document.getElementById('main-content').style.display = 'none';
        const loginModal = new bootstrap.Modal(document.getElementById('loginModal'));
        loginModal.show();
    }
    
    showMainContent() {
        document.getElementById('main-content').style.display = 'block';
        this.loadDashboardData();
    }
    
    async verifyToken() {
        const response = await fetch(`${this.apiBase}/auth/me`, {
            headers: {
                'Authorization': `Bearer ${this.accessToken}`
            }
        });
        
        if (!response.ok) {
            throw new Error('Token invalid');
        }
        
        const userInfo = await response.json();
        document.getElementById('user-info').innerHTML = `
            <i class="bi bi-person-circle"></i>
            ${userInfo.username} (${userInfo.role})
        `;
        
        return userInfo;
    }
    
    async login() {
        const btn = event.target;
        btn.classList.add('loading');
        
        try {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            const response = await fetch(`${this.apiBase}/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ username, password })
            });
            
            if (!response.ok) {
                throw new Error('Login failed');
            }
            
            const data = await response.json();
            this.accessToken = data.access_token;
            localStorage.setItem('aiva_token', this.accessToken);
            
            // 關閉登入模態框
            const loginModal = bootstrap.Modal.getInstance(document.getElementById('loginModal'));
            loginModal.hide();
            
            this.showMainContent();
            this.startAutoRefresh();
            
        } catch (error) {
            alert('登入失敗: ' + error.message);
        } finally {
            btn.classList.remove('loading');
        }
    }
    
    logout() {
        localStorage.removeItem('aiva_token');
        this.accessToken = null;
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        location.reload();
    }
    
    async loadDashboardData() {
        try {
            // 載入系統狀態
            await this.loadSystemStatus();
            
            // 載入統計數據
            await this.loadStats();
            
            // 載入最近掃描
            await this.loadRecentScans();
            
        } catch (error) {
            console.error('Failed to load dashboard data:', error);
        }
    }
    
    async loadSystemStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            const statusElement = document.getElementById('system-status');
            const updateElement = document.getElementById('last-update');
            
            if (data.status === 'healthy') {
                statusElement.className = 'badge bg-success';
                statusElement.textContent = '正常運行';
            } else {
                statusElement.className = 'badge bg-warning';
                statusElement.textContent = data.status;
            }
            
            updateElement.textContent = `最後更新: ${new Date().toLocaleTimeString()}`;
            
        } catch (error) {
            const statusElement = document.getElementById('system-status');
            statusElement.className = 'badge bg-danger';
            statusElement.textContent = '連接失敗';
        }
    }
    
    async loadStats() {
        try {
            const response = await fetch(`${this.apiBase}/api/v1/admin/stats`, {
                headers: {
                    'Authorization': `Bearer ${this.accessToken}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                
                document.getElementById('total-scans').textContent = data.total_scans || 0;
                
                // 計算活躍掃描數量
                const activeScanCount = Object.values(data.scan_statuses || {})
                    .filter(status => status === 'started' || status === 'running')
                    .reduce((sum, count) => sum + count, 0);
                
                document.getElementById('active-scans').textContent = activeScanCount;
                
                // 系統負載（如果有的話）
                if (data.system_resources && data.system_resources.cpu_percent) {
                    document.getElementById('system-load').textContent = 
                        `${Math.round(data.system_resources.cpu_percent)}%`;
                } else {
                    document.getElementById('system-load').textContent = '正常';
                }
            }
            
        } catch (error) {
            console.error('Failed to load stats:', error);
        }
    }
    
    async loadRecentScans() {
        try {
            const response = await fetch(`${this.apiBase}/api/v1/scans`, {
                headers: {
                    'Authorization': `Bearer ${this.accessToken}`
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                const scansContainer = document.getElementById('recent-scans');
                
                if (data.scans && data.scans.length > 0) {
                    scansContainer.innerHTML = data.scans
                        .slice(0, 5) // 只顯示最近5個
                        .map(scan => this.createScanResultHTML(scan))
                        .join('');
                } else {
                    scansContainer.innerHTML = '<p class="text-muted text-center">暫無掃描記錄</p>';
                }
            }
            
        } catch (error) {
            console.error('Failed to load recent scans:', error);
        }
    }
    
    createScanResultHTML(scan) {
        const statusClass = {
            'completed': 'success',
            'failed': 'danger',
            'started': 'warning',
            'running': 'warning'
        }[scan.status] || 'secondary';
        
        const statusText = {
            'completed': '已完成',
            'failed': '失敗',
            'started': '進行中',
            'running': '運行中'
        }[scan.status] || scan.status;
        
        const scanTime = scan.start_time ? 
            new Date(scan.start_time).toLocaleString() : '未知';
        
        return `
            <div class="scan-result ${statusClass}">
                <div class="d-flex justify-content-between align-items-start">
                    <div>
                        <h6 class="mb-1">${this.getScanTypeName(scan.type)}</h6>
                        <small class="text-muted">${scanTime}</small>
                    </div>
                    <span class="badge bg-${statusClass === 'warning' ? 'warning' : statusClass}">
                        ${statusText}
                    </span>
                </div>
                <div class="mt-2">
                    <small class="text-muted">掃描 ID: ${scan.scan_id}</small>
                    <button class="btn btn-sm btn-outline-primary ms-2" 
                            onclick="dashboard.viewScanDetails('${scan.scan_id}')">
                        查看詳情
                    </button>
                </div>
            </div>
        `;
    }
    
    getScanTypeName(type) {
        const typeNames = {
            'mass_assignment': 'Mass Assignment',
            'jwt_confusion': 'JWT Confusion',
            'oauth_confusion': 'OAuth Confusion',
            'graphql_authz': 'GraphQL AuthZ',
            'ssrf_oob': 'SSRF OOB'
        };
        return typeNames[type] || type;
    }
    
    startScan(scanType) {
        this.currentScanType = scanType;
        this.setupScanModal(scanType);
        const scanModal = new bootstrap.Modal(document.getElementById('scanModal'));
        scanModal.show();
    }
    
    setupScanModal(scanType) {
        const modalTitle = document.getElementById('scanModalTitle');
        const additionalFields = document.getElementById('additional-fields');
        
        modalTitle.textContent = `配置 ${this.getScanTypeName(scanType)} 掃描`;
        
        // 清空之前的字段
        additionalFields.innerHTML = '';
        
        // 根據掃描類型添加特定字段
        switch (scanType) {
            case 'mass-assignment':
                additionalFields.innerHTML = `
                    <div class="mb-3">
                        <label for="update_endpoint" class="form-label">更新端點 *</label>
                        <input type="text" class="form-control" id="update_endpoint" 
                               placeholder="/api/users/update" required>
                    </div>
                    <div class="mb-3">
                        <label for="auth_header" class="form-label">認證標頭</label>
                        <input type="text" class="form-control" id="auth_header" 
                               placeholder="Bearer your-token-here">
                    </div>
                    <div class="mb-3">
                        <label for="test_fields" class="form-label">測試字段（逗號分隔）</label>
                        <input type="text" class="form-control" id="test_fields" 
                               value="admin,role,is_admin,permissions">
                    </div>
                `;
                break;
                
            case 'jwt-confusion':
                additionalFields.innerHTML = `
                    <div class="mb-3">
                        <label for="victim_token" class="form-label">目標 JWT 令牌 *</label>
                        <textarea class="form-control" id="victim_token" rows="3" 
                                  placeholder="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." required></textarea>
                    </div>
                `;
                break;
                
            case 'oauth-confusion':
                additionalFields.innerHTML = `
                    <div class="mb-3">
                        <label for="client_id" class="form-label">客戶端 ID *</label>
                        <input type="text" class="form-control" id="client_id" required>
                    </div>
                    <div class="mb-3">
                        <label for="legitimate_redirect" class="form-label">合法重定向 URL *</label>
                        <input type="url" class="form-control" id="legitimate_redirect" 
                               placeholder="https://legitimate.com/callback" required>
                    </div>
                    <div class="mb-3">
                        <label for="attacker_redirect" class="form-label">攻擊者重定向 URL *</label>
                        <input type="url" class="form-control" id="attacker_redirect" 
                               placeholder="https://attacker.com/callback" required>
                    </div>
                `;
                break;
                
            case 'graphql-authz':
                additionalFields.innerHTML = `
                    <div class="mb-3">
                        <label for="user_auth" class="form-label">用戶認證標頭</label>
                        <input type="text" class="form-control" id="user_auth" 
                               placeholder="Bearer user-token">
                    </div>
                    <div class="mb-3">
                        <label for="test_queries" class="form-label">測試查詢（一行一個）</label>
                        <textarea class="form-control" id="test_queries" rows="4" 
                                  placeholder="query { users { id email } }
query { adminUsers { id email role } }"></textarea>
                    </div>
                `;
                break;
                
            case 'ssrf-oob':
                additionalFields.innerHTML = `
                    <div class="mb-3">
                        <label for="oob_callback" class="form-label">OOB 回調 URL *</label>
                        <input type="url" class="form-control" id="oob_callback" 
                               placeholder="https://your-oob-server.com/callback" required>
                    </div>
                `;
                break;
        }
    }
    
    async submitScan() {
        const btn = event.target;
        btn.classList.add('loading');
        
        try {
            const scanData = this.getScanFormData();
            const endpoint = this.getScanEndpoint(this.currentScanType);
            
            const response = await fetch(`${this.apiBase}${endpoint}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${this.accessToken}`
                },
                body: JSON.stringify(scanData)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Scan failed');
            }
            
            const result = await response.json();
            
            // 關閉模態框
            const scanModal = bootstrap.Modal.getInstance(document.getElementById('scanModal'));
            scanModal.hide();
            
            // 顯示成功消息
            this.showAlert('success', `掃描已開始！掃描 ID: ${result.scan_id}`);
            
            // 刷新數據
            setTimeout(() => {
                this.loadDashboardData();
            }, 1000);
            
        } catch (error) {
            this.showAlert('danger', '掃描啟動失敗: ' + error.message);
        } finally {
            btn.classList.remove('loading');
        }
    }
    
    getScanFormData() {
        const target = document.getElementById('target').value;
        const scanData = { target };
        
        switch (this.currentScanType) {
            case 'mass-assignment':
                scanData.update_endpoint = document.getElementById('update_endpoint').value;
                const authHeader = document.getElementById('auth_header').value;
                if (authHeader) {
                    scanData.auth_headers = { 'Authorization': authHeader };
                } else {
                    scanData.auth_headers = {};
                }
                const testFields = document.getElementById('test_fields').value;
                if (testFields) {
                    scanData.test_fields = testFields.split(',').map(f => f.trim());
                }
                break;
                
            case 'jwt-confusion':
                scanData.victim_token = document.getElementById('victim_token').value;
                break;
                
            case 'oauth-confusion':
                scanData.client_id = document.getElementById('client_id').value;
                scanData.legitimate_redirect = document.getElementById('legitimate_redirect').value;
                scanData.attacker_redirect = document.getElementById('attacker_redirect').value;
                break;
                
            case 'graphql-authz':
                const userAuth = document.getElementById('user_auth').value;
                if (userAuth) {
                    scanData.user_headers = { 'Authorization': userAuth };
                } else {
                    scanData.user_headers = {};
                }
                const queries = document.getElementById('test_queries').value;
                if (queries) {
                    scanData.test_queries = queries.split('\n').filter(q => q.trim());
                } else {
                    scanData.test_queries = [];
                }
                break;
                
            case 'ssrf-oob':
                scanData.oob_callback = document.getElementById('oob_callback').value;
                break;
        }
        
        return scanData;
    }
    
    getScanEndpoint(scanType) {
        const endpoints = {
            'mass-assignment': '/api/v1/security/mass-assignment',
            'jwt-confusion': '/api/v1/security/jwt-confusion',
            'oauth-confusion': '/api/v1/security/oauth-confusion',
            'graphql-authz': '/api/v1/security/graphql-authz',
            'ssrf-oob': '/api/v1/security/ssrf-oob'
        };
        return endpoints[scanType];
    }
    
    async viewScanDetails(scanId) {
        try {
            const response = await fetch(`${this.apiBase}/api/v1/scans/${scanId}`, {
                headers: {
                    'Authorization': `Bearer ${this.accessToken}`
                }
            });
            
            if (response.ok) {
                const scanData = await response.json();
                this.showScanDetailsModal(scanData);
            } else {
                this.showAlert('warning', '無法載入掃描詳情');
            }
            
        } catch (error) {
            this.showAlert('danger', '載入掃描詳情失敗: ' + error.message);
        }
    }
    
    showScanDetailsModal(scanData) {
        // 創建詳情模態框（簡化版本）
        const modalHTML = `
            <div class="modal fade" id="scanDetailsModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">掃描詳情</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <h6>基本信息</h6>
                            <ul class="list-unstyled">
                                <li><strong>掃描 ID:</strong> ${scanData.scan_id}</li>
                                <li><strong>類型:</strong> ${this.getScanTypeName(scanData.type)}</li>
                                <li><strong>狀態:</strong> ${scanData.status}</li>
                                <li><strong>開始時間:</strong> ${new Date(scanData.start_time).toLocaleString()}</li>
                                ${scanData.end_time ? `<li><strong>結束時間:</strong> ${new Date(scanData.end_time).toLocaleString()}</li>` : ''}
                            </ul>
                            
                            ${scanData.result ? `
                                <h6>掃描結果</h6>
                                <pre class="bg-light p-3 rounded">${JSON.stringify(scanData.result, null, 2)}</pre>
                            ` : ''}
                            
                            ${scanData.error ? `
                                <h6>錯誤信息</h6>
                                <div class="alert alert-danger">${scanData.error}</div>
                            ` : ''}
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">關閉</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // 移除舊的模態框
        const oldModal = document.getElementById('scanDetailsModal');
        if (oldModal) {
            oldModal.remove();
        }
        
        // 添加新的模態框
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // 顯示模態框
        const modal = new bootstrap.Modal(document.getElementById('scanDetailsModal'));
        modal.show();
    }
    
    showAlert(type, message) {
        const alertHTML = `
            <div class="alert alert-${type} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;
        
        // 在導航欄下方插入警告
        const navbar = document.querySelector('.navbar');
        navbar.insertAdjacentHTML('afterend', alertHTML);
        
        // 5秒後自動移除
        setTimeout(() => {
            const alert = document.querySelector('.alert');
            if (alert) {
                alert.remove();
            }
        }, 5000);
    }
    
    startAutoRefresh() {
        // 每30秒刷新一次數據
        this.refreshInterval = setInterval(() => {
            this.loadDashboardData();
        }, 30000);
    }
}

// 全域實例
const dashboard = new AIVADashboard();

// 全域函數（供 HTML 調用）
function login() {
    dashboard.login();
}

function logout() {
    dashboard.logout();
}

function startScan(scanType) {
    dashboard.startScan(scanType);
}

function submitScan() {
    dashboard.submitScan();
}