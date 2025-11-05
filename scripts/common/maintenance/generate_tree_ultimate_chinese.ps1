# -*- coding: utf-8 -*-
# AIVA 專案程式碼樹狀圖生成腳本（終極整合版）
# 功能：
# 1. 生成僅程式碼的樹狀架構圖
# 2. 與上一版比對，標記新增(綠色)/刪除(紅色)/不變(白色)
# 3. 統計資訊顯示前後對比
# 4. 檔案名稱後面附加中文說明
# 5. 同時輸出純文字檔案和彩色終端機顯示

param(
    [string]$ProjectRoot = 'C:\D\fold7\AIVA-git',
    [string]$OutputDir = 'C:\D\fold7\AIVA-git\_out',
    [string]$PreviousTreeFile = '',
    [switch]$ShowColorInTerminal,
    [switch]$AddChineseComments
)

# 設定輸出編碼為 UTF-8
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$OutputEncoding = [Text.Encoding]::UTF8

Write-Host "🚀 開始生成程式碼樹狀圖（終極整合版）..." -ForegroundColor Cyan

# 要排除的目錄
$excludeDirs = @(
    '.git', '__pycache__', '.mypy_cache', '.ruff_cache',
    'node_modules', '.venv', 'venv', 'env', '.env',
    '.pytest_cache', '.tox', 'dist', 'build', 'target',
    'bin', 'obj', '.egg-info', '.eggs', 'htmlcov',
    '.coverage', '.hypothesis', '.idea', '.vscode',
    'site-packages', '_backup', '_out', 'aiva_platform_integrated.egg-info'
)

# 只保留的程式碼檔案類型
$codeExtensions = @(
    '.py', '.go', '.rs', '.ts', '.js', '.jsx', '.tsx',
    '.c', '.cpp', '.h', '.hpp', '.java', '.cs',
    '.sql', '.html', '.css', '.scss', '.vue'
)

# 中文檔名說明對照表
$chineseComments = @{
    # Python 檔案 - 基礎架構
    '__init__.py' = '模組初始化'
    'models.py' = '資料模型'
    'schemas.py' = '資料結構定義'
    'config.py' = '配置管理'
    'worker.py' = '工作執行器'
    'app.py' = '應用程式入口'
    'main.py' = '主程式'
    'server.py' = '伺服器'
    'client.py' = '客戶端'
    'utils.py' = '工具函數'
    'helper.py' = '輔助函數'
    'manager.py' = '管理器'
    'handler.py' = '處理器'
    'controller.py' = '控制器'
    'service.py' = '服務層'
    'api.py' = 'API 介面'
    'test.py' = '測試程式'
    'demo.py' = '示範程式'
    'example.py' = '範例程式'
    'settings.py' = '設定檔'
    'constants.py' = '常數定義'
    'exceptions.py' = '例外處理'
    'enums.py' = '列舉定義'
    'types.py' = '型別定義'
    'base.py' = '基礎模組'
    'core.py' = '核心模組'
    'common.py' = '通用模組'
    'shared.py' = '共享模組'
    'interface.py' = '介面模組'
    'abstract.py' = '抽象類別'
    'factory.py' = '工廠模式'
    'singleton.py' = '單例模式'
    'decorator.py' = '裝飾器'
    'middleware.py' = '中介軟體'
    'plugin.py' = '插件模組'
    'extension.py' = '擴展模組'
    'adapter.py' = '適配器'
    'bridge.py' = '橋接器'
    'proxy.py' = '代理模組'
    'facade.py' = '外觀模式'
    'strategy.py' = '策略模式'
    'observer.py' = '觀察者模式'
    'command.py' = '命令模式'
    'state.py' = '狀態模式'
    'template.py' = '模板模式'
    'visitor.py' = '訪問者模式'
    'iterator.py' = '迭代器模式'
    'builder.py' = '建造者模式'
    'prototype.py' = '原型模式'
    
    # 特定檔案
    'bio_neuron_core.py' = '生物神經元核心'
    'bio_neuron_core_v2.py' = '生物神經元核心 v2'
    'bio_neuron_master.py' = '生物神經元主控'
    'ai_commander.py' = 'AI 指揮官'
    'ai_controller.py' = 'AI 控制器'
    'ai_integration_test.py' = 'AI 整合測試'
    'ai_schemas.py' = 'AI 資料結構'
    'ai_ui_schemas.py' = 'AI UI 資料結構'
    'multilang_coordinator.py' = '多語言協調器'
    'nlg_system.py' = '自然語言生成系統'
    'optimized_core.py' = '最佳化核心'
    'business_schemas.py' = '業務資料結構'
    
    # 新增更多檔案識別
    'aiva_launcher.py' = 'AIVA 啟動器'
    'comprehensive_pentest_runner.py' = '全面滲透測試執行器'
    'pentest_runner.py' = '滲透測試執行器'
    'live_pentest_runner.py' = '即時滲透測試執行器'
    'fixed_pentest_runner.py' = '修復版滲透測試執行器'
    'comprehensive_system_validation.py' = '全面系統驗證'
    'ai_autonomous_testing_loop.py' = 'AI 自主測試循環'
    'ai_component_explorer.py' = 'AI 元件探索器'
    'ai_functionality_validator.py' = 'AI 功能驗證器'
    'ai_security_test.py' = 'AI 安全測試'
    'ai_system_explorer.py' = 'AI 系統探索器'
    'health_check.py' = '健康檢查'
    'final_validation.py' = '最終驗證'
    'full_validation_test.py' = '完整驗證測試'
    'features_ai_cli.py' = '功能 AI 命令列介面'
    'core_scan_integration_cli.py' = '核心掃描整合 CLI'
    'scanner_statistics.py' = '掃描器統計'
    'launch_offline_mode.py' = '離線模式啟動器'
    'fix_environment_dependencies.py' = '環境依賴修復'
    'fix_offline_dependencies.py' = '離線依賴修復'
    'check_documentation_errors.py' = '文件錯誤檢查'
    'analyze_ai_performance.py' = 'AI 效能分析'
    'apply_performance_optimizations.py' = '效能優化應用'
    'aiva_package_validator.py' = 'AIVA 套件驗證器'
    
    # 功能模組
    'smart_detection_manager.py' = '智慧檢測管理器'
    'smart_idor_detector.py' = '智慧 IDOR 檢測器'
    'smart_ssrf_detector.py' = '智慧 SSRF 檢測器'
    'enhanced_worker.py' = '增強型工作器'
    'detection_models.py' = '檢測模型'
    'payload_generator.py' = '攻擊載荷生成器'
    'result_publisher.py' = '結果發布器'
    'task_queue.py' = '任務佇列'
    'telemetry.py' = '遙測'
    
    # 資料庫相關
    'database.py' = '資料庫操作'
    'db.py' = '資料庫連接'
    'orm.py' = '物件關聯映射'
    'migration.py' = '資料庫遷移'
    'seed.py' = '資料種子'
    'backup.py' = '備份模組'
    'restore.py' = '還原模組'
    'connection.py' = '連接管理'
    'session.py' = '會話管理'
    'transaction.py' = '交易管理'
    'query.py' = '查詢建構器'
    'repository.py' = '資料存取層'
    'dao.py' = '資料存取物件'
    
    # 網路相關
    'network.py' = '網路模組'
    'http.py' = 'HTTP 模組'
    'https.py' = 'HTTPS 模組'
    'websocket.py' = 'WebSocket 模組'
    'tcp.py' = 'TCP 模組'
    'udp.py' = 'UDP 模組'
    'socket.py' = 'Socket 模組'
    'request.py' = '請求處理'
    'response.py' = '回應處理'
    'router.py' = '路由器'
    'routing.py' = '路由管理'
    'url.py' = 'URL 處理'
    'endpoint.py' = '端點定義'
    'filter.py' = '過濾器'
    'interceptor.py' = '攔截器'
    
    # 安全相關
    'security.py' = '安全模組'
    'auth.py' = '身份驗證'
    'authentication.py' = '身份驗證'
    'authorization.py' = '授權管理'
    'permission.py' = '權限管理'
    'role.py' = '角色管理'
    'token.py' = '令牌管理'
    'jwt.py' = 'JWT 令牌'
    'oauth.py' = 'OAuth 認證'
    'crypto.py' = '加密模組'
    'encryption.py' = '加密處理'
    'decryption.py' = '解密處理'
    'hash.py' = '雜湊處理'
    'ssl.py' = 'SSL 模組'
    'tls.py' = 'TLS 模組'
    'certificate.py' = '憑證管理'
    'signature.py' = '數位簽章'
    'validation.py' = '驗證模組'
    'sanitization.py' = '資料清理'
    'firewall.py' = '防火牆'
    'antivirus.py' = '防毒模組'
    
    # 快取相關
    'cache.py' = '快取模組'
    'memory.py' = '記憶體管理'
    'redis.py' = 'Redis 快取'
    'memcached.py' = 'Memcached 快取'
    'storage.py' = '儲存模組'
    'filesystem.py' = '檔案系統'
    'file.py' = '檔案處理'
    'upload.py' = '檔案上傳'
    'download.py' = '檔案下載'
    'stream.py' = '串流處理'
    
    # 日誌相關
    'logger.py' = '日誌記錄器'
    'logging.py' = '日誌模組'
    'log.py' = '日誌處理'
    'audit.py' = '稽核日誌'
    'monitoring.py' = '監控模組'
    'metrics.py' = '指標收集'
    'analytics.py' = '分析模組'
    'statistics.py' = '統計模組'
    'report.py' = '報告生成'
    'dashboard.py' = '儀表板'
    
    # 引擎類
    'boolean_detection_engine.py' = '布林檢測引擎'
    'error_detection_engine.py' = '錯誤檢測引擎'
    'time_detection_engine.py' = '時間檢測引擎'
    'union_detection_engine.py' = '聯合檢測引擎'
    'oob_detection_engine.py' = '帶外檢測引擎'
    
    # Go 檔案
    'main.go' = '主程式'
    'config.go' = '配置管理'
    'models.go' = '資料模型'
    'schemas.go' = '資料結構'
    'client.go' = '客戶端'
    'server.go' = '伺服器'
    'worker.go' = '工作器'
    'handler.go' = '處理器'
    'service.go' = '服務'
    'logger.go' = '日誌記錄器'
    'message.go' = '訊息處理'
    
    # 特定 Go 檔案
    'sca_scanner.go' = 'SCA 掃描器'
    'cspm_scanner.go' = 'CSPM 掃描器'
    'brute_forcer.go' = '暴力破解器'
    'token_analyzer.go' = 'Token 分析器'
    'ssrf.go' = 'SSRF 檢測'
    
    # Rust 檔案
    'main.rs' = '主程式'
    'lib.rs' = '程式庫'
    'models.rs' = '資料模型'
    'config.rs' = '配置'
    'worker.rs' = '工作器'
    'analyzers.rs' = '分析器'
    'parsers.rs' = '解析器'
    'rules.rs' = '規則引擎'
    'scanner.rs' = '掃描器'
    'git_history_scanner.rs' = 'Git 歷史掃描器'
    'secret_detector.rs' = '機密檢測器'
    
    # TypeScript/JavaScript 檔案
    'index.ts' = '入口檔案'
    'index.js' = '入口檔案'
    'main.ts' = '主程式'
    'main.js' = '主程式'
    'config.ts' = '配置管理'
    'types.ts' = '型別定義'
    'interfaces.ts' = '介面定義'
    'service.ts' = '服務'
    'controller.ts' = '控制器'
    'utils.ts' = '工具函數'
    'logger.ts' = '日誌記錄器'
    
    # 特定 TS 檔案
    'dynamic-scan.interfaces.ts' = '動態掃描介面'
    'enhanced-content-extractor.service.ts' = '增強內容提取服務'
    'enhanced-dynamic-scan.service.ts' = '增強動態掃描服務'
    'interaction-simulator.service.ts' = '互動模擬服務'
    'network-interceptor.service.ts' = '網路攔截服務'
    'scan-service.ts' = '掃描服務'
    
    # SQL 檔案
    '001_schema.sql' = '資料庫結構初始化'
    '002_enhanced_schema.sql' = '增強資料庫結構'
    '001_initial_schema.py' = '初始資料庫遷移'
    
    # HTML/CSS 檔案
    'index.html' = '首頁'
    'main.css' = '主樣式表'
    'style.css' = '樣式表'
    'app.css' = '應用樣式'
    
    # 目錄中文說明
    'aiva_common' = 'AIVA 共用模組'
    'aiva_core' = 'AIVA 核心模組'
    'aiva_integration' = 'AIVA 整合模組'
    'aiva_scan' = 'AIVA 掃描模組'
    'aiva_scan_node' = 'AIVA Node.js 掃描模組'
    'aiva_func_idor' = 'IDOR 功能模組'
    'aiva_func_sqli' = 'SQL 注入功能模組'
    'aiva_func_ssrf' = 'SSRF 功能模組'
    'aiva_func_xss' = 'XSS 功能模組'
    'aiva_common_go' = 'Go 共用模組'
    
    'ai_engine' = 'AI 引擎'
    'ai_engine_backup' = 'AI 引擎備份'
    'ai_model' = 'AI 模型'
    'analysis' = '分析模組'
    'authz' = '授權模組'
    'bizlogic' = '業務邏輯'
    'execution' = '執行模組'
    'execution_tracer' = '執行追蹤器'
    'ingestion' = '資料接收'
    'learning' = '學習模組'
    'messaging' = '訊息處理'
    'output' = '輸出模組'
    'planner' = '規劃器'
    'rag' = 'RAG 檢索增強'
    'state' = '狀態管理'
    'storage' = '儲存模組'
    'training' = '訓練模組'
    'ui_panel' = 'UI 面板'
    
    'function_authn_go' = 'Go 身份驗證功能'
    'function_crypto_go' = 'Go 密碼學功能'
    'function_cspm_go' = 'Go CSPM 功能'
    'function_idor' = 'IDOR 功能'
    'function_postex' = '後滲透功能'
    'function_sca_go' = 'Go SCA 功能'
    'function_sqli' = 'SQL 注入功能'
    'function_ssrf' = 'SSRF 功能'
    'function_ssrf_go' = 'Go SSRF 功能'
    'function_xss' = 'XSS 功能'
    
    'attack_path_analyzer' = '攻擊路徑分析器'
    'config_template' = '配置範本'
    'middlewares' = '中介軟體'
    'observability' = '可觀測性'
    'perf_feedback' = '效能回饋'
    'reception' = '接收模組'
    'remediation' = '修復建議'
    'reporting' = '報告生成'
    'security' = '安全模組'
    'threat_intel' = '威脅情報'
    
    'core_crawling_engine' = '核心爬蟲引擎'
    'dynamic_engine' = '動態引擎'
    'info_gatherer' = '資訊收集器'
    'info_gatherer_rust' = 'Rust 資訊收集器'
    
    'cmd' = '命令列工具'
    'internal' = '內部模組'
    'pkg' = '套件'
    'src' = '原始碼'
    'config' = '配置'
    'logger' = '日誌'
    'mq' = '訊息佇列'
    'schemas' = '資料結構'
    'models' = '資料模型'
    'scanner' = '掃描器'
    'analyzer' = '分析器'
    'detector' = '檢測器'
    'brute_force' = '暴力破解'
    'token_test' = 'Token 測試'
    
    'engines' = '檢測引擎'
    'interfaces' = '介面定義'
    'services' = '服務模組'
    'utils' = '工具函數'
    'examples' = '範例程式'
    'versions' = '版本管理'
    'alembic' = '資料庫遷移'
    'api_gateway' = 'API 閘道'
    
    'dedup' = '去重複'
    'network' = '網路模組'
    'standards' = '標準規範'
    'types' = '型別定義'
    'tools' = '工具集'
    'docker' = 'Docker 容器'
    'initdb' = '資料庫初始化'
    'docs' = '文件'
}

function Test-ShouldIncludeFile {
    param([string]$FileName)
    
    $ext = [System.IO.Path]::GetExtension($FileName).ToLower()
    
    if ([string]::IsNullOrEmpty($ext)) {
        return $false
    }
    
    return $codeExtensions -contains $ext
}

function Get-ChineseComment {
    param([string]$FileName, [string]$IsDirectory = $false, [int]$AlignPosition = 50)
    
    if (-not $AddChineseComments) {
        return ""
    }
    
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($FileName)
    $fullName = $FileName
    $comment = ""
    
    # 完全匹配
    if ($chineseComments.ContainsKey($fullName)) {
        $comment = $chineseComments[$fullName]
    }
    # 基本檔名匹配
    elseif ($chineseComments.ContainsKey($baseName)) {
        $comment = $chineseComments[$baseName]
    }
    else {
        # 模式匹配
        foreach ($pattern in $chineseComments.Keys) {
            if ($fullName -like "*$pattern*" -or $baseName -like "*$pattern*") {
                $comment = $chineseComments[$pattern]
                break
            }
        }
        
        # 根據副檔名和檔名模式進行智慧推測
        if (-not $comment) {
            $ext = [System.IO.Path]::GetExtension($FileName).ToLower()
            switch ($ext) {
                '.py' { 
                    if ($fullName -match "(test|spec).*\.py$") { $comment = "測試程式" }
                    elseif ($fullName -match "(demo|example).*\.py$") { $comment = "示範程式" }
                    elseif ($fullName -match "(validation|validator).*\.py$") { $comment = "驗證程式" }
                    elseif ($fullName -match "(check|checker).*\.py$") { $comment = "檢查程式" }
                    elseif ($fullName -match "(fix|fixer|repair).*\.py$") { $comment = "修復程式" }
                    elseif ($fullName -match "(worker|executor).*\.py$") { $comment = "工作器" }
                    elseif ($fullName -match "(manager|coordinator).*\.py$") { $comment = "管理器" }
                    elseif ($fullName -match "(handler|processor).*\.py$") { $comment = "處理器" }
                    elseif ($fullName -match "(detector|finder).*\.py$") { $comment = "檢測器" }
                    elseif ($fullName -match "(analyzer|parser).*\.py$") { $comment = "分析器" }
                    elseif ($fullName -match "(scanner|crawler).*\.py$") { $comment = "掃描器" }
                    elseif ($fullName -match "(engine|core).*\.py$") { $comment = "引擎" }
                    elseif ($fullName -match "(generator|builder).*\.py$") { $comment = "生成器" }
                    elseif ($fullName -match "(launcher|starter).*\.py$") { $comment = "啟動器" }
                    elseif ($fullName -match "(runner|executor).*\.py$") { $comment = "執行器" }
                    elseif ($fullName -match "(monitor|watcher|observer).*\.py$") { $comment = "監控器" }
                    elseif ($fullName -match "(optimizer|enhancer).*\.py$") { $comment = "優化器" }
                    elseif ($fullName -match "(trainer|learning).*\.py$") { $comment = "訓練器" }
                    elseif ($fullName -match "(simulator|emulator).*\.py$") { $comment = "模擬器" }
                    elseif ($fullName -match "pentest.*\.py$") { $comment = "滲透測試" }
                    elseif ($fullName -match "ai.*\.py$") { $comment = "AI 模組" }
                    elseif ($fullName -match "neuron.*\.py$") { $comment = "神經網路" }
                    # 更多智慧推測
                    elseif ($fullName -match "(api|endpoint|route).*\.py$") { $comment = "API 介面" }
                    elseif ($fullName -match "(model|entity|schema).*\.py$") { $comment = "資料模型" }
                    elseif ($fullName -match "(service|business|logic).*\.py$") { $comment = "業務邏輯" }
                    elseif ($fullName -match "(util|helper|tool).*\.py$") { $comment = "工具函數" }
                    elseif ($fullName -match "(auth|security|permission).*\.py$") { $comment = "安全模組" }
                    elseif ($fullName -match "(cache|redis|memory).*\.py$") { $comment = "快取模組" }
                    elseif ($fullName -match "(db|database|sql).*\.py$") { $comment = "資料庫" }
                    elseif ($fullName -match "(log|audit|monitor).*\.py$") { $comment = "日誌監控" }
                    elseif ($fullName -match "(config|setting|env).*\.py$") { $comment = "配置管理" }
                    elseif ($fullName -match "(client|http|request).*\.py$") { $comment = "客戶端" }
                    elseif ($fullName -match "(server|app|web).*\.py$") { $comment = "伺服器" }
                    elseif ($fullName -match "(queue|task|job).*\.py$") { $comment = "任務佇列" }
                    elseif ($fullName -match "(plugin|extension|addon).*\.py$") { $comment = "插件模組" }
                    elseif ($fullName -match "(migration|upgrade|patch).*\.py$") { $comment = "遷移腳本" }
                    elseif ($fullName -match "(seed|fixture|sample).*\.py$") { $comment = "資料種子" }
                    elseif ($fullName -match "(backup|restore|archive).*\.py$") { $comment = "備份還原" }
                    elseif ($fullName -match "(deploy|install|setup).*\.py$") { $comment = "部署安裝" }
                    elseif ($fullName -match "(clean|clear|reset).*\.py$") { $comment = "清理重置" }
                    elseif ($fullName -match "(export|import|convert).*\.py$") { $comment = "資料轉換" }
                    elseif ($fullName -match "(notify|alert|message).*\.py$") { $comment = "通知系統" }
                    elseif ($fullName -match "(schedule|cron|timer).*\.py$") { $comment = "排程任務" }
                    elseif ($fullName -match "(report|summary|stats).*\.py$") { $comment = "報告統計" }
                    elseif ($fullName -match "(interface|abstract|base).*\.py$") { $comment = "介面定義" }
                    elseif ($fullName -match "(factory|builder|creator).*\.py$") { $comment = "工廠模式" }
                    elseif ($fullName -match "(singleton|global|shared).*\.py$") { $comment = "單例模式" }
                    elseif ($fullName -match "(decorator|wrapper|proxy).*\.py$") { $comment = "裝飾器" }
                    elseif ($fullName -match "(middleware|filter|guard).*\.py$") { $comment = "中介軟體" }
                    elseif ($fullName -match "(error|exception|fault).*\.py$") { $comment = "錯誤處理" }
                    elseif ($fullName -match "(constant|enum|type).*\.py$") { $comment = "常數定義" }
                    elseif ($fullName -match "(crypto|encrypt|hash).*\.py$") { $comment = "加密模組" }
                    elseif ($fullName -match "(stream|pipe|flow).*\.py$") { $comment = "串流處理" }
                    elseif ($fullName -match "(socket|tcp|udp).*\.py$") { $comment = "網路通訊" }
                    elseif ($fullName -match "(thread|async|concurrent).*\.py$") { $comment = "並發處理" }
                    elseif ($fullName -match "(serialize|json|xml).*\.py$") { $comment = "序列化" }
                    else { $comment = "Python 模組" }
                }
                '.go' { 
                    if ($fullName -match "test.*\.go$") { $comment = "測試程式" }
                    elseif ($fullName -match "main\.go$") { $comment = "主程式" }
                    elseif ($fullName -match "(server|client).*\.go$") { $comment = "服務程式" }
                    elseif ($fullName -match "(worker|handler).*\.go$") { $comment = "工作器" }
                    elseif ($fullName -match "(scanner|detector).*\.go$") { $comment = "掃描器" }
                    else { $comment = "Go 模組" }
                }
                '.rs' { 
                    if ($fullName -match "main\.rs$") { $comment = "主程式" }
                    elseif ($fullName -match "lib\.rs$") { $comment = "程式庫" }
                    elseif ($fullName -match "mod\.rs$") { $comment = "模組" }
                    elseif ($fullName -match "(scanner|detector).*\.rs$") { $comment = "掃描器" }
                    elseif ($fullName -match "(analyzer|parser).*\.rs$") { $comment = "分析器" }
                    else { $comment = "Rust 模組" }
                }
                '.ts' { 
                    if ($fullName -match "interface.*\.ts$") { $comment = "介面定義" }
                    elseif ($fullName -match "service.*\.ts$") { $comment = "服務模組" }
                    elseif ($fullName -match "index\.ts$") { $comment = "入口檔案" }
                    elseif ($fullName -match "types.*\.ts$") { $comment = "型別定義" }
                    elseif ($fullName -match "config.*\.ts$") { $comment = "配置模組" }
                    else { $comment = "TypeScript 模組" }
                }
                '.js' { 
                    if ($fullName -match "index\.js$") { $comment = "入口檔案" }
                    else { $comment = "JavaScript 模組" }
                }
                '.sql' { 
                    if ($fullName -match "schema.*\.sql$") { $comment = "資料庫結構" }
                    elseif ($fullName -match "init.*\.sql$") { $comment = "初始化腳本" }
                    else { $comment = "資料庫腳本" }
                }
                '.html' { 
                    if ($fullName -match "index\.html$") { $comment = "首頁" }
                    else { $comment = "網頁檔案" }
                }
                '.css' { 
                    if ($fullName -match "main.*\.css$") { $comment = "主樣式表" }
                    else { $comment = "樣式表" }
                }
                '.vue' { $comment = "Vue 元件" }
                '.jsx' { $comment = "React 元件" }
                '.tsx' { $comment = "React TypeScript 元件" }
                '.c' { $comment = "C 程式" }
                '.cpp' { $comment = "C++ 程式" }
                '.h' { $comment = "標頭檔案" }
                '.hpp' { $comment = "C++ 標頭檔案" }
                '.java' { $comment = "Java 程式" }
                '.cs' { $comment = "C# 程式" }
                '.scss' { $comment = "Sass 樣式表" }
                default { return "" }
            }
        }
    }
    
    if ($comment) {
        # 使用傳入的空格數（已在調用處計算好對齊位置）
        $spaces = " " * $AlignPosition
        return "$spaces# $comment"
    }

    return ""
}

# 全域變數：儲存當前檔案樹結構
$script:currentTree = @{}

function Get-CodeTree {
    param(
        [string]$Path,
        [string]$Prefix = "",
        [string]$RelativePath = "",
        [int]$Level = 0,
        [int]$MaxLevel = 10,
        [ref]$FileCount,
        [ref]$DirCount,
        [hashtable]$PreviousTree = @{}
    )

    if ($Level -ge $MaxLevel) { return @() }

    $results = @()
    
    try {
        $items = Get-ChildItem -Path $Path -Force -ErrorAction Stop |
            Where-Object {
                $name = $_.Name
                if ($_.PSIsContainer) {
                    if ($excludeDirs -contains $name) {
                        return $false
                    }
                    $DirCount.Value++
                    return $true
                } else {
                    if (Test-ShouldIncludeFile -FileName $name) {
                        $FileCount.Value++
                        return $true
                    }
                    return $false
                }
            } |
            Sort-Object @{Expression={$_.PSIsContainer}; Descending=$true}, Name

        $itemCount = $items.Count
        for ($i = 0; $i -lt $itemCount; $i++) {
            $item = $items[$i]
            $isLast = ($i -eq $itemCount - 1)

            $connector = if ($isLast) { "└─" } else { "├─" }
            $extension = if ($isLast) { "    " } else { "│   " }

            $itemRelPath = if ($RelativePath) { "$RelativePath/$($item.Name)" } else { $item.Name }
            
            # 記錄到當前樹結構
            $script:currentTree[$itemRelPath] = $true
            
            # 判斷是新增、刪除還是不變
            $status = "unchanged"  # unchanged, added
            if ($PreviousTree.Count -gt 0 -and -not $PreviousTree.ContainsKey($itemRelPath)) {
                $status = "added"
            }
            
            # 添加中文註解 - 動態對齊計算
            $linePrefix = "$Prefix$connector"
            $alignPosition = 65  # 基準對齊位置
            $currentLength = $linePrefix.Length + $item.Name.Length
            $spacesNeeded = [Math]::Max(1, $alignPosition - $currentLength)
            $chineseComment = Get-ChineseComment -FileName $item.Name -IsDirectory $item.PSIsContainer -AlignPosition $spacesNeeded
            $itemNameWithComment = "$($item.Name)$chineseComment"
            
            $outputLine = "$linePrefix$itemNameWithComment"
            
            # 根據狀態添加標記
            $markedLine = switch ($status) {
                "added" { "[+] $outputLine" }  # 新增
                default { "    $outputLine" }  # 不變
            }
            
            # 輸出（根據狀態決定顏色）
            if ($ShowColorInTerminal) {
                switch ($status) {
                    "added" { Write-Host $outputLine -ForegroundColor Green }
                    default { Write-Host $outputLine -ForegroundColor White }
                }
            }
            
            # 添加到結果集
            $results += $markedLine

            if ($item.PSIsContainer) {
                $subTreeResults = Get-CodeTree -Path $item.FullName -Prefix "$Prefix$extension" -RelativePath $itemRelPath -Level ($Level + 1) -MaxLevel $MaxLevel -FileCount $FileCount -DirCount $DirCount -PreviousTree $PreviousTree
                $results += $subTreeResults
            }
        }
    } catch {
        # 忽略無法存取的目錄
        Write-Verbose "無法存取目錄: $Path - $_"
    }
    
    return $results
}

# 驗證專案根目錄
if (-not (Test-Path $ProjectRoot)) {
    Write-Error "專案根目錄不存在: $ProjectRoot"
    return
}

Write-Host "📂 專案路徑: $ProjectRoot" -ForegroundColor Cyan
Write-Host "📁 輸出目錄: $OutputDir" -ForegroundColor Cyan

# 收集統計資料
Write-Host "📊 收集統計資料..." -ForegroundColor Yellow

# 統計各語言檔案數和行數
$allCodeFiles = @()
try {
    $allCodeFiles = Get-ChildItem -Path $ProjectRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $path = $_.FullName
            $shouldExclude = $false
            foreach ($dir in $excludeDirs) {
                if ($path -like "*\$dir\*") {
                    $shouldExclude = $true
                    break
                }
            }
            if ($shouldExclude) { return $false }
            Test-ShouldIncludeFile -FileName $_.Name
        }
    
    Write-Host "✅ 找到 $($allCodeFiles.Count) 個程式碼檔案" -ForegroundColor Green
} catch {
    Write-Error "掃描檔案時發生錯誤: $_"
    return
}

$langStats = @()
$totalFiles = 0
$totalLines = 0

if ($allCodeFiles.Count -gt 0) {
    $langStats = $allCodeFiles | 
        Group-Object Extension |
        ForEach-Object {
            $ext = $_.Name
            $files = $_.Group
            $totalLinesForExt = 0
            foreach ($file in $files) {
                try {
                    $lines = (Get-Content $file.FullName -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
                    $totalLinesForExt += $lines
                } catch {
                    Write-Verbose "無法讀取檔案 $($file.FullName): $_"
                }
            }
            [PSCustomObject]@{
                Extension = $ext
                FileCount = $files.Count
                TotalLines = $totalLinesForExt
                AvgLines = if ($files.Count -gt 0) { [math]::Round($totalLinesForExt / $files.Count, 1) } else { 0 }
            }
        } |
        Sort-Object TotalLines -Descending

    # 計算總計
    $totalFiles = ($langStats | Measure-Object -Property FileCount -Sum).Sum
    $totalLines = ($langStats | Measure-Object -Property TotalLines -Sum).Sum
}

Write-Host "📈 統計完成: $totalFiles 個檔案，共 $totalLines 行程式碼" -ForegroundColor Green

# 讀取上一版統計和樹狀結構
$previousStats = $null
$previousTree = @{}
if ($PreviousTreeFile -and (Test-Path $PreviousTreeFile)) {
    Write-Host "📖 讀取上一版數據..." -ForegroundColor Yellow
    try {
        $previousContent = Get-Content $PreviousTreeFile -Encoding utf8
        
        # 解析上一版的統計資料
        $prevTotalFiles = 0
        $prevTotalLines = 0
        
        foreach ($line in $previousContent) {
            # 格式: "總檔案數: 456 → 320" 或 "總檔案數: 456"
            if ($line -match "總檔案數[：:]\s*(\d+)") {
                $prevTotalFiles = [int]$matches[1]
            }
            # 格式: "專案檔案數: 456 個"
            elseif ($line -match "專案檔案數[：:]\s*(\d+)") {
                $prevTotalFiles = [int]$matches[1]
            }
            
            # 總程式碼行數
            if ($line -match "總程式碼行數[：:]\s*(\d+)") {
                $prevTotalLines = [int]$matches[1]
            }
            elseif ($line -match "~(\d+)K\+?\s*行") {
                $prevTotalLines = [int]$matches[1] * 1000
            }
        }
        
        # 讀取上一版的樹狀結構（用於差異對比）
        $inTreeSection = $false
        
        foreach ($line in $previousContent) {
            if ($line -match "^(程式碼結構樹狀圖|專案結構樹狀圖)") {
                $inTreeSection = $true
                continue
            }
            
            if ($inTreeSection) {
                # 解析樹狀結構行
                # 格式: "    ├─檔案名 # 中文說明" 或 "[+] ├─檔案名 # 中文說明"
                if ($line -match "[\[+ \-\]]*\s*[├└│─\s]*([^#]+)") {
                    $itemName = $matches[1].Trim()
                    if ($itemName -and $itemName -ne "AIVA" -and -not ($itemName -match "^=+$")) {
                        $previousTree[$itemName] = $true
                    }
                }
            }
        }
        
        if ($prevTotalFiles -gt 0 -or $prevTotalLines -gt 0) {
            $previousStats = @{
                TotalFiles = $prevTotalFiles
                TotalLines = $prevTotalLines
            }
            Write-Host "✅ 已載入上一版數據 (檔案: $prevTotalFiles, 行數: $prevTotalLines, 樹節點: $($previousTree.Count))" -ForegroundColor Green
        } else {
            Write-Host "⚠️ 無法解析上一版統計數據" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️ 讀取上一版數據時發生錯誤: $_" -ForegroundColor Yellow
    }
}

# 檢查已刪除的項目
$deletedItems = @()
if ($previousTree.Count -gt 0) {
    foreach ($item in $previousTree.Keys) {
        if (-not $script:currentTree.ContainsKey($item)) {
            $deletedItems += $item
        }
    }
}

# 生成樹狀結構
Write-Host "🌳 生成樹狀結構..." -ForegroundColor Yellow
if ($ShowColorInTerminal) {
    Write-Host "   (終端機將顯示彩色輸出，檔名含中文說明)" -ForegroundColor Gray
}

$fileCountRef = [ref]0
$dirCountRef = [ref]0

$rootName = Split-Path $ProjectRoot -Leaf
$output = @()

# 添加標題和統計
$output += "================================================================================"
$output += "AIVA 專案程式碼樹狀架構圖（終極整合版 - 含中文檔名說明）"
$output += "================================================================================"
$output += "生成日期: $(Get-Date -Format 'yyyy年MM月dd日 HH:mm:ss')"
$output += "專案路徑: $ProjectRoot"
$output += ""
$output += "📊 程式碼統計"
$output += "────────────────────────────────────────────────────────────────────────────────"

# 顯示新舊對比
if ($previousStats) {
    $fileDiff = $totalFiles - $previousStats.TotalFiles
    $lineDiff = $totalLines - $previousStats.TotalLines
    $fileSymbol = if ($fileDiff -gt 0) { "📈" } elseif ($fileDiff -lt 0) { "📉" } else { "➡️" }
    $lineSymbol = if ($lineDiff -gt 0) { "📈" } elseif ($lineDiff -lt 0) { "📉" } else { "➡️" }
    
    # 格式化差異值（正數加+，負數已經有-）
    $fileDiffStr = if ($fileDiff -gt 0) { "+$fileDiff" } else { "$fileDiff" }
    $lineDiffStr = if ($lineDiff -gt 0) { "+$lineDiff" } else { "$lineDiff" }
    
    $output += "總檔案數: $($previousStats.TotalFiles) → $totalFiles $fileSymbol ($fileDiffStr)"
    $output += "總程式碼行數: $($previousStats.TotalLines) → $totalLines $lineSymbol ($lineDiffStr)"
} else {
    $output += "總檔案數: $totalFiles"
    $output += "總程式碼行數: $totalLines"
}

$output += ""
$output += "💻 語言分布:"

foreach ($stat in $langStats) {
    $pct = [math]::Round(($stat.TotalLines / $totalLines) * 100, 1)
    $output += "   • $($stat.Extension): $($stat.FileCount) 檔案, $($stat.TotalLines) 行 ($pct%)"
}

$output += ""
$output += "🔧 排除項目"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "已排除："
$output += "• 虛擬環境: .venv, venv, env"
$output += "• 快取: __pycache__, .mypy_cache, .ruff_cache"
$output += "• 建置產物: dist, build, target, bin, obj"
$output += "• 文件: .md, .txt"
$output += "• 配置檔: .json, .yaml, .toml, .ini"
$output += "• 腳本: .ps1, .sh, .bat"
$output += ""
$output += "💡 說明"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "「行」(Line) = 程式碼的一行，以換行符號 (\n) 結束"
$output += "「字」(Character) = 單一字元（含中文、英文、符號）"
$output += "「檔案數」= 符合條件的程式碼檔案總數"
$output += "「程式碼行數」= 所有程式碼檔案的總行數（包含空行和註解）"
$output += ""
$output += "🎨 差異標記說明"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "[+] = 🟢 新增的檔案或目錄（綠色顯示於終端機）"
$output += "[-] = 🔴 已刪除的檔案或目錄（紅色顯示於終端機）"
$output += "    = ⚪ 保持不變（白色顯示於終端機）"
$output += ""
$output += "🌏 中文檔名說明"
$output += "────────────────────────────────────────────────────────────────────────────────"
$output += "每個檔案名稱後面會自動添加中文說明，格式：檔案名 # 中文說明"
$output += "• 根據檔案名稱和目錄結構智慧推測功能"
$output += "• 涵蓋 Python、Go、Rust、TypeScript 等多語言"
$output += "• 包含 AIVA 專案特定的模組和功能說明"
$output += ""
$output += "注意：文字檔案輸出含 [+]/[-] 標記和中文說明"
$output += "終端機執行時會顯示對應顏色但不含 [+]/[-] 標記"
$output += "下一版本更新時，[-] 項目將被移除，[+] 項目將變為不變（空格）"
$output += ""
$output += "================================================================================"
$output += "程式碼結構樹狀圖（含中文檔名說明）"
$output += "================================================================================"
$output += ""

# 顯示標題（終端機）
if ($ShowColorInTerminal) {
    Write-Host ""
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host "程式碼結構樹狀圖（彩色輸出 + 中文檔名說明）" -ForegroundColor Cyan
    Write-Host "================================================================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "$rootName # AIVA 安全檢測平台" -ForegroundColor White
}

$rootComment = Get-ChineseComment -FileName $rootName -AlignPosition 60
$output += "$rootName$rootComment"

# 生成樹狀結構
$treeOutput = Get-CodeTree -Path $ProjectRoot -FileCount $fileCountRef -DirCount $dirCountRef -PreviousTree $previousTree
if ($treeOutput -and $treeOutput.Count -gt 0) {
    $output += $treeOutput
} else {
    $output += "    (空目錄或無符合的程式碼檔案)"
}

# 如果有刪除的項目，在最後列出
if ($deletedItems.Count -gt 0) {
    $output += ""
    $output += "────────────────────────────────────────────────────────────────────────────────"
    $output += "🔴 已刪除的項目 (共 $($deletedItems.Count) 個):"
    $output += "────────────────────────────────────────────────────────────────────────────────"
    
    if ($ShowColorInTerminal) {
        Write-Host ""
        Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Yellow
        Write-Host "🔴 已刪除的項目 (共 $($deletedItems.Count) 個):" -ForegroundColor Yellow
        Write-Host "────────────────────────────────────────────────────────────────────────────────" -ForegroundColor Yellow
    }
    
    foreach ($item in ($deletedItems | Sort-Object)) {
        $deletedComment = Get-ChineseComment -FileName $item -AlignPosition 60
        $deletedLine = "[-] $item$deletedComment"
        $output += $deletedLine
        if ($ShowColorInTerminal) {
            Write-Host $deletedLine -ForegroundColor Red
        }
    }
}

# 確保輸出目錄存在
if (-not (Test-Path $OutputDir)) {
    New-Item -Path $OutputDir -ItemType Directory -Force | Out-Null
}

# 儲存到檔案
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outputFile = Join-Path $OutputDir "tree_ultimate_chinese_$timestamp.txt"
try {
    $output | Out-File $outputFile -Encoding utf8 -ErrorAction Stop
} catch {
    Write-Error "無法寫入檔案 $outputFile : $_"
    return
}

# 統計
$lineCount = $output.Count

Write-Host ""
Write-Host "✅ 程式碼樹狀圖已生成（終極整合版 + 中文檔名）！" -ForegroundColor Green
Write-Host "   檔案位置: $outputFile" -ForegroundColor White
Write-Host "   樹狀圖行數: $lineCount 行" -ForegroundColor White
if ($previousStats) {
    $fileDiff = $totalFiles - $previousStats.TotalFiles
    $lineDiff = $totalLines - $previousStats.TotalLines
    
    # 格式化差異值（正數加+，負數已經有-）
    $fileDiffStr = if ($fileDiff -gt 0) { "+$fileDiff" } else { "$fileDiff" }
    $lineDiffStr = if ($lineDiff -gt 0) { "+$lineDiff" } else { "$lineDiff" }
    
    Write-Host "   程式碼檔案數: $($previousStats.TotalFiles) → $totalFiles ($fileDiffStr)" -ForegroundColor $(if($fileDiff -gt 0){"Green"}elseif($fileDiff -lt 0){"Red"}else{"White"})
    Write-Host "   總程式碼行數: $($previousStats.TotalLines) → $totalLines ($lineDiffStr)" -ForegroundColor $(if($lineDiff -gt 0){"Green"}elseif($lineDiff -lt 0){"Red"}else{"White"})
    if ($deletedItems.Count -gt 0) {
        Write-Host "   已刪除項目: $($deletedItems.Count) 個" -ForegroundColor Red
    }
} else {
    Write-Host "   程式碼檔案數: $totalFiles" -ForegroundColor White
    Write-Host "   總程式碼行數: $totalLines" -ForegroundColor White
}
Write-Host ""
Write-Host "📋 語言分布:" -ForegroundColor Cyan
foreach ($stat in $langStats | Select-Object -First 5) {
    $pct = [math]::Round(($stat.TotalLines / $totalLines) * 100, 1)
    Write-Host "   $($stat.Extension): $($stat.FileCount) 檔案, $($stat.TotalLines) 行 ($pct%)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🌏 中文檔名說明功能:" -ForegroundColor Cyan
Write-Host "   • 自動為 $(($chineseComments.Keys | Measure-Object).Count) 種檔案/目錄添加中文說明" -ForegroundColor Gray
Write-Host "   • 支援智慧模式匹配和副檔名推測" -ForegroundColor Gray
Write-Host "   • 涵蓋 AIVA 專案特有的模組和功能" -ForegroundColor Gray

if (-not $PreviousTreeFile) {
    Write-Host ""
    Write-Host "💡 提示：下次執行時可以指定上一版檔案進行比對：" -ForegroundColor Yellow
    Write-Host "   .\generate_tree_ultimate_chinese.ps1 -PreviousTreeFile `"$outputFile`"" -ForegroundColor Gray
}

Write-Host ""
Write-Host "📌 本次輸出檔案: $outputFile" -ForegroundColor Cyan
Write-Host "🎉 終極整合版完成！包含所有功能：差異對比 + 彩色顯示 + 中文檔名說明" -ForegroundColor Green