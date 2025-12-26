#!/usr/bin/env python3
"""深度分析 typescript_engine 所有依賴的實際功用和優化空間"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_dependencies():
    """分析依賴的功用和必要性"""
    
    package_json = Path('services/scan/engines/typescript_engine/package.json')
    with open(package_json, 'r', encoding='utf-8') as f:
        pkg = json.load(f)
    
    deps = pkg.get('dependencies', {})
    dev_deps = pkg.get('devDependencies', {})
    
    print("=" * 80)
    print("TypeScript Engine 依賴功用深度分析")
    print("=" * 80)
    print()
    
    # 1. 運行時依賴分析
    print("=" * 80)
    print("1️⃣  運行時依賴 (dependencies) - 4 個")
    print("=" * 80)
    print()
    
    runtime_deps = {
        'playwright': {
            'version': deps.get('playwright'),
            'purpose': '🌐 瀏覽器自動化引擎',
            'function': [
                '啟動 Chromium/Firefox/WebKit 瀏覽器',
                '執行 JavaScript 並渲染動態內容',
                '模擬使用者互動（點擊、輸入、滾動）',
                '攔截網路請求（Network Interception）',
                '截圖和錄影功能',
                '處理 SPA (Single Page Application)'
            ],
            'size': '~12 MB (含瀏覽器驅動)',
            'can_replace': '❌ 核心功能，無法替代',
            'alternatives': [
                'Puppeteer (僅 Chrome，功能類似)',
                'Selenium (較舊，速度慢)',
                'Cypress (僅測試用途)'
            ],
            'why_playwright': [
                '支援多瀏覽器（Chromium/Firefox/WebKit）',
                '速度快，API 設計好',
                '自動等待機制',
                '網路攔截功能強大'
            ],
            'necessary': '✅ 絕對必要',
            'optimization': '無法減少（已是最小安裝）'
        },
        'amqplib': {
            'version': deps.get('amqplib'),
            'purpose': '📨 RabbitMQ 訊息佇列客戶端',
            'function': [
                '連接到 RabbitMQ 伺服器',
                '接收掃描任務（從 task.scan.dynamic 隊列）',
                '發送掃描結果（到 findings.new 隊列）',
                '實現微服務間的異步通訊'
            ],
            'size': '~200 KB',
            'can_replace': '⚠️ 可替代，但不建議',
            'alternatives': [
                'HTTP API (同步，無法處理大量任務)',
                'Redis Pub/Sub (功能較弱)',
                'Kafka (太重量級)',
                'Bull/BullMQ (基於 Redis 的任務佇列)'
            ],
            'why_amqplib': [
                'RabbitMQ 是訊息佇列標準',
                '可靠性高（訊息持久化）',
                '支援任務重試和死信隊列',
                '與 Python 服務整合良好'
            ],
            'necessary': '✅ 架構必需（微服務通訊）',
            'optimization': '無法減少（已是官方客戶端）'
        },
        'pino': {
            'version': deps.get('pino'),
            'purpose': '📝 高性能日誌記錄器',
            'function': [
                '結構化日誌輸出（JSON 格式）',
                '效能優化（異步寫入）',
                '支援日誌級別（debug/info/warn/error）',
                '可擴展（支援多種輸出）'
            ],
            'size': '~100 KB',
            'can_replace': '✅ 可替代',
            'alternatives': [
                'console.log (最簡單，但功能弱)',
                'winston (功能強，但較慢)',
                'bunyan (類似 pino)',
                'log4js (類似 Java Log4j)'
            ],
            'why_pino': [
                '性能最佳（比 winston 快 5-10x）',
                'Node.js 社群首選',
                'JSON 格式易於解析'
            ],
            'necessary': '⚠️ 建議保留（但可用 console.log）',
            'optimization': '可移除（改用 console.log 節省 ~100 KB）'
        },
        'pino-pretty': {
            'version': deps.get('pino-pretty'),
            'purpose': '🎨 Pino 日誌美化器',
            'function': [
                '將 JSON 日誌轉換為人類可讀格式',
                '彩色輸出（方便開發）',
                '格式化時間戳'
            ],
            'size': '~50 KB',
            'can_replace': '✅ 可完全移除',
            'alternatives': [
                '不使用（直接看 JSON）',
                '使用外部工具（如 pino-colada）'
            ],
            'why_pino_pretty': [
                '開發時方便閱讀日誌'
            ],
            'necessary': '❌ 非必需（僅開發便利）',
            'optimization': '⚠️ 應移到 devDependencies'
        }
    }
    
    for name, info in runtime_deps.items():
        print(f"📦 {name} {info['version']}")
        print(f"   {info['purpose']}")
        print()
        print("   🎯 實際功用:")
        for func in info['function']:
            print(f"      • {func}")
        print()
        print(f"   💾 大小: {info['size']}")
        print(f"   🔄 可替代性: {info['can_replace']}")
        print(f"   ✅ 必要性: {info['necessary']}")
        
        if info['alternatives']:
            print(f"   🔀 替代方案:")
            for alt in info['alternatives']:
                print(f"      • {alt}")
        
        if info.get('why_playwright') or info.get('why_amqplib') or info.get('why_pino') or info.get('why_pino_pretty'):
            print(f"   💡 為什麼選這個:")
            for reason in (info.get('why_playwright') or info.get('why_amqplib') or info.get('why_pino') or info.get('why_pino_pretty') or []):
                print(f"      • {reason}")
        
        print(f"   ⚡ 優化空間: {info['optimization']}")
        print()
        print("-" * 80)
        print()
    
    # 2. 開發依賴分析
    print("=" * 80)
    print("2️⃣  開發依賴 (devDependencies) - 9 個")
    print("=" * 80)
    print()
    
    dev_deps_info = {
        'typescript': {
            'version': dev_deps.get('typescript'),
            'purpose': '🔷 TypeScript 編譯器',
            'function': [
                '將 .ts 檔案編譯為 .js',
                '型別檢查',
                '生成型別定義檔 (.d.ts)',
                '提供 IDE 支援'
            ],
            'size': '~23 MB',
            'necessary': '✅ 絕對必要',
            'can_remove': '❌ 不能移除',
            'optimization': '無（核心工具）',
            'note': 'TypeScript 專案的基礎'
        },
        '@types/node': {
            'version': dev_deps.get('@types/node'),
            'purpose': '📘 Node.js API 型別定義',
            'function': [
                '提供 Node.js 內建 API 的 TypeScript 型別',
                '讓 IDE 能自動完成和檢查'
            ],
            'size': '~2 MB',
            'necessary': '✅ 強烈建議',
            'can_remove': '⚠️ 可移除但會失去型別檢查',
            'optimization': '無',
            'note': '沒有它會失去 TypeScript 的優勢'
        },
        '@types/amqplib': {
            'version': dev_deps.get('@types/amqplib'),
            'purpose': '📘 amqplib 型別定義',
            'function': [
                '提供 amqplib 的 TypeScript 型別'
            ],
            'size': '~50 KB',
            'necessary': '✅ 建議保留',
            'can_remove': '⚠️ 可移除',
            'optimization': '無',
            'note': '提升開發體驗'
        },
        '@typescript-eslint/eslint-plugin': {
            'version': dev_deps.get('@typescript-eslint/eslint-plugin'),
            'purpose': '🔍 TypeScript ESLint 規則',
            'function': [
                '提供 TypeScript 專用的 lint 規則',
                '檢查程式碼品質',
                '統一程式碼風格'
            ],
            'size': '~2.5 MB',
            'necessary': '⚠️ 開發時建議',
            'can_remove': '✅ 可移除（不影響運行）',
            'optimization': '如果不需要 lint 可移除',
            'note': '146 個 MD 文件就是這個套件的規則文檔'
        },
        '@typescript-eslint/parser': {
            'version': dev_deps.get('@typescript-eslint/parser'),
            'purpose': '🔍 TypeScript ESLint 解析器',
            'function': [
                '讓 ESLint 能解析 TypeScript 語法'
            ],
            'size': '~1 MB',
            'necessary': '⚠️ 如果用 ESLint 則必需',
            'can_remove': '✅ 可移除（與 eslint-plugin 一起）',
            'optimization': '同上',
            'note': '與 eslint-plugin 配套'
        },
        'eslint': {
            'version': dev_deps.get('eslint'),
            'purpose': '🔍 JavaScript/TypeScript Linter',
            'function': [
                '檢查程式碼品質',
                '發現潛在 bug',
                '強制程式碼風格'
            ],
            'size': '~3 MB',
            'necessary': '⚠️ 開發時建議',
            'can_remove': '✅ 可移除（不影響運行）',
            'optimization': '如果有嚴格開發流程可保留',
            'note': '幫助維護程式碼品質'
        },
        'prettier': {
            'version': dev_deps.get('prettier'),
            'purpose': '🎨 程式碼格式化工具',
            'function': [
                '自動格式化程式碼',
                '統一縮排、空格、換行',
                '消除格式爭議'
            ],
            'size': '~8 MB',
            'necessary': '⚠️ 開發便利',
            'can_remove': '✅ 可移除',
            'optimization': '團隊開發建議保留',
            'note': '自動化格式，節省時間'
        },
        'tsx': {
            'version': dev_deps.get('tsx'),
            'purpose': '⚡ TypeScript 執行器',
            'function': [
                '直接執行 .ts 檔案（無需先編譯）',
                '開發模式熱重載',
                '快速測試'
            ],
            'size': '~5 MB',
            'necessary': '✅ 開發時非常有用',
            'can_remove': '⚠️ 可移除（用 tsc + node）',
            'optimization': '建議保留（開發效率）',
            'note': '讓 npm run dev 能直接執行 .ts'
        },
        'vitest': {
            'version': dev_deps.get('vitest'),
            'purpose': '🧪 測試框架',
            'function': [
                '單元測試',
                '整合測試',
                '覆蓋率報告'
            ],
            'size': '~2 MB',
            'necessary': '⚠️ 如果有測試則必需',
            'can_remove': '✅ 可移除（如果沒寫測試）',
            'optimization': '建議保留並寫測試',
            'note': '目前可能沒有測試檔案'
        }
    }
    
    for name, info in dev_deps_info.items():
        print(f"📦 {name} {info['version']}")
        print(f"   {info['purpose']}")
        print()
        print("   🎯 實際功用:")
        for func in info['function']:
            print(f"      • {func}")
        print()
        print(f"   💾 大小: {info['size']}")
        print(f"   ✅ 必要性: {info['necessary']}")
        print(f"   🗑️  可移除: {info['can_remove']}")
        print(f"   ⚡ 優化空間: {info['optimization']}")
        print(f"   📝 備註: {info['note']}")
        print()
        print("-" * 80)
        print()
    
    # 3. 傳遞依賴分析
    print("=" * 80)
    print("3️⃣  傳遞依賴 (間接依賴) - ~220 個")
    print("=" * 80)
    print()
    
    print("這些是上述 13 個套件自己依賴的其他套件，你無法直接控制。")
    print()
    
    transitive_deps = {
        'playwright 的依賴': {
            'count': '~30 個',
            'examples': [
                'playwright-core (核心引擎)',
                '@playwright/test (測試工具)',
                'ws (WebSocket 通訊)',
                'jpeg-js (圖片處理)'
            ],
            'note': '支援多瀏覽器需要的底層工具'
        },
        'TypeScript ESLint 的依賴': {
            'count': '~80 個',
            'examples': [
                '@typescript-eslint/utils',
                '@typescript-eslint/scope-manager',
                '@typescript-eslint/types',
                '@typescript-eslint/visitor-keys'
            ],
            'note': '146 個 MD 檔案主要來自這裡（規則文檔）'
        },
        'ESLint 的依賴': {
            'count': '~40 個',
            'examples': [
                'eslint-scope',
                'eslint-visitor-keys',
                'espree (JavaScript 解析器)',
                'esquery (AST 查詢)'
            ],
            'note': '程式碼分析工具鏈'
        },
        'Vitest 的依賴': {
            'count': '~30 個',
            'examples': [
                '@vitest/expect',
                '@vitest/runner',
                'vite (構建工具)',
                'chai (斷言庫)'
            ],
            'note': '測試框架的完整工具鏈'
        },
        '其他工具鏈依賴': {
            'count': '~40 個',
            'examples': [
                'typescript (自己的依賴)',
                'prettier (自己的依賴)',
                '各種 @types/* (型別定義)',
                '工具函式庫 (lodash, chalk 等)'
            ],
            'note': '開發工具的基礎設施'
        }
    }
    
    for category, info in transitive_deps.items():
        print(f"📂 {category}")
        print(f"   數量: {info['count']}")
        print(f"   範例:")
        for ex in info['examples']:
            print(f"      • {ex}")
        print(f"   📝 說明: {info['note']}")
        print()
    
    print("⚠️  重要: 傳遞依賴是自動管理的，你只能透過移除直接依賴來減少它們")
    print()
    
    # 4. 優化建議
    print("=" * 80)
    print("4️⃣  優化建議和加強功能空間")
    print("=" * 80)
    print()
    
    print("🎯 立即可優化（不影響功能）:")
    print()
    print("1. 移動 pino-pretty 到 devDependencies")
    print("   原因: 僅開發時需要，生產環境用 JSON 日誌")
    print("   效果: 生產環境減少 ~50 KB")
    print()
    print("2. 檢查是否有測試檔案")
    print("   如果沒有測試: 可移除 vitest 及其依賴")
    print("   效果: 減少 ~30 個套件，~10 MB")
    print()
    print("3. 評估 ESLint 需求")
    print("   如果是個人專案: 可移除 eslint + @typescript-eslint/*")
    print("   效果: 減少 ~120 個套件，~15 MB，且移除 146 個 MD 檔案")
    print()
    
    print("⚠️  謹慎優化（會影響開發體驗）:")
    print()
    print("4. 移除 prettier")
    print("   影響: 需手動格式化程式碼")
    print("   效果: 減少 ~8 MB")
    print()
    print("5. 移除 tsx，改用 tsc")
    print("   影響: 開發時需先編譯才能測試")
    print("   效果: 減少 ~5 MB，但開發變慢")
    print()
    
    print("❌ 不建議移除:")
    print()
    print("6. typescript")
    print("   理由: 核心工具，沒有它就不是 TypeScript 專案")
    print()
    print("7. playwright")
    print("   理由: 核心功能，移除後無法掃描動態網頁")
    print()
    print("8. amqplib")
    print("   理由: 架構依賴，移除後無法與其他服務通訊")
    print()
    
    print("=" * 80)
    print("5️⃣  加強功能的空間")
    print("=" * 80)
    print()
    
    enhancements = [
        {
            'feature': '🔐 安全掃描增強',
            'packages': [
                'helmet (HTTP 安全標頭檢查)',
                'retire.js (檢測過時的 JS 庫)',
                'snyk (依賴漏洞掃描)'
            ],
            'size': '+2-5 MB',
            'value': '高'
        },
        {
            'feature': '📊 效能監控',
            'packages': [
                'clinic (Node.js 效能分析)',
                'prom-client (Prometheus 指標)',
                'newrelic (APM 監控)'
            ],
            'size': '+1-3 MB',
            'value': '中'
        },
        {
            'feature': '🎭 進階爬蟲功能',
            'packages': [
                'cheerio (HTML 解析，輕量級)',
                'axios (HTTP 請求，補充 Playwright)',
                'user-agents (隨機 UA)'
            ],
            'size': '+1-2 MB',
            'value': '高'
        },
        {
            'feature': '💾 資料庫整合',
            'packages': [
                'mongoose (MongoDB ORM)',
                'pg (PostgreSQL 客戶端)',
                'redis (快取)'
            ],
            'size': '+2-5 MB',
            'value': '中（看需求）'
        },
        {
            'feature': '🧪 測試增強',
            'packages': [
                '@playwright/test (E2E 測試)',
                'supertest (API 測試)',
                'sinon (Mock/Stub)'
            ],
            'size': '+3-5 MB',
            'value': '高（品質保證）'
        },
        {
            'feature': '📝 文檔生成',
            'packages': [
                'typedoc (TypeScript 文檔)',
                'swagger-jsdoc (API 文檔)'
            ],
            'size': '+2-3 MB',
            'value': '中'
        }
    ]
    
    for enh in enhancements:
        print(f"💡 {enh['feature']}")
        print(f"   建議套件:")
        for pkg in enh['packages']:
            print(f"      • {pkg}")
        print(f"   增加大小: {enh['size']}")
        print(f"   價值評估: {enh['value']}")
        print()
    
    # 6. 總結建議
    print("=" * 80)
    print("6️⃣  總結建議")
    print("=" * 80)
    print()
    
    print("🎯 最小化配置（僅核心功能）:")
    print("   保留:")
    print("     ✅ playwright, amqplib, pino")
    print("     ✅ typescript, @types/node, @types/amqplib, tsx")
    print("   移除:")
    print("     ❌ pino-pretty, eslint*, prettier, vitest")
    print("   結果:")
    print("     • 從 235 個套件 → 約 60 個套件")
    print("     • 從 100 MB → 約 40 MB")
    print("     • 從 439 個 MD → 約 50 個 MD")
    print()
    
    print("⚖️  平衡配置（推薦）:")
    print("   保留:")
    print("     ✅ 所有運行時依賴")
    print("     ✅ TypeScript 工具鏈")
    print("     ✅ tsx (開發便利)")
    print("     ⚠️  移動 pino-pretty 到 devDependencies")
    print("   移除:")
    print("     ❌ eslint* (如果是個人專案)")
    print("     ❌ vitest (如果沒寫測試)")
    print("   結果:")
    print("     • 從 235 個套件 → 約 100 個套件")
    print("     • 從 100 MB → 約 60 MB")
    print("     • 保留良好開發體驗")
    print()
    
    print("🚀 完整配置（團隊開發）:")
    print("   保留:")
    print("     ✅ 所有現有依賴")
    print("   新增:")
    print("     ➕ @playwright/test (E2E 測試)")
    print("     ➕ cheerio (HTML 解析)")
    print("     ➕ typedoc (文檔生成)")
    print("   結果:")
    print("     • 從 235 個套件 → 約 250 個套件")
    print("     • 從 100 MB → 約 110 MB")
    print("     • 完整的開發/測試/文檔工具鏈")
    print()
    
    print("=" * 80)
    print("💡 最終建議")
    print("=" * 80)
    print()
    print("1. ✅ 立即刪除 node_modules/")
    print("   原因: 已在 .gitignore，不應存在")
    print()
    print("2. 📝 調整 package.json (可選):")
    print("   • 移動 pino-pretty 到 devDependencies")
    print("   • 如果沒測試，移除 vitest")
    print("   • 個人專案可考慮移除 eslint*")
    print()
    print("3. 🔄 重新安裝 (優化後):")
    print("   npm install")
    print()
    print("4. 📦 未來擴充:")
    print("   • 需要時再加入增強功能套件")
    print("   • 保持依賴最小化原則")
    print()

if __name__ == '__main__':
    analyze_dependencies()
