# AIVA Security Platform - Web 管理界面

現代化的 Web 管理控制台，提供直觀的用戶界面來管理 AIVA 安全掃描功能。

## 🔧 修復原則

**保留未使用函數原則**: 在程式碼修復過程中，若發現有定義但尚未使用的函數或方法，只要不影響程式正常運作，建議予以保留。這些函數可能是：
- 預留的 API 端點或介面
- 未來功能的基礎架構
- 測試或除錯用途的輔助函數
- 向下相容性考量的舊版介面

說不定未來會用到，保持程式碼的擴展性和靈活性。

## 🎯 功能特點

### ✅ 已實現功能
- **用戶認證**: JWT 令牌登入系統
- **儀表板**: 系統狀態和統計數據展示
- **高價值掃描**: 5個高價值功能模組的 Web 界面
- **掃描管理**: 查看掃描歷史和詳細結果
- **實時更新**: 自動刷新掃描狀態
- **響應式設計**: 支援桌面和移動設備

### 🚀 高價值功能模組界面
1. **Mass Assignment** - 權限提升檢測 ($2.1K-$8.2K)
2. **JWT Confusion** - JWT 混淆攻擊 ($1.8K-$7.5K)
3. **OAuth Confusion** - OAuth 配置錯誤 ($2.5K-$10.2K)
4. **GraphQL AuthZ** - GraphQL 權限檢測 ($1.9K-$7.8K)
5. **SSRF OOB** - SSRF Out-of-Band 檢測 ($2.2K-$8.7K)

## 🔧 技術架構

- **前端框架**: 純 JavaScript + Bootstrap 5
- **UI 組件**: Bootstrap Icons
- **認證**: JWT 令牌
- **通訊**: REST API (XMLHttpRequest/Fetch)
- **設計**: 響應式設計，深色/淺色主題

## 📁 文件結構

```
web/
├── index.html                 # 主頁面
├── js/
│   └── aiva-dashboard.js     # 主要 JavaScript 邏輯
└── README.md                 # 本文檔
```

## 🚀 快速開始

### 1. 啟動 API 服務
```bash
cd ../api
python start_api.py
```

### 2. 開啟 Web 界面
```bash
# 使用 Python 內建服務器
cd web
python -m http.server 8080

# 或使用 Node.js 服務器
npx http-server -p 8080

# 或直接在瀏覽器中打開
# file:///path/to/AIVA-git/web/index.html
```

### 3. 訪問管理界面
- URL: http://localhost:8080
- 預設帳戶:
  - 管理員: `admin` / `aiva-admin-2025`
  - 一般用戶: `user` / `aiva-user-2025`
  - 檢視者: `viewer` / `aiva-viewer-2025`

## 📊 界面功能

### 🏠 儀表板
- **系統狀態**: API 服務健康檢查
- **統計數據**: 總掃描次數、活躍掃描、潛在價值
- **系統監控**: CPU 使用率、記憶體狀態

### 💎 安全掃描
- **快速啟動**: 點擊按鈕開始掃描
- **參數配置**: 為每種掃描類型提供專用表單
- **實時狀態**: 掃描進度和狀態更新
- **結果展示**: 詳細的掃描結果和錯誤信息

### 📈 掃描管理
- **歷史記錄**: 查看所有掃描歷史
- **詳細結果**: 展示完整的掃描輸出
- **狀態過濾**: 依狀態篩選掃描記錄
- **導出功能**: 下載掃描報告

## 🔒 安全特性

### 認證和授權
- **JWT 令牌**: 安全的無狀態認證
- **角色權限**: 不同角色有不同的訪問權限
- **自動登出**: 令牌過期自動重定向到登入頁面
- **CSRF 保護**: 使用 Authorization 標頭防止 CSRF

### 數據保護
- **敏感數據遮罩**: API 密鑰和令牌在界面中隱藏
- **本地存儲**: 令牌安全存儲在 localStorage
- **HTTPS 就緒**: 支援 HTTPS 部署

## 🎨 用戶體驗

### 現代化設計
- **Bootstrap 5**: 現代化 UI 組件
- **響應式**: 自適應各種螢幕尺寸
- **圖標系統**: Bootstrap Icons 提供一致的視覺語言
- **載入動畫**: 操作回饋和載入狀態

### 易用性
- **直觀操作**: 簡單的點擊操作開始掃描
- **即時回饋**: 操作結果立即顯示
- **錯誤處理**: 友好的錯誤信息提示
- **幫助信息**: 內建的使用說明

## 📱 部署選項

### 開發環境
```bash
# 本地開發
python -m http.server 8080
```

### 生產環境
```bash
# 使用 Nginx
server {
    listen 80;
    server_name your-domain.com;
    root /path/to/AIVA-git/web;
    index index.html;
    
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Docker 部署
```dockerfile
FROM nginx:alpine
COPY web/ /usr/share/nginx/html/
COPY nginx.conf /etc/nginx/nginx.conf
EXPOSE 80
```

## 🔧 自訂配置

### API 端點配置
在 `js/aiva-dashboard.js` 中修改:
```javascript
this.apiBase = 'https://your-api-server.com';
```

### 主題自訂
在 `index.html` 的 CSS 變數中修改:
```css
:root {
    --aiva-primary: #2563eb;    /* 主要顏色 */
    --aiva-secondary: #64748b;  /* 次要顏色 */
    --aiva-success: #059669;    /* 成功顏色 */
}
```

## 📊 使用統計

### 目標用戶
- **滲透測試人員**: 使用高價值功能模組進行專業測試
- **安全研究員**: 研究和發現新的安全漏洞
- **Bug Bounty 獵人**: 尋找高價值漏洞獲得獎勵
- **企業安全團隊**: 內部安全評估和合規檢查

### 商業價值
- **每次掃描潛在價值**: $1.8K-$10.2K
- **平台總價值**: $10.5K-$41K+
- **投資回報**: 單次成功漏洞發現即可回收成本

## 🚀 未來規劃

### 短期計劃 (1-2 週)
- [ ] 掃描結果導出功能
- [ ] 批量掃描支援
- [ ] 掃描模板保存
- [ ] 進階過濾和搜尋

### 中期計劃 (1-2 月)
- [ ] 實時通知系統
- [ ] 掃描排程功能
- [ ] 多用戶協作
- [ ] 報告生成系統

### 長期計劃 (3-6 月)
- [ ] 機器學習整合
- [ ] 自動化工作流程
- [ ] 第三方系統整合
- [ ] 移動 App 支援

## 🛠 故障排除

### 常見問題

**Q: 無法連接到 API**
A: 檢查 API 服務是否運行在 localhost:8000，或修改 `apiBase` 設定

**Q: 登入失敗**
A: 確認使用正確的用戶名和密碼，檢查 API 服務狀態

**Q: 掃描無法啟動**
A: 檢查必填欄位是否完整，確認目標 URL 格式正確

**Q: 結果不顯示**
A: 等待掃描完成，或檢查瀏覽器開發者工具的錯誤信息

### 調試模式
在瀏覽器開發者工具中輸入:
```javascript
// 啟用詳細日誌
dashboard.debug = true;

// 檢查當前狀態
console.log(dashboard);
```

## 📞 技術支援

- 查看瀏覽器開發者工具的 Console 標籤頁獲取錯誤信息
- 檢查 Network 標籤頁確認 API 請求狀態
- 確保 API 服務正常運行且可訪問
- 驗證用戶權限和令牌有效性

---

**AIVA Web 界面讓安全測試變得簡單直觀！** 🚀