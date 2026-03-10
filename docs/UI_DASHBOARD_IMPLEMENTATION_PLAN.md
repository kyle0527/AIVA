# AIVA UI Dashboard 完善實施計畫

> **文件版本**: 1.0  
> **建立日期**: 2026-02-13  
> **負責模組**: services/dashboard + services/core/ui  
> **預估工期**: 3-5 天（含測試）

---

## 📋 目錄

1. [現況分析](#1-現況分析)
2. [目標與需求](#2-目標與需求)
3. [架構設計](#3-架構設計)
4. [實施計劃](#4-實施計劃)
5. [技術細節](#5-技術細節)
6. [部署方案](#6-部署方案)
7. [測試計劃](#7-測試計劃)
8. [維護方案](#8-維護方案)

---

## 1. 現況分析

### 1.1 現有 UI 組件盤點

#### ✅ **終端 UI (Rich Library)**
```
services/core/ui/
├── rich_console.py    # Console 單例 + AIVA_THEME
├── components.py      # 10+ 個 UI 組件
├── themes.py          # 配色方案
└── __init__.py        # 模組匯出
```

**可用組件**:
- `show_banner()` - 開場橫幅
- `show_menu()` - 選單系統
- `show_table()` - 資料表格
- `show_panel()` - 資訊面板
- `show_progress()` - 進度條
- `confirm_action()` - 確認對話
- 狀態訊息: `show_success/error/warning/info()`

**主題配色** (AIVA_THEME):
```python
primary: "#00D9FF"    # 青色
accent: "#FF00FF"     # 洋紅
success: "#00FF00"    # 綠色
warning: "#FFAA00"    # 橘色
error: "#FF0000"      # 紅色
```

#### ✅ **FastAPI REST API**
```
services/core/aiva_core/service_backbone/api/app.py (Port 8000)
```

**現有端點**:
```
GET  /health               # 健康檢查
POST /scan                 # 啟動掃描
  Request:  {target, scan_type, max_depth, timeout}
  Response: {scan_id, status, message, estimated_time}

GET  /status/{scan_id}     # 查詢掃描狀態
  Response: {scan_id, status, progress, phase, findings}
```

#### ✅ **資料模型** (services/core/models.py)
完整的 Pydantic 模型庫:
- `ScanRequest/ScanResponse` - 掃描請求/響應
- `EnhancedVulnerability` - 漏洞詳情（含 CVE/CWE/CVSS）
- `Task` - 任務資訊
- `RiskAssessment` - 風險評估
- `AttackPath/AttackPathNode` - 攻擊路徑
- `ModuleStatus` - 模組狀態

#### ✅ **狀態追蹤機制**
```python
# services/core/aiva_core/service_backbone/state/session_state_manager.py
SessionStateManager:
  - get_scan_status(scan_id) -> Dict  # 即時狀態
  - list_all_scans() -> List          # 歷史記錄
  - get_scan_results(scan_id) -> Dict # 完整結果
```

### 1.2 問題與限制

❌ **缺少視覺化界面**:
- 無 Web Dashboard（需手動調用 API）
- 無即時監控面板（無法查看進度）
- 無結果視覺化（漏洞圖表、攻擊路徑）
- 無歷史管理（需手動查詢資料庫）

❌ **用戶體驗不足**:
- 終端 UI 僅適用於 CLI 用戶
- 無圖形化操作介面
- 無批次掃描管理
- 無報告匯出功能

❌ **監控能力有限**:
- 無即時日誌串流
- 無系統資源監控
- 無任務佇列可視化
- 無錯誤告警機制

---

## 2. 目標與需求

### 2.1 核心目標

1. **降低使用門檻**: 從 `curl` 調用 → 點擊按鈕
2. **即時可見性**: 掃描進度、日誌、任務狀態即時顯示
3. **結果視覺化**: 漏洞分析、風險評估、攻擊路徑圖表化
4. **歷史管理**: 掃描記錄、結果對比、報告匯出

### 2.2 功能需求

#### 🎯 **掃描控制台**
- [ ] 輸入目標 URL/IP（支援單個/批次）
- [ ] 選擇掃描類型（快速/標準/深度）
- [ ] 設定掃描參數（深度、超時、並發數）
- [ ] 一鍵啟動/停止/暫停掃描
- [ ] 顯示當前掃描列表

#### 📊 **即時監控面板**
- [ ] 掃描進度條 (Phase 0/1 分別顯示)
- [ ] 任務佇列狀態（等待/執行/完成/失敗）
- [ ] 即時日誌串流（WebSocket/SSE）
- [ ] 系統資源監控（CPU/記憶體）
- [ ] 模組運行狀態（5 大子系統）

#### 🔍 **結果分析頁**
- [ ] 漏洞列表（可篩選、排序、搜尋）
- [ ] 嚴重度統計（圓餅圖）
- [ ] CVSS 分數分布（長條圖）
- [ ] 攻擊路徑圖（Graphviz 流程圖）
- [ ] 漏洞詳情彈窗（CVE/CWE/POC/修復建議）

#### 📜 **歷史記錄**
- [ ] 掃描歷史列表（時間/目標/狀態）
- [ ] 結果對比（多次掃描差異）
- [ ] 報告匯出（JSON/HTML/PDF）
- [ ] 歷史資料清理

#### ⚙️ **系統設定**
- [ ] API 金鑰管理
- [ ] 掃描引擎啟用/停用
- [ ] 通知設定（Email/Webhook）
- [ ] 主題切換（亮色/暗色）

### 2.3 非功能需求

- **效能**: 支援 10+ 並發掃描，監控延遲 < 500ms
- **穩定性**: 99% 可用性，自動重連機制
- **安全性**: API Token 認證，HTTPS 支援
- **可擴展**: 模組化設計，易於新增頁面
- **相容性**: 支援 Chrome/Firefox/Edge 最新版

---

## 3. 架構設計

### 3.1 整體架構圖

```
┌─────────────────────────────────────────────────────────────┐
│                     AIVA 完整系統架構                          │
└─────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │   使用者界面層      │
                    └─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   CLI 終端 UI         Streamlit Dashboard    FastAPI Web
 (Rich Console)         (Port 8501)          (Port 8000 /ui)
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌─────────────────────┐
                    │   FastAPI Core API  │ ← 統一入口
                    │    (Port 8000)      │
                    └─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
   REST API            WebSocket/SSE         狀態管理
  /scan, /status      /api/logs/stream   SessionStateManager
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌─────────────────────┐
                    │  AIVA Core Engine   │
                    │   (5 大子系統)       │
                    └─────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │         │          │         │          │
     資料接收   分析引擎   任務協調  狀態管理  結果處理
```

### 3.2 Dashboard 技術架構

#### 🔥 **方案選擇: Streamlit + FastAPI**

**Streamlit 優勢**:
- ✅ Python 全棧開發（無需前端技能）
- ✅ 內建組件豐富（圖表、表格、表單）
- ✅ 即時更新機制（WebSocket 內建）
- ✅ 快速原型開發（2-3 天完成）
- ✅ 易於維護與擴展

**FastAPI 職責**:
- ✅ 核心業務邏輯（掃描、分析）
- ✅ 資料存取層（SessionStateManager）
- ✅ 即時通訊（SSE 日誌串流）
- ✅ API 閘道（統一鑑權）

### 3.3 目錄結構設計

```
AIVA-git/
├── services/
│   ├── core/
│   │   ├── aiva_core/
│   │   │   └── service_backbone/
│   │   │       └── api/
│   │   │           ├── app.py          # FastAPI 主程式（新增端點）
│   │   │           ├── websocket.py    # WebSocket 處理器（新建）
│   │   │           └── sse.py          # SSE 端點（新建）
│   │   └── ui/
│   │       ├── rich_console.py         # 終端 UI（保留）
│   │       ├── components.py           # 組件庫（保留）
│   │       └── themes.py               # 主題（保留）
│   │
│   └── dashboard/                      # 🆕 Streamlit Dashboard
│       ├── streamlit_app.py            # 主程式入口
│       ├── config.py                   # Dashboard 配置
│       ├── utils.py                    # 工具函式
│       ├── api_client.py               # FastAPI 客戶端
│       ├── pages/                      # 多頁面應用
│       │   ├── 1_🎯_掃描控制台.py
│       │   ├── 2_📊_即時監控.py
│       │   ├── 3_🔍_結果分析.py
│       │   └── 4_📜_歷史記錄.py
│       ├── components/                 # 自定義組件
│       │   ├── scan_form.py            # 掃描表單
│       │   ├── progress_tracker.py     # 進度追蹤
│       │   ├── vulnerability_table.py  # 漏洞表格
│       │   └── attack_graph.py         # 攻擊路徑圖
│       └── assets/                     # 靜態資源
│           ├── logo.png
│           └── styles.css
│
├── data/
│   └── dashboard/                      # Dashboard 資料
│       ├── cache/                      # Streamlit 快取
│       └── exports/                    # 報告匯出
│
├── config/
│   └── dashboard_config.yaml           # Dashboard 配置檔
│
└── docs/
    ├── UI_DASHBOARD_IMPLEMENTATION_PLAN.md  # 本文件
    └── UI_DASHBOARD_USER_GUIDE.md           # 使用手冊（待建）
```

---

## 4. 實施計劃

### 4.1 開發階段劃分

#### **Phase 1: 基礎設施建設** (1.5 天)

**目標**: 建立 FastAPI 擴展端點 + Streamlit 骨架

**任務清單**:
- [ ] **Task 1.1**: 新增 FastAPI SSE 端點
  - 檔案: `services/core/aiva_core/service_backbone/api/sse.py`
  - 端點: `GET /api/logs/stream` (SSE 日誌串流)
  - 端點: `GET /api/status/stream` (SSE 狀態更新)

- [ ] **Task 1.2**: 擴展 app.py 端點
  - 新增: `GET /api/scans` (歷史掃描列表)
  - 新增: `GET /api/scans/{scan_id}/results` (完整結果)
  - 新增: `DELETE /api/scans/{scan_id}` (刪除掃描)
  - 新增: `POST /api/scans/{scan_id}/stop` (停止掃描)

- [ ] **Task 1.3**: 建立 Streamlit 主程式
  - 檔案: `services/dashboard/streamlit_app.py`
  - 功能: 側邊欄導航 + 首頁儀表板
  - 整合: AIVA_THEME 配色

- [ ] **Task 1.4**: API 客戶端封裝
  - 檔案: `services/dashboard/api_client.py`
  - 功能: 封裝所有 FastAPI 端點調用
  - 功能: 錯誤處理 + 重試機制

**產出物**:
- ✅ FastAPI 新增 6 個端點
- ✅ Streamlit 可運行骨架
- ✅ API 客戶端庫

---

#### **Phase 2: 核心功能開發** (2 天)

**目標**: 完成 4 個主要頁面功能

##### **2.1 掃描控制台** (0.5 天)
**檔案**: `services/dashboard/pages/1_🎯_掃描控制台.py`

**功能模組**:
```python
# 1. 掃描表單區
st.form("scan_form"):
    target = st.text_input("目標 URL/IP")
    scan_type = st.selectbox(["快速", "標準", "深度"])
    advanced = st.expander("進階設定"):
        max_depth = st.slider(1-10)
        timeout = st.number_input(600-7200)
        engines = st.multiselect(["Nuclei", "SQLMap", ...])
    
    if st.form_submit_button("🚀 啟動掃描"):
        api_client.start_scan(...)

# 2. 當前掃描列表
active_scans = api_client.get_active_scans()
for scan in active_scans:
    col1, col2, col3 = st.columns([3,1,1])
    col1.write(scan.target)
    col2.metric("進度", f"{scan.progress}%")
    col3.button("⏹️ 停止", key=scan.id)

# 3. 快速統計卡片
col1, col2, col3, col4 = st.columns(4)
col1.metric("執行中", active_count, delta="+2")
col2.metric("今日完成", today_count)
col3.metric("發現漏洞", vuln_count)
col4.metric("高危漏洞", high_risk)
```

**組件依賴**:
- `components/scan_form.py` - 表單驗證邏輯
- API: `POST /scan`, `GET /api/scans`

---

##### **2.2 即時監控** (0.5 天)
**檔案**: `services/dashboard/pages/2_📊_即時監控.py`

**功能模組**:
```python
# 1. 掃描選擇器
scan_id = st.selectbox("選擇掃描任務", scan_list)

# 2. 雙階段進度條
col1, col2 = st.columns(2)
col1.progress(phase0_progress, text="Phase 0: 快速偵察")
col2.progress(phase1_progress, text="Phase 1: 深度掃描")

# 3. 任務佇列狀態（即時更新）
@st.experimental_fragment(run_every=2)  # 2秒刷新
def task_queue_status():
    tasks = api_client.get_task_queue(scan_id)
    st.dataframe(tasks[["name", "status", "progress"]])

# 4. 日誌串流（SSE 連接）
log_container = st.empty()
with st.spinner("連接日誌串流..."):
    for log_line in api_client.stream_logs(scan_id):
        log_container.text_area(
            "執行日誌", 
            value=log_line, 
            height=400
        )

# 5. 系統資源監控
col1, col2, col3 = st.columns(3)
col1.metric("CPU", f"{cpu_percent}%")
col2.metric("記憶體", f"{mem_used}/{mem_total} GB")
col3.metric("並發任務", task_count)
```

**技術要點**:
- 使用 `st.experimental_fragment` 實現局部刷新
- SSE 連接處理（錯誤重連）
- 日誌緩衝（最多顯示 1000 行）

---

##### **2.3 結果分析** (0.5 天)
**檔案**: `services/dashboard/pages/3_🔍_結果分析.py`

**功能模組**:
```python
# 1. 掃描結果概覽
st.subheader("📊 漏洞統計")
col1, col2, col3 = st.columns([2,2,1])

with col1:
    # 嚴重度圓餅圖（Plotly）
    fig = px.pie(
        values=[critical, high, medium, low],
        names=["Critical", "High", "Medium", "Low"],
        color_discrete_map={
            "Critical": "#FF0000",
            "High": "#FFAA00",
            "Medium": "#FFFF00",
            "Low": "#00FF00"
        }
    )
    st.plotly_chart(fig)

with col2:
    # CVSS 分數分布（長條圖）
    fig = px.histogram(
        vulnerabilities,
        x="cvss_score",
        nbins=10,
        title="CVSS 分數分布"
    )
    st.plotly_chart(fig)

with col3:
    # 關鍵指標
    st.metric("總漏洞數", len(vulnerabilities))
    st.metric("平均 CVSS", avg_cvss)
    st.metric("風險評級", risk_level)

# 2. 漏洞列表（可篩選/排序/搜尋）
st.subheader("🔍 漏洞清單")

# 篩選器
col1, col2, col3 = st.columns(3)
severity_filter = col1.multiselect("嚴重度", ["Critical", ...])
cwe_filter = col2.multiselect("CWE 類別", cwe_list)
search = col3.text_input("🔎 搜尋", placeholder="CVE-2024...")

# 表格顯示（可點選查看詳情）
selected_row = st.dataframe(
    vulnerabilities_df,
    use_container_width=True,
    on_select="rerun",
    selection_mode="single-row"
)

# 3. 漏洞詳情彈窗
if selected_row:
    vuln = get_vulnerability_details(selected_row.id)
    with st.expander("🔬 詳細資訊", expanded=True):
        st.markdown(f"**CVE**: {vuln.cve_id}")
        st.markdown(f"**CWE**: {vuln.cwe_id}")
        st.code(vuln.poc, language="python")
        st.markdown("**修復建議**:")
        st.info(vuln.recommendation)

# 4. 攻擊路徑圖（Graphviz）
st.subheader("🗺️ 攻擊路徑分析")
attack_path = api_client.get_attack_path(scan_id)
graph = generate_graphviz_graph(attack_path)
st.graphviz_chart(graph)
```

**依賴套件**:
```bash
plotly          # 互動圖表
graphviz        # 攻擊路徑圖
pandas          # 資料處理
```

---

##### **2.4 歷史記錄** (0.5 天)
**檔案**: `services/dashboard/pages/4_📜_歷史記錄.py`

**功能模組**:
```python
# 1. 歷史掃描列表
st.subheader("📜 掃描歷史")

# 時間範圍篩選
date_range = st.date_input("日期範圍", [start_date, end_date])
scans = api_client.get_scan_history(date_range)

# 表格顯示
st.dataframe(scans[[
    "scan_id", "target", "status", 
    "start_time", "duration", "vulnerabilities"
]])

# 2. 結果對比（多選）
st.subheader("🔄 結果對比")
selected_scans = st.multiselect("選擇掃描", scan_list, max_selections=3)

if len(selected_scans) >= 2:
    comparison = api_client.compare_scans(selected_scans)
    
    # 新增/減少漏洞
    col1, col2 = st.columns(2)
    col1.metric("新增漏洞", comparison.new_count, delta="+5")
    col2.metric("已修復", comparison.fixed_count, delta="-3")
    
    # 詳細對比表
    st.dataframe(comparison.diff_table)

# 3. 報告匯出
st.subheader("📥 匯出報告")
export_format = st.radio("格式", ["JSON", "HTML", "PDF"])
scan_ids = st.multiselect("選擇掃描", scan_list)

if st.button("📥 匯出"):
    report = api_client.export_report(scan_ids, export_format)
    st.download_button(
        "下載報告",
        data=report,
        file_name=f"aiva_report_{timestamp}.{export_format.lower()}"
    )

# 4. 歷史資料清理
with st.expander("🗑️ 資料管理"):
    days_old = st.slider("刪除多久前的資料", 7, 365, 30)
    if st.button("清理歷史"):
        api_client.cleanup_old_scans(days_old)
        st.success(f"已清理 {days_old} 天前的資料")
```

---

#### **Phase 3: 優化與測試** (1 天)

**任務清單**:
- [ ] **Task 3.1**: 效能優化
  - 實作 Streamlit 快取 (`@st.cache_data`)
  - 資料分頁載入（大型掃描結果）
  - 圖表渲染優化

- [ ] **Task 3.2**: 錯誤處理
  - API 連接失敗處理
  - SSE 斷線重連
  - 資料驗證與異常提示

- [ ] **Task 3.3**: 整合測試
  - 完整掃描流程測試（啟動 → 監控 → 結果）
  - 並發掃描測試（5+ 同時運行）
  - 長時間運行測試（24 小時）

- [ ] **Task 3.4**: 文件撰寫
  - 使用手冊 (`docs/UI_DASHBOARD_USER_GUIDE.md`)
  - API 文件更新
  - 部署說明

---

#### **Phase 4: 部署與交付** (0.5 天)

**任務清單**:
- [ ] **Task 4.1**: Docker 容器化
  ```dockerfile
  # Dockerfile.dashboard
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install -r requirements.txt
  COPY services/dashboard/ ./dashboard/
  CMD ["streamlit", "run", "dashboard/streamlit_app.py", "--server.port=8501"]
  ```

- [ ] **Task 4.2**: Docker Compose 整合
  ```yaml
  # docker-compose.yml (新增)
  services:
    aiva-core:
      build: .
      ports:
        - "8000:8000"
    
    aiva-dashboard:
      build:
        context: .
        dockerfile: Dockerfile.dashboard
      ports:
        - "8501:8501"
      environment:
        - AIVA_API_URL=http://aiva-core:8000
      depends_on:
        - aiva-core
  ```

- [ ] **Task 4.3**: 啟動腳本
  ```bash
  # scripts/start_dashboard.sh
  #!/bin/bash
  echo "🚀 啟動 AIVA Dashboard..."
  
  # 啟動 FastAPI (後台)
  python services/core/aiva_core/service_backbone/api/app.py &
  CORE_PID=$!
  
  # 啟動 Streamlit
  streamlit run services/dashboard/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0
  
  # 清理
  kill $CORE_PID
  ```

---

### 4.2 時程規劃甘特圖

```
┌─────────────────────────────────────────────────────────────┐
│               AIVA Dashboard 開發時程表                       │
│          預計工期: 5 天 (2026-02-13 ~ 2026-02-17)          │
└─────────────────────────────────────────────────────────────┘

Day 1 (02-13):
  ├─ Task 1.1: FastAPI SSE 端點開發         [████████] 完成
  ├─ Task 1.2: 擴展 app.py 端點            [████████] 完成
  └─ Task 1.3: Streamlit 主程式骨架        [████░░░░] 50%

Day 2 (02-14):
  ├─ Task 1.3: Streamlit 骨架完成          [████████] 完成
  ├─ Task 1.4: API 客戶端封裝              [████████] 完成
  └─ Task 2.1: 掃描控制台頁面              [████░░░░] 50%

Day 3 (02-15):
  ├─ Task 2.1: 掃描控制台完成              [████████] 完成
  ├─ Task 2.2: 即時監控頁面                [████████] 完成
  └─ Task 2.3: 結果分析頁面                [████░░░░] 50%

Day 4 (02-16):
  ├─ Task 2.3: 結果分析完成                [████████] 完成
  ├─ Task 2.4: 歷史記錄頁面                [████████] 完成
  └─ Task 3.1-3.2: 優化與錯誤處理          [████░░░░] 50%

Day 5 (02-17):
  ├─ Task 3.3: 整合測試                    [████████] 完成
  ├─ Task 3.4: 文件撰寫                    [████████] 完成
  └─ Task 4.1-4.3: 部署準備                [████████] 完成
```

---

## 5. 技術細節

### 5.1 FastAPI 新增端點規格

#### **5.1.1 SSE 日誌串流**
```python
# services/core/aiva_core/service_backbone/api/sse.py

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
import asyncio

router = APIRouter(prefix="/api")

@router.get("/logs/stream")
async def stream_logs(scan_id: str):
    """串流掃描日誌（SSE）
    
    事件格式:
        event: log
        data: {"timestamp": "...", "level": "INFO", "message": "..."}
    """
    async def log_generator():
        log_file = Path(f"data/logs/{scan_id}.log")
        
        # 讀取現有日誌
        if log_file.exists():
            with open(log_file) as f:
                for line in f:
                    yield {
                        "event": "log",
                        "data": json.dumps(parse_log_line(line))
                    }
        
        # 持續監控新日誌
        while True:
            if log_file.exists():
                with open(log_file) as f:
                    f.seek(0, 2)  # 移到檔尾
                    while True:
                        line = f.readline()
                        if line:
                            yield {
                                "event": "log",
                                "data": json.dumps(parse_log_line(line))
                            }
                        else:
                            await asyncio.sleep(0.5)
                            break
            else:
                await asyncio.sleep(1)
    
    return EventSourceResponse(log_generator())
```

#### **5.1.2 掃描狀態串流**
```python
@router.get("/status/stream")
async def stream_status(scan_id: str):
    """串流掃描狀態（SSE）
    
    事件格式:
        event: status
        data: {
            "scan_id": "...",
            "status": "running",
            "progress": 0.45,
            "phase": "phase1",
            "current_task": "...",
            "findings_count": 12
        }
    """
    async def status_generator():
        while True:
            status = await session_state_manager.get_scan_status(scan_id)
            
            yield {
                "event": "status",
                "data": json.dumps(status.dict())
            }
            
            # 如果掃描結束，發送完成事件
            if status["status"] in ["completed", "failed"]:
                yield {
                    "event": "completed",
                    "data": json.dumps({"scan_id": scan_id})
                }
                break
            
            await asyncio.sleep(2)  # 每 2 秒更新
    
    return EventSourceResponse(status_generator())
```

#### **5.1.3 歷史掃描管理**
```python
@router.get("/api/scans")
async def list_scans(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """查詢歷史掃描列表
    
    Returns:
        {
            "total": 123,
            "scans": [
                {
                    "scan_id": "...",
                    "target": "...",
                    "status": "completed",
                    "start_time": "...",
                    "duration": 1234,
                    "findings_count": 12
                },
                ...
            ]
        }
    """
    scans = await session_state_manager.list_scans(
        start_date=start_date,
        end_date=end_date,
        status=status,
        limit=limit,
        offset=offset
    )
    return scans

@router.get("/api/scans/{scan_id}/results")
async def get_scan_results(scan_id: str):
    """取得完整掃描結果（含漏洞詳情）"""
    results = await session_state_manager.get_scan_results(scan_id)
    return results

@router.post("/api/scans/{scan_id}/stop")
async def stop_scan(scan_id: str):
    """停止執行中的掃描"""
    await coordinator.stop_scan(scan_id)
    return {"message": "Scan stopped"}

@router.delete("/api/scans/{scan_id}")
async def delete_scan(scan_id: str):
    """刪除掃描記錄"""
    await session_state_manager.delete_scan(scan_id)
    return {"message": "Scan deleted"}
```

---

### 5.2 Streamlit 關鍵技術

#### **5.2.1 API 客戶端封裝**
```python
# services/dashboard/api_client.py

import requests
from typing import Iterator
import sseclient

class AIVAApiClient:
    """AIVA FastAPI 客戶端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def start_scan(
        self, 
        target: str, 
        scan_type: str = "comprehensive",
        **kwargs
    ) -> dict:
        """啟動掃描"""
        response = self.session.post(
            f"{self.base_url}/scan",
            json={
                "target": target,
                "scan_type": scan_type,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_scan_status(self, scan_id: str) -> dict:
        """查詢掃描狀態"""
        response = self.session.get(f"{self.base_url}/status/{scan_id}")
        response.raise_for_status()
        return response.json()
    
    def stream_logs(self, scan_id: str) -> Iterator[str]:
        """串流日誌（SSE）"""
        response = self.session.get(
            f"{self.base_url}/api/logs/stream",
            params={"scan_id": scan_id},
            stream=True
        )
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.event == "log":
                yield json.loads(event.data)["message"]
    
    def stream_status(self, scan_id: str) -> Iterator[dict]:
        """串流狀態更新（SSE）"""
        response = self.session.get(
            f"{self.base_url}/api/status/stream",
            params={"scan_id": scan_id},
            stream=True
        )
        
        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.event == "status":
                yield json.loads(event.data)
            elif event.event == "completed":
                break
```

#### **5.2.2 即時更新機制**
```python
# services/dashboard/pages/2_📊_即時監控.py

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# 方法 1: 使用 st_autorefresh（全頁刷新）
st_autorefresh(interval=2000, key="status_refresh")  # 2 秒刷新

# 方法 2: 使用 st.experimental_fragment（局部刷新）
@st.experimental_fragment(run_every=2)
def update_progress():
    status = api_client.get_scan_status(scan_id)
    st.progress(status["progress"])
    st.metric("當前階段", status["phase"])

# 方法 3: 使用 SSE 串流（真正即時）
with st.container():
    log_area = st.empty()
    status_area = st.empty()
    
    for status in api_client.stream_status(scan_id):
        status_area.metric("進度", f"{status['progress']*100:.1f}%")
        
        # 同時顯示日誌
        for log in api_client.stream_logs(scan_id):
            log_area.text(log)
```

#### **5.2.3 圖表生成範例**
```python
# services/dashboard/components/vulnerability_charts.py

import plotly.express as px
import plotly.graph_objects as go

def create_severity_pie_chart(vulnerabilities: list) -> go.Figure:
    """嚴重度圓餅圖"""
    severity_counts = Counter([v["severity"] for v in vulnerabilities])
    
    fig = px.pie(
        values=list(severity_counts.values()),
        names=list(severity_counts.keys()),
        title="漏洞嚴重度分布",
        color_discrete_map={
            "Critical": "#FF0000",
            "High": "#FFAA00",
            "Medium": "#FFFF00",
            "Low": "#00FF00"
        }
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig

def create_cvss_histogram(vulnerabilities: list) -> go.Figure:
    """CVSS 分數分布直方圖"""
    cvss_scores = [v["cvss_score"] for v in vulnerabilities if v.get("cvss_score")]
    
    fig = px.histogram(
        x=cvss_scores,
        nbins=10,
        title="CVSS 分數分布",
        labels={"x": "CVSS 分數", "y": "漏洞數量"}
    )
    
    # 添加嚴重度區間標記
    fig.add_vrect(x0=0, x1=3.9, fillcolor="green", opacity=0.1)
    fig.add_vrect(x0=4, x1=6.9, fillcolor="yellow", opacity=0.1)
    fig.add_vrect(x0=7, x1=8.9, fillcolor="orange", opacity=0.1)
    fig.add_vrect(x0=9, x1=10, fillcolor="red", opacity=0.1)
    
    return fig

def create_attack_path_graph(attack_path: dict) -> str:
    """攻擊路徑圖（Graphviz）"""
    from graphviz import Digraph
    
    dot = Digraph(comment='Attack Path')
    dot.attr(rankdir='LR')
    
    # 添加節點
    for node in attack_path["nodes"]:
        color = {
            "entry_point": "green",
            "vulnerability": "red",
            "privilege": "orange",
            "data": "blue"
        }.get(node["type"], "gray")
        
        dot.node(
            node["id"],
            node["label"],
            color=color,
            style="filled",
            fillcolor=f"{color}22"
        )
    
    # 添加邊
    for edge in attack_path["edges"]:
        dot.edge(
            edge["from"],
            edge["to"],
            label=edge.get("method", "")
        )
    
    return dot.source
```

---

### 5.3 配置檔案設計

#### **config/dashboard_config.yaml**
```yaml
# AIVA Dashboard 配置檔

api:
  base_url: http://localhost:8000
  timeout: 30
  retry_attempts: 3
  retry_delay: 1

dashboard:
  title: AIVA Security Dashboard
  port: 8501
  host: 0.0.0.0
  theme: dark  # dark | light
  
  # 頁面配置
  pages:
    - id: scan_console
      title: 掃描控制台
      icon: 🎯
      enabled: true
    
    - id: monitoring
      title: 即時監控
      icon: 📊
      enabled: true
    
    - id: results
      title: 結果分析
      icon: 🔍
      enabled: true
    
    - id: history
      title: 歷史記錄
      icon: 📜
      enabled: true

# 掃描預設設定
scan_defaults:
  scan_type: comprehensive
  max_depth: 3
  timeout: 1800
  concurrent_limit: 5

# 視覺化設定
visualization:
  chart_theme: plotly  # plotly | seaborn
  color_scheme: aiva   # 使用 AIVA_THEME 配色
  
  attack_graph:
    layout: LR  # LR (左右) | TB (上下)
    node_shape: box
    edge_style: solid

# 效能設定
performance:
  cache_ttl: 300  # 快取 5 分鐘
  log_buffer_size: 1000  # 最多顯示 1000 行日誌
  status_refresh_interval: 2  # 狀態更新間隔（秒）

# 匯出設定
export:
  formats: [json, html, pdf]
  output_dir: data/dashboard/exports/
  template_dir: services/dashboard/templates/
```

---

## 6. 部署方案

### 6.1 本地開發環境

#### **安裝依賴**
```bash
# 1. 安裝 Dashboard 依賴
pip install -r requirements-dashboard.txt

# requirements-dashboard.txt 內容:
streamlit>=1.31.0
plotly>=5.18.0
graphviz>=0.20.1
sseclient-py>=1.8.0
requests>=2.31.0
pandas>=2.1.0
streamlit-autorefresh>=1.0.1
sse-starlette>=1.8.2  # FastAPI SSE 支援
```

#### **啟動方式**
```bash
# 方式 1: 分別啟動（開發階段）
# Terminal 1: FastAPI
python app.py

# Terminal 2: Streamlit
streamlit run services/dashboard/streamlit_app.py

# 方式 2: 一鍵啟動（使用腳本）
bash scripts/start_dashboard.sh

# 方式 3: Python 啟動器
python -m services.dashboard.launcher
```

---

### 6.2 Docker 部署

#### **Dockerfile.dashboard**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安裝依賴
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# 複製 Dashboard 代碼
COPY services/dashboard/ ./services/dashboard/
COPY config/dashboard_config.yaml ./config/

# 暴露端口
EXPOSE 8501

# 啟動命令
CMD ["streamlit", "run", \
     "services/dashboard/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

#### **docker-compose.yml**
```yaml
version: '3.8'

services:
  aiva-core:
    build: .
    container_name: aiva-core
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    networks:
      - aiva-network
  
  aiva-dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    container_name: aiva-dashboard
    ports:
      - "8501:8501"
    environment:
      - AIVA_API_URL=http://aiva-core:8000
      - STREAMLIT_SERVER_PORT=8501
    depends_on:
      - aiva-core
    networks:
      - aiva-network
    volumes:
      - ./config:/app/config:ro

networks:
  aiva-network:
    driver: bridge
```

#### **啟動命令**
```bash
# 構建並啟動
docker-compose up -d

# 查看日誌
docker-compose logs -f aiva-dashboard

# 停止服務
docker-compose down
```

---

### 6.3 生產環境部署

#### **Nginx 反向代理**
```nginx
# /etc/nginx/sites-available/aiva

upstream aiva_core {
    server localhost:8000;
}

upstream aiva_dashboard {
    server localhost:8501;
}

server {
    listen 80;
    server_name aiva.example.com;
    
    # 重定向到 HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name aiva.example.com;
    
    ssl_certificate /etc/ssl/certs/aiva.crt;
    ssl_certificate_key /etc/ssl/private/aiva.key;
    
    # Dashboard (主界面)
    location / {
        proxy_pass http://aiva_dashboard;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # API 端點
    location /api/ {
        proxy_pass http://aiva_core;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # SSE 串流（特殊處理）
    location /api/logs/stream {
        proxy_pass http://aiva_core;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

---

## 7. 測試計劃

### 7.1 單元測試

```python
# tests/dashboard/test_api_client.py

import pytest
from services.dashboard.api_client import AIVAApiClient

@pytest.fixture
def api_client():
    return AIVAApiClient(base_url="http://localhost:8000")

def test_start_scan(api_client):
    result = api_client.start_scan(
        target="https://example.com",
        scan_type="quick"
    )
    assert "scan_id" in result
    assert result["status"] == "started"

def test_get_scan_status(api_client):
    scan_id = "test_scan_123"
    status = api_client.get_scan_status(scan_id)
    assert "progress" in status
    assert 0 <= status["progress"] <= 1

def test_stream_logs(api_client):
    scan_id = "test_scan_123"
    logs = list(api_client.stream_logs(scan_id))
    assert len(logs) > 0
```

### 7.2 整合測試

```python
# tests/dashboard/test_integration.py

import pytest
import streamlit as st
from services.dashboard.pages.scan_console import start_scan_workflow

def test_full_scan_workflow():
    """測試完整掃描流程"""
    # 1. 啟動掃描
    result = start_scan_workflow(target="https://example.com")
    scan_id = result["scan_id"]
    
    # 2. 監控進度
    status = get_scan_progress(scan_id)
    assert status["status"] in ["running", "completed"]
    
    # 3. 查看結果
    results = get_scan_results(scan_id)
    assert "vulnerabilities" in results
    
    # 4. 匯出報告
    report = export_report(scan_id, format="json")
    assert report is not None
```

### 7.3 效能測試

```python
# tests/dashboard/test_performance.py

import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor

def test_concurrent_scans():
    """測試並發掃描處理能力"""
    targets = [f"https://example{i}.com" for i in range(10)]
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(api_client.start_scan, target)
            for target in targets
        ]
        
        results = [f.result() for f in futures]
    
    assert len(results) == 10
    assert all("scan_id" in r for r in results)

def test_sse_stream_performance():
    """測試 SSE 串流效能"""
    import time
    
    start = time.time()
    log_count = 0
    
    for log in api_client.stream_logs("test_scan"):
        log_count += 1
        if log_count >= 100:
            break
    
    duration = time.time() - start
    assert duration < 5  # 100 條日誌應在 5 秒內完成
```

---

## 8. 維護方案

### 8.1 監控指標

建立 Grafana Dashboard 監控以下指標:
- Dashboard 響應時間
- API 調用成功率
- SSE 連線數
- 並發掃描數
- 資料庫查詢延遲

### 8.2 日誌管理

```python
# services/dashboard/utils/logger.py

import logging
from logging.handlers import RotatingFileHandler

def setup_dashboard_logger():
    logger = logging.getLogger("aiva_dashboard")
    logger.setLevel(logging.INFO)
    
    # 檔案處理器（自動輪替）
    handler = RotatingFileHandler(
        "logs/dashboard.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    return logger
```

### 8.3 備份策略

- **掃描資料**: 每日自動備份到 `data/dashboard/backups/`
- **配置檔**: Git 版本控制
- **使用者設定**: 儲存在 SQLite（本地 Streamlit session）

---

## 9. 附錄

### 9.1 依賴套件清單

```txt
# requirements-dashboard.txt

# Streamlit 核心
streamlit>=1.31.0
streamlit-autorefresh>=1.0.1

# 視覺化
plotly>=5.18.0
graphviz>=0.20.1
matplotlib>=3.8.0
seaborn>=0.13.0

# 資料處理
pandas>=2.1.0
numpy>=1.26.0

# API 通訊
requests>=2.31.0
sseclient-py>=1.8.0
websocket-client>=1.7.0

# FastAPI SSE 支援
sse-starlette>=1.8.2

# 工具
pyyaml>=6.0.1
python-dotenv>=1.0.0
```

### 9.2 參考資源

- [Streamlit 官方文件](https://docs.streamlit.io/)
- [FastAPI SSE 教學](https://fastapi.tiangolo.com/advanced/custom-response/#server-sent-events)
- [Plotly 圖表範例](https://plotly.com/python/)
- [Graphviz 語法](https://graphviz.org/doc/info/lang.html)

### 9.3 常見問題 (FAQ)

**Q1: Dashboard 無法連接到 FastAPI?**
- 檢查 `config/dashboard_config.yaml` 的 `api.base_url`
- 確認 FastAPI 服務運行在 Port 8000
- 查看防火牆設定

**Q2: SSE 串流斷線?**
- 實作自動重連機制（見 `api_client.py`）
- 檢查 Nginx `proxy_buffering` 設定
- 增加 `proxy_read_timeout`

**Q3: 圖表顯示異常?**
- 清除 Streamlit 快取: `streamlit cache clear`
- 檢查資料格式是否符合 Plotly 要求
- 更新 Plotly 到最新版本

---

## 10. 下一步行動

立即開始實施！執行以下命令啟動開發:

```bash
# 建立目錄結構
mkdir -p services/dashboard/pages
mkdir -p services/dashboard/components
mkdir -p services/dashboard/assets
mkdir -p data/dashboard/exports

# 安裝依賴
pip install streamlit plotly graphviz sseclient-py sse-starlette

# 建立配置檔
cp config/dashboard_config.yaml.example config/dashboard_config.yaml

# 啟動開發
python scripts/dev_dashboard.py
```

---

**文件結束** | 準備開始實施 ✅
