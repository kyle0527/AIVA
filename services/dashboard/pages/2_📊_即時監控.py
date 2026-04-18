# -*- coding: utf-8 -*-
# 即時監控頁面 - 提供掃描任務的即時監控功能
#   - 掃描進度實時追蹤
#   - 任務狀態監控
#   - 日誌串流顯示

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import time

from services.dashboard.config import config
from services.dashboard.api_client import AIVAApiClient

# 頁面配置
st.set_page_config(
    page_title="即時監控 - AIVA Dashboard",
    page_icon="📊",
    layout="wide"
)

# 初始化 API 客戶端
if 'api_client' not in st.session_state:
    st.session_state.api_client = AIVAApiClient(
        base_url=config.api_base_url,
        timeout=config.api_timeout
    )

api_client = st.session_state.api_client

# ==================== 頁面標題 ====================

st.title("📊 即時監控")
st.markdown("即時查看掃描進度，監控任務狀態，查看執行日誌。")
st.divider()

# ==================== 掃描選擇器 ====================

try:
    # 獲取運行中和最近完成的掃描
    running_scans = api_client.list_scans(status="running", limit=10)
    recent_scans = api_client.list_scans(limit=20)
    
    all_scans = running_scans.get("scans", []) + recent_scans.get("scans", [])
    
    # 去重（基於 scan_id）
    seen = set()
    unique_scans = []
    for scan in all_scans:
        scan_id = scan.get("scan_id")
        if scan_id and scan_id not in seen:
            seen.add(scan_id)
            unique_scans.append(scan)
    
    if unique_scans:
        # 創建選項列表
        scan_options = {
            f"{scan.get('scan_id', 'N/A')} - {scan.get('target', 'N/A')}": scan.get("scan_id")
            for scan in unique_scans
        }
        
        selected_label = st.selectbox(
            "🎯 選擇掃描任務",
            options=list(scan_options.keys()),
            help="選擇要監控的掃描任務"
        )
        
        selected_scan_id = scan_options[selected_label]
    else:
        st.warning("📭 沒有可監控的掃描任務")
        st.info("請前往「掃描控制台」啟動新掃描")
        st.stop()

except Exception as e:
    st.error(f"❌ 無法載入掃描列表: {e}")
    st.stop()

st.divider()

# ==================== 掃描狀態與進度 ====================

st.subheader("📈 掃描進度")

# 自動刷新設定
auto_refresh = st.checkbox(
    "🔄 自動刷新",
    value=config.auto_refresh,
    help=f"每 {config.status_refresh_interval} 秒自動更新狀態"
)

# 建立狀態顯示區域
status_container = st.container()

with status_container:
    try:
        # 獲取詳細狀態
        status_data = api_client.get_scan_status(selected_scan_id)
        
        # 基本資訊
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("狀態", status_data.get("status", "unknown"))
        
        with col2:
            progress = status_data.get("progress", 0)
            st.metric("進度", f"{progress*100:.1f}%")
        
        with col3:
            phase = status_data.get("phase", "unknown")
            st.metric("當前階段", phase)
        
        with col4:
            findings = status_data.get("findings_count", 0)
            st.metric("發現漏洞", findings)
        
        # 進度條
        st.progress(progress if 0 <= progress <= 1 else 0)
        
        # 當前任務（如果有）
        current_task = status_data.get("current_task")
        if current_task:
            st.info(f"🔄 **當前任務**: {current_task}")
        
        # 詳細狀態（可展開）
        with st.expander("🔍 查看詳細狀態"):
            st.json(status_data)
    
    except Exception as e:
        st.error(f"❌ 無法獲取狀態: {e}")

st.divider()

# ==================== 任務佇列狀態 ====================

st.subheader("📋 任務佇列")

# NOTE: 當 app.py 提供任務佇列 API 後可實作此功能
st.info("⚠️  任務佇列詳情尚未實作（需要 app.py 新增對應 API）")

st.divider()

# ==================== 日誌串流 ====================

st.subheader("📜 執行日誌")

# 日誌控制
col1, col2 = st.columns([1, 3])

with col1:
    stream_logs = st.checkbox("📡 啟用日誌串流", value=False, help="即時顯示掃描日誌（SSE）")

with col2:
    if stream_logs:
        st.info("🔴 日誌串流已啟用")

# 日誌顯示區域
log_container = st.empty()

if stream_logs:
    try:
        # 使用 SSE 串流日誌
        with log_container.container():
            st.caption("🔴 即時日誌串流")
            log_text = st.empty()
            
            log_lines = []
            max_lines = config.get("performance.log_buffer_size", 1000)
            
            # 串流日誌（限制顯示行數）
            for log_event in api_client.stream_logs(selected_scan_id):
                timestamp = log_event.get("timestamp", "")
                level = log_event.get("level", "INFO")
                message = log_event.get("message", "")
                
                # 格式化日誌行
                log_line = f"[{timestamp}] {level} - {message}"
                log_lines.append(log_line)
                
                # 限制行數（保留最新的 N 行）
                if len(log_lines) > max_lines:
                    log_lines = log_lines[-max_lines:]
                
                # 更新顯示
                log_text.text_area(
                    label="日誌內容",
                    value="\n".join(log_lines),
                    height=400,
                    label_visibility="collapsed"
                )
    
    except Exception as e:
        st.error(f"❌ 日誌串流失敗: {e}")
else:
    # 靜態日誌顯示（不使用 SSE）
    with log_container.container():
        st.caption("📄 靜態日誌顯示（不即時更新）")
        
        # NOTE: 日志串流功能需要 SSE 支援，從日誌檔案讀取並顯示
        st.text_area(
            label="日誌內容",
            value="[提示] 請啟用「日誌串流」以查看即時日誌\n[提示] 或實作靜態日誌讀取功能",
            height=400,
            label_visibility="collapsed"
        )

st.divider()

# ==================== 系統資源監控 ====================

st.subheader("💻 系統資源")

# NOTE: 當有系統監控 API 後實作此功能
API_PENDING = "需要系統監控 API"
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("CPU 使用率", "N/A", help=API_PENDING)

with col2:
    st.metric("記憶體使用", "N/A", help=API_PENDING)

with col3:
    st.metric("並發任務", "N/A", help="需要系統監控 API")

st.divider()

# ==================== 自動刷新邏輯 ====================

if auto_refresh:
    # 使用 st.empty 作為刷新觸發器
    time.sleep(config.status_refresh_interval)
    st.rerun()
