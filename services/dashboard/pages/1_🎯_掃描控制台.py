# -*- coding: utf-8 -*-
# 掃描控制台頁面 - 提供掃描任務的創建和管理功能

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from datetime import datetime

from services.dashboard.config import config
from services.dashboard.api_client import AIVAApiClient

# 頁面配置
st.set_page_config(
    page_title="掃描控制台 - AIVA Dashboard",
    page_icon="🎯",
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

st.title("🎯 掃描控制台")
st.markdown("啟動新的掃描任務，配置掃描參數，管理執行中的掃描。")
st.divider()

# ==================== 掃描表單 ====================

st.subheader("🚀 啟動新掃描")

with st.form("scan_form"):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        target = st.text_input(
            "🎯 目標 URL/IP",
            placeholder="https://example.com 或 192.168.1.1",
            help="輸入要掃描的目標網址或 IP 地址"
        )
    
    with col2:
        scan_type = st.selectbox(
            "📋 掃描類型",
            options=["quick", "comprehensive", "deep"],
            index=1,  # 預設 comprehensive
            format_func=lambda x: {
                "quick": "🏃 快速掃描",
                "comprehensive": "🔍 標準掃描",
                "deep": "🔬 深度掃描"
            }[x]
        )
    
    # 進階設定
    with st.expander("⚙️ 進階設定"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_depth = st.slider(
                "掃描深度",
                min_value=1,
                max_value=10,
                value=config.scan_max_depth_default,
                help="爬蟲深度限制"
            )
        
        with col2:
            timeout = st.number_input(
                "超時時間（秒）",
                min_value=60,
                max_value=7200,
                value=config.scan_timeout_default,
                step=60,
                help="掃描超時時間"
            )
    
    # 提交按鈕
    submit = st.form_submit_button(
        "🚀 啟動掃描",
        use_container_width=True,
        type="primary"
    )
    
    if submit:
        if not target:
            st.error("❌ 請輸入目標 URL/IP")
        else:
            try:
                with st.spinner("正在啟動掃描..."):
                    result = api_client.start_scan(
                        target=target,
                        scan_type=scan_type,
                        max_depth=max_depth,
                        timeout=timeout
                    )
                
                st.success("✅ 掃描已啟動！")
                st.info(f"**Scan ID**: `{result['scan_id']}`")
                st.caption(f"預估時間: {result.get('estimated_time', 600)} 秒")
                
                # 提供跳轉到監控頁面的按鈕
                st.page_link(
                    "pages/2_📊_即時監控.py",
                    label="前往監控頁面 →",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ 啟動掃描失敗: {e}")

st.divider()

# ==================== 當前掃描列表 ====================

st.subheader("📋 當前掃描任務")

# 刷新按鈕
if st.button("🔄 刷新", use_container_width=False):
    st.rerun()

try:
    # 獲取運行中的掃描
    running_scans_data = api_client.list_scans(status="running", limit=20)
    running_scans = running_scans_data.get("scans", [])
    
    if running_scans:
        # 顯示每個掃描任務
        for scan in running_scans:
            scan_id = scan.get("scan_id", "N/A")
            target = scan.get("target", "N/A")
            status = scan.get("status", "unknown")
            start_time = scan.get("start_time", "N/A")
            
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**🎯 {target}**")
                    st.caption(f"ID: {scan_id} | 開始: {start_time}")
                
                with col2:
                    # 獲取詳細狀態（含進度）
                    try:
                        detail = api_client.get_scan_status(scan_id)
                        progress = detail.get("progress", 0)
                        st.metric("進度", f"{progress*100:.0f}%")
                    except Exception:
                        st.metric("狀態", status)
                
                with col3:
                    # 停止按鈕
                    if st.button("⏹️ 停止", key=f"stop_{scan_id}"):
                        try:
                            api_client.stop_scan(scan_id)
                            st.success("已發送停止請求")
                            st.rerun()
                        except Exception as e:
                            st.error(f"停止失敗: {e}")
                
                st.divider()
    else:
        st.info("📭 目前沒有執行中的掃描任務")

except Exception as e:
    st.error(f"❌ 無法載入掃描列表: {e}")

st.divider()

# ==================== 快速統計 ====================

st.subheader("📊 快速統計")

col1, col2, col3 = st.columns(3)

try:
    all_scans = api_client.list_scans(limit=1000)
    total = all_scans.get("total", 0)
    
    running_data = api_client.list_scans(status="running", limit=100)
    running = running_data.get("total", 0)
    
    completed_data = api_client.list_scans(status="completed", limit=100)
    completed = completed_data.get("total", 0)
    
    with col1:
        st.metric("總掃描數", total)
    
    with col2:
        st.metric("執行中", running, delta=f"+{running}" if running > 0 else None)
    
    with col3:
        st.metric("已完成", completed)

except Exception as e:
    st.error(f"❌ 統計載入失敗: {e}")
