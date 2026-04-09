# -*- coding: utf-8 -*-
# 歷史記錄頁面 - 提供掃描歷史記錄的查詢和管理功能

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:

import streamlit as st
from datetime import datetime, timedelta

from services.dashboard.config import config
from services.dashboard.api_client import AIVAApiClient

# 頁面配置
st.set_page_config(
    page_title="歷史記錄 - AIVA Dashboard",
    page_icon="📜",
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

st.title("📜 歷史記錄")
st.markdown("查詢過往掃描記錄，對比結果，匯出報告。")
st.divider()

# ==================== 篩選器 ====================

st.subheader("🔍 篩選條件")

col1, col2, col3 = st.columns(3)

with col1:
    STATUS_FORMAT_MAP = {
        "running": "▶️ 執行中",
        "completed": "✅ 已完成",
        "failed": "❌ 失敗",
        "cancelled": "⏹️ 已取消"
    }
    
    def format_status(x: str) -> str:
        """Format status with fallback"""
        return STATUS_FORMAT_MAP.get(x, x) or x
    
    status_filter = st.multiselect(
        "狀態",
        options=["running", "completed", "failed", "cancelled"],
        default=[],
        format_func=format_status
    )

with col2:
    # 時間範圍（簡化版）
    time_range = st.selectbox(
        "時間範圍",
        options=["全部", "今日", "最近 7 天", "最近 30 天"],
        index=0
    )

with col3:
    search = st.text_input("🔎 搜尋目標", placeholder="example.com")

# 刷新按鈕
if st.button("🔄 刷新", use_container_width=False):
    st.rerun()

st.divider()

# ==================== 掃描歷史列表 ====================

st.subheader("📋 掃描歷史")

try:
    # 獲取掃描列表
    all_scans = api_client.list_scans(limit=100)
    scans = all_scans.get("scans", [])
    
    # 應用篩選
    if status_filter:
        scans = [s for s in scans if s.get("status") in status_filter]
    
    if search:
        scans = [
            s for s in scans 
            if search.lower() in s.get("target", "").lower() or 
               search.lower() in s.get("scan_id", "").lower()
        ]
    
    # 顯示統計
    st.caption(f"📊 找到 {len(scans)} 筆記錄")
    
    if scans:
        # 建立表格數據
        import pandas as pd
        
        table_data = []
        for scan in scans:
            table_data.append({
                "掃描 ID": scan.get("scan_id", "N/A"),
                "目標": scan.get("target", "N/A"),
                "狀態": scan.get("status", "unknown"),
                "開始時間": scan.get("start_time", "N/A"),
                "持續時間": scan.get("duration", "N/A"),
                "發現漏洞": scan.get("vulnerabilities_count", 0)
            })
        
        df = pd.DataFrame(table_data)
        
        # 顯示表格（可選擇行）
        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # 如果有選擇，顯示操作按鈕
        if hasattr(event, 'selection'):
            selection = event.get('selection', {})
            rows = selection.get('rows', [])
            if rows:
                selected_index = rows[0]
                selected_scan = scans[selected_index]
            selected_scan_id = selected_scan.get("scan_id")
            
            st.divider()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 查看詳情", use_container_width=True, type="primary"):
                    st.info(f"🔄 跳轉到結果分析頁面（Scan ID: {selected_scan_id}）")
                    # NOTE: Streamlit 頁面跳轉需要使用 st.switch_page() 並傳遞 scan_id
            
            with col2:
                if st.button("🗑️ 刪除記錄", use_container_width=True):
                    try:
                        api_client.delete_scan(selected_scan_id)
                        st.success("✅ 已刪除")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 刪除失敗: {e}")
            
            with col3:
                if st.button("📥 匯出報告", use_container_width=True):
                    st.info("⚠️  報告匯出功能尚未實作")
    else:
        st.info("📭 沒有符合條件的掃描記錄")

except Exception as e:
    st.error(f"❌ 無法載入歷史記錄: {e}")

st.divider()

# ==================== 結果對比 ====================

st.subheader("🔄 結果對比")

# 多選掃描進行對比
st.caption("選擇 2-3 個掃描結果進行對比")

# NOTE: Phase 3 實作掃描對比功能
st.info("⚠️  結果對比功能尚未實作")

selected_scans_for_compare = st.multiselect(
    "選擇掃描",
    options=[],  # NOTE: 需要從 API 獲取掃描列表
    max_selections=3
)

if len(selected_scans_for_compare) >= 2 and st.button("🔄 開始對比", type="primary"):
    st.info("⚠️  對比功能尚未實作")

st.divider()

# ==================== 批次操作 ====================

st.subheader("🗑️ 資料管理")

with st.expander("⚠️ 危險操作"):
    st.warning("以下操作將永久刪除資料，請謹慎操作！")
    
    days_old = st.slider(
        "刪除多久前的資料",
        min_value=7,
        max_value=365,
        value=30,
        help="刪除指定天數之前的掃描記錄"
    )
    
    if st.button("🗑️ 確認清理", type="primary"):
        # NOTE: Phase 3 - 實作批次刪除功能
        st.info(f"⚠️  批次清理功能尚未實作（將刪除 {days_old} 天前的資料）")
