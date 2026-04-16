# -*- coding: utf-8 -*-
# 結果分析頁面 - 提供掃描結果的分析和視覺化功能

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:

import streamlit as st

from services.dashboard.config import config
from services.dashboard.api_client import AIVAApiClient

# 頁面配置
st.set_page_config(
    page_title="結果分析 - AIVA Dashboard",
    page_icon="🔍",
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

st.title("🔍 結果分析")
st.markdown("查看漏洞詳情，分析攻擊路徑，生成安全報告。")
st.divider()

# ==================== 掃描選擇器 ====================

try:
    # 獲取已完成的掃描
    completed_scans = api_client.list_scans(status="completed", limit=50)
    all_scans = api_client.list_scans(limit=50)
    
    scans = completed_scans.get("scans", []) + all_scans.get("scans", [])
    
    # 去重
    seen = set()
    unique_scans = []
    for scan in scans:
        scan_id = scan.get("scan_id")
        if scan_id and scan_id not in seen:
            seen.add(scan_id)
            unique_scans.append(scan)
    
    if unique_scans:
        scan_options = {
            f"{scan.get('scan_id', 'N/A')} - {scan.get('target', 'N/A')} ({scan.get('status', 'unknown')})": scan.get("scan_id")
            for scan in unique_scans
        }
        
        selected_label = st.selectbox(
            "🎯 選擇掃描結果",
            options=list(scan_options.keys())
        )
        
        selected_scan_id = scan_options[selected_label]
    else:
        st.warning("📭 沒有可分析的掃描結果")
        st.stop()

except Exception as e:
    st.error(f"❌ 無法載入掃描列表: {e}")
    st.stop()

st.divider()

# ==================== 掃描概覽 ====================

st.subheader("📊 掃描概覽")

try:
    results = api_client.get_scan_results(selected_scan_id)
    
    # 基本統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_vulns = len(results.get("vulnerabilities", []))
        st.metric("總漏洞數", total_vulns)
    
    with col2:
        # 計算高危漏洞
        high_risk = sum(1 for v in results.get("vulnerabilities", []) if str(v.get("severity", "")).lower() in ["high", "critical", "高危", "嚴重"])
        st.metric("高危漏洞", high_risk)
    
    with col3:
        assets = len(results.get("assets", []))
        st.metric("發現資產", assets)
    
    with col4:
        risk_level = results.get("risk_level", "Unknown")
        st.metric("風險評級", risk_level)
    
    st.divider()
    
    # ==================== 漏洞列表 ====================
    
    st.subheader("🔍 漏洞清單")
    
    vulnerabilities = results.get("vulnerabilities", [])
    
    if vulnerabilities:
        # 篩選器
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.multiselect(
                "嚴重度",
                options=["Critical", "High", "Medium", "Low", "Info"],
                default=[]
            )
        
        with col2:
            # NOTE: CWE 篩選需要從數據中提取 CWE ID
            st.multiselect("CWE 類別", options=[], default=[])
        
        with col3:
            search = st.text_input("🔎 搜尋", placeholder="CVE-2024...")
        
        # 顯示漏洞表格
        import pandas as pd
        
        vuln_data = []
        for vuln in vulnerabilities:
            # 適配不同的數據結構
            vuln_data.append({
                "ID": vuln.get("id", "N/A"),
                "標題": vuln.get("title", vuln.get("name", "N/A")),
                "嚴重度": vuln.get("severity", "Unknown"),
                "CVSS": vuln.get("cvss_score", "N/A"),
                "狀態": vuln.get("status", "Open")
            })
        
        df = pd.DataFrame(vuln_data)
        
        # 應用篩選
        if severity_filter:
            df = df[df["嚴重度"].isin(severity_filter)]
        
        if search:
            df = df[
                df["ID"].str.contains(search, case=False, na=False) |
                df["標題"].str.contains(search, case=False, na=False)
            ]
        
        # 顯示表格
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # 漏洞詳情（選擇第一個作為示例）
        if not df.empty:
            with st.expander("🔬 查看詳細資訊（第一筆）"):
                first_vuln = vulnerabilities[0]
                st.json(first_vuln)
    else:
        st.info("📭 此次掃描未發現漏洞")
    
    st.divider()
    
    # ==================== 視覺化圖表 ====================
    
    st.subheader("📈 統計圖表")
    
    # NOTE: Phase 2 實作 - 使用 Plotly 生成互動式圖表
    st.info("⚠️  圖表功能尚未實作（需要安裝 plotly 並實作圖表生成邏輯）")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 嚴重度分布")
        # NOTE: Phase 2 - 實作嚴重度分佈圓餅圖
        st.caption("需要 Plotly 實作")
    
    with col2:
        st.markdown("#### CVSS 分數分布")
        # NOTE: Phase 2 - 實作 CVSS 分數分佈直方圖
        st.caption("需要 Plotly 實作")
    
    st.divider()
    
    # ==================== 攻擊路徑圖 ====================
    
    st.subheader("🗺️ 攻擊路徑分析")
    
    attack_paths = results.get("attack_paths", [])
    
    if attack_paths:
        # NOTE: Phase 2 - 使用 Graphviz 顯示攻擊路徑
        st.info("⚠️  攻擊路徑圖尚未實作（需要安裝 graphviz 並實作圖形生成）")
    else:
        st.info("📭 此次掃描未生成攻擊路徑")
    
    st.divider()
    
    # ==================== 報告匯出 ====================
    
    st.subheader("📥 匯出報告")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "選擇格式",
            options=["JSON", "HTML", "PDF"]
        )
    
    with col2:
        if st.button("📥 匯出報告", use_container_width=True, type="primary"):
            # NOTE: Phase 3 - 實作報告匯出功能 (JSON/HTML/PDF)
            st.info(f"⚠️  報告匯出功能尚未實作（格式: {export_format}）")

except Exception as e:
    st.error(f"❌ 無法載入掃描結果: {e}")
    st.exception(e)
