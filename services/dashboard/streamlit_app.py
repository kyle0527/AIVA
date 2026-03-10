"""AIVA Security Dashboard - Streamlit 主程式

此為 AIVA 系統的 Web 管理界面主程式。

功能：
- 掃描控制台
- 即時監控
- 結果分析
- 歷史記錄
"""

import sys
from pathlib import Path

# 添加專案路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "services"))

import streamlit as st

# 必須在最開始設定頁面配置
st.set_page_config(
    page_title="AIVA Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
from pathlib import Path

# 添加專案根目錄到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# NOTE: Pylance 可能報告 "無法解析匯入" 警告
# 這是因為 sys.path 在運行時動態添加，Pylance 靜態分析時無法識別
# 實際運行時不會有問題
from services.dashboard.config import config
from services.dashboard.api_client import AIVAApiClient

# ==================== 初始化 ====================

# 初始化 API 客戶端（使用 Session State 確保單例）
if 'api_client' not in st.session_state:
    st.session_state.api_client = AIVAApiClient(
        base_url=config.api_base_url,
        timeout=config.api_timeout,
        retry_attempts=config.api_retry_attempts
    )

api_client: AIVAApiClient = st.session_state.api_client


# ==================== 側邊欄 ====================

def render_sidebar():
    """渲染側邊欄"""
    with st.sidebar:
        # Logo 和標題
        st.title("🛡️ AIVA Dashboard")
        st.caption("AI-Powered Vulnerability Intelligence & Assessment")
        st.divider()
        
        # API 連線狀態
        st.subheader("🔌 API 狀態")
        
        try:
            if api_client.is_api_available():
                st.success("✅ API 連線正常")
                health = api_client.health_check()
                
                # 顯示各組件狀態
                with st.expander("查看詳情"):
                    for component, status in health.get("components", {}).items():
                        st.text(f"• {component}: {status}")
            else:
                st.error("❌ API 無法連線")
                st.caption(f"請確認 API 服務運行在 {config.api_base_url}")
        except Exception as e:
            st.error("❌ API 連線失敗")
            st.caption(str(e))
        
        st.divider()
        
        # 系統資訊
        st.subheader("ℹ️ 系統資訊")
        st.caption(f"**API URL**: {config.api_base_url}")
        st.caption(f"**主題**: {config.dashboard_theme}")
        st.caption("**版本**: v4.4.1")


# ==================== 主頁面 ====================

def render_home_page():
    """渲染首頁"""
    # 標題
    st.title("🛡️ AIVA Security Dashboard")
    st.markdown("### AI-Powered Vulnerability Intelligence & Assessment")
    st.divider()
    
    # 快速統計（使用列佈局）
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        # 獲取掃描統計
        all_scans = api_client.list_scans(limit=1000)
        total_scans = all_scans.get("total", 0)
        
        # 運行中的掃描
        running_scans = api_client.list_scans(status="running", limit=100)
        running_count = running_scans.get("total", 0)
        
        # 今日完成（簡化處理）
        completed_scans = api_client.list_scans(status="completed", limit=100)
        completed_count = completed_scans.get("total", 0)
        
        # 顯示統計
        with col1:
            st.metric("📊 總掃描數", total_scans)
        
        with col2:
            st.metric("▶️ 執行中", running_count, delta=f"+{running_count}" if running_count > 0 else None)
        
        with col3:
            st.metric("✅ 已完成", completed_count)
        
        with col4:
            st.metric("⚠️ 高危漏洞", 0, help="需要實作漏洞統計")
    
    except Exception as e:
        st.error(f"❌ 無法載入統計資訊: {e}")
        
        # 顯示預設值
        with col1:
            st.metric("📊 總掃描數", "N/A")
        with col2:
            st.metric("▶️ 執行中", "N/A")
        with col3:
            st.metric("✅ 已完成", "N/A")
        with col4:
            st.metric("⚠️ 高危漏洞", "N/A")
    
    st.divider()
    
    # 功能導航卡片
    st.subheader("📋 功能導航")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container():
            st.markdown("#### 🎯 掃描控制台")
            st.markdown("啟動新的掃描任務，配置掃描參數，管理執行中的掃描。")
            st.page_link("pages/1_🎯_掃描控制台.py", label="前往 →", use_container_width=True)
        
        with st.container():
            st.markdown("#### 🔍 結果分析")
            st.markdown("查看漏洞詳情，分析攻擊路徑，生成安全報告。")
            st.page_link("pages/3_🔍_結果分析.py", label="前往 →", use_container_width=True)
    
    with col2:
        with st.container():
            st.markdown("#### 📊 即時監控")
            st.markdown("即時查看掃描進度，監控任務狀態，查看執行日誌。")
            st.page_link("pages/2_📊_即時監控.py", label="前往 →", use_container_width=True)
        
        with st.container():
            st.markdown("#### 📜 歷史記錄")
            st.markdown("查詢過往掃描記錄，對比結果，匯出報告。")
            st.page_link("pages/4_📜_歷史記錄.py", label="前往 →", use_container_width=True)
    
    st.divider()
    
    # 最近掃描
    st.subheader("🕒 最近掃描")
    
    try:
        recent_scans = api_client.list_scans(limit=5)
        scans = recent_scans.get("scans", [])
        
        if scans:
            # 準備表格數據
            import pandas as pd
            
            table_data = []
            for scan in scans:
                table_data.append({
                    "掃描 ID": scan.get("scan_id", "N/A"),
                    "目標": scan.get("target", "N/A"),
                    "狀態": scan.get("status", "unknown"),
                    "開始時間": scan.get("start_time", "N/A"),
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("📭 暫無掃描記錄")
    
    except Exception as e:
        st.error(f"❌ 無法載入最近掃描: {e}")
    
    st.divider()
    
    # 使用說明
    with st.expander("📖 使用說明"):
        st.markdown("""
        ### 快速開始
        
        1. **啟動 AIVA Core API**
           ```bash
           cd services/core/aiva_core/service_backbone/api
           python app.py
           ```
        
        2. **訪問 Dashboard**
           ```bash
           streamlit run services/dashboard/streamlit_app.py
           ```
        
        3. **開始掃描**
           - 前往「掃描控制台」
           - 輸入目標 URL
           - 點擊「🚀 啟動掃描」
        
        ### 功能概覽
        
        - **🎯 掃描控制台**: 管理掃描任務
        - **📊 即時監控**: 查看執行進度和日誌
        - **🔍 結果分析**: 分析漏洞和風險
        - **📜 歷史記錄**: 管理掃描記錄
        
        ### 技術支援
        
        如遇問題，請查閱 `docs/UI_DASHBOARD_IMPLEMENTATION_PLAN.md`
        """)


# ==================== 主程式 ====================

def main():
    """主程式入口"""
    # 渲染側邊欄
    render_sidebar()
    
    # 渲染主頁面
    render_home_page()


if __name__ == "__main__":
    main()
