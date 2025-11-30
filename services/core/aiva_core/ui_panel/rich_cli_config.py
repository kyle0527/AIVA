"""
AIVA Rich CLI 配置檔案

定義 Rich CLI 的外觀、主題、佈局和行為設定。
整合 HackingTool 的視覺設計元素。
"""

from typing import Dict, List, Tuple, Any
from rich.theme import Theme
from rich import box

# =============================================================================
# 視覺主題配置
# =============================================================================

# AIVA 品牌色彩
AIVA_COLORS = {
    "primary": "#00D4AA",      # AIVA 標誌青綠色
    "secondary": "#7B61FF",    # 紫色輔助色
    "success": "#00C851",      # 成功狀態綠色
    "warning": "#FF8800",      # 警告狀態橙色
    "error": "#FF4444",        # 錯誤狀態紅色
    "info": "#33B5E5",         # 資訊狀態藍色
    "muted": "#666666",        # 次要文字灰色
    "accent": "#FFD700",       # 強調色金色
    "background": "#1a1a1a",   # 深色背景
    "surface": "#2d2d2d",      # 介面表面色
}

# Rich 主題定義
RICH_THEME = Theme({
    "aiva.primary": AIVA_COLORS["primary"],
    "aiva.secondary": AIVA_COLORS["secondary"],
    "aiva.success": AIVA_COLORS["success"],
    "aiva.warning": AIVA_COLORS["warning"],
    "aiva.error": AIVA_COLORS["error"],
    "aiva.info": AIVA_COLORS["info"],
    "aiva.muted": AIVA_COLORS["muted"],
    "aiva.accent": AIVA_COLORS["accent"],
    "aiva.bg": AIVA_COLORS["background"],
    "aiva.surface": AIVA_COLORS["surface"],
    
    # 組合樣式
    "aiva.title": f"bold {AIVA_COLORS['primary']}",
    "aiva.subtitle": f"italic {AIVA_COLORS['muted']}",
    "aiva.highlight": f"bold {AIVA_COLORS['accent']}",
    "aiva.status.online": f"bold {AIVA_COLORS['success']}",
    "aiva.status.offline": f"bold {AIVA_COLORS['error']}",
    "aiva.status.pending": f"bold {AIVA_COLORS['warning']}",
})

# =============================================================================
# 版面配置配置
# =============================================================================

# 控制台設定
CONSOLE_CONFIG = {
    "width": 120,
    "theme": RICH_THEME,
    "force_terminal": True,
    "legacy_windows": False,
}

# 表格樣式配置
TABLE_STYLES = {
    "main_menu": {
        "box": box.MINIMAL_DOUBLE_HEAD,
        "title_style": "bold aiva.primary", 
        "header_style": "bold aiva.accent",
        "row_styles": ["", "aiva.surface"],
    },
    "capability_list": {
        "box": box.SIMPLE,
        "title_style": "bold aiva.info",
        "header_style": "bold aiva.accent",
        "row_styles": ["", "dim"],
    },
    "scan_results": {
        "box": box.DOUBLE,
        "title_style": "bold aiva.success",
        "header_style": "bold",
        "row_styles": ["", "aiva.muted"],
    },
    "status": {
        "box": box.SIMPLE,
        "title_style": "bold aiva.info",
        "header_style": "bold aiva.accent",
        "row_styles": None,
    }
}

# 面板樣式配置
PANEL_STYLES = {
    "banner": {
        "border_style": "aiva.primary",
        "box": box.DOUBLE,
        "title_style": "bold aiva.accent",
        "subtitle_style": "aiva.muted",
    },
    "info": {
        "border_style": "aiva.info",
        "box": box.SIMPLE,
        "title_style": "bold aiva.info",
    },
    "success": {
        "border_style": "aiva.success",
        "box": box.SIMPLE,
        "title_style": "bold aiva.success",
    },
    "warning": {
        "border_style": "aiva.warning",
        "box": box.SIMPLE,
        "title_style": "bold aiva.warning",
    },
    "error": {
        "border_style": "aiva.error",
        "box": box.SIMPLE,
        "title_style": "bold aiva.error",
    }
}

# =============================================================================
# 功能選單配置
# =============================================================================

# 主選單項目
MAIN_MENU_ITEMS = [
    {
        "key": "1",
        "name": "漏洞掃描",
        "description": "啟動 AI 驅動的安全評估",
        "icon": "🔍",
        "handler": "handle_vulnerability_scan",
        "status": "active"
    },
    {
        "key": "2", 
        "name": "能力管理",
        "description": "管理註冊的安全工具和能力",
        "icon": "⚙️",
        "handler": "handle_capability_management",
        "status": "active"
    },
    {
        "key": "3",
        "name": "AI 對話",
        "description": "與 AIVA AI 引擎互動",
        "icon": "🤖",
        "handler": "handle_ai_interaction", 
        "status": "active"
    },
    {
        "key": "4",
        "name": "AI 能力查詢",
        "description": "查詢 AIVA 的功能與能力",
        "icon": "🔍",
        "handler": "handle_ai_capability_query",
        "status": "active"
    },
    {
        "key": "5",
        "name": "工具集成",
        "description": "整合新的安全工具",
        "icon": "🔧",
        "handler": "handle_tool_integration",
        "status": "development"
    },
    {
        "key": "6",
        "name": "系統監控",
        "description": "查看系統狀態和日誌",
        "icon": "📊",
        "handler": "handle_system_monitoring",
        "status": "development"
    },
    {
        "key": "7",
        "name": "設定配置",
        "description": "調整 AIVA 系統設定",
        "icon": "⚙️",
        "handler": "handle_settings",
        "status": "development"
    },
    {
        "key": "8",
        "name": "報告生成",
        "description": "生成掃描和評估報告",
        "icon": "📄",
        "handler": "handle_report_generation",
        "status": "development"
    },
    {
        "key": "8",
        "name": "幫助文檔",
        "description": "查看使用指南和 API 文檔",
        "icon": "❓",
        "handler": "show_help",
        "status": "active"
    },
    {
        "key": "9",
        "name": "關於 AIVA",
        "description": "版本資訊和開發團隊",
        "icon": "ℹ️",
        "handler": "show_about",
        "status": "active"
    },
    {
        "key": "0",
        "name": "退出",
        "description": "安全退出 AIVA CLI",
        "icon": "🚪",
        "handler": "exit_application",
        "status": "active"
    }
]

# 掃描類型配置
SCAN_TYPES = [
    {
        "id": "quick",
        "name": "快速掃描",
        "description": "基本端口掃描和服務識別",
        "duration": "1-3 分鐘",
        "tools": ["nmap", "service_detection"],
        "complexity": "low"
    },
    {
        "id": "standard", 
        "name": "標準掃描",
        "description": "標準漏洞檢測和常見攻擊向量",
        "duration": "5-15 分鐘",
        "tools": ["nmap", "nikto", "sqlmap", "xss_scanner"],
        "complexity": "medium"
    },
    {
        "id": "deep",
        "name": "深度掃描", 
        "description": "全面安全評估和高級威脅檢測",
        "duration": "30-60 分鐘",
        "tools": ["comprehensive_scan", "ai_analysis", "custom_payloads"],
        "complexity": "high"
    },
    {
        "id": "custom",
        "name": "自定義掃描",
        "description": "用戶自定義掃描策略",
        "duration": "可變",
        "tools": ["user_selected"],
        "complexity": "variable"
    }
]

# =============================================================================
# 狀態指示器配置
# =============================================================================

STATUS_INDICATORS = {
    "online": {"symbol": "●", "style": "aiva.success", "text": "在線"},
    "offline": {"symbol": "●", "style": "aiva.error", "text": "離線"},
    "pending": {"symbol": "◐", "style": "aiva.warning", "text": "處理中"},
    "active": {"symbol": "✓", "style": "aiva.success", "text": "活躍"},
    "inactive": {"symbol": "○", "style": "aiva.muted", "text": "非活躍"},
    "error": {"symbol": "✗", "style": "aiva.error", "text": "錯誤"},
    "warning": {"symbol": "⚠", "style": "aiva.warning", "text": "警告"},
    "info": {"symbol": "ℹ", "style": "aiva.info", "text": "資訊"},
}

# =============================================================================
# 進度指示器配置  
# =============================================================================

PROGRESS_STYLES = {
    "scan": {
        "spinner": "dots",
        "color": "aiva.primary",
        "description_style": "aiva.info",
    },
    "install": {
        "spinner": "line",
        "color": "aiva.warning", 
        "description_style": "aiva.muted",
    },
    "analysis": {
        "spinner": "arc",
        "color": "aiva.accent",
        "description_style": "aiva.info",
    }
}

# =============================================================================
# 鍵盤快捷鍵配置
# =============================================================================

KEYBOARD_SHORTCUTS = {
    "quit": ["ctrl+c", "q"],
    "help": ["h", "?"],
    "refresh": ["r", "f5"],
    "back": ["b", "esc"], 
    "home": ["ctrl+h"],
    "clear": ["ctrl+l"],
}

# =============================================================================
# 動畫和過渡效果配置
# =============================================================================

ANIMATION_CONFIG = {
    "enable_animations": True,
    "transition_delay": 0.3,
    "typing_speed": 0.05,
    "fade_duration": 0.5,
}

# =============================================================================
# 輸出格式配置
# =============================================================================

OUTPUT_FORMATS = {
    "timestamp_format": "%Y-%m-%d %H:%M:%S",
    "date_format": "%Y-%m-%d",
    "log_format": "[{timestamp}] {level}: {message}",
    "max_table_width": 100,
    "max_description_length": 80,
}

# =============================================================================
# HackingTool 整合配置
# =============================================================================

HACKINGTOOL_CONFIG = {
    "enable_hackingtool_ui": True,
    "inherit_color_scheme": True,
    "use_hackingtool_panels": True,
    "enable_project_urls": True,
    "show_installation_status": True,
}