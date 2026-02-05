"""
AIVA UI 主題配置

定義統一的顏色主題和視覺樣式
"""

from rich.theme import Theme
from rich import box

# AIVA 品牌色彩方案
COLORS = {
    "primary": "#00D4AA",      # AIVA 青綠色
    "secondary": "#7B61FF",    # 紫色
    "success": "#00C851",      # 成功綠
    "warning": "#FF8800",      # 警告橙
    "error": "#FF4444",        # 錯誤紅
    "info": "#33B5E5",         # 資訊藍
    "muted": "#666666",        # 次要灰
    "accent": "#FFD700",       # 強調金
    "bg": "#1a1a1a",          # 深色背景
    "surface": "#2d2d2d",     # 表面色
}

# Rich 主題定義
AIVA_THEME = Theme({
    # 基本顏色
    "aiva.primary": COLORS["primary"],
    "aiva.secondary": COLORS["secondary"],
    "aiva.success": COLORS["success"],
    "aiva.warning": COLORS["warning"],
    "aiva.error": COLORS["error"],
    "aiva.info": COLORS["info"],
    "aiva.muted": COLORS["muted"],
    "aiva.accent": COLORS["accent"],
    
    # 組合樣式
    "aiva.title": f"bold {COLORS['primary']}",
    "aiva.subtitle": f"italic {COLORS['muted']}",
    "aiva.highlight": f"bold {COLORS['accent']}",
    
    # 狀態樣式
    "status.online": f"bold {COLORS['success']}",
    "status.offline": f"bold {COLORS['error']}",
    "status.pending": f"bold {COLORS['warning']}",
    
    # 特殊標記
    "mark.success": f"bold {COLORS['success']}",
    "mark.fail": f"bold {COLORS['error']}",
    "mark.info": f"bold {COLORS['info']}",
})

# 表格樣式配置 - 使用直接顏色值而非主題變數
TABLE_STYLES = {
    "main_menu": {
        "box": box.ROUNDED,
        "border_style": COLORS["primary"],
        "header_style": f"bold {COLORS['accent']}",
        "title_style": f"bold {COLORS['primary']}",
    },
    "data": {
        "box": box.SIMPLE,
        "border_style": COLORS["info"],
        "header_style": "bold",
        "show_lines": False,
    },
    "status": {
        "box": box.MINIMAL,
        "border_style": COLORS["muted"],
        "header_style": f"bold {COLORS['primary']}",
    }
}

# 面板樣式配置 - 使用直接顏色值
PANEL_STYLES = {
    "banner": {
        "box": box.DOUBLE,
        "border_style": COLORS["primary"],
        "padding": (1, 2),
    },
    "info": {
        "box": box.ROUNDED,
        "border_style": COLORS["info"],
        "padding": (1, 2),
    },
    "success": {
        "box": box.ROUNDED,
        "border_style": COLORS["success"],
        "padding": (1, 2),
    },
    "warning": {
        "box": box.ROUNDED,
        "border_style": COLORS["warning"],
        "padding": (1, 2),
    },
    "error": {
        "box": box.ROUNDED,
        "border_style": COLORS["error"],
        "padding": (1, 2),
    }
}

# 圖標配置
ICONS = {
    "success": "✓",
    "error": "✗",
    "warning": "⚠",
    "info": "ℹ",
    "bullet": "●",
    "arrow": "►",
    "check": "[OK]",
    "cross": "[ERROR]",
    "star": "[STAR]",
    "sparkle": "[SPARKLE]",
    "rocket": "[START]",
    "shield": "[SHIELD]",
    "gear": "[CONFIG]",
    "magnify": "[SEARCH]",
}
