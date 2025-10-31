"""
用戶界面和用戶體驗枚舉

遵循以下標準：
- W3C Web Content Accessibility Guidelines (WCAG) 2.1
- Material Design Guidelines
- Apple Human Interface Guidelines
- Microsoft Fluent Design System
- ISO 9241-11:2018 Usability Guidelines
- Nielsen's 10 Usability Heuristics
- Web Content Accessibility Guidelines (WCAG) AA
- Section 508 Accessibility Standards
"""

from enum import Enum

# ==================== 用戶界面元素 ====================


class UIComponentType(Enum):
    """UI 組件類型 - 基於通用設計系統"""

    BUTTON = "BUTTON"
    INPUT = "INPUT"
    SELECT = "SELECT"
    CHECKBOX = "CHECKBOX"
    RADIO = "RADIO"
    TEXTAREA = "TEXTAREA"
    LABEL = "LABEL"
    LINK = "LINK"
    IMAGE = "IMAGE"
    ICON = "ICON"
    CARD = "CARD"
    MODAL = "MODAL"
    TOOLTIP = "TOOLTIP"
    DROPDOWN = "DROPDOWN"
    MENU = "MENU"
    NAVIGATION = "NAVIGATION"
    BREADCRUMB = "BREADCRUMB"
    PAGINATION = "PAGINATION"
    PROGRESS_BAR = "PROGRESS_BAR"
    SPINNER = "SPINNER"
    ALERT = "ALERT"
    NOTIFICATION = "NOTIFICATION"
    TAB = "TAB"
    ACCORDION = "ACCORDION"
    SLIDER = "SLIDER"
    TOGGLE = "TOGGLE"
    DATEPICKER = "DATEPICKER"
    TABLE = "TABLE"
    LIST = "LIST"
    TREE = "TREE"


class ButtonType(Enum):
    """按鈕類型"""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    TERTIARY = "TERTIARY"
    DANGER = "DANGER"
    WARNING = "WARNING"
    SUCCESS = "SUCCESS"
    INFO = "INFO"
    GHOST = "GHOST"
    LINK = "LINK"
    ICON = "ICON"
    FLOATING_ACTION = "FLOATING_ACTION"


class InputType(Enum):
    """輸入框類型 - 基於 HTML5 標準"""

    TEXT = "TEXT"
    PASSWORD = "PASSWORD"
    EMAIL = "EMAIL"
    NUMBER = "NUMBER"
    TEL = "TEL"
    URL = "URL"
    SEARCH = "SEARCH"
    DATE = "DATE"
    TIME = "TIME"
    DATETIME_LOCAL = "DATETIME_LOCAL"
    MONTH = "MONTH"
    WEEK = "WEEK"
    COLOR = "COLOR"
    FILE = "FILE"
    RANGE = "RANGE"
    HIDDEN = "HIDDEN"


class LayoutType(Enum):
    """佈局類型"""

    GRID = "GRID"
    FLEXBOX = "FLEXBOX"
    FIXED = "FIXED"
    FLUID = "FLUID"
    RESPONSIVE = "RESPONSIVE"
    ADAPTIVE = "ADAPTIVE"
    MASONRY = "MASONRY"
    CARD_LAYOUT = "CARD_LAYOUT"
    LIST_LAYOUT = "LIST_LAYOUT"
    SIDEBAR = "SIDEBAR"
    HEADER_FOOTER = "HEADER_FOOTER"
    SINGLE_COLUMN = "SINGLE_COLUMN"
    MULTI_COLUMN = "MULTI_COLUMN"


# ==================== 用戶交互模式 ====================


class InteractionPattern(Enum):
    """交互模式 - 基於 UX 設計模式"""

    CLICK = "CLICK"
    DOUBLE_CLICK = "DOUBLE_CLICK"
    HOVER = "HOVER"
    DRAG_AND_DROP = "DRAG_AND_DROP"
    SWIPE = "SWIPE"
    PINCH_TO_ZOOM = "PINCH_TO_ZOOM"
    SCROLL = "SCROLL"
    LONG_PRESS = "LONG_PRESS"
    KEYBOARD_SHORTCUT = "KEYBOARD_SHORTCUT"
    VOICE_COMMAND = "VOICE_COMMAND"
    GESTURE = "GESTURE"
    TOUCH = "TOUCH"
    MULTI_TOUCH = "MULTI_TOUCH"


class NavigationPattern(Enum):
    """導航模式"""

    HIERARCHICAL = "HIERARCHICAL"
    FLAT = "FLAT"
    TABBED = "TABBED"
    MODAL = "MODAL"
    DRAWER = "DRAWER"
    BOTTOM_NAVIGATION = "BOTTOM_NAVIGATION"
    BREADCRUMB = "BREADCRUMB"
    PAGINATION = "PAGINATION"
    INFINITE_SCROLL = "INFINITE_SCROLL"
    STEP_BY_STEP = "STEP_BY_STEP"
    WIZARD = "WIZARD"


class FeedbackType(Enum):
    """回饋類型"""

    VISUAL = "VISUAL"
    AUDITORY = "AUDITORY"
    HAPTIC = "HAPTIC"
    TEXTUAL = "TEXTUAL"
    ANIMATION = "ANIMATION"
    COLOR_CHANGE = "COLOR_CHANGE"
    SIZE_CHANGE = "SIZE_CHANGE"
    POSITION_CHANGE = "POSITION_CHANGE"
    LOADING_STATE = "LOADING_STATE"
    SUCCESS_STATE = "SUCCESS_STATE"
    ERROR_STATE = "ERROR_STATE"


# ==================== 響應式設計 ====================


class DeviceType(Enum):
    """設備類型"""

    MOBILE = "MOBILE"
    TABLET = "TABLET"
    DESKTOP = "DESKTOP"
    LAPTOP = "LAPTOP"
    SMARTWATCH = "SMARTWATCH"
    TV = "TV"
    KIOSK = "KIOSK"
    VR_HEADSET = "VR_HEADSET"
    AR_DEVICE = "AR_DEVICE"


class ScreenSize(Enum):
    """螢幕尺寸 - 基於響應式斷點"""

    XS = "XS"  # < 576px
    SM = "SM"  # 576px - 768px
    MD = "MD"  # 768px - 992px
    LG = "LG"  # 992px - 1200px
    XL = "XL"  # 1200px - 1400px
    XXL = "XXL"  # > 1400px


class Orientation(Enum):
    """螢幕方向"""

    PORTRAIT = "PORTRAIT"
    LANDSCAPE = "LANDSCAPE"
    AUTO = "AUTO"


class ViewportUnit(Enum):
    """視窗單位"""

    PX = "PX"  # 像素
    EM = "EM"  # 相對單位
    REM = "REM"  # 根元素相對單位
    VW = "VW"  # 視窗寬度百分比
    VH = "VH"  # 視窗高度百分比
    VMIN = "VMIN"  # 視窗最小尺寸百分比
    VMAX = "VMAX"  # 視窗最大尺寸百分比
    PERCENT = "PERCENT"  # 百分比


# ==================== 無障礙設計 ====================


class AccessibilityRole(Enum):
    """無障礙角色 - 基於 ARIA 標準"""

    BUTTON = "BUTTON"
    LINK = "LINK"
    HEADING = "HEADING"
    BANNER = "BANNER"
    MAIN = "MAIN"
    NAVIGATION = "NAVIGATION"
    COMPLEMENTARY = "COMPLEMENTARY"
    CONTENTINFO = "CONTENTINFO"
    SEARCH = "SEARCH"
    FORM = "FORM"
    REGION = "REGION"
    ARTICLE = "ARTICLE"
    SECTION = "SECTION"
    ASIDE = "ASIDE"
    DIALOG = "DIALOG"
    ALERTDIALOG = "ALERTDIALOG"
    ALERT = "ALERT"
    STATUS = "STATUS"
    PROGRESSBAR = "PROGRESSBAR"
    SLIDER = "SLIDER"
    SPINBUTTON = "SPINBUTTON"
    TEXTBOX = "TEXTBOX"
    CHECKBOX = "CHECKBOX"
    RADIO = "RADIO"
    LISTBOX = "LISTBOX"
    OPTION = "OPTION"
    COMBOBOX = "COMBOBOX"
    MENU = "MENU"
    MENUITEM = "MENUITEM"
    TAB = "TAB"
    TABLIST = "TABLIST"
    TABPANEL = "TABPANEL"
    TREE = "TREE"
    TREEITEM = "TREEITEM"
    GRID = "GRID"
    GRIDCELL = "GRIDCELL"
    ROW = "ROW"
    COLUMNHEADER = "COLUMNHEADER"
    ROWHEADER = "ROWHEADER"


class AccessibilityLevel(Enum):
    """無障礙等級 - 基於 WCAG 2.1"""

    A = "A"  # 最低等級
    AA = "AA"  # 中等等級 (推薦)
    AAA = "AAA"  # 最高等級


class ColorContrastRatio(Enum):
    """色彩對比度比例 - 基於 WCAG 標準"""

    RATIO_3_1 = "RATIO_3_1"  # 大文字 AA 等級
    RATIO_4_5_1 = "RATIO_4_5_1"  # 一般文字 AA 等級
    RATIO_7_1 = "RATIO_7_1"  # AAA 等級


class KeyboardNavigation(Enum):
    """鍵盤導航"""

    TAB = "TAB"
    SHIFT_TAB = "SHIFT_TAB"
    ENTER = "ENTER"
    SPACE = "SPACE"
    ARROW_UP = "ARROW_UP"
    ARROW_DOWN = "ARROW_DOWN"
    ARROW_LEFT = "ARROW_LEFT"
    ARROW_RIGHT = "ARROW_RIGHT"
    HOME = "HOME"
    END = "END"
    PAGE_UP = "PAGE_UP"
    PAGE_DOWN = "PAGE_DOWN"
    ESCAPE = "ESCAPE"


# ==================== 視覺設計系統 ====================


class ColorScheme(Enum):
    """色彩方案"""

    LIGHT = "LIGHT"
    DARK = "DARK"
    HIGH_CONTRAST = "HIGH_CONTRAST"
    AUTO = "AUTO"
    SYSTEM = "SYSTEM"


class ColorPalette(Enum):
    """調色盤"""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    ACCENT = "ACCENT"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    INFO = "INFO"
    NEUTRAL = "NEUTRAL"
    BACKGROUND = "BACKGROUND"
    SURFACE = "SURFACE"
    ON_PRIMARY = "ON_PRIMARY"
    ON_SECONDARY = "ON_SECONDARY"
    ON_BACKGROUND = "ON_BACKGROUND"
    ON_SURFACE = "ON_SURFACE"


class Typography(Enum):
    """字體系統"""

    DISPLAY_LARGE = "DISPLAY_LARGE"
    DISPLAY_MEDIUM = "DISPLAY_MEDIUM"
    DISPLAY_SMALL = "DISPLAY_SMALL"
    HEADLINE_LARGE = "HEADLINE_LARGE"
    HEADLINE_MEDIUM = "HEADLINE_MEDIUM"
    HEADLINE_SMALL = "HEADLINE_SMALL"
    TITLE_LARGE = "TITLE_LARGE"
    TITLE_MEDIUM = "TITLE_MEDIUM"
    TITLE_SMALL = "TITLE_SMALL"
    BODY_LARGE = "BODY_LARGE"
    BODY_MEDIUM = "BODY_MEDIUM"
    BODY_SMALL = "BODY_SMALL"
    LABEL_LARGE = "LABEL_LARGE"
    LABEL_MEDIUM = "LABEL_MEDIUM"
    LABEL_SMALL = "LABEL_SMALL"


class SpacingSystem(Enum):
    """間距系統 - 基於 8px 網格"""

    NONE = "NONE"  # 0px
    XS = "XS"  # 4px
    SM = "SM"  # 8px
    MD = "MD"  # 16px
    LG = "LG"  # 24px
    XL = "XL"  # 32px
    XXL = "XXL"  # 48px
    XXXL = "XXXL"  # 64px


class Elevation(Enum):
    """陰影層級 - 基於 Material Design"""

    LEVEL_0 = "LEVEL_0"  # 無陰影
    LEVEL_1 = "LEVEL_1"  # 輕微陰影
    LEVEL_2 = "LEVEL_2"  # 小陰影
    LEVEL_3 = "LEVEL_3"  # 中陰影
    LEVEL_4 = "LEVEL_4"  # 大陰影
    LEVEL_5 = "LEVEL_5"  # 重陰影


# ==================== 動畫和過渡 ====================


class AnimationType(Enum):
    """動畫類型"""

    FADE = "FADE"
    SLIDE = "SLIDE"
    SCALE = "SCALE"
    ROTATE = "ROTATE"
    BOUNCE = "BOUNCE"
    ELASTIC = "ELASTIC"
    SPRING = "SPRING"
    MORPH = "MORPH"
    PARALLAX = "PARALLAX"
    PARTICLE = "PARTICLE"


class EasingFunction(Enum):
    """緩動函數"""

    LINEAR = "LINEAR"
    EASE = "EASE"
    EASE_IN = "EASE_IN"
    EASE_OUT = "EASE_OUT"
    EASE_IN_OUT = "EASE_IN_OUT"
    CUBIC_BEZIER = "CUBIC_BEZIER"
    BOUNCE = "BOUNCE"
    ELASTIC = "ELASTIC"
    BACK = "BACK"


class AnimationDuration(Enum):
    """動畫持續時間"""

    INSTANT = "INSTANT"  # 0ms
    FAST = "FAST"  # 100ms
    NORMAL = "NORMAL"  # 300ms
    SLOW = "SLOW"  # 500ms
    EXTRA_SLOW = "EXTRA_SLOW"  # 1000ms


class TransitionTrigger(Enum):
    """過渡觸發器"""

    HOVER = "HOVER"
    CLICK = "CLICK"
    FOCUS = "FOCUS"
    SCROLL = "SCROLL"
    LOAD = "LOAD"
    RESIZE = "RESIZE"
    STATE_CHANGE = "STATE_CHANGE"
    USER_INPUT = "USER_INPUT"


# ==================== 表單設計 ====================


class FormLayout(Enum):
    """表單佈局"""

    VERTICAL = "VERTICAL"
    HORIZONTAL = "HORIZONTAL"
    INLINE = "INLINE"
    MULTI_COLUMN = "MULTI_COLUMN"
    STEPPED = "STEPPED"
    TABBED = "TABBED"


class ValidationState(Enum):
    """驗證狀態"""

    VALID = "VALID"
    INVALID = "INVALID"
    PENDING = "PENDING"
    UNTOUCHED = "UNTOUCHED"
    TOUCHED = "TOUCHED"
    DIRTY = "DIRTY"
    PRISTINE = "PRISTINE"


class ValidationTiming(Enum):
    """驗證時機"""

    ON_SUBMIT = "ON_SUBMIT"
    ON_BLUR = "ON_BLUR"
    ON_CHANGE = "ON_CHANGE"
    ON_FOCUS = "ON_FOCUS"
    REAL_TIME = "REAL_TIME"
    DEBOUNCED = "DEBOUNCED"


class FieldSize(Enum):
    """欄位尺寸"""

    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    FULL_WIDTH = "FULL_WIDTH"
    AUTO = "AUTO"


# ==================== 數據可視化 ====================


class ChartType(Enum):
    """圖表類型"""

    LINE = "LINE"
    BAR = "BAR"
    COLUMN = "COLUMN"
    AREA = "AREA"
    PIE = "PIE"
    DOUGHNUT = "DOUGHNUT"
    SCATTER = "SCATTER"
    BUBBLE = "BUBBLE"
    HEATMAP = "HEATMAP"
    TREEMAP = "TREEMAP"
    HISTOGRAM = "HISTOGRAM"
    BOX_PLOT = "BOX_PLOT"
    VIOLIN_PLOT = "VIOLIN_PLOT"
    CANDLESTICK = "CANDLESTICK"
    GAUGE = "GAUGE"
    RADAR = "RADAR"
    FUNNEL = "FUNNEL"
    WATERFALL = "WATERFALL"
    SANKEY = "SANKEY"
    NETWORK = "NETWORK"


class DataVisualizationPurpose(Enum):
    """數據可視化目的"""

    COMPARISON = "COMPARISON"  # 比較
    COMPOSITION = "COMPOSITION"  # 組成
    DISTRIBUTION = "DISTRIBUTION"  # 分佈
    RELATIONSHIP = "RELATIONSHIP"  # 關係
    TREND = "TREND"  # 趨勢
    RANKING = "RANKING"  # 排名
    PART_TO_WHOLE = "PART_TO_WHOLE"  # 部分與整體
    CORRELATION = "CORRELATION"  # 相關性
    GEOGRAPHIC = "GEOGRAPHIC"  # 地理空間
    FLOW = "FLOW"  # 流程


class InteractionType(Enum):
    """交互類型"""

    ZOOM = "ZOOM"
    PAN = "PAN"
    FILTER = "FILTER"
    DRILL_DOWN = "DRILL_DOWN"
    DRILL_UP = "DRILL_UP"
    BRUSH = "BRUSH"
    SELECT = "SELECT"
    HIGHLIGHT = "HIGHLIGHT"
    TOOLTIP = "TOOLTIP"
    CROSSFILTER = "CROSSFILTER"


# ==================== 用戶體驗指標 ====================


class UXMetric(Enum):
    """用戶體驗指標"""

    TASK_SUCCESS_RATE = "TASK_SUCCESS_RATE"
    TASK_COMPLETION_TIME = "TASK_COMPLETION_TIME"
    ERROR_RATE = "ERROR_RATE"
    LEARNABILITY = "LEARNABILITY"
    MEMORABILITY = "MEMORABILITY"
    USER_SATISFACTION = "USER_SATISFACTION"
    ACCESSIBILITY_SCORE = "ACCESSIBILITY_SCORE"
    USABILITY_SCORE = "USABILITY_SCORE"
    NET_PROMOTER_SCORE = "NET_PROMOTER_SCORE"
    CONVERSION_RATE = "CONVERSION_RATE"
    BOUNCE_RATE = "BOUNCE_RATE"
    SESSION_DURATION = "SESSION_DURATION"
    PAGE_VIEWS = "PAGE_VIEWS"
    CLICK_THROUGH_RATE = "CLICK_THROUGH_RATE"


class UserFeedbackType(Enum):
    """用戶回饋類型"""

    SURVEY = "SURVEY"
    INTERVIEW = "INTERVIEW"
    USABILITY_TEST = "USABILITY_TEST"
    A_B_TEST = "A_B_TEST"
    ANALYTICS = "ANALYTICS"
    HEATMAP = "HEATMAP"
    SESSION_RECORDING = "SESSION_RECORDING"
    BUG_REPORT = "BUG_REPORT"
    FEATURE_REQUEST = "FEATURE_REQUEST"
    RATING = "RATING"
    REVIEW = "REVIEW"


class PersonaType(Enum):
    """用戶角色類型"""

    PRIMARY = "PRIMARY"
    SECONDARY = "SECONDARY"
    NEGATIVE = "NEGATIVE"
    SERVED = "SERVED"
    CUSTOMER = "CUSTOMER"
    ANTI_PERSONA = "ANTI_PERSONA"
