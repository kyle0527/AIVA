"""
SQLi 功能模組命令處理器

符合 aiva_common 命令系統規範,實現統一的命令處理接口。
包裝現有的 SQLi 檢測功能,使其可以被 AI 通過 AICommandCenter 調用。

Usage:
    from services.aiva_common import get_command_center
    from services.features.function_sqli.command_handler import SQLiCommandHandler
    
    command_center = get_command_center()
    sqli_handler = SQLiCommandHandler()
    command_center.register_module("features.sqli", sqli_handler)
"""
