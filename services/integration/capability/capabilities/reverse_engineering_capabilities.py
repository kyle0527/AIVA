#!/usr/bin/env python3
"""
反向工程工具能力定義
基於 aiva_common v2.0 規範的標準化能力註冊
"""

from services.aiva_common.schemas.capability import InputParameter, OutputParameter
from services.integration.capability.models import CapabilityRecord, CapabilityType
from services.aiva_common.enums import ProgrammingLanguage, TaskStatus

# ================================================
# Reverse Engineering Tools Capabilities
# ================================================

ANDROGUARD_CAPABILITY = CapabilityRecord(
    id="reverse_engineering.androguard",
    name="Androguard",
    description="Reverse engineering, Malware and goodware analysis of Android applications. "
                "Supports APK, DEX, and ODEX file formats.",
    version="1.0.0",
    module="integration",
    language=ProgrammingLanguage.PYTHON,
    entrypoint="services.integration.capability.reverse_engineering_tools:AndroGuard",
    capability_type=CapabilityType.UTILITY,
    
    # 輸入參數（不可運行，僅安裝）
    inputs=[],
    outputs=[],
    
    # 標籤化分類
    tags=["android", "reverse-engineering", "malware-analysis", "python", "apk", "dex"],
    category="Reverse Engineering",
    priority=70,
    
    # 運行時配置
    timeout_seconds=300,
    retry_count=3,
    status=TaskStatus.PENDING,
    
    # 額外配置
    config={
        "project_url": "https://github.com/androguard/androguard",
        "install_command": "sudo pip3 install -U androguard",
        "runnable": False,
        "requires_manual_setup": True
    }
)

APK2GOLD_CAPABILITY = CapabilityRecord(
    id="reverse_engineering.apk2gold",
    name="Apk2Gold",
    description="CLI tool for decompiling Android apps to Java source code. "
                "Converts APK files to readable Java code for security analysis.",
    version="1.0.0",
    module="integration",
    language=ProgrammingLanguage.PYTHON,
    entrypoint="services.integration.capability.reverse_engineering_tools:Apk2Gold",
    capability_type=CapabilityType.UTILITY,
    
    # 輸入參數（含參數提示）
    inputs=[
        InputParameter(
            name="apk_file",
            type="string",
            required=True,
            description="Path to the APK file to decompile (must end with .apk)",
            default=None,
            validation_rules={
                "pattern": r".*\.apk$",
                "file_must_exist": True
            }
        )
    ],
    outputs=[
        OutputParameter(
            name="decompiled_code",
            type="directory",
            description="Directory containing decompiled Java source code"
        ),
        OutputParameter(
            name="manifest",
            type="file",
            description="AndroidManifest.xml file"
        )
    ],
    
    # 標籤化分類
    tags=["android", "decompiler", "apk", "java", "security-analysis"],
    category="Reverse Engineering",
    priority=80,
    
    # 運行時配置
    timeout_seconds=600,  # 反編譯可能需要較長時間
    retry_count=2,
    status=TaskStatus.PENDING,
    
    # 額外配置
    config={
        "project_url": "https://github.com/lxdvs/apk2gold",
        "install_commands": [
            "sudo git clone https://github.com/lxdvs/apk2gold.git",
            "cd apk2gold && sudo bash make.sh"
        ],
        "runnable": True
    }
)

JADX_CAPABILITY = CapabilityRecord(
    id="reverse_engineering.jadx",
    name="JadX",
    description="Dex to Java decompiler with GUI and CLI support. Decompiles Dalvik bytecode "
                "to Java classes from APK, dex, aar and zip files. Also decodes AndroidManifest.xml "
                "and other resources from resources.arsc.",
    version="1.0.0",
    module="integration",
    language=ProgrammingLanguage.PYTHON,
    entrypoint="services.integration.capability.reverse_engineering_tools:Jadx",
    capability_type=CapabilityType.UTILITY,
    
    # 輸入參數（不可運行，僅安裝）
    inputs=[],
    outputs=[],
    
    # 標籤化分類
    tags=["android", "decompiler", "dex", "java", "apk", "gui", "dalvik"],
    category="Reverse Engineering",
    priority=85,
    
    # 運行時配置
    timeout_seconds=300,
    retry_count=3,
    status=TaskStatus.PENDING,
    
    # 額外配置
    config={
        "project_url": "https://github.com/skylot/jadx",
        "install_commands": [
            "sudo git clone https://github.com/skylot/jadx.git",
            "cd jadx && ./gradlew dist"
        ],
        "runnable": False,
        "requires_manual_setup": True,
        "supports_gui": True
    }
)

# ================================================
# 能力列表
# ================================================

REVERSE_ENGINEERING_CAPABILITIES = [
    ANDROGUARD_CAPABILITY,
    APK2GOLD_CAPABILITY,
    JADX_CAPABILITY
]

# ================================================
# 輔助函數
# ================================================

def get_capability_by_id(capability_id: str) -> CapabilityRecord | None:
    """根據ID獲取能力記錄"""
    for cap in REVERSE_ENGINEERING_CAPABILITIES:
        if cap.id == capability_id:
            return cap
    return None

def get_capabilities_by_tag(tag: str) -> list[CapabilityRecord]:
    """根據標籤獲取能力列表"""
    return [cap for cap in REVERSE_ENGINEERING_CAPABILITIES if tag in cap.tags]

def get_runnable_capabilities() -> list[CapabilityRecord]:
    """獲取所有可運行的能力"""
    return [cap for cap in REVERSE_ENGINEERING_CAPABILITIES 
            if cap.config.get("runnable", False)]
