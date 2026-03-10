"""
Reverse Engineering Manager
"""

import hashlib
import logging
import os
import re
import shutil
import struct
import subprocess
import zipfile
from datetime import datetime
from typing import Optional

from .models import (
    AnalysisMode,
    APKInfo,
    BinaryInfo,
    BinaryType,
    CodeAnalysisResult,
    DecompileConfig,
    DecompileResult,
    DecompilerType,
    MalwareAnalysisResult,
    ThreatLevel,
)

logger = logging.getLogger(__name__)

# ── Magic bytes → (BinaryType, architecture) 映射 ──
_ELF_MAGIC = b"\x7fELF"
_PE_MAGIC = b"MZ"
_MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce": ("macho", "x86"),
    b"\xce\xfa\xed\xfe": ("macho", "x86"),
    b"\xfe\xed\xfa\xcf": ("macho", "x64"),
    b"\xcf\xfa\xed\xfe": ("macho", "x64"),
}
# 可疑 API/字串模式（惡意軟體指示器）
_SUSPICIOUS_APIS = [
    b"CreateRemoteThread", b"VirtualAllocEx", b"WriteProcessMemory",
    b"SetWindowsHookEx", b"keybd_event", b"GetAsyncKeyState",
    b"InternetOpen", b"URLDownloadToFile", b"WinExec", b"ShellExecute",
    b"RegSetValueEx", b"CryptEncrypt", b"socket", b"connect",
    b"CreateService", b"OpenSCManager",
]


class ReverseEngineeringManager:
    """逆向工程管理器"""
    
    def __init__(self):
        logger.info("ReverseEngineeringManager initialized")
    
    async def analyze_binary(
        self,
        file_path: str,
        mode: AnalysisMode = AnalysisMode.STATIC
    ) -> BinaryInfo:
        """分析二進制檔案（magic bytes 識別類型/架構，SHA-256 雜湊）"""
        try:
            logger.info(f"Analyzing binary: {file_path} [mode={mode.value}]")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            file_size = os.path.getsize(file_path)

            # SHA-256
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as fh:
                header = fh.read(8)
                fh.seek(0)
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest()

            # 類型 / 架構識別
            binary_type = BinaryType.EXE
            architecture = "unknown"
            is_packed = False

            if header[:4] == _ELF_MAGIC and len(header) >= 5:
                binary_type = BinaryType.ELF
                ei_class = header[4]
                architecture = "x64" if ei_class == 2 else "x86"
            elif header[:2] == _PE_MAGIC:
                ext = os.path.splitext(file_path)[1].lower()
                binary_type = BinaryType.DLL if ext == ".dll" else BinaryType.EXE
                # 讀取 PE Machine 欄位
                with open(file_path, "rb") as fh:
                    fh.seek(60)
                    pe_offset_raw = fh.read(4)
                    if len(pe_offset_raw) == 4:
                        pe_offset = struct.unpack("<I", pe_offset_raw)[0]
                        fh.seek(pe_offset + 4)
                        machine_raw = fh.read(2)
                        if len(machine_raw) == 2:
                            machine = struct.unpack("<H", machine_raw)[0]
                            architecture = "x64" if machine == 0x8664 else "ARM" if machine in (0x01C0, 0xAA64) else "x86"
            else:
                for magic, (btype, arch) in _MACHO_MAGICS.items():
                    if header[:4] == magic:
                        binary_type = BinaryType.MACH_O
                        architecture = arch
                        break
                else:
                    if file_path.lower().endswith(".apk"):
                        binary_type = BinaryType.APK
                    elif file_path.lower().endswith(".so"):
                        binary_type = BinaryType.SO

            # 簡單 packer 啟發（高entropy區間 → UPX/加殼）
            with open(file_path, "rb") as fh:
                sample = fh.read(4096)
            byte_counts = [sample.count(bytes([i])) for i in range(256)]
            total = sum(byte_counts)
            import math
            entropy = -sum((c / total) * math.log2(c / total) for c in byte_counts if c > 0) if total else 0.0
            if entropy > 7.2:
                is_packed = True

            logger.info(f"Binary analyzed: {binary_type.value}, arch={architecture}, hash={file_hash[:12]}…, packed={is_packed}")
            return BinaryInfo(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                file_size=file_size,
                file_hash=file_hash,
                binary_type=binary_type,
                architecture=architecture,
                is_packed=is_packed,
                metadata={"entropy": round(entropy, 4), "mode": mode.value},
            )

        except Exception as e:
            logger.error(f"Binary analysis failed: {str(e)}", exc_info=True)
            raise
    
    async def analyze_apk(
        self,
        apk_path: str
    ) -> APKInfo:
        """分析 APK 檔案（APK 是 ZIP，讀取 AndroidManifest 和 META-INF 憑證清單）"""
        try:
            logger.info(f"Analyzing APK: {apk_path}")
            if not os.path.isfile(apk_path):
                raise FileNotFoundError(f"APK not found: {apk_path}")

            package_name = "unknown"
            version_name = "unknown"
            version_code = 0
            permissions: list[str] = []
            dangerous_permissions: list[str] = []
            certificates: list[dict] = []
            is_signed = False
            filenames: list[str] = []

            # 危險 Android 權限集合
            _DANGEROUS = {
                "READ_CONTACTS", "WRITE_CONTACTS", "READ_CALL_LOG",
                "WRITE_CALL_LOG", "READ_SMS", "RECEIVE_SMS", "SEND_SMS",
                "RECORD_AUDIO", "CAMERA", "READ_EXTERNAL_STORAGE",
                "WRITE_EXTERNAL_STORAGE", "ACCESS_FINE_LOCATION",
                "ACCESS_COARSE_LOCATION", "READ_PHONE_STATE",
                "CALL_PHONE", "PROCESS_OUTGOING_CALLS",
            }

            with zipfile.ZipFile(apk_path, "r") as zf:
                filenames = zf.namelist()
                is_signed = any(f.startswith("META-INF/") and f.endswith((".RSA", ".DSA", ".EC")) for f in filenames)

                # 憑證摘要（列出 META-INF 簽章檔名即可，不解析 DER）
                for f in filenames:
                    if f.startswith("META-INF/") and f.endswith((".RSA", ".DSA", ".EC")):
                        info = zf.getinfo(f)
                        certificates.append({"file": f, "size": info.file_size})

                # 嘗試以純文字方式讀取 AndroidManifest（部分 APK 以 binary XML 儲存）
                try:
                    raw = zf.read("AndroidManifest.xml")
                    # 嘗試提取可讀字串（UTF-8 和 UTF-16LE）
                    text_utf8 = raw.decode("utf-8", errors="ignore")
                    text_utf16 = raw.decode("utf-16-le", errors="ignore")
                    combined = text_utf8 + " " + text_utf16

                    # package=
                    m = re.search(r'package[=\s]+"?([a-zA-Z][a-zA-Z0-9_.]+)"?', combined)
                    if m:
                        package_name = m.group(1)

                    # versionName=
                    m = re.search(r'versionName[=\s]+"?([^\s"<>]+)"?', combined)
                    if m:
                        version_name = m.group(1)

                    # versionCode=
                    m = re.search(r'versionCode[=\s]+"?(\d+)"?', combined)
                    if m:
                        version_code = int(m.group(1))

                    # uses-permission
                    for perm in re.findall(r'android\.permission\.([A-Z_]+)', combined):
                        full = f"android.permission.{perm}"
                        if full not in permissions:
                            permissions.append(full)
                        if perm in _DANGEROUS and full not in dangerous_permissions:
                            dangerous_permissions.append(full)

                except Exception:
                    pass  # binary XML，無法解析 — 保留 unknown 預設值

            logger.info(f"APK analyzed: pkg={package_name}, signed={is_signed}, perms={len(permissions)}")
            return APKInfo(
                file_path=apk_path,
                package_name=package_name,
                version_name=version_name,
                version_code=version_code,
                is_signed=is_signed,
                certificates=certificates,
                permissions=permissions,
                dangerous_permissions=dangerous_permissions,
                metadata={"total_files": len(filenames)},
            )

        except Exception as e:
            logger.error(f"APK analysis failed: {str(e)}", exc_info=True)
            raise
    
    async def decompile_apk(
        self,
        apk_path: str,
        output_dir: str,
        decompiler: DecompilerType = DecompilerType.JADX
    ) -> DecompileResult:
        """反編譯 APK（呼叫系統安裝的 jadx/apktool，不在 PATH 時回報友善錯誤）"""
        try:
            logger.info(f"Decompiling APK with {decompiler.value}: {apk_path}")
            if not os.path.isfile(apk_path):
                return DecompileResult(success=False, output_dir=output_dir, error=f"APK not found: {apk_path}")

            os.makedirs(output_dir, exist_ok=True)

            # 工具命令映射
            cmd_map: dict[DecompilerType, list[str]] = {
                DecompilerType.JADX: ["jadx", "-d", output_dir, apk_path],
                DecompilerType.JD_GUI: ["jd-cli", "-od", output_dir, apk_path],
                DecompilerType.RADARE2: ["r2", "-qc", "aaa; s main; pdc", apk_path],
            }

            cmd = cmd_map.get(decompiler)
            if cmd is None:
                return DecompileResult(
                    success=False,
                    output_dir=output_dir,
                    error=f"{decompiler.value} requires a licensed install (Ghidra/IDA/BinaryNinja); not auto-invocable.",
                )

            tool_name = cmd[0]
            if not shutil.which(tool_name):
                return DecompileResult(
                    success=False,
                    output_dir=output_dir,
                    error=f"'{tool_name}' not found in PATH. Install it first (e.g. brew install jadx).",
                )

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )
            if result.returncode != 0:
                return DecompileResult(
                    success=False,
                    output_dir=output_dir,
                    error=result.stderr.strip() or f"{tool_name} exited with code {result.returncode}",
                )

            # 統計輸出檔案數
            total_files = sum(len(files) for _, _, files in os.walk(output_dir))
            java_files = sum(
                1 for _, _, files in os.walk(output_dir)
                for f in files if f.endswith((".java", ".kt", ".smali"))
            )

            logger.info(f"Decompile complete: {total_files} files ({java_files} source)")
            return DecompileResult(
                success=True,
                output_dir=output_dir,
                classes_decompiled=java_files,
                total_files=total_files,
                decompile_rate=1.0 if java_files > 0 else 0.0,
            )

        except subprocess.TimeoutExpired:
            return DecompileResult(success=False, output_dir=output_dir, error="Decompile timed out (>300s)")
        except Exception as e:
            logger.error(f"Decompile failed: {str(e)}", exc_info=True)
            return DecompileResult(success=False, output_dir=output_dir, error=str(e))
    
    async def detect_malware(
        self,
        file_path: str
    ) -> MalwareAnalysisResult:
        """惡意軟體靜態啟發式檢測（可疑 API + 網路指示器 + packer entropy）"""
        try:
            logger.info(f"Detecting malware: {file_path}")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # SHA-256
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as fh:
                raw = fh.read()
            sha256.update(raw)
            file_hash = sha256.hexdigest()

            suspicious_apis: list[str] = []
            network_indicators: list[str] = []
            behaviors: list[str] = []
            suspicious_strings_found: list[str] = []

            # 掃描可疑 API
            for api in _SUSPICIOUS_APIS:
                if api in raw:
                    suspicious_apis.append(api.decode("ascii", errors="ignore"))

            # 網路指示器（IP 模式、URL）
            for m in re.finditer(rb"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", raw):
                ip = m.group().decode()
                if ip not in ("127.0.0.1", "0.0.0.0") and ip not in network_indicators:
                    network_indicators.append(ip)

            for m in re.finditer(rb"https?://[^\x00-\x1f\s]{4,80}", raw):
                url = m.group().decode("ascii", errors="ignore")
                if url not in network_indicators:
                    network_indicators.append(url)

            # Entropy（packer 指示）
            total = len(raw)
            byte_counts = [raw.count(bytes([i])) for i in range(256)]
            import math
            entropy = -sum((c / total) * math.log2(c / total) for c in byte_counts if c > 0) if total else 0.0
            if entropy > 7.2:
                behaviors.append("high_entropy_packing")
                suspicious_strings_found.append(f"entropy={entropy:.2f} (likely packed/encrypted)")

            # 額外行為推斷
            if any(a in ("CreateRemoteThread", "VirtualAllocEx", "WriteProcessMemory") for a in suspicious_apis):
                behaviors.append("process_injection")
            if any(a in ("keybd_event", "GetAsyncKeyState", "SetWindowsHookEx") for a in suspicious_apis):
                behaviors.append("keylogging")
            if network_indicators:
                behaviors.append("network_communication")
            if any(a in ("RegSetValueEx", "CreateService") for a in suspicious_apis):
                behaviors.append("persistence")

            # 威脅等級評分
            score = len(suspicious_apis) * 2 + len(network_indicators) + len(behaviors) * 3
            if score >= 15:
                threat_level = ThreatLevel.CRITICAL
            elif score >= 8:
                threat_level = ThreatLevel.HIGH
            elif score >= 4:
                threat_level = ThreatLevel.MEDIUM
            elif score >= 1:
                threat_level = ThreatLevel.LOW
            else:
                threat_level = ThreatLevel.BENIGN

            is_malicious = threat_level not in (ThreatLevel.BENIGN, ThreatLevel.LOW)
            confidence = min(1.0, score / 20)

            logger.info(f"Malware scan: {threat_level.value}, suspicious_apis={len(suspicious_apis)}, confidence={confidence:.2f}")
            return MalwareAnalysisResult(
                file_hash=file_hash,
                threat_level=threat_level,
                is_malicious=is_malicious,
                confidence=round(confidence, 4),
                suspicious_apis=suspicious_apis,
                suspicious_strings=suspicious_strings_found,
                network_indicators=network_indicators[:20],  # 最多20條
                behaviors=behaviors,
            )

        except Exception as e:
            logger.error(f"Malware detection failed: {str(e)}", exc_info=True)
            raise
    
    async def extract_strings(
        self,
        file_path: str,
        min_length: int = 4
    ) -> list:
        """從二進制檔案提取可印字串（等同 strings 工具的 ASCII/UTF-16LE 掃描）"""
        try:
            logger.info(f"Extracting strings from: {file_path} (min_len={min_length})")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            with open(file_path, "rb") as fh:
                raw = fh.read()

            found: list[str] = []

            # ASCII 可印字元連續序列
            ascii_pattern = re.compile(rb"[\x20-\x7e]{" + str(min_length).encode() + rb",}")
            for m in ascii_pattern.finditer(raw):
                s = m.group().decode("ascii", errors="ignore").strip()
                if len(s) >= min_length:
                    found.append(s)

            # UTF-16LE（Windows 字串常見編碼）
            try:
                text_utf16 = raw.decode("utf-16-le", errors="ignore")
                utf16_pattern = re.compile(r"[\x20-\x7e]{" + str(min_length) + r",}")
                for m in utf16_pattern.finditer(text_utf16):
                    s = m.group().strip()
                    if len(s) >= min_length and s not in found:
                        found.append(s)
            except Exception:
                pass

            # 去重並保持順序
            seen: set[str] = set()
            unique: list[str] = []
            for s in found:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)

            logger.info(f"Extracted {len(unique)} strings from {file_path}")
            return unique

        except Exception as e:
            logger.error(f"String extraction failed: {str(e)}", exc_info=True)
            return []
