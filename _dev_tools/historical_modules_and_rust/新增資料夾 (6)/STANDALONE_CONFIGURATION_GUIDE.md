# 無線攻擊工具 - 獨立版本配置指南

本文檔說明如何將 wireless_attack_tools.py 從 AIVA 系統中分離，作為獨立工具使用。

---

## 🎯 目標

將當前依賴 AIVA 框架的模組轉換為：
1. **獨立命令行工具** - 可在任何 Linux 系統上運行
2. **最小依賴** - 只依賴必要的第三方庫
3. **保持功能完整** - 所有攻擊功能保持可用

---

## 📋 需要移除的 AIVA 依賴

### 當前 AIVA 依賴

```python
# 需要移除的導入
from ...core.base_capability import BaseCapability
from ...aiva_common.schemas import APIResponse, CapabilityType
from ...core.registry import CapabilityRegistry
```

### 替代方案

```python
# 獨立版本不需要這些
# 功能將通過命令行參數直接調用
```

---

## 🔧 轉換步驟

### Step 1: 創建獨立版本目錄結構

```bash
wireless_attack_tools_standalone/
│
├── wireless_tools.py              # 主程序 (重構後)
├── requirements.txt               # Python 依賴
├── install.sh                     # 自動安裝腳本
├── README.md                      # 使用說明
├── LICENSE                        # 授權協議
│
├── config/
│   ├── settings.json              # 配置文件
│   └── hardware_db.json           # 硬體相容性資料庫
│
├── data/
│   └── wireless_attacks/          # 攻擊結果存儲
│
├── templates/
│   └── report_template.html       # 報告模板
│
└── utils/
    ├── __init__.py
    ├── hardware.py                # 硬體檢測
    ├── network.py                 # 網路操作
    └── report.py                  # 報告生成
```

### Step 2: 修改主程序結構

**原始版本** (AIVA 整合):
```python
class WirelessCapability(BaseCapability):
    """AIVA 能力類"""
    
    def initialize(self) -> bool:
        """初始化"""
        
    def execute(self, command: str, parameters: Dict) -> Dict:
        """執行命令"""
        
    def cleanup(self) -> bool:
        """清理"""
```

**獨立版本**:
```python
class WirelessTools:
    """獨立無線工具類"""
    
    def __init__(self, config_file: str = "config/settings.json"):
        """初始化工具"""
        self.config = self.load_config(config_file)
        self.console = Console(theme=self.config.get('theme'))
        self.logger = self.setup_logging()
        
    def run_interactive(self):
        """運行互動式選單"""
        # 原 WirelessManager.interactive_menu() 的內容
        
    def scan_wifi(self, duration: int = 30, output: str = None):
        """WiFi 掃描命令"""
        # 原 WifiScanner.scan_networks() 的內容
        
    def wps_attack(self, target_bssid: str, channel: int):
        """WPS 攻擊命令"""
        # 原 WPSAttack.pixie_dust_attack() 的內容
        
    def capture_handshake(self, target_bssid: str, timeout: int = 120):
        """捕獲握手包命令"""
        # 原 HandshakeCapture.capture_handshake() 的內容

def main():
    """主入口函數"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Wireless Penetration Testing Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('-i', '--interactive', action='store_true',
                       help='Launch interactive menu')
    parser.add_argument('--scan', action='store_true',
                       help='Scan WiFi networks')
    parser.add_argument('--wps-attack', metavar='BSSID',
                       help='Perform WPS Pixie Dust attack')
    parser.add_argument('--capture-handshake', metavar='BSSID',
                       help='Capture WPA handshake')
    parser.add_argument('-c', '--channel', type=int,
                       help='Target channel')
    parser.add_argument('-t', '--timeout', type=int, default=120,
                       help='Timeout in seconds')
    parser.add_argument('-o', '--output', metavar='FILE',
                       help='Output file path')
    
    args = parser.parse_args()
    
    tools = WirelessTools()
    
    if args.interactive:
        tools.run_interactive()
    elif args.scan:
        tools.scan_wifi(output=args.output)
    elif args.wps_attack:
        tools.wps_attack(args.wps_attack, args.channel)
    elif args.capture_handshake:
        tools.capture_handshake(args.capture_handshake, args.timeout)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

### Step 3: 創建 requirements.txt

```txt
# Core dependencies
rich>=13.0.0
asyncio>=3.4.3

# Optional dependencies (for advanced features)
# jinja2>=3.1.0        # For report generation
# matplotlib>=3.5.0    # For visualization
# folium>=0.14.0       # For geographic heatmaps
```

### Step 4: 創建自動安裝腳本

**install.sh**:
```bash
#!/bin/bash

# Wireless Attack Tools - Installation Script
# Supports: Ubuntu/Debian, Arch Linux, Kali Linux

set -e

echo "========================================="
echo "Wireless Attack Tools - Installation"
echo "========================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VER=$VERSION_ID
else
    echo "❌ Cannot detect OS"
    exit 1
fi

echo "✅ Detected OS: $OS $VER"

# Install system dependencies
echo ""
echo "Installing system dependencies..."

case $OS in
    ubuntu|debian|kali)
        apt update
        apt install -y \
            python3 python3-pip python3-venv \
            aircrack-ng reaver pixiewps \
            hcxtools hcxdumptool \
            wireless-tools net-tools \
            bluez bluetooth \
            macchanger
        ;;
    arch|manjaro)
        pacman -Sy --noconfirm \
            python python-pip \
            aircrack-ng reaver \
            hcxtools hcxdumptool \
            wireless_tools net-tools \
            bluez bluez-utils \
            macchanger
        ;;
    *)
        echo "⚠️  Unsupported OS: $OS"
        echo "Please install dependencies manually"
        exit 1
        ;;
esac

echo "✅ System dependencies installed"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."

pip3 install -r requirements.txt

echo "✅ Python dependencies installed"

# Create directories
echo ""
echo "Creating directories..."

mkdir -p data/wireless_attacks
mkdir -p config

echo "✅ Directories created"

# Create symbolic link
echo ""
echo "Creating symbolic link..."

ln -sf "$(pwd)/wireless_tools.py" /usr/local/bin/wireless-tools
chmod +x wireless_tools.py

echo "✅ Symbolic link created"

# Hardware check
echo ""
echo "Checking wireless hardware..."

python3 wireless_tools.py --hardware-check

echo ""
echo "========================================="
echo "✅ Installation completed!"
echo "========================================="
echo ""
echo "Usage:"
echo "  sudo wireless-tools -i              # Interactive mode"
echo "  sudo wireless-tools --scan          # Scan WiFi"
echo "  sudo wireless-tools --help          # Show help"
echo ""
echo "⚠️  Remember: Only use on authorized networks!"
echo ""
```

### Step 5: 簡化配置文件

**config/settings.json**:
```json
{
  "version": "1.0.0",
  "theme": {
    "purple": "#7B61FF",
    "warning": "yellow",
    "error": "red",
    "success": "green"
  },
  "scan": {
    "default_duration": 30,
    "default_interface": "wlan0",
    "auto_monitor_mode": true
  },
  "attack": {
    "wps_timeout": 300,
    "handshake_timeout": 120,
    "deauth_count": 10
  },
  "output": {
    "data_dir": "data/wireless_attacks",
    "report_format": "json",
    "auto_save": true
  },
  "logging": {
    "level": "INFO",
    "file": "wireless_tools.log",
    "console": true
  }
}
```

---

## 🚀 使用方式

### 基本命令

**1. 互動式模式** (推薦):
```bash
sudo wireless-tools -i
```

**2. WiFi 掃描**:
```bash
# 基本掃描
sudo wireless-tools --scan

# 指定掃描時間
sudo wireless-tools --scan --timeout 60

# 保存結果
sudo wireless-tools --scan -o scan_results.json
```

**3. WPS 攻擊**:
```bash
# 基本攻擊
sudo wireless-tools --wps-attack AA:BB:CC:DD:EE:FF -c 6

# 帶超時設置
sudo wireless-tools --wps-attack AA:BB:CC:DD:EE:FF -c 6 -t 600
```

**4. 握手包捕獲**:
```bash
# 捕獲握手包
sudo wireless-tools --capture-handshake AA:BB:CC:DD:EE:FF -c 6

# 指定輸出文件
sudo wireless-tools --capture-handshake AA:BB:CC:DD:EE:FF -c 6 -o handshake.cap
```

**5. 硬體檢測**:
```bash
sudo wireless-tools --hardware-check
```

**6. 生成報告**:
```bash
# HTML 報告
sudo wireless-tools --generate-report --format html -o report.html

# PDF 報告
sudo wireless-tools --generate-report --format pdf -o report.pdf
```

---

## 📦 打包與分發

### 方法 1: Python Package

**創建 setup.py**:
```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="wireless-attack-tools",
    version="1.0.0",
    author="AIVA Team",
    author_email="contact@example.com",
    description="Professional wireless penetration testing tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wireless-attack-tools",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.11",
    install_requires=[
        "rich>=13.0.0",
        "asyncio>=3.4.3",
    ],
    entry_points={
        "console_scripts": [
            "wireless-tools=wireless_tools:main",
        ],
    },
)
```

**安裝方式**:
```bash
pip install wireless-attack-tools
```

### 方法 2: Docker Container

**Dockerfile**:
```dockerfile
FROM kalilinux/kali-rolling

# Install dependencies
RUN apt update && apt install -y \
    python3 python3-pip \
    aircrack-ng reaver pixiewps \
    wireless-tools bluez \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /opt/wireless-tools

# Copy files
COPY . .

# Install Python dependencies
RUN pip3 install -r requirements.txt

# Entry point
ENTRYPOINT ["python3", "wireless_tools.py"]
CMD ["-i"]
```

**使用方式**:
```bash
# Build
docker build -t wireless-tools .

# Run (需要 --privileged 和 --net=host)
docker run --rm -it --privileged --net=host wireless-tools
```

### 方法 3: AppImage (Linux 單一可執行文件)

**使用 PyInstaller**:
```bash
# 安裝 PyInstaller
pip install pyinstaller

# 打包
pyinstaller --onefile \
            --name wireless-tools \
            --add-data "config:config" \
            --add-data "templates:templates" \
            wireless_tools.py

# 輸出: dist/wireless-tools
```

---

## 🔒 安全與合規

### 使用前檢查清單

```bash
# 執行安全檢查
sudo wireless-tools --security-check

輸出:
✅ Root 權限: 已授予
✅ 必要工具: aircrack-ng, reaver, pixiewps
✅ 硬體支援: wlan0 支援監控模式
⚠️  授權確認: 請確保已取得測試授權
⚠️  法律聲明: 已閱讀並同意使用條款

繼續嗎? [y/N]:
```

### 強制授權確認

**修改代碼增加授權檢查**:
```python
def check_authorization(self):
    """強制授權確認"""
    
    console.print("[warning]⚠️  重要法律聲明[/warning]")
    console.print("")
    console.print("未經授權的無線網路攻擊在全球範圍內均屬違法行為。")
    console.print("使用本工具前，您必須：")
    console.print("  1. 擁有目標網路的所有權")
    console.print("  2. 或已取得書面授權")
    console.print("  3. 在合法的測試環境中使用")
    console.print("")
    
    consent = Confirm.ask("您確認已取得必要授權嗎?")
    
    if not consent:
        console.print("[error]❌ 未經授權，程序退出[/error]")
        sys.exit(1)
    
    # 記錄授權確認
    self.logger.info(f"User confirmed authorization at {datetime.now()}")
    
    # 要求輸入授權編號 (可選)
    auth_code = Prompt.ask("請輸入授權編號 (可選)", default="N/A")
    self.logger.info(f"Authorization code: {auth_code}")
```

### 使用日誌記錄

**完整的稽核日誌**:
```python
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(self):
    """設置詳細日誌"""
    
    logger = logging.getLogger('WirelessTools')
    logger.setLevel(logging.INFO)
    
    # 文件處理器 (輪替)
    file_handler = RotatingFileHandler(
        'wireless_tools.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # 控制台處理器
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

# 使用範例
self.logger.info("=== Session Start ===")
self.logger.info(f"User: {os.getenv('USER')}")
self.logger.info(f"Host: {socket.gethostname()}")
self.logger.info(f"IP: {get_local_ip()}")
self.logger.info(f"Command: {' '.join(sys.argv)}")
```

---

## 🧪 測試與驗證

### 單元測試

**tests/test_wireless_tools.py**:
```python
import unittest
from wireless_tools import WirelessTools

class TestWirelessTools(unittest.TestCase):
    
    def setUp(self):
        self.tools = WirelessTools()
    
    def test_hardware_detection(self):
        """測試硬體檢測"""
        adapters = self.tools.detect_wireless_adapters()
        self.assertIsInstance(adapters, list)
    
    def test_config_loading(self):
        """測試配置加載"""
        config = self.tools.load_config("config/settings.json")
        self.assertIn('version', config)
    
    def test_network_parsing(self):
        """測試網路資訊解析"""
        csv_data = "BSSID,ESSID,Channel,Encryption\n"
        csv_data += "AA:BB:CC:DD:EE:FF,TestNetwork,6,WPA2\n"
        
        networks = self.tools.parse_networks(csv_data)
        self.assertEqual(len(networks), 1)
        self.assertEqual(networks[0].essid, "TestNetwork")

if __name__ == '__main__':
    unittest.main()
```

### 整合測試

**使用測試 AP**:
```bash
# 設置測試環境
# 1. 使用舊路由器作為測試目標
# 2. 在隔離網路中測試
# 3. 驗證所有功能

sudo wireless-tools --test-mode --target TEST_AP
```

---

## 📝 文檔

### 用戶手冊

創建詳細的用戶手冊:
- 安裝指南
- 快速開始
- 功能詳解
- 故障排除
- FAQ

### 開發者文檔

- API 參考
- 架構設計
- 貢獻指南
- 代碼規範

---

## 🔄 從 AIVA 遷移檢查清單

- [ ] 移除 AIVA 核心依賴
- [ ] 重構為獨立類結構
- [ ] 添加命令行參數解析
- [ ] 創建配置文件系統
- [ ] 實現日誌記錄
- [ ] 添加授權確認機制
- [ ] 編寫安裝腳本
- [ ] 創建測試套件
- [ ] 編寫用戶文檔
- [ ] 打包與發布

---

## 📞 支援

**問題回報**: [GitHub Issues]  
**討論**: [GitHub Discussions]  
**電子郵件**: support@example.com

---

**最後更新**: 2025年11月25日  
**版本**: v1.0.0

© 2025 AIVA Team. All rights reserved.
