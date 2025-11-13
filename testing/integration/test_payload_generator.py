"""
AIVA Payload Generator Module Tests
===================================

Task 10: 整合載荷生成工具 - 測試用例
基於 HackingTool 的 payload_creator.py 功能測試
"""

import unittest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from payload_generator import (
    PayloadType,
    PayloadFormat,
    PayloadEncoder,
    PayloadArchitecture,
    PayloadConfig,
    PayloadResult,
    MSFVenomGenerator,
    CustomPayloadGenerator,
    AndroidPayloadGenerator,
    PayloadManager,
    PayloadCLI
)


class TestPayloadConfig(unittest.TestCase):
    """載荷配置測試"""
    
    def test_payload_config_creation(self):
        """測試載荷配置創建"""
        config = PayloadConfig(
            name="test_payload",
            payload_type=PayloadType.WINDOWS_EXECUTABLE,
            format=PayloadFormat.EXE,
            lhost="192.168.1.100",
            lport=8080
        )
        
        self.assertEqual(config.name, "test_payload")
        self.assertEqual(config.payload_type, PayloadType.WINDOWS_EXECUTABLE)
        self.assertEqual(config.format, PayloadFormat.EXE)
        self.assertEqual(config.lhost, "192.168.1.100")
        self.assertEqual(config.lport, 8080)
    
    def test_payload_config_to_dict(self):
        """測試載荷配置轉字典"""
        config = PayloadConfig(
            name="test_payload",
            payload_type=PayloadType.POWERSHELL_SCRIPT,
            format=PayloadFormat.PS1
        )
        
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["name"], "test_payload")
        self.assertEqual(config_dict["payload_type"], "POWERSHELL_SCRIPT")
        self.assertEqual(config_dict["format"], "ps1")


class TestPayloadResult(unittest.TestCase):
    """載荷結果測試"""
    
    def test_payload_result_creation(self):
        """測試載荷結果創建"""
        config = PayloadConfig(
            name="test",
            payload_type=PayloadType.PYTHON_SCRIPT,
            format=PayloadFormat.PY
        )
        
        result = PayloadResult(
            config=config,
            success=True,
            payload_data=b"test payload data"
        )
        
        self.assertTrue(result.success)
        self.assertEqual(result.payload_data, b"test payload data")
        self.assertIsNotNone(result.creation_time)
    
    def test_payload_result_get_hash(self):
        """測試載荷哈希計算"""
        config = PayloadConfig(
            name="test",
            payload_type=PayloadType.PYTHON_SCRIPT,
            format=PayloadFormat.PY
        )
        
        result = PayloadResult(
            config=config,
            success=True,
            payload_data=b"test payload data"
        )
        
        hash_value = result.get_hash()
        self.assertIsNotNone(hash_value)
        self.assertEqual(len(hash_value), 64)  # SHA256 hash length
    
    def test_payload_result_save_to_file(self):
        """測試載荷保存到文件"""
        config = PayloadConfig(
            name="test",
            payload_type=PayloadType.PYTHON_SCRIPT,
            format=PayloadFormat.PY
        )
        
        result = PayloadResult(
            config=config,
            success=True,
            payload_data=b"test payload data"
        )
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        success = result.save_to_file(temp_path)
        
        self.assertTrue(success)
        
        # 驗證文件內容
        with open(temp_path, 'rb') as f:
            content = f.read()
        
        self.assertEqual(content, b"test payload data")
        
        # 清理
        Path(temp_path).unlink()


class TestMSFVenomGenerator(unittest.TestCase):
    """MSFVenom生成器測試"""
    
    def setUp(self):
        self.generator = MSFVenomGenerator()
    
    @patch('subprocess.run')
    def test_find_msfvenom_success(self, mock_run):
        """測試找到MSFVenom"""
        mock_run.return_value.returncode = 0
        
        generator = MSFVenomGenerator()
        # 重新初始化以觸發查找
        generator.msfvenom_path = generator._find_msfvenom()
        
        self.assertIsNotNone(generator.msfvenom_path)
    
    @patch('subprocess.run')
    def test_find_msfvenom_not_found(self, mock_run):
        """測試未找到MSFVenom"""
        mock_run.side_effect = FileNotFoundError()
        
        generator = MSFVenomGenerator()
        generator.msfvenom_path = generator._find_msfvenom()
        
        self.assertIsNone(generator.msfvenom_path)
    
    def test_get_payload_name(self):
        """測試獲取載荷名稱"""
        config = PayloadConfig(
            name="test",
            payload_type=PayloadType.WINDOWS_EXECUTABLE,
            format=PayloadFormat.EXE
        )
        
        payload_name = self.generator._get_payload_name(config)
        self.assertEqual(payload_name, "windows/meterpreter/reverse_tcp")
    
    def test_build_msfvenom_command(self):
        """測試構建MSFVenom命令"""
        config = PayloadConfig(
            name="test",
            payload_type=PayloadType.LINUX_EXECUTABLE,
            format=PayloadFormat.ELF,
            lhost="192.168.1.100",
            lport=9999,
            encoder=PayloadEncoder.SHIKATA_GA_NAI,
            iterations=3
        )
        
        self.generator.msfvenom_path = "/usr/bin/msfvenom"
        cmd = self.generator._build_msfvenom_command(config)
        
        self.assertIn("/usr/bin/msfvenom", cmd)
        self.assertIn("-p", cmd)
        self.assertIn("linux/x86/meterpreter/reverse_tcp", cmd)
        self.assertIn("LHOST=192.168.1.100", cmd)
        self.assertIn("LPORT=9999", cmd)
        self.assertIn("-f", cmd)
        self.assertIn("elf", cmd)
        self.assertIn("-e", cmd)
        self.assertIn("x86/shikata_ga_nai", cmd)
        self.assertIn("-i", cmd)
        self.assertIn("3", cmd)
    
    @patch('asyncio.create_subprocess_exec')
    async def test_generate_payload_success(self, mock_subprocess):
        """測試成功生成載荷"""
        # 模擬成功的subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"fake payload data", b"")
        mock_subprocess.return_value = mock_process
        
        self.generator.msfvenom_path = "/usr/bin/msfvenom"
        
        config = PayloadConfig(
            name="test",
            payload_type=PayloadType.WINDOWS_EXECUTABLE,
            format=PayloadFormat.EXE
        )
        
        result = await self.generator.generate_payload(config)
        
        self.assertTrue(result.success)
        self.assertEqual(result.payload_data, b"fake payload data")
        self.assertIsNotNone(result.execution_time)
    
    @patch('asyncio.create_subprocess_exec')
    async def test_generate_payload_failure(self, mock_subprocess):
        """測試載荷生成失敗"""
        # 模擬失敗的subprocess
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Error: invalid payload")
        mock_subprocess.return_value = mock_process
        
        self.generator.msfvenom_path = "/usr/bin/msfvenom"
        
        config = PayloadConfig(
            name="test",
            payload_type=PayloadType.WINDOWS_EXECUTABLE,
            format=PayloadFormat.EXE
        )
        
        result = await self.generator.generate_payload(config)
        
        self.assertFalse(result.success)
        self.assertIn("Error: invalid payload", result.error_message)
    
    async def test_generate_payload_no_msfvenom(self):
        """測試沒有MSFVenom時的處理"""
        self.generator.msfvenom_path = None
        
        config = PayloadConfig(
            name="test",
            payload_type=PayloadType.WINDOWS_EXECUTABLE,
            format=PayloadFormat.EXE
        )
        
        result = await self.generator.generate_payload(config)
        
        self.assertFalse(result.success)
        self.assertIn("MSFVenom未安裝或不可用", result.error_message)


class TestCustomPayloadGenerator(unittest.TestCase):
    """自定義載荷生成器測試"""
    
    def setUp(self):
        self.generator = CustomPayloadGenerator()
    
    async def test_generate_powershell_payload(self):
        """測試生成PowerShell載荷"""
        config = PayloadConfig(
            name="test_ps",
            payload_type=PayloadType.POWERSHELL_SCRIPT,
            format=PayloadFormat.PS1,
            lhost="192.168.1.100",
            lport=8080
        )
        
        result = await self.generator.generate_powershell_payload(config)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.payload_data)
        
        # 檢查載荷內容包含期望的元素
        payload_text = result.payload_data.decode()
        self.assertIn("192.168.1.100", payload_text)
        self.assertIn("8080", payload_text)
        self.assertIn("TCPClient", payload_text)
    
    async def test_generate_python_payload(self):
        """測試生成Python載荷"""
        config = PayloadConfig(
            name="test_py",
            payload_type=PayloadType.PYTHON_SCRIPT,
            format=PayloadFormat.PY,
            lhost="10.0.0.1",
            lport=4444
        )
        
        result = await self.generator.generate_python_payload(config)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.payload_data)
        
        payload_text = result.payload_data.decode()
        self.assertIn("10.0.0.1", payload_text)
        self.assertIn("4444", payload_text)
        self.assertIn("socket", payload_text)
    
    async def test_generate_bash_payload(self):
        """測試生成Bash載荷"""
        config = PayloadConfig(
            name="test_bash",
            payload_type=PayloadType.BASH_SCRIPT,
            format=PayloadFormat.SH,
            lhost="172.16.1.1",
            lport=9999
        )
        
        result = await self.generator.generate_bash_payload(config)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.payload_data)
        
        payload_text = result.payload_data.decode()
        self.assertIn("172.16.1.1", payload_text)
        self.assertIn("9999", payload_text)
        self.assertIn("/dev/tcp", payload_text)
    
    def test_create_powershell_reverse_shell(self):
        """測試創建PowerShell反向Shell"""
        ps_code = self.generator._create_powershell_reverse_shell("192.168.1.1", 443)
        
        self.assertIn("192.168.1.1", ps_code)
        self.assertIn("443", ps_code)
        self.assertIn("TCPClient", ps_code)
        self.assertIn("GetStream", ps_code)
    
    def test_create_python_reverse_shell(self):
        """測試創建Python反向Shell""" 
        py_code = self.generator._create_python_reverse_shell("10.10.10.10", 1337)
        
        self.assertIn("10.10.10.10", py_code)
        self.assertIn("1337", py_code)
        self.assertIn("socket.socket", py_code)
        self.assertIn("subprocess", py_code)
    
    def test_create_bash_reverse_shell(self):
        """測試創建Bash反向Shell"""
        bash_code = self.generator._create_bash_reverse_shell("172.16.0.1", 8888)
        
        self.assertIn("172.16.0.1", bash_code)
        self.assertIn("8888", bash_code)
        self.assertIn("/dev/tcp", bash_code)
        self.assertIn("bash -i", bash_code)


class TestAndroidPayloadGenerator(unittest.TestCase):
    """Android載荷生成器測試"""
    
    def setUp(self):
        self.generator = AndroidPayloadGenerator()
    
    @patch('subprocess.run')
    def test_check_apktool_available(self, mock_run):
        """測試檢查APKTool可用性"""
        mock_run.return_value.returncode = 0
        
        generator = AndroidPayloadGenerator()
        generator.apktool_available = generator._check_apktool()
        
        self.assertTrue(generator.apktool_available)
    
    @patch('subprocess.run')
    def test_check_apktool_not_available(self, mock_run):
        """測試APKTool不可用"""
        mock_run.side_effect = FileNotFoundError()
        
        generator = AndroidPayloadGenerator()
        generator.apktool_available = generator._check_apktool()
        
        self.assertFalse(generator.apktool_available)
    
    async def test_generate_android_payload_no_tools(self):
        """測試沒有工具時的Android載荷生成"""
        self.generator.apktool_available = False
        
        config = PayloadConfig(
            name="test_android",
            payload_type=PayloadType.ANDROID_APK,
            format=PayloadFormat.APK
        )
        
        # 模擬MSFVenom也不可用
        with patch.object(MSFVenomGenerator, 'is_available', return_value=False):
            result = await self.generator.generate_android_payload(config)
        
        self.assertFalse(result.success)
        self.assertIn("APKTool和MSFVenom都不可用", result.error_message)


class TestPayloadManager(unittest.TestCase):
    """載荷管理器測試"""
    
    def setUp(self):
        self.manager = PayloadManager()
    
    def test_create_payload_config(self):
        """測試創建載荷配置"""
        config = self.manager.create_payload_config(
            name="test_payload",
            payload_type=PayloadType.WINDOWS_EXECUTABLE,
            lhost="192.168.1.1",
            lport=4444
        )
        
        self.assertEqual(config.name, "test_payload")
        self.assertEqual(config.payload_type, PayloadType.WINDOWS_EXECUTABLE)
        self.assertEqual(config.format, PayloadFormat.EXE)
        self.assertEqual(config.lhost, "192.168.1.1")
        self.assertEqual(config.lport, 4444)
    
    def test_get_supported_payload_types(self):
        """測試獲取支持的載荷類型"""
        supported_types = self.manager.get_supported_payload_types()
        
        # 至少應該包含自定義載荷類型
        self.assertIn(PayloadType.POWERSHELL_SCRIPT, supported_types)
        self.assertIn(PayloadType.PYTHON_SCRIPT, supported_types)
        self.assertIn(PayloadType.BASH_SCRIPT, supported_types)
    
    async def test_generate_powershell_payload(self):
        """測試生成PowerShell載荷"""
        config = self.manager.create_payload_config(
            name="test_ps",
            payload_type=PayloadType.POWERSHELL_SCRIPT
        )
        
        result = await self.manager.generate_payload(config)
        
        self.assertTrue(result.success)
        self.assertEqual(len(self.manager.results_history), 1)
    
    def test_get_generation_stats_empty(self):
        """測試空統計"""
        stats = self.manager.get_generation_stats()
        
        self.assertEqual(stats["total_generated"], 0)
        self.assertEqual(stats["successful"], 0)
        self.assertEqual(stats["failed"], 0)
        self.assertEqual(stats["success_rate"], 0)
    
    async def test_get_generation_stats_with_data(self):
        """測試有數據的統計"""
        # 生成一個成功的載荷
        config = self.manager.create_payload_config(
            name="test",
            payload_type=PayloadType.PYTHON_SCRIPT
        )
        
        await self.manager.generate_payload(config)
        
        stats = self.manager.get_generation_stats()
        
        self.assertEqual(stats["total_generated"], 1)
        self.assertEqual(stats["successful"], 1)
        self.assertEqual(stats["success_rate"], 100.0)
    
    def test_export_payload(self):
        """測試導出載荷"""
        config = PayloadConfig(
            name="test_export",
            payload_type=PayloadType.PYTHON_SCRIPT,
            format=PayloadFormat.PY
        )
        
        result = PayloadResult(
            config=config,
            success=True,
            payload_data=b"test payload content"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            success = self.manager.export_payload(result, temp_dir)
            
            self.assertTrue(success)
            
            # 檢查文件是否存在
            payload_file = Path(temp_dir) / "test_export.py"
            metadata_file = Path(temp_dir) / "test_export_metadata.json"
            
            self.assertTrue(payload_file.exists())
            self.assertTrue(metadata_file.exists())
            
            # 檢查文件內容
            with open(payload_file, 'rb') as f:
                content = f.read()
            self.assertEqual(content, b"test payload content")
            
            # 檢查元數據
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.assertEqual(metadata["config"]["name"], "test_export")


class TestPayloadEnums(unittest.TestCase):
    """載荷枚舉測試"""
    
    def test_payload_type_enum(self):
        """測試載荷類型枚舉"""
        self.assertTrue(hasattr(PayloadType, 'WINDOWS_EXECUTABLE'))
        self.assertTrue(hasattr(PayloadType, 'LINUX_EXECUTABLE'))
        self.assertTrue(hasattr(PayloadType, 'ANDROID_APK'))
        self.assertTrue(hasattr(PayloadType, 'POWERSHELL_SCRIPT'))
    
    def test_payload_format_enum(self):
        """測試載荷格式枚舉"""
        self.assertEqual(PayloadFormat.EXE.value, "exe")
        self.assertEqual(PayloadFormat.ELF.value, "elf") 
        self.assertEqual(PayloadFormat.APK.value, "apk")
        self.assertEqual(PayloadFormat.PS1.value, "ps1")
    
    def test_payload_encoder_enum(self):
        """測試載荷編碼器枚舉"""
        self.assertEqual(PayloadEncoder.SHIKATA_GA_NAI.value, "x86/shikata_ga_nai")
        self.assertEqual(PayloadEncoder.ALPHA_MIXED.value, "x86/alpha_mixed")
    
    def test_payload_architecture_enum(self):
        """測試載荷架構枚舉"""
        self.assertEqual(PayloadArchitecture.X86.value, "x86")
        self.assertEqual(PayloadArchitecture.X64.value, "x64")
        self.assertEqual(PayloadArchitecture.ARM.value, "arm")


class TestPayloadIntegration(unittest.TestCase):
    """載荷集成測試"""
    
    async def test_end_to_end_python_payload(self):
        """端到端Python載荷測試"""
        manager = PayloadManager()
        
        # 創建配置
        config = manager.create_payload_config(
            name="e2e_test",
            payload_type=PayloadType.PYTHON_SCRIPT,
            lhost="127.0.0.1",
            lport=4444
        )
        
        # 生成載荷
        result = await manager.generate_payload(config)
        
        # 驗證結果
        self.assertTrue(result.success)
        self.assertIsNotNone(result.payload_data)
        self.assertGreater(len(result.payload_data), 0)
        
        # 驗證載荷內容
        payload_text = result.payload_data.decode()
        self.assertIn("127.0.0.1", payload_text)
        self.assertIn("4444", payload_text)
        self.assertIn("socket", payload_text)
        
        # 導出測試
        with tempfile.TemporaryDirectory() as temp_dir:
            export_success = manager.export_payload(result, temp_dir)
            self.assertTrue(export_success)
            
            payload_file = Path(temp_dir) / "e2e_test.py" 
            self.assertTrue(payload_file.exists())
    
    async def test_multiple_payload_generation(self):
        """多載荷生成測試"""
        manager = PayloadManager()
        
        payload_configs = [
            ("python_test", PayloadType.PYTHON_SCRIPT),
            ("powershell_test", PayloadType.POWERSHELL_SCRIPT),
            ("bash_test", PayloadType.BASH_SCRIPT)
        ]
        
        results = []
        for name, ptype in payload_configs:
            config = manager.create_payload_config(name=name, payload_type=ptype)
            result = await manager.generate_payload(config)
            results.append(result)
        
        # 驗證所有載荷都成功生成
        for result in results:
            self.assertTrue(result.success)
        
        # 檢查統計
        stats = manager.get_generation_stats()
        self.assertEqual(stats["total_generated"], 3)
        self.assertEqual(stats["successful"], 3)
        self.assertEqual(stats["success_rate"], 100.0)


# 異步測試運行器
class AsyncTestCase(unittest.TestCase):
    """異步測試基類"""
    
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def tearDown(self):
        self.loop.close()
    
    def async_test(self, coro):
        """運行異步測試"""
        return self.loop.run_until_complete(coro)


# 為異步測試方法添加裝飾器
def async_test_method(test_method):
    """異步測試方法裝飾器"""
    def wrapper(self):
        return self.async_test(test_method(self))
    return wrapper


# 應用裝飾器到異步測試方法
TestMSFVenomGenerator.test_generate_payload_success = async_test_method(TestMSFVenomGenerator.test_generate_payload_success)
TestMSFVenomGenerator.test_generate_payload_failure = async_test_method(TestMSFVenomGenerator.test_generate_payload_failure)
TestMSFVenomGenerator.test_generate_payload_no_msfvenom = async_test_method(TestMSFVenomGenerator.test_generate_payload_no_msfvenom)

TestCustomPayloadGenerator.test_generate_powershell_payload = async_test_method(TestCustomPayloadGenerator.test_generate_powershell_payload)
TestCustomPayloadGenerator.test_generate_python_payload = async_test_method(TestCustomPayloadGenerator.test_generate_python_payload)
TestCustomPayloadGenerator.test_generate_bash_payload = async_test_method(TestCustomPayloadGenerator.test_generate_bash_payload)

TestAndroidPayloadGenerator.test_generate_android_payload_no_tools = async_test_method(TestAndroidPayloadGenerator.test_generate_android_payload_no_tools)

TestPayloadManager.test_generate_powershell_payload = async_test_method(TestPayloadManager.test_generate_powershell_payload)
TestPayloadManager.test_get_generation_stats_with_data = async_test_method(TestPayloadManager.test_get_generation_stats_with_data)

TestPayloadIntegration.test_end_to_end_python_payload = async_test_method(TestPayloadIntegration.test_end_to_end_python_payload)
TestPayloadIntegration.test_multiple_payload_generation = async_test_method(TestPayloadIntegration.test_multiple_payload_generation)


if __name__ == "__main__":
    # 運行測試
    unittest.main(verbosity=2)