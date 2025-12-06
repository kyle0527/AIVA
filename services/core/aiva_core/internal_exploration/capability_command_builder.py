"""Capability Command Builder - CLI 指令構建器

基於數據流路徑構建可執行的 CLI 指令序列
支援路徑驗證、參數傳遞和指令串聯執行

作者: AIVA Team
版本: 1.0
日期: 2025-12-06
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class CommandStep:
    """單個指令步驟"""
    script_name: str              # 腳本名稱
    script_path: str              # 腳本路徑
    command: str                  # 執行指令
    parameters: Dict[str, Any]    # 參數映射
    input_params: List[str]       # 輸入參數名稱
    output_params: List[str]      # 輸出參數名稱
    is_executable: bool           # 是否可執行
    execution_time: float = 0.0   # 預估執行時間


@dataclass
class CommandSequence:
    """指令序列"""
    sequence_id: str                    # 序列 ID
    capability_name: str                # 能力名稱
    steps: List[CommandStep]           # 指令步驟
    total_estimated_time: float        # 總預估時間
    validation_status: str             # 驗證狀態 (valid/invalid/warning)
    validation_messages: List[str]     # 驗證消息
    execution_history: List[Dict] = field(default_factory=list)  # 執行歷史


class ScriptInterface(ABC):
    """腳本接口抽象類"""
    
    @abstractmethod
    def get_input_parameters(self) -> List[str]:
        """獲取輸入參數"""
        pass
        
    @abstractmethod
    def get_output_parameters(self) -> List[str]:
        """獲取輸出參數"""
        pass
        
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """驗證參數"""
        pass
        
    @abstractmethod
    def build_command(self, params: Dict[str, Any]) -> str:
        """構建執行指令"""
        pass


class PythonScriptInterface(ScriptInterface):
    """Python 腳本接口"""
    
    def __init__(self, script_path: Path):
        self.script_path = Path(script_path)
        self.script_name = self.script_path.stem
        self._analyze_script()
        
    def _analyze_script(self) -> None:
        """分析腳本以提取接口信息"""
        try:
            with open(self.script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 簡單的靜態分析，實際應用中可以用 AST
            self.input_params = self._extract_cli_arguments(content)
            self.output_params = self._extract_output_patterns(content)
            
        except Exception as e:
            logger.warning(f"分析腳本失敗 {self.script_path}: {e}")
            self.input_params = []
            self.output_params = []
            
    def _extract_cli_arguments(self, content: str) -> List[str]:
        """提取 CLI 參數"""
        import re
        
        # 查找 argparse 參數定義
        arg_patterns = [
            r"add_argument\(['\"]--?([^'\"]+)['\"]",
            r"add_argument\(['\"]([^'\"]+)['\"]",
        ]
        
        arguments = []
        for pattern in arg_patterns:
            matches = re.findall(pattern, content)
            arguments.extend(matches)
            
        # 移除常見的通用參數
        common_args = {'help', 'verbose', 'output', 'debug'}
        filtered_args = [arg for arg in arguments if arg not in common_args]
        
        return filtered_args[:10]  # 限制數量
        
    def _extract_output_patterns(self, content: str) -> List[str]:
        """提取輸出模式"""
        import re
        
        # 查找常見的輸出模式
        output_patterns = [
            r"print\(['\"]([^'\"]*result[^'\"]*)['\"]",
            r"\.write\(['\"]([^'\"]*\.json)['\"]",
            r"save[^(]*\(['\"]([^'\"]*)['\"]",
            r"export[^(]*\(['\"]([^'\"]*)['\"]"
        ]
        
        outputs = []
        for pattern in output_patterns:
            matches = re.findall(pattern, content)
            outputs.extend(matches)
            
        return outputs[:5]  # 限制數量
        
    def get_input_parameters(self) -> List[str]:
        return self.input_params.copy()
        
    def get_output_parameters(self) -> List[str]:
        return self.output_params.copy()
        
    def validate_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """驗證參數"""
        # 檢查腳本是否存在
        if not self.script_path.exists():
            return False, f"腳本文件不存在: {self.script_path}"
            
        # 檢查是否為 Python 文件
        if self.script_path.suffix != '.py':
            return False, f"不是 Python 文件: {self.script_path}"
            
        # 基本語法檢查
        try:
            with open(self.script_path, 'r', encoding='utf-8') as f:
                compile(f.read(), str(self.script_path), 'exec')
            return True, "腳本驗證成功"
        except SyntaxError as e:
            return False, f"語法錯誤: {e}"
        except Exception as e:
            return False, f"驗證失敗: {e}"
            
    def build_command(self, params: Dict[str, Any]) -> str:
        """構建執行指令"""
        cmd_parts = ["python", f'"{self.script_path}"']
        
        # 添加參數
        for param_name, param_value in params.items():
            if param_name in self.input_params:
                cmd_parts.extend([f"--{param_name}", str(param_value)])
                
        return " ".join(cmd_parts)


class CapabilityCommandBuilder:
    """能力指令構建器
    
    核心功能：
    1. 基於數據流路徑構建 CLI 指令序列
    2. 驗證路徑的可執行性
    3. 處理參數傳遞和數據流轉
    4. 支援指令串聯執行
    5. 提供執行歷史和性能統計
    """
    
    def __init__(self, project_root: Path):
        """初始化指令構建器
        
        Args:
            project_root: 項目根目錄
        """
        self.project_root = Path(project_root)
        self.script_interfaces: Dict[str, ScriptInterface] = {}
        self.command_sequences: Dict[str, CommandSequence] = {}
        
        # 腳本查找路徑
        self.script_search_paths = [
            self.project_root / "services" / "core" / "aiva_core",
            self.project_root / "services",
            self.project_root,
        ]
        
        logger.info(f"初始化指令構建器: {project_root}")
        self._scan_available_scripts()
        
    def _scan_available_scripts(self) -> None:
        """掃描可用的腳本"""
        for search_path in self.script_search_paths:
            if not search_path.exists():
                continue
                
            # 遞歸查找 Python 腳本
            for py_file in search_path.rglob("*.py"):
                # 跳過特殊文件
                if py_file.name.startswith('__') or py_file.name.startswith('test_'):
                    continue
                    
                script_name = py_file.stem
                try:
                    interface = PythonScriptInterface(py_file)
                    self.script_interfaces[script_name] = interface
                except Exception as e:
                    logger.debug(f"無法創建腳本接口 {py_file}: {e}")
                    
        logger.info(f"發現 {len(self.script_interfaces)} 個可用腳本")
        
    def find_script_path(self, script_name: str) -> Optional[Path]:
        """查找腳本路徑"""
        # 首先嘗試從已知接口中查找
        if script_name in self.script_interfaces:
            interface = self.script_interfaces[script_name]
            if hasattr(interface, 'script_path'):
                return interface.script_path
                
        # 在搜索路徑中查找
        for search_path in self.script_search_paths:
            if not search_path.exists():
                continue
                
            # 查找確切匹配
            exact_match = search_path / f"{script_name}.py"
            if exact_match.exists():
                return exact_match
                
            # 遞歸查找
            for py_file in search_path.rglob(f"*{script_name}*.py"):
                if py_file.stem == script_name:
                    return py_file
                    
        return None
        
    def validate_path_executability(self, nodes: List[str]) -> Tuple[bool, List[str]]:
        """驗證路徑的可執行性"""
        validation_messages = []
        all_valid = True
        
        for node in nodes:
            script_path = self.find_script_path(node)
            
            if not script_path:
                validation_messages.append(f"❌ 未找到腳本: {node}")
                all_valid = False
                continue
                
            # 檢查腳本接口
            if node in self.script_interfaces:
                interface = self.script_interfaces[node]
                is_valid, message = interface.validate_parameters({})
                
                if is_valid:
                    validation_messages.append(f"✅ {node}: {message}")
                else:
                    validation_messages.append(f"❌ {node}: {message}")
                    all_valid = False
            else:
                validation_messages.append(f"⚠️  {node}: 無法創建腳本接口")
                
        return all_valid, validation_messages
        
    def build_command_sequence(self, capability_name: str, 
                             nodes: List[str], 
                             base_params: Dict[str, Any] = None) -> CommandSequence:
        """構建指令序列"""
        if base_params is None:
            base_params = {}
            
        sequence_id = f"{capability_name}_{len(self.command_sequences)}"
        steps = []
        validation_messages = []
        total_time = 0.0
        
        # 驗證路徑可執行性
        is_valid, path_validation = self.validate_path_executability(nodes)
        validation_messages.extend(path_validation)
        
        # 構建各個步驟
        current_params = base_params.copy()
        
        for i, node in enumerate(nodes):
            try:
                step = self._build_command_step(node, current_params, i)
                steps.append(step)
                total_time += step.execution_time
                
                # 模擬參數傳遞 (簡化版本)
                for output_param in step.output_params:
                    # 輸出參數成為下一步的輸入
                    current_params[f"{node}_{output_param}"] = f"output_from_{node}"
                    
            except Exception as e:
                validation_messages.append(f"❌ 構建步驟失敗 {node}: {e}")
                is_valid = False
                
        # 確定驗證狀態
        if is_valid and len(steps) == len(nodes):
            validation_status = "valid"
        elif steps:
            validation_status = "warning"
        else:
            validation_status = "invalid"
            
        sequence = CommandSequence(
            sequence_id=sequence_id,
            capability_name=capability_name,
            steps=steps,
            total_estimated_time=total_time,
            validation_status=validation_status,
            validation_messages=validation_messages
        )
        
        self.command_sequences[sequence_id] = sequence
        logger.info(f"構建指令序列: {sequence_id} ({validation_status})")
        
        return sequence
        
    def _build_command_step(self, script_name: str, 
                          available_params: Dict[str, Any], 
                          step_index: int) -> CommandStep:
        """構建單個指令步驟"""
        script_path = self.find_script_path(script_name)
        
        if not script_path:
            raise ValueError(f"未找到腳本: {script_name}")
            
        # 獲取腳本接口
        if script_name in self.script_interfaces:
            interface = self.script_interfaces[script_name]
            input_params = interface.get_input_parameters()
            output_params = interface.get_output_parameters()
        else:
            # 創建臨時接口
            interface = PythonScriptInterface(script_path)
            input_params = interface.get_input_parameters()
            output_params = interface.get_output_parameters()
            
        # 構建參數映射
        step_params = {}
        for param in input_params:
            if param in available_params:
                step_params[param] = available_params[param]
            else:
                # 提供默認值
                step_params[param] = f"default_{param}"
                
        # 構建執行指令
        command = interface.build_command(step_params)
        
        # 驗證可執行性
        is_valid, _ = interface.validate_parameters(step_params)
        
        # 估算執行時間 (簡化版本)
        estimated_time = self._estimate_execution_time(script_name, step_params)
        
        return CommandStep(
            script_name=script_name,
            script_path=str(script_path),
            command=command,
            parameters=step_params,
            input_params=input_params,
            output_params=output_params,
            is_executable=is_valid,
            execution_time=estimated_time
        )
        
    def _estimate_execution_time(self, script_name: str, params: Dict[str, Any]) -> float:
        """估算腳本執行時間"""
        # 基於腳本類型的基礎時間 (秒)
        base_times = {
            "analyzer": 30.0,    # 分析類腳本較耗時
            "trainer": 300.0,    # 訓練腳本非常耗時
            "core": 10.0,        # 核心腳本中等耗時
            "models": 60.0,      # 模型相關腳本較耗時
            "orchestrator": 15.0, # 編排腳本中等耗時
            "store": 5.0,        # 存儲腳本較快
            "registry": 5.0,     # 註冊腳本較快
        }
        
        # 基於腳本名稱匹配
        for keyword, base_time in base_times.items():
            if keyword in script_name.lower():
                return base_time
                
        return 10.0  # 默認時間
        
    def execute_command_sequence(self, sequence_id: str, 
                               dry_run: bool = True) -> Dict[str, Any]:
        """執行指令序列"""
        if sequence_id not in self.command_sequences:
            raise ValueError(f"序列不存在: {sequence_id}")
            
        sequence = self.command_sequences[sequence_id]
        execution_result = {
            "sequence_id": sequence_id,
            "start_time": "2025-12-06",
            "dry_run": dry_run,
            "steps_executed": 0,
            "steps_failed": 0,
            "total_steps": len(sequence.steps),
            "execution_log": [],
            "success": False
        }
        
        if sequence.validation_status == "invalid":
            execution_result["execution_log"].append("❌ 序列驗證失敗，無法執行")
            return execution_result
            
        logger.info(f"開始執行序列: {sequence_id} (dry_run: {dry_run})")
        
        for i, step in enumerate(sequence.steps):
            step_result = self._execute_step(step, dry_run)
            execution_result["execution_log"].append(step_result)
            
            if step_result["success"]:
                execution_result["steps_executed"] += 1
            else:
                execution_result["steps_failed"] += 1
                
                # 失敗時是否繼續執行 (可配置)
                if not dry_run:
                    break
                    
        execution_result["success"] = execution_result["steps_failed"] == 0
        
        # 記錄執行歷史
        sequence.execution_history.append(execution_result)
        
        logger.info(f"序列執行完成: {execution_result['success']}")
        return execution_result
        
    def _execute_step(self, step: CommandStep, dry_run: bool) -> Dict[str, Any]:
        """執行單個步驟"""
        step_result = {
            "script_name": step.script_name,
            "command": step.command,
            "success": False,
            "output": "",
            "error": "",
            "execution_time": 0.0
        }
        
        if dry_run:
            # 乾運行模式
            step_result["success"] = step.is_executable
            step_result["output"] = f"[DRY RUN] 模擬執行: {step.command}"
            return step_result
            
        if not step.is_executable:
            step_result["error"] = "步驟不可執行"
            return step_result
            
        try:
            # 實際執行
            import time
            start_time = time.time()
            
            result = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=300  # 5分鐘超時
            )
            
            step_result["execution_time"] = time.time() - start_time
            step_result["success"] = result.returncode == 0
            step_result["output"] = result.stdout
            step_result["error"] = result.stderr
            
        except subprocess.TimeoutExpired:
            step_result["error"] = "執行超時"
        except Exception as e:
            step_result["error"] = f"執行異常: {e}"
            
        return step_result
        
    def get_command_sequences_by_capability(self, capability_name: str) -> List[CommandSequence]:
        """獲取特定能力的指令序列"""
        return [seq for seq in self.command_sequences.values() 
                if seq.capability_name == capability_name]
        
    def export_command_sequences(self, output_dir: Path) -> None:
        """導出指令序列"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 準備導出數據
        export_data = {
            "export_timestamp": "2025-12-06",
            "project_root": str(self.project_root),
            "total_sequences": len(self.command_sequences),
            "available_scripts": list(self.script_interfaces.keys()),
            "sequences": {}
        }
        
        for seq_id, sequence in self.command_sequences.items():
            export_data["sequences"][seq_id] = {
                "capability_name": sequence.capability_name,
                "validation_status": sequence.validation_status,
                "total_steps": len(sequence.steps),
                "estimated_time": sequence.total_estimated_time,
                "steps": [
                    {
                        "script_name": step.script_name,
                        "command": step.command,
                        "is_executable": step.is_executable,
                        "input_params": step.input_params,
                        "output_params": step.output_params
                    }
                    for step in sequence.steps
                ],
                "validation_messages": sequence.validation_messages,
                "execution_count": len(sequence.execution_history)
            }
            
        # 保存 JSON 文件
        json_file = output_dir / "command_sequences.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"指令序列已導出至: {json_file}")
        
        # 生成可執行腳本
        self._generate_executable_scripts(output_dir)
        
    def _generate_executable_scripts(self, output_dir: Path) -> None:
        """生成可執行腳本"""
        scripts_dir = output_dir / "executable_scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        for seq_id, sequence in self.command_sequences.items():
            if sequence.validation_status != "valid":
                continue
                
            script_content = [
                "#!/bin/bash",
                f"# 自動生成的執行腳本: {sequence.capability_name}",
                f"# 序列 ID: {seq_id}",
                f"# 預估執行時間: {sequence.total_estimated_time:.1f} 秒",
                "",
                "set -e  # 遇到錯誤時停止",
                ""
            ]
            
            for i, step in enumerate(sequence.steps):
                script_content.extend([
                    f"echo '步驟 {i+1}/{len(sequence.steps)}: 執行 {step.script_name}'",
                    step.command,
                    "echo '步驟完成'",
                    ""
                ])
                
            script_content.append("echo '所有步驟執行完成'")
            
            # 保存腳本
            script_file = scripts_dir / f"{sequence.capability_name}_execution.sh"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(script_content))
                
            # 設置可執行權限 (在 Unix 系統上)
            try:
                script_file.chmod(0o755)
            except:
                pass  # Windows 系統忽略
                
        logger.info(f"可執行腳本已生成至: {scripts_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CLI 指令構建器")
    parser.add_argument("--project-root", default=".", help="項目根目錄")
    parser.add_argument("--capability", help="指定能力名稱")
    parser.add_argument("--nodes", nargs="+", help="節點序列")
    parser.add_argument("--execute", help="執行指令序列 ID")
    parser.add_argument("--dry-run", action="store_true", help="乾運行模式")
    parser.add_argument("--output", default="./command_sequences_results", help="輸出目錄")
    parser.add_argument("--verbose", action="store_true", help="詳細輸出")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        
    # 初始化構建器
    builder = CapabilityCommandBuilder(Path(args.project_root))
    
    if args.capability and args.nodes:
        # 構建指令序列
        sequence = builder.build_command_sequence(args.capability, args.nodes)
        print(f"構建完成: {sequence.sequence_id} ({sequence.validation_status})")
        for msg in sequence.validation_messages:
            print(f"  {msg}")
            
    if args.execute:
        # 執行指令序列
        result = builder.execute_command_sequence(args.execute, args.dry_run)
        print(f"執行結果: {'成功' if result['success'] else '失敗'}")
        for log in result["execution_log"]:
            print(f"  {log['script_name']}: {'✅' if log['success'] else '❌'}")
            
    # 導出結果
    builder.export_command_sequences(Path(args.output))
    print(f"結果已保存至: {args.output}")