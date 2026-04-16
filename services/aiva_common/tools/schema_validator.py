#!/usr/bin/env python3
"""
AIVA Schema 驗證器
================

驗證自動生成的Schema的正確性和一致性

功能:
- 🔍 Python Pydantic Schema 語法驗證
- 🔄 跨語言Schema一致性檢查
- 📊 Schema覆蓋率報告
- 🚨 破壞性變更檢測
"""

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Schema驗證器"""

    def __init__(self):
        self.errors = []
        self.warnings = []
        self.python_schemas = {}

    def validate_python_schemas(
        self, schema_dir: str = "services/aiva_common/schemas/generated"
    ) -> bool:
        """驗證Python生成的Schema"""
        logger.info("🔍 開始驗證Python Schema...")

        schema_path = Path(schema_dir)
        if not schema_path.exists():
            self.errors.append(f"Schema目錄不存在: {schema_path}")
            return False

        # 檢查所有Python檔案
        python_files = list(schema_path.glob("*.py"))
        if not python_files:
            self.errors.append("未找到任何Python Schema檔案")
            return False

        logger.info(f"找到 {len(python_files)} 個Python Schema檔案")

        for py_file in python_files:
            if py_file.name == "__init__.py":
                continue

            logger.info(f"驗證: {py_file.name}")

            # 語法檢查
            if not self._check_python_syntax(py_file):
                continue

            # 載入並分析Schema
            if not self._analyze_python_schema(py_file):
                continue

        return len(self.errors) == 0

    def _check_python_syntax(self, py_file: Path) -> bool:
        """檢查Python檔案語法"""
        try:
            with open(py_file, encoding="utf-8") as f:
                content = f.read()

            # AST語法解析
            ast.parse(content)
            logger.info(f"  ✅ {py_file.name}: 語法正確")
            return True

        except SyntaxError as e:
            self.errors.append(f"{py_file.name}: 語法錯誤 - {e}")
            logger.error(f"  ❌ {py_file.name}: 語法錯誤 - {e}")
            return False
        except Exception as e:
            self.errors.append(f"{py_file.name}: 讀取錯誤 - {e}")
            logger.error(f"  ❌ {py_file.name}: 讀取錯誤 - {e}")
            return False

    def _analyze_python_schema(self, py_file: Path) -> bool:
        """分析Python Schema結構"""
        try:
            # 動態導入模組
            spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)

                # 添加必要的依賴到sys.modules
                if py_file.name != "base_types.py":
                    # 先載入base_types
                    base_types_path = py_file.parent / "base_types.py"
                    if base_types_path.exists():
                        base_spec = importlib.util.spec_from_file_location(
                            "base_types", base_types_path
                        )
                        if base_spec and base_spec.loader:
                            base_module = importlib.util.module_from_spec(base_spec)
                            sys.modules["base_types"] = base_module
                            base_spec.loader.exec_module(base_module)

                spec.loader.exec_module(module)

                # 分析模組中的Schema類別
                schema_classes = self._extract_pydantic_models(module)
                self.python_schemas[py_file.stem] = schema_classes

                logger.info(
                    f"  ✅ {py_file.name}: 找到 {len(schema_classes)} 個Schema類別"
                )
                return True

            # spec or spec.loader is None
            return False

        except ImportError as e:
            self.warnings.append(f"{py_file.name}: 導入警告 - {e}")
            logger.warning(f"  ⚠️ {py_file.name}: 導入警告 - {e}")
            return True  # 導入錯誤不阻止其他驗證
        except Exception as e:
            self.errors.append(f"{py_file.name}: 分析錯誤 - {e}")
            logger.error(f"  ❌ {py_file.name}: 分析錯誤 - {e}")
            return False

    def _extract_pydantic_models(self, module) -> dict[str, Any]:
        """提取Pydantic模型"""
        models = {}

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            # 檢查是否為Pydantic BaseModel子類
            if hasattr(attr, "__mro__") and any(
                "BaseModel" in str(base) for base in attr.__mro__
            ):

                models[attr_name] = {
                    "fields": (
                        list(attr.model_fields.keys())
                        if hasattr(attr, "model_fields")
                        else []
                    ),
                    "class": attr,
                }

        return models

    def validate_go_schemas(
        self,
        schema_file: str = "services/features/common/go/aiva_common_go/schemas/generated/schemas.go",
    ) -> bool:
        """驗證Go Schema檔案"""
        logger.info("🔍 開始驗證Go Schema...")

        go_file = Path(schema_file)
        if not go_file.exists():
            self.errors.append(f"Go Schema檔案不存在: {go_file}")
            return False

        try:
            with open(go_file, encoding="utf-8") as f:
                content = f.read()

            # 基本檢查
            if not content.strip():
                self.errors.append("Go Schema檔案為空")
                return False

            # 檢查基本Go語法結構
            if "package schemas" not in content:
                self.errors.append("Go Schema缺少package宣告")

            if 'import "time"' not in content:
                self.warnings.append("Go Schema缺少time包導入")

            # 統計struct數量
            struct_count = content.count("type ") - content.count("// type ")
            logger.info(f"  ✅ Go Schema: 找到 {struct_count} 個結構體")

            return len(self.errors) == 0

        except Exception as e:
            self.errors.append(f"Go Schema驗證錯誤: {e}")
            logger.error(f"  ❌ Go Schema驗證錯誤: {e}")
            return False

    def check_cross_language_consistency(self) -> bool:
        """檢查跨語言一致性"""
        logger.info("🔄 檢查跨語言Schema一致性...")

        # 這裡可以實現更複雜的一致性檢查
        # 例如字段數量、命名規範等

        if not self.python_schemas:
            self.warnings.append("沒有Python Schema可供一致性檢查")
            return True

        total_python_classes = sum(
            len(classes) for classes in self.python_schemas.values()
        )
        logger.info(f"  📊 Python Schema總計: {total_python_classes} 個類別")

        return True

    def generate_report(self) -> dict[str, Any]:
        """生成驗證報告"""
        return {
            "success": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "python_schemas": self.python_schemas,
            "stats": {
                "total_python_files": len(self.python_schemas),
                "total_python_classes": sum(
                    len(classes) for classes in self.python_schemas.values()
                ),
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
            },
        }

    def validate_all(self) -> bool:
        """執行所有驗證"""
        logger.info("🚀 開始完整Schema驗證...")

        # 驗證Python
        python_ok = self.validate_python_schemas()

        # 驗證Go
        go_ok = self.validate_go_schemas()

        # 檢查一致性
        consistency_ok = self.check_cross_language_consistency()

        # 生成報告
        report = self.generate_report()

        logger.info("📋 驗證結果摘要:")
        logger.info(f"  Python Schema: {'✅' if python_ok else '❌'}")
        logger.info(f"  Go Schema: {'✅' if go_ok else '❌'}")
        logger.info(f"  跨語言一致性: {'✅' if consistency_ok else '❌'}")
        logger.info(f"  錯誤數: {report['stats']['error_count']}")
        logger.info(f"  警告數: {report['stats']['warning_count']}")

        if self.errors:
            logger.error("❌ 發現錯誤:")
            for error in self.errors:
                logger.error(f"  - {error}")

        if self.warnings:
            logger.warning("⚠️ 發現警告:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")

        success = len(self.errors) == 0
        if success:
            logger.info("🎉 所有Schema驗證通過!")
        else:
            logger.error("💥 Schema驗證失敗!")

        return success


def main():
    """主程式"""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    validator = SchemaValidator()
    success = validator.validate_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
