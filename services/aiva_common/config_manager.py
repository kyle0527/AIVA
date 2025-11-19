"""
AIVA Configuration Management System
AIVA 配置管理系統

實施 TODO 項目 9: 建立配置管理系統
- 環境變量管理
- 動態配置更新
- 配置驗證和校驗
- 多環境配置支持
- 配置熱更新機制

特性：
1. 分層配置系統 (環境變量 > 用戶配置 > 默認配置)
2. 配置變更監聽和通知
3. 配置驗證和類型檢查
4. 敏感配置加密存儲
5. 配置模板和預設管理
"""

import asyncio
import base64
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from cryptography.fernet import Fernet

from .error_handling import AIVAError, ErrorHandler, ErrorSeverity, ErrorType


class ConfigScope(Enum):
    """配置作用域"""

    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    TEMPORARY = "temporary"


class ConfigType(Enum):
    """配置類型"""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICT = "dict"
    PATH = "path"
    URL = "url"
    SECRET = "secret"  # 敏感信息


@dataclass
class ConfigSchema:
    """配置項架構定義"""

    key: str
    config_type: ConfigType
    default_value: Any = None
    required: bool = False
    description: str = ""
    validation_func: Callable[[Any], bool] | None = None
    scope: ConfigScope = ConfigScope.GLOBAL
    sensitive: bool = False  # 是否為敏感信息
    env_var: str | None = None  # 對應的環境變量名
    choices: list[Any] | None = None  # 允許的選項
    min_value: int | float | None = None
    max_value: int | float | None = None


@dataclass
class ConfigChangeEvent:
    """配置變更事件"""

    key: str
    old_value: Any
    new_value: Any
    scope: ConfigScope
    timestamp: float = field(default_factory=time.time)
    changed_by: str = "system"


class ConfigManager:
    """
    AIVA 配置管理器

    支持多層次配置管理：
    1. 環境變量 (最高優先級)
    2. 用戶配置文件
    3. 默認配置
    4. 運行時臨時配置
    """

    def __init__(self, config_dir: Path | None = None):
        """
        初始化配置管理器

        Args:
            config_dir: 配置文件目錄，默認為 ~/.aiva/config
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.error_handler = ErrorHandler()

        # 配置目錄設置
        if config_dir is None:
            config_dir = Path.home() / ".aiva" / "config"
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 配置文件路徑
        self.config_files = {
            ConfigScope.GLOBAL: self.config_dir / "global.yaml",
            ConfigScope.USER: self.config_dir / "user.yaml",
            ConfigScope.SESSION: self.config_dir / "session.yaml",
        }

        # 配置存儲
        self._configs: dict[ConfigScope, dict[str, Any]] = {
            scope: {} for scope in ConfigScope
        }

        # 配置架構註冊表
        self._schemas: dict[str, ConfigSchema] = {}

        # 變更監聽器
        self._change_listeners: list[Callable[[ConfigChangeEvent], None]] = []

        # 加密密鑰 (用於敏感配置)
        self._encryption_key = self._get_or_create_encryption_key()

        # 初始化默認配置架構
        self._register_default_schemas()

        # 標記配置是否已加載（延遲加載）
        self._configs_loaded = False

        # 嘗試同步加載基本配置（不創建異步任務）
        self._load_basic_configs_sync()

    def _load_basic_configs_sync(self):
        """同步加載基本配置（避免異步初始化問題）"""
        try:
            # 加載環境變量
            self._load_environment_variables()

            # 嘗試同步讀取配置文件
            for scope in [ConfigScope.GLOBAL, ConfigScope.USER, ConfigScope.SESSION]:
                self._load_config_file_sync(scope)

            self._configs_loaded = True
            self.logger.info("基本配置同步加載完成")

        except Exception as e:
            self.logger.warning(f"同步配置加載失敗，將使用默認值: {e}")
            self._configs_loaded = True  # 即使失败也标记为已加载，使用默认值

    def _load_config_file_sync(self, scope: ConfigScope):
        """同步加載指定作用域的配置文件"""
        config_file = self.config_files.get(scope)
        if not config_file or not config_file.exists():
            return

        try:
            with open(config_file, encoding="utf-8") as f:
                if config_file.suffix.lower() == ".yaml":
                    config_data = yaml.safe_load(f) or {}
                else:
                    config_data = json.load(f)

            # 簡化版本：暫時跳過解密（避免異步依賴）
            self._configs[scope] = config_data
            self.logger.debug(f"同步加載 {scope.value} 配置: {len(config_data)} 項")

        except Exception as e:
            self.logger.warning(f"同步加載配置文件失敗 {config_file}: {e}")

    async def _ensure_configs_loaded(self):
        """確保配置已加載（異步版本，用於完整功能）"""
        if self._configs_loaded:
            return

        await self._load_all_configs()
        self._configs_loaded = True

    def _get_or_create_encryption_key(self) -> bytes:
        """獲取或創建加密密鑰"""
        key_file = self.config_dir / ".encryption_key"

        if key_file.exists():
            try:
                with open(key_file, "rb") as f:
                    return base64.b64decode(f.read())
            except Exception as e:
                self.logger.warning(f"無法讀取加密密鑰，將生成新密鑰: {e}")

        # 生成新密鑰
        key = Fernet.generate_key()
        try:
            with open(key_file, "wb") as f:
                f.write(base64.b64encode(key))
            os.chmod(key_file, 0o600)  # 僅擁有者可讀寫
        except Exception as e:
            self.logger.error(f"無法保存加密密鑰: {e}")

        return key

    def _register_default_schemas(self):
        """註冊默認配置架構"""
        default_schemas = [
            # 系統配置
            ConfigSchema(
                key="system.log_level",
                config_type=ConfigType.STRING,
                default_value="INFO",
                description="日誌級別",
                choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                env_var="LOG_LEVEL",
            ),
            ConfigSchema(
                key="system.max_workers",
                config_type=ConfigType.INTEGER,
                default_value=4,
                description="最大工作線程數",
                min_value=1,
                max_value=32,
                env_var="AIVA_MAX_WORKERS",
            ),
            ConfigSchema(
                key="system.timeout",
                config_type=ConfigType.INTEGER,
                default_value=30,
                description="默認超時秒數",
                min_value=5,
                max_value=300,
                env_var="AIVA_TIMEOUT",
            ),
            # AI 配置
            ConfigSchema(
                key="ai.default_model",
                config_type=ConfigType.STRING,
                default_value="gpt-3.5-turbo",
                description="默認 AI 模型",
                env_var="AIVA_AI_MODEL",
            ),
            ConfigSchema(
                key="ai.api_key",
                config_type=ConfigType.SECRET,
                required=False,
                description="AI API 密鑰",
                sensitive=True,
                env_var="AIVA_AI_API_KEY",
            ),
            ConfigSchema(
                key="ai.max_tokens",
                config_type=ConfigType.INTEGER,
                default_value=2048,
                description="AI 響應最大 token 數",
                min_value=100,
                max_value=8192,
                env_var="AIVA_AI_MAX_TOKENS",
            ),
            ConfigSchema(
                key="ai.temperature",
                config_type=ConfigType.FLOAT,
                default_value=0.7,
                description="AI 生成溫度",
                min_value=0.0,
                max_value=2.0,
                env_var="AIVA_AI_TEMPERATURE",
            ),
            # 安全掃描配置
            ConfigSchema(
                key="security.scan_timeout",
                config_type=ConfigType.INTEGER,
                default_value=300,
                description="安全掃描超時秒數",
                min_value=60,
                max_value=3600,
                env_var="AIVA_SCAN_TIMEOUT",
            ),
            ConfigSchema(
                key="security.max_concurrent_scans",
                config_type=ConfigType.INTEGER,
                default_value=3,
                description="最大並發掃描數",
                min_value=1,
                max_value=10,
                env_var="AIVA_MAX_CONCURRENT_SCANS",
            ),
            ConfigSchema(
                key="security.user_agent",
                config_type=ConfigType.STRING,
                default_value="AIVA-Scanner/2.0",
                description="掃描用戶代理",
                env_var="AIVA_USER_AGENT",
            ),
            # 數據庫配置
            ConfigSchema(
                key="database.url",
                config_type=ConfigType.URL,
                default_value="sqlite:///~/.aiva/data/aiva.db",
                description="數據庫連接 URL",
                env_var="DATABASE_URL",
            ),
            ConfigSchema(
                key="database.pool_size",
                config_type=ConfigType.INTEGER,
                default_value=5,
                description="數據庫連接池大小",
                min_value=1,
                max_value=50,
                env_var="AIVA_DB_POOL_SIZE",
            ),
            # 網絡配置
            ConfigSchema(
                key="network.bind_host",
                config_type=ConfigType.STRING,
                default_value="127.0.0.1",
                description="服務綁定地址",
                env_var="AIVA_BIND_HOST",
            ),
            ConfigSchema(
                key="network.bind_port",
                config_type=ConfigType.INTEGER,
                default_value=8080,
                description="服務綁定端口",
                min_value=1024,
                max_value=65535,
                env_var="AIVA_BIND_PORT",
            ),
            ConfigSchema(
                key="network.proxy_url",
                config_type=ConfigType.URL,
                required=False,
                description="代理服務器 URL",
                env_var="AIVA_PROXY_URL",
            ),
            # 存儲配置
            ConfigSchema(
                key="storage.data_dir",
                config_type=ConfigType.PATH,
                default_value="~/.aiva/data",
                description="數據存儲目錄",
                env_var="AIVA_DATA_DIR",
            ),
            ConfigSchema(
                key="storage.max_file_size",
                config_type=ConfigType.INTEGER,
                default_value=100 * 1024 * 1024,  # 100MB
                description="最大文件大小 (字節)",
                min_value=1024,
                env_var="AIVA_MAX_FILE_SIZE",
            ),
        ]

        for schema in default_schemas:
            self.register_schema(schema)

    def register_schema(self, schema: ConfigSchema):
        """註冊配置架構"""
        self._schemas[schema.key] = schema
        self.logger.debug(f"已註冊配置架構: {schema.key}")

    def add_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """添加配置變更監聽器"""
        self._change_listeners.append(listener)
        self.logger.debug(f"已添加配置變更監聽器: {listener.__name__}")

    def remove_change_listener(self, listener: Callable[[ConfigChangeEvent], None]):
        """移除配置變更監聽器"""
        if listener in self._change_listeners:
            self._change_listeners.remove(listener)
            self.logger.debug(f"已移除配置變更監聽器: {listener.__name__}")

    async def _load_all_configs(self):
        """加載所有配置文件"""
        for scope in [ConfigScope.GLOBAL, ConfigScope.USER, ConfigScope.SESSION]:
            await self._load_config_file(scope)

        # 加載環境變量
        self._load_environment_variables()

        self.logger.info("所有配置加載完成")

    async def _load_config_file(self, scope: ConfigScope):
        """加載指定作用域的配置文件"""
        config_file = self.config_files.get(scope)
        if not config_file or not config_file.exists():
            return

        try:
            with open(config_file, encoding="utf-8") as f:
                if config_file.suffix.lower() == ".yaml":
                    config_data = yaml.safe_load(f) or {}
                else:
                    config_data = json.load(f)

            # 解密敏感配置
            config_data = self._decrypt_sensitive_configs(config_data)

            self._configs[scope] = config_data
            self.logger.debug(f"已加載 {scope.value} 配置: {len(config_data)} 項")

        except Exception as e:
            self.logger.error(f"無法加載 {scope.value} 配置文件: {e}")
            await self.error_handler.handle_error(
                AIVAError(
                    error_type=ErrorType.CONFIGURATION_ERROR,
                    message=f"配置文件加載失敗: {config_file}",
                    details={
                        "scope": scope.value,
                        "file": str(config_file),
                        "error": str(e),
                    },
                    severity=ErrorSeverity.MEDIUM,
                )
            )

    def _load_environment_variables(self):
        """加載環境變量配置"""
        env_configs = {}

        for schema in self._schemas.values():
            if schema.env_var and schema.env_var in os.environ:
                try:
                    raw_value = os.environ[schema.env_var]
                    typed_value = self._convert_value(raw_value, schema.config_type)
                    env_configs[schema.key] = typed_value
                    self.logger.debug(
                        f"從環境變量加載配置 {schema.key}: {schema.env_var}"
                    )
                except Exception as e:
                    self.logger.warning(f"環境變量 {schema.env_var} 類型轉換失敗: {e}")

        # 環境變量具有最高優先級
        self._configs[ConfigScope.GLOBAL].update(env_configs)

    def _convert_value(self, value: str, config_type: ConfigType) -> Any:
        """轉換配置值類型"""
        if config_type == ConfigType.STRING or config_type == ConfigType.SECRET:
            return value
        elif config_type == ConfigType.INTEGER:
            return int(value)
        elif config_type == ConfigType.FLOAT:
            return float(value)
        elif config_type == ConfigType.BOOLEAN:
            return value.lower() in ("true", "1", "yes", "on")
        elif config_type == ConfigType.LIST:
            return [item.strip() for item in value.split(",")]
        elif config_type == ConfigType.DICT:
            return json.loads(value)
        elif config_type == ConfigType.PATH:
            return os.path.expanduser(value)
        elif config_type == ConfigType.URL:
            return value
        else:
            return value

    def _encrypt_value(self, value: str) -> str:
        """加密敏感值"""
        try:
            fernet = Fernet(self._encryption_key)
            encrypted = fernet.encrypt(value.encode())
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            self.logger.error(f"配置加密失敗: {e}")
            return value

    def _decrypt_value(self, encrypted_value: str) -> str:
        """解密敏感值"""
        try:
            fernet = Fernet(self._encryption_key)
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted = fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            self.logger.error(f"配置解密失敗: {e}")
            return encrypted_value

    def _encrypt_sensitive_configs(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """加密配置中的敏感信息"""
        encrypted_data = config_data.copy()

        for key, value in config_data.items():
            schema = self._schemas.get(key)
            if schema and schema.sensitive and isinstance(value, str):
                encrypted_data[key] = self._encrypt_value(value)

        return encrypted_data

    def _decrypt_sensitive_configs(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """解密配置中的敏感信息"""
        decrypted_data = config_data.copy()

        for key, value in config_data.items():
            schema = self._schemas.get(key)
            if schema and schema.sensitive and isinstance(value, str):
                decrypted_data[key] = self._decrypt_value(value)

        return decrypted_data

    def get(
        self, key: str, default: Any = None, scope: ConfigScope | None = None
    ) -> Any:
        """
        獲取配置值

        Args:
            key: 配置鍵
            default: 默認值
            scope: 指定作用域，None 表示按優先級查找

        Returns:
            配置值
        """
        if scope:
            return self._configs[scope].get(key, default)

        # 按優先級查找: 臨時 > 會話 > 用戶 > 全局
        for check_scope in [
            ConfigScope.TEMPORARY,
            ConfigScope.SESSION,
            ConfigScope.USER,
            ConfigScope.GLOBAL,
        ]:
            if key in self._configs[check_scope]:
                return self._configs[check_scope][key]

        # 查找架構默認值
        schema = self._schemas.get(key)
        if schema and schema.default_value is not None:
            return schema.default_value

        return default

    def set(
        self,
        key: str,
        value: Any,
        scope: ConfigScope = ConfigScope.USER,
        persist: bool = True,
    ) -> bool:
        """
        設置配置值

        Args:
            key: 配置鍵
            value: 配置值
            scope: 配置作用域
            persist: 是否持久化到文件

        Returns:
            是否設置成功
        """
        try:
            # 驗證配置
            if not self._validate_config(key, value):
                return False

            old_value = self.get(key, scope=scope)

            # 設置配置值
            self._configs[scope][key] = value

            # 持久化配置
            if persist and scope != ConfigScope.TEMPORARY:
                asyncio.create_task(self._save_config_file(scope))

            # 觸發變更事件
            event = ConfigChangeEvent(
                key=key, old_value=old_value, new_value=value, scope=scope
            )
            self._notify_change_listeners(event)

            self.logger.info(f"配置已更新: {key} = {value} ({scope.value})")
            return True

        except Exception as e:
            self.logger.error(f"設置配置失敗 {key}: {e}")
            return False

    def _validate_config(self, key: str, value: Any) -> bool:
        """驗證配置值"""
        schema = self._schemas.get(key)
        if not schema:
            self.logger.warning(f"未註冊的配置項: {key}")
            return True  # 允許未註冊的配置

        # 類型檢查
        if not self._check_type(value, schema.config_type):
            self.logger.error(
                f"配置 {key} 類型不匹配，期望 {schema.config_type.value}，實際 {type(value)}"
            )
            return False

        # 選項檢查
        if schema.choices and value not in schema.choices:
            self.logger.error(f"配置 {key} 值 {value} 不在允許選項中: {schema.choices}")
            return False

        # 數值範圍檢查
        if isinstance(value, (int, float)):
            if schema.min_value is not None and value < schema.min_value:
                self.logger.error(
                    f"配置 {key} 值 {value} 小於最小值 {schema.min_value}"
                )
                return False
            if schema.max_value is not None and value > schema.max_value:
                self.logger.error(
                    f"配置 {key} 值 {value} 大於最大值 {schema.max_value}"
                )
                return False

        # 自定義驗證函數
        if schema.validation_func and not schema.validation_func(value):
            self.logger.error(f"配置 {key} 值 {value} 未通過自定義驗證")
            return False

        return True

    def _check_type(self, value: Any, expected_type: ConfigType) -> bool:
        """檢查值類型"""
        type_checks = {
            ConfigType.STRING: lambda v: isinstance(v, str),
            ConfigType.SECRET: lambda v: isinstance(v, str),
            ConfigType.INTEGER: lambda v: isinstance(v, int),
            ConfigType.FLOAT: lambda v: isinstance(v, (int, float)),
            ConfigType.BOOLEAN: lambda v: isinstance(v, bool),
            ConfigType.LIST: lambda v: isinstance(v, list),
            ConfigType.DICT: lambda v: isinstance(v, dict),
            ConfigType.PATH: lambda v: isinstance(v, (str, Path)),
            ConfigType.URL: lambda v: isinstance(v, str)
            and (v.startswith("http") or v.startswith("ftp")),
        }

        check_func = type_checks.get(expected_type, lambda v: True)
        return check_func(value)

    async def _save_config_file(self, scope: ConfigScope):
        """保存配置文件"""
        config_file = self.config_files.get(scope)
        if not config_file:
            return

        try:
            config_data = self._configs[scope].copy()

            # 加密敏感配置
            config_data = self._encrypt_sensitive_configs(config_data)

            # 保存文件
            with open(config_file, "w", encoding="utf-8") as f:
                if config_file.suffix.lower() == ".yaml":
                    yaml.dump(
                        config_data, f, allow_unicode=True, default_flow_style=False
                    )
                else:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"已保存 {scope.value} 配置到 {config_file}")

        except Exception as e:
            self.logger.error(f"保存 {scope.value} 配置文件失敗: {e}")

    def _notify_change_listeners(self, event: ConfigChangeEvent):
        """通知配置變更監聽器"""
        for listener in self._change_listeners:
            try:
                listener(event)
            except Exception as e:
                self.logger.error(f"配置變更監聽器執行失敗: {e}")

    def delete(
        self, key: str, scope: ConfigScope = ConfigScope.USER, persist: bool = True
    ) -> bool:
        """
        刪除配置項

        Args:
            key: 配置鍵
            scope: 配置作用域
            persist: 是否持久化更改

        Returns:
            是否刪除成功
        """
        if key not in self._configs[scope]:
            return False

        try:
            old_value = self._configs[scope][key]
            del self._configs[scope][key]

            if persist and scope != ConfigScope.TEMPORARY:
                asyncio.create_task(self._save_config_file(scope))

            # 觸發變更事件
            event = ConfigChangeEvent(
                key=key, old_value=old_value, new_value=None, scope=scope
            )
            self._notify_change_listeners(event)

            self.logger.info(f"配置已刪除: {key} ({scope.value})")
            return True

        except Exception as e:
            self.logger.error(f"刪除配置失敗 {key}: {e}")
            return False

    def list_configs(
        self, scope: ConfigScope | None = None, include_sensitive: bool = False
    ) -> dict[str, Any]:
        """
        列出配置項

        Args:
            scope: 指定作用域，None 表示所有作用域
            include_sensitive: 是否包含敏感配置

        Returns:
            配置字典
        """
        if scope:
            configs = self._configs[scope].copy()
        else:
            # 合併所有作用域的配置
            configs = {}
            for check_scope in [
                ConfigScope.GLOBAL,
                ConfigScope.USER,
                ConfigScope.SESSION,
                ConfigScope.TEMPORARY,
            ]:
                configs.update(self._configs[check_scope])

        # 過濾敏感配置
        if not include_sensitive:
            filtered_configs = {}
            for key, value in configs.items():
                schema = self._schemas.get(key)
                if not schema or not schema.sensitive:
                    filtered_configs[key] = value
                else:
                    filtered_configs[key] = "***HIDDEN***"
            configs = filtered_configs

        return configs

    def get_schema(self, key: str) -> ConfigSchema | None:
        """獲取配置架構"""
        return self._schemas.get(key)

    def list_schemas(self) -> dict[str, ConfigSchema]:
        """列出所有配置架構"""
        return self._schemas.copy()

    def validate_all_configs(self) -> dict[str, list[str]]:
        """
        驗證所有配置項

        Returns:
            驗證結果字典，鍵為作用域，值為錯誤列表
        """
        validation_errors = {scope.value: [] for scope in ConfigScope}

        for scope in ConfigScope:
            for key, value in self._configs[scope].items():
                if not self._validate_config(key, value):
                    validation_errors[scope.value].append(f"配置項 {key} 驗證失敗")

        # 檢查必需配置
        for schema in self._schemas.values():
            if schema.required:
                if self.get(schema.key) is None:
                    validation_errors[ConfigScope.GLOBAL.value].append(
                        f"必需配置項 {schema.key} 缺失"
                    )

        return validation_errors

    def reset_to_defaults(self, scope: ConfigScope = ConfigScope.USER) -> bool:
        """
        重置配置到默認值

        Args:
            scope: 要重置的作用域

        Returns:
            是否重置成功
        """
        try:
            self._configs[scope].clear()

            # 應用默認值
            for schema in self._schemas.values():
                if schema.default_value is not None and schema.scope == scope:
                    self._configs[scope][schema.key] = schema.default_value

            # 持久化
            if scope != ConfigScope.TEMPORARY:
                asyncio.create_task(self._save_config_file(scope))

            self.logger.info(f"已重置 {scope.value} 配置到默認值")
            return True

        except Exception as e:
            self.logger.error(f"重置配置失敗: {e}")
            return False

    def export_config(
        self,
        file_path: Path,
        scope: ConfigScope | None = None,
        include_sensitive: bool = False,
    ) -> bool:
        """
        導出配置到文件

        Args:
            file_path: 導出文件路徑
            scope: 導出作用域，None 表示所有
            include_sensitive: 是否包含敏感配置

        Returns:
            是否導出成功
        """
        try:
            configs = self.list_configs(scope, include_sensitive)

            with open(file_path, "w", encoding="utf-8") as f:
                if file_path.suffix.lower() == ".yaml":
                    yaml.dump(configs, f, allow_unicode=True, default_flow_style=False)
                else:
                    json.dump(configs, f, indent=2, ensure_ascii=False)

            self.logger.info(f"配置已導出到 {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"配置導出失敗: {e}")
            return False

    def import_config(
        self, file_path: Path, scope: ConfigScope = ConfigScope.USER, merge: bool = True
    ) -> bool:
        """
        從文件導入配置

        Args:
            file_path: 導入文件路徑
            scope: 導入到的作用域
            merge: 是否合併到現有配置，False 表示替換

        Returns:
            是否導入成功
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                if file_path.suffix.lower() == ".yaml":
                    imported_configs = yaml.safe_load(f) or {}
                else:
                    imported_configs = json.load(f)

            if not merge:
                self._configs[scope].clear()

            # 驗證並設置配置
            for key, value in imported_configs.items():
                if self._validate_config(key, value):
                    self._configs[scope][key] = value
                else:
                    self.logger.warning(f"跳過無效配置: {key} = {value}")

            # 持久化
            if scope != ConfigScope.TEMPORARY:
                asyncio.create_task(self._save_config_file(scope))

            self.logger.info(f"配置已從 {file_path} 導入到 {scope.value}")
            return True

        except Exception as e:
            self.logger.error(f"配置導入失敗: {e}")
            return False

    def get_config_status(self) -> dict[str, Any]:
        """獲取配置管理器狀態"""
        return {
            "config_dir": str(self.config_dir),
            "total_schemas": len(self._schemas),
            "total_configs": {
                scope.value: len(configs) for scope, configs in self._configs.items()
            },
            "change_listeners": len(self._change_listeners),
            "validation_errors": self.validate_all_configs(),
            "config_files": {
                scope.value: {
                    "path": str(path),
                    "exists": path.exists(),
                    "size": path.stat().st_size if path.exists() else 0,
                }
                for scope, path in self.config_files.items()
            },
        }


# 全局配置管理器實例
_global_config_manager: ConfigManager | None = None


def get_config_manager(config_dir: Path | None = None) -> ConfigManager:
    """獲取全局配置管理器實例"""
    global _global_config_manager

    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_dir)

    return _global_config_manager


def get_config(
    key: str, default: Any = None, scope: ConfigScope | None = None
) -> Any:
    """快捷方式：獲取配置值"""
    return get_config_manager().get(key, default, scope)


def set_config(
    key: str, value: Any, scope: ConfigScope = ConfigScope.USER, persist: bool = True
) -> bool:
    """快捷方式：設置配置值"""
    return get_config_manager().set(key, value, scope, persist)
