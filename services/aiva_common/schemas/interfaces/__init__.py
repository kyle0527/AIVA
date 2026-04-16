"""
AIVA Interfaces Domain Schemas
==============================

外部接口領域模型，包含：
- API標準 (OpenAPI/AsyncAPI/GraphQL)
- 命令行界面
- 異步工具

此領域專注於與外部系統的交互接口。
"""

from .api_standards import *
from .async_utils import *
from .cli import *

__all__ = [
    # API標準 (api_standards.py)
    "OpenAPIDocument",
    "OpenAPIInfo",
    "OpenAPIServer",
    "OpenAPIPathItem",
    "OpenAPIOperation",
    "OpenAPIParameter",
    "OpenAPISchema",
    "OpenAPIComponents",
    "OpenAPISecurityScheme",
    "AsyncAPIDocument",
    "AsyncAPIInfo",
    "AsyncAPIServer",
    "AsyncAPIChannel",
    "AsyncAPIMessage",
    "AsyncAPIOperation",
    "AsyncAPIComponents",
    "GraphQLSchema",
    "GraphQLTypeDefinition",
    "GraphQLFieldDefinition",
    "GraphQLDirectiveDefinition",
    # CLI界面 (cli.py)
    "CLIParameter",
    "CLICommand",
    "CLIExecutionResult",
    "CLISession",
    "CLIConfiguration",
    "CLIMetrics",
    # 異步工具 (async_utils.py)
    "AsyncTaskConfig",
    "AsyncTaskResult",
    "RetryConfig",
    "ResourceLimits",
    "AsyncBatchConfig",
    "AsyncBatchResult",
]
