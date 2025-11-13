"""
API 標準 Schema 模型 - 基於 OpenAPI 3.1、AsyncAPI 3.0、GraphQL 等官方標準

此模組實現了現代 API 標準的完整支援，用於 API 安全測試、
規範驗證、漏洞檢測等功能。

參考標準：
- OpenAPI 3.1 (https://spec.openapis.org/oas/v3.1.0)
- AsyncAPI 3.0 (https://www.asyncapi.com/docs/reference/specification/v3.0.0)
- GraphQL (https://spec.graphql.org/)
- JSON Schema Draft 2020-12 (https://json-schema.org/draft/2020-12/schema)
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

# 常用描述字串常量
DEPRECATED_DESCRIPTION = "是否已棄用"
SERVER_LIST_DESCRIPTION = "伺服器列表"

from pydantic import BaseModel, Field, HttpUrl

from ..enums.web_api_standards import (
    OpenAPIFormat,
    OpenAPIParameterLocation,
    OpenAPISchemaType,
    OpenAPISecuritySchemeType,
)

# ==================== OpenAPI 3.1 支援 ====================


class OpenAPIInfo(BaseModel):
    """OpenAPI Info 物件"""

    title: str = Field(description="API 標題")
    summary: str | None = Field(default=None, description="API 摘要")
    description: str | None = Field(default=None, description="API 描述")
    terms_of_service: HttpUrl | None = Field(
        default=None, description="服務條款 URL"
    )
    contact: Optional["OpenAPIContact"] = Field(default=None, description="聯絡資訊")
    license: Optional["OpenAPILicense"] = Field(default=None, description="授權資訊")
    version: str = Field(description="API 版本")


class OpenAPIContact(BaseModel):
    """OpenAPI Contact 物件"""

    name: str | None = Field(default=None, description="聯絡人名稱")
    url: HttpUrl | None = Field(default=None, description="聯絡 URL")
    email: str | None = Field(default=None, description="聯絡電子郵件")


class OpenAPILicense(BaseModel):
    """OpenAPI License 物件"""

    name: str = Field(description="授權名稱")
    identifier: str | None = Field(default=None, description="SPDX 授權識別符")
    url: HttpUrl | None = Field(default=None, description="授權 URL")


class OpenAPIServer(BaseModel):
    """OpenAPI Server 物件"""

    url: str = Field(description="伺服器 URL")
    description: str | None = Field(default=None, description="伺服器描述")
    variables: dict[str, "OpenAPIServerVariable"] | None = Field(
        default=None, description="伺服器變數"
    )


class OpenAPIServerVariable(BaseModel):
    """OpenAPI Server Variable 物件"""

    enum: list[str] | None = Field(default=None, description="可能的值")
    default: str = Field(description="預設值")
    description: str | None = Field(default=None, description="變數描述")


class OpenAPISchema(BaseModel):
    """OpenAPI Schema 物件（基於 JSON Schema Draft 2020-12）"""

    # JSON Schema 核心關鍵字
    type: Union["OpenAPISchemaType", list["OpenAPISchemaType"]] | None = Field(
        default=None, description="資料類型"
    )
    format: Optional["OpenAPIFormat"] = Field(default=None, description="資料格式")
    enum: list[Any] | None = Field(default=None, description="枚舉值")
    const: Any | None = Field(default=None, description="常數值")

    # 字串相關
    max_length: int | None = Field(default=None, ge=0, description="最大長度")
    min_length: int | None = Field(default=None, ge=0, description="最小長度")
    pattern: str | None = Field(default=None, description="正則表達式模式")

    # 數字相關
    multiple_of: float | None = Field(default=None, gt=0, description="倍數")
    maximum: float | None = Field(default=None, description="最大值")
    exclusive_maximum: float | None = Field(default=None, description="排他最大值")
    minimum: float | None = Field(default=None, description="最小值")
    exclusive_minimum: float | None = Field(default=None, description="排他最小值")

    # 陣列相關
    items: Union["OpenAPISchema", list["OpenAPISchema"]] | None = Field(
        default=None, description="陣列項目 Schema"
    )
    max_items: int | None = Field(default=None, ge=0, description="最大項目數")
    min_items: int | None = Field(default=None, ge=0, description="最小項目數")
    unique_items: bool | None = Field(default=None, description="唯一項目")

    # 物件相關
    properties: dict[str, "OpenAPISchema"] | None = Field(
        default=None, description="物件屬性"
    )
    additional_properties: Union[bool, "OpenAPISchema"] | None = Field(
        default=None, description="額外屬性"
    )
    required: list[str] | None = Field(default=None, description="必需屬性")
    max_properties: int | None = Field(default=None, ge=0, description="最大屬性數")
    min_properties: int | None = Field(default=None, ge=0, description="最小屬性數")

    # 組合 Schema
    all_of: list["OpenAPISchema"] | None = Field(
        default=None, description="所有條件"
    )
    any_of: list["OpenAPISchema"] | None = Field(
        default=None, description="任一條件"
    )
    one_of: list["OpenAPISchema"] | None = Field(
        default=None, description="單一條件"
    )
    not_schema: Optional["OpenAPISchema"] = Field(
        default=None, alias="not", description="否定條件"
    )

    # 條件 Schema
    if_schema: Optional["OpenAPISchema"] = Field(
        default=None, alias="if", description="條件判斷"
    )
    then_schema: Optional["OpenAPISchema"] = Field(
        default=None, alias="then", description="條件成立"
    )
    else_schema: Optional["OpenAPISchema"] = Field(
        default=None, alias="else", description="條件不成立"
    )

    # 註解
    title: str | None = Field(default=None, description="標題")
    description: str | None = Field(default=None, description="描述")
    default: Any | None = Field(default=None, description="預設值")
    examples: list[Any] | None = Field(default=None, description="範例值")
    deprecated: bool | None = Field(default=None, description=DEPRECATED_DESCRIPTION)
    read_only: bool | None = Field(default=None, description="是否唯讀")
    write_only: bool | None = Field(default=None, description="是否唯寫")

    # OpenAPI 擴展
    discriminator: Optional["OpenAPIDiscriminator"] = Field(
        default=None, description="判別器"
    )
    xml: Optional["OpenAPIXML"] = Field(default=None, description="XML 元資料")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )
    example: Any | None = Field(default=None, description="範例（單一）")


class OpenAPIDiscriminator(BaseModel):
    """OpenAPI Discriminator 物件"""

    property_name: str = Field(description="判別屬性名稱")
    mapping: dict[str, str] | None = Field(default=None, description="值對應映射")


class OpenAPIXML(BaseModel):
    """OpenAPI XML 物件"""

    name: str | None = Field(default=None, description="XML 元素名稱")
    namespace: str | None = Field(default=None, description="XML 命名空間")
    prefix: str | None = Field(default=None, description="XML 前綴")
    attribute: bool | None = Field(default=None, description="是否為屬性")
    wrapped: bool | None = Field(default=None, description="是否包裝")


class OpenAPIExternalDocumentation(BaseModel):
    """OpenAPI External Documentation 物件"""

    description: str | None = Field(default=None, description="文件描述")
    url: HttpUrl = Field(description="文件 URL")


class OpenAPIParameter(BaseModel):
    """OpenAPI Parameter 物件"""

    name: str = Field(description="參數名稱")
    in_: OpenAPIParameterLocation = Field(alias="in", description="參數位置")
    description: str | None = Field(default=None, description="參數描述")
    required: bool | None = Field(default=None, description="是否必需")
    deprecated: bool | None = Field(default=None, description=DEPRECATED_DESCRIPTION)
    allow_empty_value: bool | None = Field(default=None, description="是否允許空值")

    # Parameter Serialization
    style: str | None = Field(default=None, description="序列化樣式")
    explode: bool | None = Field(default=None, description="是否展開")
    allow_reserved: bool | None = Field(default=None, description="是否允許保留字元")
    schema_: Optional["OpenAPISchema"] = Field(
        default=None, alias="schema", description="參數 Schema"
    )
    example: Any | None = Field(default=None, description="範例值")
    examples: dict[str, "OpenAPIExample"] | None = Field(
        default=None, description="範例值集合"
    )

    # Media Type
    content: dict[str, "OpenAPIMediaType"] | None = Field(
        default=None, description="內容類型"
    )


class OpenAPIExample(BaseModel):
    """OpenAPI Example 物件"""

    summary: str | None = Field(default=None, description="範例摘要")
    description: str | None = Field(default=None, description="範例描述")
    value: Any | None = Field(default=None, description="範例值")
    external_value: HttpUrl | None = Field(
        default=None, description="外部範例值 URL"
    )


class OpenAPIMediaType(BaseModel):
    """OpenAPI Media Type 物件"""

    schema_: OpenAPISchema | None = Field(
        default=None, alias="schema", description="媒體類型 Schema"
    )
    example: Any | None = Field(default=None, description="範例值")
    examples: dict[str, OpenAPIExample] | None = Field(
        default=None, description="範例值集合"
    )
    encoding: dict[str, "OpenAPIEncoding"] | None = Field(
        default=None, description="編碼資訊"
    )


class OpenAPIEncoding(BaseModel):
    """OpenAPI Encoding 物件"""

    content_type: str | None = Field(default=None, description="內容類型")
    headers: dict[str, Union["OpenAPIParameter", "OpenAPIReference"]] | None = Field(
        default=None, description="標頭"
    )
    style: str | None = Field(default=None, description="序列化樣式")
    explode: bool | None = Field(default=None, description="是否展開")
    allow_reserved: bool | None = Field(default=None, description="是否允許保留字元")


class OpenAPIReference(BaseModel):
    """OpenAPI Reference 物件"""

    ref: str = Field(alias="$ref", description="引用路徑")
    summary: str | None = Field(default=None, description="引用摘要")
    description: str | None = Field(default=None, description="引用描述")


class OpenAPIRequestBody(BaseModel):
    """OpenAPI Request Body 物件"""

    description: str | None = Field(default=None, description="請求體描述")
    content: dict[str, "OpenAPIMediaType"] = Field(description="媒體類型對應")
    required: bool | None = Field(default=None, description="是否必需")


class OpenAPIResponse(BaseModel):
    """OpenAPI Response 物件"""

    description: str = Field(description="回應描述")
    headers: dict[str, Union["OpenAPIParameter", "OpenAPIReference"]] | None = Field(
        default=None, description="回應標頭"
    )
    content: dict[str, "OpenAPIMediaType"] | None = Field(
        default=None, description="回應內容"
    )
    links: dict[str, Union["OpenAPILink", "OpenAPIReference"]] | None = Field(
        default=None, description="回應連結"
    )


class OpenAPILink(BaseModel):
    """OpenAPI Link 物件"""

    operation_ref: str | None = Field(default=None, description="操作引用")
    operation_id: str | None = Field(default=None, description="操作 ID")
    parameters: dict[str, Any] | None = Field(default=None, description="參數映射")
    request_body: Any | None = Field(default=None, description="請求體")
    description: str | None = Field(default=None, description="連結描述")
    server: OpenAPIServer | None = Field(default=None, description="連結伺服器")


class OpenAPIOperation(BaseModel):
    """OpenAPI Operation 物件"""

    tags: list[str] | None = Field(default=None, description="標籤")
    summary: str | None = Field(default=None, description="操作摘要")
    description: str | None = Field(default=None, description="操作描述")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )
    operation_id: str | None = Field(default=None, description="操作 ID")
    parameters: list[Union["OpenAPIParameter", "OpenAPIReference"]] | None = Field(
        default=None, description="參數列表"
    )
    request_body: Union["OpenAPIRequestBody", "OpenAPIReference"] | None = Field(
        default=None, description="請求體"
    )
    responses: dict[str, Union["OpenAPIResponse", "OpenAPIReference"]] = Field(
        description="回應對應"
    )
    callbacks: dict[str, Union["OpenAPICallback", "OpenAPIReference"]] | None = (
        Field(default=None, description="回調")
    )
    deprecated: bool | None = Field(default=None, description=DEPRECATED_DESCRIPTION)
    security: list[dict[str, list[str]]] | None = Field(
        default=None, description="安全需求"
    )
    servers: list[OpenAPIServer] | None = Field(
        default=None, description=SERVER_LIST_DESCRIPTION
    )


class OpenAPICallback(BaseModel):
    """OpenAPI Callback 物件"""

    expression: dict[str, "OpenAPIPathItem"] = Field(description="回調表達式對應")


class OpenAPIPathItem(BaseModel):
    """OpenAPI Path Item 物件"""

    ref: str | None = Field(default=None, alias="$ref", description="路徑引用")
    summary: str | None = Field(default=None, description="路徑摘要")
    description: str | None = Field(default=None, description="路徑描述")
    get: OpenAPIOperation | None = Field(default=None, description="GET 操作")
    put: OpenAPIOperation | None = Field(default=None, description="PUT 操作")
    post: OpenAPIOperation | None = Field(default=None, description="POST 操作")
    delete: OpenAPIOperation | None = Field(default=None, description="DELETE 操作")
    options: OpenAPIOperation | None = Field(
        default=None, description="OPTIONS 操作"
    )
    head: OpenAPIOperation | None = Field(default=None, description="HEAD 操作")
    patch: OpenAPIOperation | None = Field(default=None, description="PATCH 操作")
    trace: OpenAPIOperation | None = Field(default=None, description="TRACE 操作")
    servers: list[OpenAPIServer] | None = Field(
        default=None, description=SERVER_LIST_DESCRIPTION
    )
    parameters: list[Union["OpenAPIParameter", "OpenAPIReference"]] | None = Field(
        default=None, description="參數列表"
    )


class OpenAPISecurityScheme(BaseModel):
    """OpenAPI Security Scheme 物件"""

    type: OpenAPISecuritySchemeType = Field(description="安全方案類型")
    description: str | None = Field(default=None, description="安全方案描述")
    name: str | None = Field(default=None, description="API 金鑰名稱")
    in_: OpenAPIParameterLocation | None = Field(
        default=None, alias="in", description="API 金鑰位置"
    )
    scheme: str | None = Field(default=None, description="HTTP 認證方案")
    bearer_format: str | None = Field(default=None, description="Bearer 格式")
    flows: Optional["OpenAPIOAuthFlows"] = Field(
        default=None, description="OAuth2 流程"
    )
    open_id_connect_url: HttpUrl | None = Field(
        default=None, description="OpenID Connect URL"
    )


class OpenAPIOAuthFlows(BaseModel):
    """OpenAPI OAuth Flows 物件"""

    implicit: Optional["OpenAPIOAuthFlow"] = Field(
        default=None, description="Implicit 流程"
    )
    password: Optional["OpenAPIOAuthFlow"] = Field(
        default=None, description="Password 流程"
    )
    client_credentials: Optional["OpenAPIOAuthFlow"] = Field(
        default=None, description="Client Credentials 流程"
    )
    authorization_code: Optional["OpenAPIOAuthFlow"] = Field(
        default=None, description="Authorization Code 流程"
    )


class OpenAPIOAuthFlow(BaseModel):
    """OpenAPI OAuth Flow 物件"""

    authorization_url: HttpUrl | None = Field(default=None, description="授權 URL")
    token_url: HttpUrl | None = Field(default=None, description="令牌 URL")
    refresh_url: HttpUrl | None = Field(default=None, description="刷新 URL")
    scopes: dict[str, str] = Field(description="範圍對應")


class OpenAPITag(BaseModel):
    """OpenAPI Tag 物件"""

    name: str = Field(description="標籤名稱")
    description: str | None = Field(default=None, description="標籤描述")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )


class OpenAPIComponents(BaseModel):
    """OpenAPI Components 物件"""

    schemas: dict[str, Union["OpenAPISchema", "OpenAPIReference"]] | None = Field(
        default=None, description="Schema 元件"
    )
    responses: dict[str, Union["OpenAPIResponse", "OpenAPIReference"]] | None = (
        Field(default=None, description="Response 元件")
    )
    parameters: dict[str, Union["OpenAPIParameter", "OpenAPIReference"]] | None = (
        Field(default=None, description="Parameter 元件")
    )
    examples: dict[str, Union["OpenAPIExample", "OpenAPIReference"]] | None = Field(
        default=None, description="Example 元件"
    )
    request_bodies: dict[str, Union["OpenAPIRequestBody", "OpenAPIReference"]] | None = Field(default=None, description="Request Body 元件")
    headers: dict[str, Union["OpenAPIParameter", "OpenAPIReference"]] | None = Field(
        default=None, description="Header 元件"
    )
    security_schemes: dict[str, Union["OpenAPISecurityScheme", "OpenAPIReference"]] | None = Field(default=None, description="Security Scheme 元件")
    links: dict[str, Union["OpenAPILink", "OpenAPIReference"]] | None = Field(
        default=None, description="Link 元件"
    )
    callbacks: dict[str, Union["OpenAPICallback", "OpenAPIReference"]] | None = (
        Field(default=None, description="Callback 元件")
    )
    path_items: dict[str, Union["OpenAPIPathItem", "OpenAPIReference"]] | None = (
        Field(default=None, description="Path Item 元件")
    )


class OpenAPIDocument(BaseModel):
    """OpenAPI 3.1 文件根物件"""

    openapi: str = Field(default="3.1.0", description="OpenAPI 版本")
    info: "OpenAPIInfo" = Field(description="API 資訊")
    json_schema_dialect: str | None = Field(
        default=None, description="JSON Schema 方言"
    )
    servers: list["OpenAPIServer"] | None = Field(
        default=None, description=SERVER_LIST_DESCRIPTION
    )
    paths: dict[str, "OpenAPIPathItem"] | None = Field(
        default=None, description="路徑對應"
    )
    webhooks: dict[str, Union["OpenAPIPathItem", "OpenAPIReference"]] | None = Field(
        default=None, description="Webhook 對應"
    )
    components: Optional["OpenAPIComponents"] = Field(
        default=None, description="可重用元件"
    )
    security: list[dict[str, list[str]]] | None = Field(
        default=None, description="安全需求"
    )
    tags: list["OpenAPITag"] | None = Field(default=None, description="標籤列表")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )


# ==================== AsyncAPI 3.0 支援 ====================


class AsyncAPIInfo(BaseModel):
    """AsyncAPI Info 物件"""

    title: str = Field(description="應用程式標題")
    version: str = Field(description="應用程式版本")
    description: str | None = Field(default=None, description="應用程式描述")
    terms_of_service: HttpUrl | None = Field(default=None, description="服務條款")
    contact: OpenAPIContact | None = Field(default=None, description="聯絡資訊")
    license: Optional["OpenAPILicense"] = Field(default=None, description="授權資訊")
    tags: list["AsyncAPITag"] | None = Field(default=None, description="標籤")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )


class AsyncAPITag(BaseModel):
    """AsyncAPI Tag 物件"""

    name: str = Field(description="標籤名稱")
    description: str | None = Field(default=None, description="標籤描述")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )


class AsyncAPIServer(BaseModel):
    """AsyncAPI Server 物件"""

    host: str = Field(description="伺服器主機")
    protocol: str = Field(description="協議")
    protocol_version: str | None = Field(default=None, description="協議版本")
    pathname: str | None = Field(default=None, description="路徑")
    description: str | None = Field(default=None, description="伺服器描述")
    title: str | None = Field(default=None, description="伺服器標題")
    summary: str | None = Field(default=None, description="伺服器摘要")
    variables: dict[str, "AsyncAPIServerVariable"] | None = Field(
        default=None, description="伺服器變數"
    )
    security: list[dict[str, list[str]]] | None = Field(
        default=None, description="安全需求"
    )
    tags: list["AsyncAPITag"] | None = Field(default=None, description="標籤")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )
    bindings: dict[str, Any] | None = Field(default=None, description="協議綁定")


class AsyncAPIServerVariable(BaseModel):
    """AsyncAPI Server Variable 物件"""

    enum: list[str] | None = Field(default=None, description="可能的值")
    default: str | None = Field(default=None, description="預設值")
    description: str | None = Field(default=None, description="變數描述")
    examples: list[str] | None = Field(default=None, description="範例值")


class AsyncAPIChannel(BaseModel):
    """AsyncAPI Channel 物件"""

    address: str | None = Field(default=None, description="通道地址")
    messages: dict[str, Union["AsyncAPIMessage", "OpenAPIReference"]] | None = Field(
        default=None, description="訊息對應"
    )
    title: str | None = Field(default=None, description="通道標題")
    summary: str | None = Field(default=None, description="通道摘要")
    description: str | None = Field(default=None, description="通道描述")
    servers: list[Union[AsyncAPIServer, "OpenAPIReference"]] | None = Field(
        default=None, description="伺服器引用"
    )
    parameters: dict[str, Union["AsyncAPIParameter", "OpenAPIReference"]] | None = (
        Field(default=None, description="參數對應")
    )
    tags: list["AsyncAPITag"] | None = Field(default=None, description="標籤")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )
    bindings: dict[str, Any] | None = Field(default=None, description="協議綁定")


class AsyncAPIParameter(BaseModel):
    """AsyncAPI Parameter 物件"""

    description: str | None = Field(default=None, description="參數描述")
    schema_: Optional["OpenAPISchema"] = Field(
        default=None, alias="schema", description="參數 Schema"
    )
    location: str | None = Field(default=None, description="參數位置")


class AsyncAPIMessage(BaseModel):
    """AsyncAPI Message 物件"""

    headers: Union["OpenAPISchema", "OpenAPIReference"] | None = Field(
        default=None, description="訊息標頭"
    )
    payload: Union["OpenAPISchema", "OpenAPIReference"] | None = Field(
        default=None, description="訊息負載"
    )
    correlation_id: Union["AsyncAPICorrelationId", "OpenAPIReference"] | None = (
        Field(default=None, description="關聯 ID")
    )
    content_type: str | None = Field(default=None, description="內容類型")
    name: str | None = Field(default=None, description="訊息名稱")
    title: str | None = Field(default=None, description="訊息標題")
    summary: str | None = Field(default=None, description="訊息摘要")
    description: str | None = Field(default=None, description="訊息描述")
    tags: list["AsyncAPITag"] | None = Field(default=None, description="標籤")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )
    bindings: dict[str, Any] | None = Field(default=None, description="協議綁定")
    examples: list[dict[str, Any]] | None = Field(
        default=None, description="訊息範例"
    )
    traits: list[Union["AsyncAPIMessageTrait", "OpenAPIReference"]] | None = Field(
        default=None, description="訊息特徵"
    )


class AsyncAPICorrelationId(BaseModel):
    """AsyncAPI Correlation ID 物件"""

    description: str | None = Field(default=None, description="關聯 ID 描述")
    location: str = Field(description="關聯 ID 位置")


class AsyncAPIMessageTrait(BaseModel):
    """AsyncAPI Message Trait 物件"""

    headers: Union[OpenAPISchema, "OpenAPIReference"] | None = Field(
        default=None, description="訊息標頭"
    )
    correlation_id: Union[AsyncAPICorrelationId, "OpenAPIReference"] | None = Field(
        default=None, description="關聯 ID"
    )
    content_type: str | None = Field(default=None, description="內容類型")
    name: str | None = Field(default=None, description="訊息名稱")
    title: str | None = Field(default=None, description="訊息標題")
    summary: str | None = Field(default=None, description="訊息摘要")
    description: str | None = Field(default=None, description="訊息描述")
    tags: list["AsyncAPITag"] | None = Field(default=None, description="標籤")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )
    bindings: dict[str, Any] | None = Field(default=None, description="協議綁定")
    examples: list[dict[str, Any]] | None = Field(
        default=None, description="訊息範例"
    )


class AsyncAPIOperation(BaseModel):
    """AsyncAPI Operation 物件"""

    action: Literal["send", "receive"] = Field(description="操作動作")
    channel: Union[AsyncAPIChannel, "OpenAPIReference"] = Field(description="通道引用")
    title: str | None = Field(default=None, description="操作標題")
    summary: str | None = Field(default=None, description="操作摘要")
    description: str | None = Field(default=None, description="操作描述")
    security: list[dict[str, list[str]]] | None = Field(
        default=None, description="安全需求"
    )
    tags: list["AsyncAPITag"] | None = Field(default=None, description="標籤")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )
    bindings: dict[str, Any] | None = Field(default=None, description="協議綁定")
    traits: list[Union["AsyncAPIOperationTrait", "OpenAPIReference"]] | None = Field(
        default=None, description="操作特徵"
    )
    messages: list[Union[AsyncAPIMessage, "OpenAPIReference"]] | None = Field(
        default=None, description="訊息列表"
    )
    reply: Union["AsyncAPIOperationReply", "OpenAPIReference"] | None = Field(
        default=None, description="回覆配置"
    )


class AsyncAPIOperationTrait(BaseModel):
    """AsyncAPI Operation Trait 物件"""

    title: str | None = Field(default=None, description="操作標題")
    summary: str | None = Field(default=None, description="操作摘要")
    description: str | None = Field(default=None, description="操作描述")
    security: list[dict[str, list[str]]] | None = Field(
        default=None, description="安全需求"
    )
    tags: list["AsyncAPITag"] | None = Field(default=None, description="標籤")
    external_docs: Optional["OpenAPIExternalDocumentation"] = Field(
        default=None, description="外部文件"
    )
    bindings: dict[str, Any] | None = Field(default=None, description="協議綁定")


class AsyncAPIOperationReply(BaseModel):
    """AsyncAPI Operation Reply 物件"""

    address: Union["AsyncAPIOperationReplyAddress", "OpenAPIReference"] | None = (
        Field(default=None, description="回覆地址")
    )
    channel: Union[AsyncAPIChannel, "OpenAPIReference"] | None = Field(
        default=None, description="回覆通道"
    )
    messages: list[Union[AsyncAPIMessage, "OpenAPIReference"]] | None = Field(
        default=None, description="回覆訊息"
    )


class AsyncAPIOperationReplyAddress(BaseModel):
    """AsyncAPI Operation Reply Address 物件"""

    description: str | None = Field(default=None, description="回覆地址描述")
    location: str = Field(description="回覆地址位置")


class AsyncAPIDocument(BaseModel):
    """AsyncAPI 3.0 文件根物件"""

    asyncapi: str = Field(default="3.0.0", description="AsyncAPI 版本")
    id: str | None = Field(default=None, description="應用程式識別符")
    info: AsyncAPIInfo = Field(description="應用程式資訊")
    servers: dict[str, Union[AsyncAPIServer, "OpenAPIReference"]] | None = Field(
        default=None, description="伺服器對應"
    )
    default_content_type: str | None = Field(
        default=None, description="預設內容類型"
    )
    channels: dict[str, Union[AsyncAPIChannel, "OpenAPIReference"]] | None = Field(
        default=None, description="通道對應"
    )
    operations: dict[str, Union[AsyncAPIOperation, "OpenAPIReference"]] | None = (
        Field(default=None, description="操作對應")
    )
    components: Optional["AsyncAPIComponents"] = Field(
        default=None, description="可重用元件"
    )


class AsyncAPIComponents(BaseModel):
    """AsyncAPI Components 物件"""

    schemas: dict[str, Union[OpenAPISchema, "OpenAPIReference"]] | None = Field(
        default=None, description="Schema 元件"
    )
    servers: dict[str, Union[AsyncAPIServer, "OpenAPIReference"]] | None = Field(
        default=None, description="Server 元件"
    )
    channels: dict[str, Union[AsyncAPIChannel, "OpenAPIReference"]] | None = Field(
        default=None, description="Channel 元件"
    )
    operations: dict[str, Union[AsyncAPIOperation, "OpenAPIReference"]] | None = (
        Field(default=None, description="Operation 元件")
    )
    messages: dict[str, Union["AsyncAPIMessage", "OpenAPIReference"]] | None = Field(
        default=None, description="Message 元件"
    )
    security_schemes: dict[str, Union[OpenAPISecurityScheme, "OpenAPIReference"]] | None = Field(default=None, description="Security Scheme 元件")
    server_variables: dict[str, Union[AsyncAPIServerVariable, "OpenAPIReference"]] | None = Field(default=None, description="Server Variable 元件")
    parameters: dict[str, Union[AsyncAPIParameter, "OpenAPIReference"]] | None = (
        Field(default=None, description="Parameter 元件")
    )
    correlation_ids: dict[str, Union[AsyncAPICorrelationId, "OpenAPIReference"]] | None = Field(default=None, description="Correlation ID 元件")
    operation_traits: dict[str, Union["AsyncAPIOperationTrait", "OpenAPIReference"]] | None = Field(default=None, description="Operation Trait 元件")
    message_traits: dict[str, Union["AsyncAPIMessageTrait", "OpenAPIReference"]] | None = Field(default=None, description="Message Trait 元件")
    tags: dict[str, Union["AsyncAPITag", "OpenAPIReference"]] | None = Field(
        default=None, description="Tag 元件"
    )
    external_docs: dict[str, Union["OpenAPIExternalDocumentation", "OpenAPIReference"]] | None = Field(default=None, description="External Documentation 元件")


# ==================== GraphQL 支援 ====================


class GraphQLType(str, Enum):
    """GraphQL 類型枚舉"""

    SCALAR = "SCALAR"
    OBJECT = "OBJECT"
    INTERFACE = "INTERFACE"
    UNION = "UNION"
    ENUM = "ENUM"
    INPUT_OBJECT = "INPUT_OBJECT"
    LIST = "LIST"
    NON_NULL = "NON_NULL"


class GraphQLDirectiveLocation(str, Enum):
    """GraphQL 指令位置枚舉"""

    # Query 相關
    QUERY = "QUERY"
    MUTATION = "MUTATION"
    SUBSCRIPTION = "SUBSCRIPTION"
    FIELD = "FIELD"
    FRAGMENT_DEFINITION = "FRAGMENT_DEFINITION"
    FRAGMENT_SPREAD = "FRAGMENT_SPREAD"
    INLINE_FRAGMENT = "INLINE_FRAGMENT"
    VARIABLE_DEFINITION = "VARIABLE_DEFINITION"

    # Schema 相關
    SCHEMA = "SCHEMA"
    SCALAR = "SCALAR"
    OBJECT = "OBJECT"
    FIELD_DEFINITION = "FIELD_DEFINITION"
    ARGUMENT_DEFINITION = "ARGUMENT_DEFINITION"
    INTERFACE = "INTERFACE"
    UNION = "UNION"
    ENUM = "ENUM"
    ENUM_VALUE = "ENUM_VALUE"
    INPUT_OBJECT = "INPUT_OBJECT"
    INPUT_FIELD_DEFINITION = "INPUT_FIELD_DEFINITION"


class GraphQLTypeDefinition(BaseModel):
    """GraphQL 類型定義"""

    kind: GraphQLType = Field(description="類型種類")
    name: str = Field(description="類型名稱")
    description: str | None = Field(default=None, description="類型描述")
    fields: list["GraphQLFieldDefinition"] | None = Field(
        default=None, description="欄位定義"
    )
    input_fields: list["GraphQLInputValueDefinition"] | None = Field(
        default=None, description="輸入欄位定義"
    )
    interfaces: list[str] | None = Field(default=None, description="實現的介面")
    enum_values: list["GraphQLEnumValueDefinition"] | None = Field(
        default=None, description="枚舉值定義"
    )
    possible_types: list[str] | None = Field(default=None, description="可能的類型")


class GraphQLFieldDefinition(BaseModel):
    """GraphQL 欄位定義"""

    name: str = Field(description="欄位名稱")
    description: str | None = Field(default=None, description="欄位描述")
    args: list["GraphQLInputValueDefinition"] = Field(
        default_factory=list, description="參數定義"
    )
    type: "GraphQLTypeReference" = Field(description="欄位類型")
    is_deprecated: bool = Field(default=False, description=DEPRECATED_DESCRIPTION)
    deprecation_reason: str | None = Field(default=None, description="棄用原因")


class GraphQLInputValueDefinition(BaseModel):
    """GraphQL 輸入值定義"""

    name: str = Field(description="參數名稱")
    description: str | None = Field(default=None, description="參數描述")
    type: "GraphQLTypeReference" = Field(description="參數類型")
    default_value: Any | None = Field(default=None, description="預設值")


class GraphQLEnumValueDefinition(BaseModel):
    """GraphQL 枚舉值定義"""

    name: str = Field(description="枚舉值名稱")
    description: str | None = Field(default=None, description="枚舉值描述")
    is_deprecated: bool = Field(default=False, description=DEPRECATED_DESCRIPTION)
    deprecation_reason: str | None = Field(default=None, description="棄用原因")


class GraphQLTypeReference(BaseModel):
    """GraphQL 類型引用"""

    kind: GraphQLType = Field(description="類型種類")
    name: str | None = Field(default=None, description="類型名稱")
    of_type: Optional["GraphQLTypeReference"] = Field(
        default=None, description="子類型"
    )


class GraphQLDirectiveDefinition(BaseModel):
    """GraphQL 指令定義"""

    name: str = Field(description="指令名稱")
    description: str | None = Field(default=None, description="指令描述")
    locations: list[GraphQLDirectiveLocation] = Field(description="指令位置")
    args: list[GraphQLInputValueDefinition] = Field(
        default_factory=list, description="指令參數"
    )


class GraphQLSchema(BaseModel):
    """GraphQL Schema 定義"""

    query_type: str | None = Field(default=None, description="查詢根類型")
    mutation_type: str | None = Field(default=None, description="變更根類型")
    subscription_type: str | None = Field(default=None, description="訂閱根類型")
    types: list[GraphQLTypeDefinition] = Field(description="類型定義列表")
    directives: list[GraphQLDirectiveDefinition] = Field(description="指令定義列表")


# ==================== API 安全測試相關 ====================


class APISecurityTest(BaseModel):
    """API 安全測試配置"""

    test_id: str = Field(description="測試唯一標識符")
    name: str = Field(description="測試名稱")
    description: str = Field(description="測試描述")

    # 測試目標
    target_api: OpenAPIDocument | AsyncAPIDocument | GraphQLSchema = Field(
        description="目標 API"
    )
    base_url: HttpUrl = Field(description="基礎 URL")

    # 測試配置
    authentication: dict[str, Any] | None = Field(
        default=None, description="認證配置"
    )
    headers: dict[str, str] = Field(default_factory=dict, description="自定義標頭")
    timeout: int = Field(default=30, description="請求超時（秒）")

    # 測試範圍
    include_endpoints: list[str] | None = Field(
        default=None, description="包含的端點"
    )
    exclude_endpoints: list[str] | None = Field(
        default=None, description="排除的端點"
    )

    # HackerOne 優化設定
    focus_low_hanging_fruit: bool = Field(
        default=True, description="專注於低價值高概率漏洞"
    )
    target_bounty_range: str = Field(
        default="50-500", description="目標獎金範圍（美元）"
    )
    max_test_time_hours: float = Field(default=2.0, description="最大測試時間（小時）")

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# APIVulnerabilityFinding 已移除重複定義，統一使用 aiva_common.schemas.vulnerability_finding
# 原 api_standards.py 中的 APIVulnerabilityFinding 於 2024-12-19 移除
# 請使用: from aiva_common.schemas.vulnerability_finding import APIVulnerabilityFinding

from .vulnerability_finding import APIVulnerabilityFinding


# 前向引用解決
OpenAPISchema.model_rebuild()
AsyncAPIChannel.model_rebuild()
AsyncAPIMessage.model_rebuild()
AsyncAPIOperation.model_rebuild()
GraphQLTypeReference.model_rebuild()
