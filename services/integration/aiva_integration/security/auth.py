from __future__ import annotations

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..settings import IntegrationSettings


class APIKeyAuth(HTTPBearer):
    """API 密鑰認證"""

    def __init__(self, settings: IntegrationSettings):
        super().__init__(auto_error=False)
        self.settings = settings

    async def __call__(self, request: Request) -> HTTPAuthorizationCredentials | None:
        if not self.settings.api_token:
            # 如果沒有設置 API token，跳過認證
            return None

        # 檢查 X-API-Key header
        api_key = request.headers.get("X-API-Key")
        if api_key and api_key == self.settings.api_token:
            # return a minimal HTTPAuthorizationCredentials-like object
            return HTTPAuthorizationCredentials(scheme="ApiKey", credentials=api_key)

        # 檢查 Authorization header
        credentials: HTTPAuthorizationCredentials | None = await super().__call__(
            request
        )
        if credentials and credentials.credentials == self.settings.api_token:
            return credentials

        # 認證失敗
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_api_key(
    request: Request, settings: IntegrationSettings | None = None
) -> None:
    """
    依賴注入函數，用於需要認證的端點

    Args:
        request: FastAPI 請求對象
        settings: Integration 設定

    Raises:
        HTTPException: 當認證失敗時
    """
    if not settings:
        from ..settings import IntegrationSettings

        settings = IntegrationSettings.load()

    auth = APIKeyAuth(settings)
    await auth(request)
