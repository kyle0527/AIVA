# -*- coding: utf-8 -*-
"""
高價值功能模組示例配置

此文件提供了所有高價值功能模組的使用示例，
展示如何配置參數以獲得最佳的檢測效果。

使用方法：
1. 設定環境變數 ALLOWLIST_DOMAINS
2. 匯入並註冊功能模組
3. 使用 FeatureStepExecutor 執行檢測
4. 分析結果並生成報告
"""

# 環境設定
EXAMPLE_CONFIG = {
    # 必須設定的環境變數
    "environment": {
        "ALLOWLIST_DOMAINS": "example.com,api.example.com,app.example.com"
    },
    
    # Mass Assignment 檢測配置
    "mass_assignment": {
        "target": "https://api.example.com",
        "path": "/api/profile/update",
        "method": "POST",
        "headers": {
            "Authorization": "Bearer low_priv_token",
            "Content-Type": "application/json"
        },
        "baseline_body": {
            "name": "John Doe",
            "email": "john@example.com"
        },
        "check_endpoint": "/api/me",
        "check_key": "role",
        "attempt_fields": ["role", "admin", "is_admin", "privilege"]
    },
    
    # JWT 混淆檢測配置
    "jwt_confusion": {
        "target": "https://api.example.com",
        "validate_endpoint": "/api/me",
        "victim_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        "jwks_url": "https://auth.example.com/.well-known/jwks.json",
        "attempts": {
            "alg_none": True,
            "kid_injection": True,
            "symmetric_rs": True
        },
        "headers": {
            "User-Agent": "AIVA Security Scanner"
        }
    },
    
    # OAuth 混淆檢測配置
    "oauth_confusion": {
        "target": "https://auth.example.com",
        "auth_endpoint": "/oauth/authorize",
        "client_id": "your_client_id",
        "legitimate_redirect": "https://app.example.com/callback",
        "attacker_redirect": "https://attacker.com/steal",
        "scope": "openid profile email",
        "tests": {
            "redirect_uri_bypass": True,
            "pkce_downgrade": True,
            "open_redirect_chain": True
        }
    },
    
    # GraphQL 權限檢測配置
    "graphql_authz": {
        "target": "https://api.example.com",
        "endpoint": "/graphql",
        "headers_user": {
            "Authorization": "Bearer user_token"
        },
        "headers_admin": {
            "Authorization": "Bearer admin_token"
        },
        "test_queries": [
            {
                "name": "user_profile",
                "query": "query GetUser($id: ID!) { user(id: $id) { id email role salary } }",
                "variables": {"id": "1001"},
                "target_user_id": "1001"
            },
            {
                "name": "admin_panel",
                "query": "query AdminData { adminStats { userCount revenue secrets } }"
            }
        ],
        "tests": {
            "introspection": True,
            "field_level_authz": True,
            "object_level_authz": True
        }
    },
    
    # SSRF OOB 檢測配置
    "ssrf_oob": {
        "target": "https://app.example.com",
        "probe_endpoints": [
            "/api/fetch",
            "/api/screenshot", 
            "/api/pdf/generate",
            "/webhook/test"
        ],
        "url_params": ["url", "link", "fetch", "src"],
        "json_fields": ["imageUrl", "webhookUrl", "sourceUrl"],
        "headers": {
            "Authorization": "Bearer your_token"
        },
        "oob_http": "https://your-collaborator.burpcollaborator.net",
        "oob_dns": "your-token.dnslog.cn",
        "test_protocols": ["http", "https"],
        "payload_types": ["direct", "encoded"],
        "options": {
            "delay_seconds": 10,
            "timeout": 30
        }
    }
}

# 實戰攻擊路線配置
ATTACK_ROUTES = {
    # 路線 A：權限與身分
    "privilege_escalation_route": [
        {
            "tool_type": "feature",
            "tool_name": "mass_assignment",
            "params": EXAMPLE_CONFIG["mass_assignment"],
            "description": "檢測 Mass Assignment 權限提升"
        },
        {
            "tool_type": "feature", 
            "tool_name": "graphql_authz",
            "params": EXAMPLE_CONFIG["graphql_authz"],
            "description": "檢測 GraphQL 欄位級權限"
        },
        {
            "tool_type": "feature",
            "tool_name": "jwt_confusion", 
            "params": EXAMPLE_CONFIG["jwt_confusion"],
            "description": "嘗試 JWT 混淆攻擊"
        }
    ],
    
    # 路線 B：登入與授權
    "authentication_bypass_route": [
        {
            "tool_type": "feature",
            "tool_name": "oauth_confusion",
            "params": EXAMPLE_CONFIG["oauth_confusion"],
            "description": "檢測 OAuth 配置錯誤"
        },
        {
            "tool_type": "feature",
            "tool_name": "ssrf_oob",
            "params": EXAMPLE_CONFIG["ssrf_oob"],
            "description": "探測 SSRF 漏洞"
        },
        {
            "tool_type": "feature",
            "tool_name": "jwt_confusion",
            "params": EXAMPLE_CONFIG["jwt_confusion"],
            "description": "JWT 身分繞過"
        }
    ],
    
    # 路線 C：資料存取
    "data_access_route": [
        {
            "tool_type": "feature",
            "tool_name": "graphql_authz",
            "params": EXAMPLE_CONFIG["graphql_authz"],
            "description": "GraphQL 資料洩漏檢測"
        },
        {
            "tool_type": "feature",
            "tool_name": "mass_assignment",
            "params": EXAMPLE_CONFIG["mass_assignment"],
            "description": "提升權限後存取敏感資料"
        },
        {
            "tool_type": "feature",
            "tool_name": "ssrf_oob",
            "params": EXAMPLE_CONFIG["ssrf_oob"],
            "description": "透過 SSRF 探測內網"
        }
    ]
}

# HackerOne 報告模板
HACKERONE_REPORT_TEMPLATE = {
    "mass_assignment": {
        "title": "Mass Assignment leading to Privilege Escalation",
        "asset": "app.example.com",
        "weakness": "Broken Access Control",
        "severity": "Critical",
        "summary": "The application accepts undocumented JSON fields in user profile updates, allowing privilege escalation from regular user to administrator.",
        "description_template": """
## Summary
A mass assignment vulnerability exists in the user profile update API that allows privilege escalation.

## Steps to Reproduce
{reproduction_steps}

## Impact
{impact}

## Recommendation
{recommendation}
        """
    },
    
    "jwt_confusion": {
        "title": "JWT Algorithm Confusion allows Authentication Bypass",
        "asset": "api.example.com", 
        "weakness": "Authentication Bypass",
        "severity": "Critical",
        "summary": "JWT implementation accepts 'none' algorithm or is vulnerable to RS256/HS256 confusion, allowing complete authentication bypass."
    },
    
    "oauth_confusion": {
        "title": "OAuth redirect_uri Bypass enables Account Takeover", 
        "asset": "auth.example.com",
        "weakness": "Authentication Bypass",
        "severity": "High",
        "summary": "OAuth implementation doesn't properly validate redirect_uri parameter, allowing attackers to steal authorization codes."
    },
    
    "graphql_authz": {
        "title": "GraphQL Field-Level Authorization Bypass",
        "asset": "api.example.com",
        "weakness": "Broken Access Control", 
        "severity": "High",
        "summary": "GraphQL endpoint lacks proper field-level authorization, exposing sensitive user data to low-privilege users."
    },
    
    "ssrf_oob": {
        "title": "Server-Side Request Forgery (SSRF) with External Callback",
        "asset": "app.example.com",
        "weakness": "Server Side Request Forgery",
        "severity": "High", 
        "summary": "Application fetches external URLs without proper validation, enabling SSRF attacks against internal services."
    }
}

def get_config_for_feature(feature_name: str) -> dict:
    """取得指定功能的配置"""
    return EXAMPLE_CONFIG.get(feature_name, {})

def get_attack_route(route_name: str) -> list:
    """取得指定攻擊路線的步驟"""
    return ATTACK_ROUTES.get(route_name, [])

def get_report_template(feature_name: str) -> dict:
    """取得指定功能的報告模板"""
    return HACKERONE_REPORT_TEMPLATE.get(feature_name, {})