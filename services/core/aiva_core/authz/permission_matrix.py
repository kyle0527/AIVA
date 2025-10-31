"""Permission Matrix - 權限矩陣數據結構與分析

提供權限矩陣的數據結構、存儲、查詢與分析功能。
"""

from datetime import datetime
from typing import Any

import numpy as np
import structlog

# 使用統一的可選依賴管理框架
from utilities.optional_deps import deps

from services.aiva_common.enums import AccessDecision

# 註冊 pandas 依賴
deps.register("pandas", ["pandas"])

# 可選 pandas 導入
if deps.is_available("pandas"):
    import pandas as pd
else:
    # Mock pandas DataFrame
    class MockDataFrame:
        def __init__(self, data=None):
            self.data = data or []
            
        def __len__(self):
            return len(self.data)
            
        def to_dict(self, orient='records'):
            return self.data
            
        def to_json(self, *args, **kwargs):
            import json
            return json.dumps(self.data)
            
        def empty(self):
            return len(self.data) == 0
    
    class MockPandas:
        DataFrame = MockDataFrame
    
    pd = MockPandas()

logger = structlog.get_logger(__name__)


class PermissionMatrix:
    """權限矩陣

    管理角色-資源-權限的三維矩陣。
    """

    def __init__(self):
        """初始化權限矩陣"""
        self.roles: set[str] = set()
        self.resources: set[str] = set()
        self.permissions: set[str] = set()

        # 矩陣：{(role, resource, permission): decision}
        self.matrix: dict[tuple[str, str, str], AccessDecision] = {}

        # 角色繼承：{child_role: [parent_roles]}
        self.role_inheritance: dict[str, list[str]] = {}

        # 條件規則：{(role, resource, permission): condition_func}
        self.conditions: dict[tuple[str, str, str], str] = {}

        logger.info("permission_matrix_initialized")

    def add_role(self, role: str, inherits_from: list[str] | None = None) -> None:
        """添加角色

        Args:
            role: 角色名稱
            inherits_from: 繼承的父角色列表
        """
        self.roles.add(role)
        if inherits_from:
            self.role_inheritance[role] = inherits_from
            logger.info("role_added_with_inheritance", role=role, parents=inherits_from)
        else:
            logger.info("role_added", role=role)

    def add_resource(self, resource: str) -> None:
        """添加資源

        Args:
            resource: 資源名稱
        """
        self.resources.add(resource)
        logger.debug("resource_added", resource=resource)

    def add_permission(self, permission: str) -> None:
        """添加權限

        Args:
            permission: 權限名稱
        """
        self.permissions.add(permission)
        logger.debug("permission_added", permission=permission)

    def grant_permission(
        self,
        role: str,
        resource: str,
        permission: str,
        decision: AccessDecision = AccessDecision.ALLOW,
        condition: str | None = None,
    ) -> None:
        """授予權限

        Args:
            role: 角色
            resource: 資源
            permission: 權限
            decision: 訪問決策
            condition: 條件表達式（可選）
        """
        # 自動添加到集合
        self.add_role(role)
        self.add_resource(resource)
        self.add_permission(permission)

        # 設置矩陣
        key = (role, resource, permission)
        self.matrix[key] = decision

        if condition:
            self.conditions[key] = condition
            logger.info(
                "permission_granted_with_condition",
                role=role,
                resource=resource,
                permission=permission,
                decision=decision,
                condition=condition,
            )
        else:
            logger.info(
                "permission_granted",
                role=role,
                resource=resource,
                permission=permission,
                decision=decision,
            )

    def revoke_permission(self, role: str, resource: str, permission: str) -> None:
        """撤銷權限

        Args:
            role: 角色
            resource: 資源
            permission: 權限
        """
        key = (role, resource, permission)
        if key in self.matrix:
            del self.matrix[key]
            if key in self.conditions:
                del self.conditions[key]
            logger.info(
                "permission_revoked",
                role=role,
                resource=resource,
                permission=permission,
            )

    def check_permission(
        self,
        role: str,
        resource: str,
        permission: str,
        context: dict[str, Any] | None = None,
    ) -> AccessDecision:
        """檢查權限

        Args:
            role: 角色
            resource: 資源
            permission: 權限
            context: 上下文數據（用於條件評估）

        Returns:
            訪問決策
        """
        key = (role, resource, permission)

        # 直接檢查
        if key in self.matrix:
            decision = self.matrix[key]

            # 如果有條件，評估條件
            if key in self.conditions and context:
                condition = self.conditions[key]
                if self._evaluate_condition(condition, context):
                    return decision
                else:
                    return AccessDecision.DENY

            return decision

        # 檢查角色繼承
        if role in self.role_inheritance:
            for parent_role in self.role_inheritance[role]:
                parent_decision = self.check_permission(
                    parent_role, resource, permission, context
                )
                if parent_decision != AccessDecision.DENY:
                    return parent_decision

        return AccessDecision.DENY

    def _evaluate_condition(self, condition: str, context: dict[str, Any]) -> bool:
        """評估條件表達式

        Args:
            condition: 條件表達式
            context: 上下文數據

        Returns:
            條件是否滿足
        """
        try:
            # 簡單的條件評估（生產環境應使用更安全的方式）
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(
                "condition_evaluation_failed", condition=condition, error=str(e)
            )
            return False

    def get_role_permissions(self, role: str) -> list[dict[str, Any]]:
        """獲取角色的所有權限

        Args:
            role: 角色名稱

        Returns:
            權限列表
        """
        permissions = []
        for (r, resource, permission), decision in self.matrix.items():
            if r == role:
                permissions.append(
                    {
                        "resource": resource,
                        "permission": permission,
                        "decision": decision,
                        "has_condition": (r, resource, permission) in self.conditions,
                    }
                )

        logger.debug("role_permissions_retrieved", role=role, count=len(permissions))
        return permissions

    def get_resource_permissions(self, resource: str) -> list[dict[str, Any]]:
        """獲取資源的所有權限

        Args:
            resource: 資源名稱

        Returns:
            權限列表
        """
        permissions = []
        for (role, res, permission), decision in self.matrix.items():
            if res == resource:
                permissions.append(
                    {
                        "role": role,
                        "permission": permission,
                        "decision": decision,
                    }
                )

        logger.debug(
            "resource_permissions_retrieved", resource=resource, count=len(permissions)
        )
        return permissions

    def to_dataframe(self) -> Any:
        """轉換為 Pandas DataFrame

        Returns:
            DataFrame 表示
        """
        data = []
        for (role, resource, permission), decision in self.matrix.items():
            has_condition = (role, resource, permission) in self.conditions
            data.append(
                {
                    "role": role,
                    "resource": resource,
                    "permission": permission,
                    "decision": decision,
                    "conditional": has_condition,
                }
            )

        df = pd.DataFrame(data)
        logger.debug("matrix_converted_to_dataframe", rows=len(df))
        return df

    def to_numpy_matrix(self) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """轉換為 NumPy 矩陣（用於數值分析）

        Returns:
            (矩陣, 角色列表, 資源列表, 權限列表)
        """
        roles_list = sorted(self.roles)
        resources_list = sorted(self.resources)
        permissions_list = sorted(self.permissions)

        # 創建三維矩陣
        matrix = np.zeros(
            (len(roles_list), len(resources_list), len(permissions_list)),
            dtype=int,
        )

        role_idx = {role: i for i, role in enumerate(roles_list)}
        resource_idx = {res: i for i, res in enumerate(resources_list)}
        permission_idx = {perm: i for i, perm in enumerate(permissions_list)}

        for (role, resource, permission), decision in self.matrix.items():
            r_idx = role_idx[role]
            res_idx = resource_idx[resource]
            p_idx = permission_idx[permission]

            # ALLOW=1, DENY=0, CONDITIONAL=2
            if decision == AccessDecision.ALLOW:
                matrix[r_idx, res_idx, p_idx] = 1
            elif decision == AccessDecision.CONDITIONAL:
                matrix[r_idx, res_idx, p_idx] = 2

        logger.debug("matrix_converted_to_numpy", shape=matrix.shape)
        return matrix, roles_list, resources_list, permissions_list

    def analyze_coverage(self) -> dict[str, Any]:
        """分析權限覆蓋率

        Returns:
            覆蓋率分析結果
        """
        total_possible = len(self.roles) * len(self.resources) * len(self.permissions)
        total_defined = len(self.matrix)

        coverage = total_defined / total_possible if total_possible > 0 else 0

        # 統計決策類型
        decision_stats = {
            AccessDecision.ALLOW: 0,
            AccessDecision.DENY: 0,
            AccessDecision.CONDITIONAL: 0,
        }

        for decision in self.matrix.values():
            decision_stats[decision] += 1

        analysis = {
            "total_roles": len(self.roles),
            "total_resources": len(self.resources),
            "total_permissions": len(self.permissions),
            "total_possible_combinations": total_possible,
            "total_defined_rules": total_defined,
            "coverage_percentage": coverage * 100,
            "decision_statistics": {
                "allow": decision_stats[AccessDecision.ALLOW],
                "deny": decision_stats[AccessDecision.DENY],
                "conditional": decision_stats[AccessDecision.CONDITIONAL],
            },
            "conditional_rules_count": len(self.conditions),
            "role_inheritance_count": len(self.role_inheritance),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("coverage_analyzed", coverage=f"{coverage:.2%}")
        return analysis

    def find_over_privileged_roles(self) -> list[dict[str, Any]]:
        """查找過度授權的角色

        Returns:
            過度授權角色列表
        """
        role_permission_counts = {}

        for role in self.roles:
            permissions = self.get_role_permissions(role)
            allow_count = sum(
                1 for p in permissions if p["decision"] == AccessDecision.ALLOW
            )
            role_permission_counts[role] = allow_count

        # 計算平均值和標準差
        counts = list(role_permission_counts.values())
        if not counts:
            return []

        mean = np.mean(counts)
        std = np.std(counts)

        # 超過平均值 + 2 標準差的角色視為過度授權
        threshold = mean + 2 * std

        over_privileged = []
        for role, count in role_permission_counts.items():
            if count > threshold:
                over_privileged.append(
                    {
                        "role": role,
                        "permission_count": count,
                        "average": mean,
                        "threshold": threshold,
                        "excess_permissions": count - threshold,
                    }
                )

        logger.info("over_privileged_roles_found", count=len(over_privileged))
        return over_privileged

    def export_to_dict(self) -> dict[str, Any]:
        """匯出為字典格式

        Returns:
            字典表示
        """
        return {
            "roles": list(self.roles),
            "resources": list(self.resources),
            "permissions": list(self.permissions),
            "matrix": [
                {
                    "role": role,
                    "resource": resource,
                    "permission": permission,
                    "decision": decision.value,
                    "condition": self.conditions.get((role, resource, permission)),
                }
                for (role, resource, permission), decision in self.matrix.items()
            ],
            "role_inheritance": self.role_inheritance,
            "timestamp": datetime.now().isoformat(),
        }


def main():
    """測試範例"""
    matrix = PermissionMatrix()

    # 添加角色
    matrix.add_role("admin")
    matrix.add_role("user")
    matrix.add_role("guest")
    matrix.add_role("power_user", inherits_from=["user"])

    # 授予權限
    matrix.grant_permission("admin", "database", "read", AccessDecision.ALLOW)
    matrix.grant_permission("admin", "database", "write", AccessDecision.ALLOW)
    matrix.grant_permission("admin", "database", "delete", AccessDecision.ALLOW)

    matrix.grant_permission("user", "database", "read", AccessDecision.ALLOW)
    matrix.grant_permission(
        "user",
        "database",
        "write",
        AccessDecision.CONDITIONAL,
        condition="user_id == owner_id",
    )

    matrix.grant_permission("guest", "database", "read", AccessDecision.ALLOW)
    matrix.grant_permission("guest", "database", "write", AccessDecision.DENY)

    # 檢查權限
    print("=== Permission Checks ===")
    print(
        f"Admin read database: {matrix.check_permission('admin', 'database', 'read')}"
    )
    print(
        f"User write database (without context): {matrix.check_permission('user', 'database', 'write')}"
    )
    print(
        f"User write database (with context): {matrix.check_permission('user', 'database', 'write', {'user_id': 123, 'owner_id': 123})}"
    )
    print(
        f"Guest write database: {matrix.check_permission('guest', 'database', 'write')}"
    )
    print(
        f"Power user read database (inherited): {matrix.check_permission('power_user', 'database', 'read')}"
    )

    # 分析覆蓋率
    print("\n=== Coverage Analysis ===")
    analysis = matrix.analyze_coverage()
    print(f"Coverage: {analysis['coverage_percentage']:.2f}%")
    print(f"Allow: {analysis['decision_statistics']['allow']}")
    print(f"Deny: {analysis['decision_statistics']['deny']}")
    print(f"Conditional: {analysis['decision_statistics']['conditional']}")

    # 轉換為 DataFrame
    print("\n=== DataFrame ===")
    df = matrix.to_dataframe()
    print(df)


if __name__ == "__main__":
    main()
