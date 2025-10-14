"""
AuthZ Mapper - 權限映射器

測試多角色權限、映射用戶到角色、分析權限衝突。
"""

from datetime import datetime
from typing import Any

import structlog

from .permission_matrix import AccessDecision, PermissionMatrix

logger = structlog.get_logger(__name__)


class AuthZMapper:
    """
    權限映射器

    管理用戶到角色的映射，提供權限查詢和衝突檢測。
    """

    def __init__(self, permission_matrix: PermissionMatrix):
        """
        初始化權限映射器

        Args:
            permission_matrix: 權限矩陣實例
        """
        self.matrix = permission_matrix

        # 用戶角色映射：{user_id: [roles]}
        self.user_roles: dict[str, list[str]] = {}

        # 用戶屬性：{user_id: {attribute: value}}
        self.user_attributes: dict[str, dict[str, Any]] = {}

        logger.info("authz_mapper_initialized")

    def assign_role_to_user(self, user_id: str, role: str) -> None:
        """
        分配角色給用戶

        Args:
            user_id: 用戶 ID
            role: 角色名稱
        """
        if user_id not in self.user_roles:
            self.user_roles[user_id] = []

        if role not in self.user_roles[user_id]:
            self.user_roles[user_id].append(role)
            logger.info("role_assigned_to_user", user_id=user_id, role=role)

    def revoke_role_from_user(self, user_id: str, role: str) -> None:
        """
        撤銷用戶的角色

        Args:
            user_id: 用戶 ID
            role: 角色名稱
        """
        if user_id in self.user_roles and role in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role)
            logger.info("role_revoked_from_user", user_id=user_id, role=role)

    def set_user_attribute(self, user_id: str, attribute: str, value: Any) -> None:
        """
        設置用戶屬性

        Args:
            user_id: 用戶 ID
            attribute: 屬性名稱
            value: 屬性值
        """
        if user_id not in self.user_attributes:
            self.user_attributes[user_id] = {}

        self.user_attributes[user_id][attribute] = value
        logger.debug("user_attribute_set", user_id=user_id, attribute=attribute)

    def get_user_roles(self, user_id: str) -> list[str]:
        """
        獲取用戶的所有角色

        Args:
            user_id: 用戶 ID

        Returns:
            角色列表
        """
        return self.user_roles.get(user_id, [])

    def check_user_permission(
        self,
        user_id: str,
        resource: str,
        permission: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[AccessDecision, list[str]]:
        """
        檢查用戶權限

        Args:
            user_id: 用戶 ID
            resource: 資源
            permission: 權限
            context: 上下文數據

        Returns:
            (訪問決策, 使用的角色列表)
        """
        user_roles = self.get_user_roles(user_id)

        if not user_roles:
            logger.debug("user_has_no_roles", user_id=user_id)
            return AccessDecision.DENY, []

        # 準備上下文（合併用戶屬性）
        full_context = context or {}
        if user_id in self.user_attributes:
            full_context.update(self.user_attributes[user_id])
        full_context["user_id"] = user_id

        # 檢查每個角色的權限
        decisions: list[tuple[AccessDecision, str]] = []

        for role in user_roles:
            decision = self.matrix.check_permission(role, resource, permission, full_context)
            decisions.append((decision, role))

        # 決策邏輯：
        # 1. 如果任何角色明確 ALLOW，則 ALLOW
        # 2. 如果所有角色都 DENY，則 DENY
        # 3. 如果有 CONDITIONAL，則 CONDITIONAL

        allow_roles = [role for dec, role in decisions if dec == AccessDecision.ALLOW]
        if allow_roles:
            logger.info(
                "user_permission_granted",
                user_id=user_id,
                resource=resource,
                permission=permission,
                roles=allow_roles,
            )
            return AccessDecision.ALLOW, allow_roles

        conditional_roles = [role for dec, role in decisions if dec == AccessDecision.CONDITIONAL]
        if conditional_roles:
            logger.info(
                "user_permission_conditional",
                user_id=user_id,
                resource=resource,
                permission=permission,
                roles=conditional_roles,
            )
            return AccessDecision.CONDITIONAL, conditional_roles

        logger.info(
            "user_permission_denied",
            user_id=user_id,
            resource=resource,
            permission=permission,
        )
        return AccessDecision.DENY, []

    def get_user_all_permissions(self, user_id: str) -> list[dict[str, Any]]:
        """
        獲取用戶的所有權限

        Args:
            user_id: 用戶 ID

        Returns:
            權限列表
        """
        user_roles = self.get_user_roles(user_id)
        all_permissions = []

        for role in user_roles:
            role_permissions = self.matrix.get_role_permissions(role)
            for perm in role_permissions:
                all_permissions.append({
                    **perm,
                    "via_role": role,
                })

        logger.debug("user_all_permissions_retrieved", user_id=user_id, count=len(all_permissions))
        return all_permissions

    def detect_permission_conflicts(self, user_id: str) -> list[dict[str, Any]]:
        """
        檢測用戶的權限衝突

        Args:
            user_id: 用戶 ID

        Returns:
            衝突列表
        """
        user_roles = self.get_user_roles(user_id)
        conflicts = []

        # 收集所有權限決策
        permission_map: dict[tuple[str, str], list[tuple[str, AccessDecision]]] = {}

        for role in user_roles:
            role_permissions = self.matrix.get_role_permissions(role)
            for perm in role_permissions:
                key = (perm["resource"], perm["permission"])
                if key not in permission_map:
                    permission_map[key] = []
                permission_map[key].append((role, perm["decision"]))

        # 檢測衝突
        for (resource, permission), role_decisions in permission_map.items():
            if len(role_decisions) > 1:
                decisions = [dec for _, dec in role_decisions]
                # 如果有 ALLOW 和 DENY 同時存在，視為衝突
                if AccessDecision.ALLOW in decisions and AccessDecision.DENY in decisions:
                    conflicts.append({
                        "resource": resource,
                        "permission": permission,
                        "conflicting_roles": role_decisions,
                        "resolution": "ALLOW takes precedence",
                    })

        logger.info("permission_conflicts_detected", user_id=user_id, count=len(conflicts))
        return conflicts

    def analyze_role_overlap(self) -> list[dict[str, Any]]:
        """
        分析角色重疊情況

        Returns:
            角色重疊分析結果
        """
        overlaps = []
        roles = list(self.matrix.roles)

        for i, role1 in enumerate(roles):
            for role2 in roles[i + 1:]:
                role1_perms = {
                    (p["resource"], p["permission"])
                    for p in self.matrix.get_role_permissions(role1)
                }
                role2_perms = {
                    (p["resource"], p["permission"])
                    for p in self.matrix.get_role_permissions(role2)
                }

                common_perms = role1_perms & role2_perms
                if common_perms:
                    overlap_ratio = len(common_perms) / max(len(role1_perms), len(role2_perms))
                    overlaps.append({
                        "role1": role1,
                        "role2": role2,
                        "common_permissions": len(common_perms),
                        "overlap_ratio": overlap_ratio,
                        "common_items": list(common_perms),
                    })

        logger.info("role_overlap_analyzed", overlaps_count=len(overlaps))
        return overlaps

    def simulate_role_removal(self, user_id: str, role: str) -> dict[str, Any]:
        """
        模擬移除角色的影響

        Args:
            user_id: 用戶 ID
            role: 要移除的角色

        Returns:
            影響分析結果
        """
        # 當前權限
        current_permissions = self.get_user_all_permissions(user_id)
        current_count = len(current_permissions)

        # 臨時移除角色
        user_roles = self.user_roles.get(user_id, [])
        if role not in user_roles:
            return {
                "error": f"User {user_id} does not have role {role}",
            }

        temp_roles = [r for r in user_roles if r != role]
        original_roles = self.user_roles[user_id]
        self.user_roles[user_id] = temp_roles

        # 模擬後的權限
        after_permissions = self.get_user_all_permissions(user_id)
        after_count = len(after_permissions)

        # 恢復原始角色
        self.user_roles[user_id] = original_roles

        # 計算影響
        lost_permissions = current_count - after_count
        impact = {
            "user_id": user_id,
            "removed_role": role,
            "current_permission_count": current_count,
            "after_permission_count": after_count,
            "lost_permissions": lost_permissions,
            "impact_percentage": (lost_permissions / current_count * 100) if current_count > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info("role_removal_simulated", **impact)
        return impact

    def recommend_role_consolidation(self, user_id: str) -> list[dict[str, Any]]:
        """
        推薦角色整合

        Args:
            user_id: 用戶 ID

        Returns:
            推薦列表
        """
        user_roles = self.get_user_roles(user_id)
        if len(user_roles) <= 1:
            return []

        recommendations = []

        # 分析角色重疊
        overlaps = []
        for i, role1 in enumerate(user_roles):
            for role2 in user_roles[i + 1:]:
                role1_perms = self.matrix.get_role_permissions(role1)
                role2_perms = self.matrix.get_role_permissions(role2)

                role1_set = {(p["resource"], p["permission"]) for p in role1_perms}
                role2_set = {(p["resource"], p["permission"]) for p in role2_perms}

                overlap = role1_set & role2_set
                if overlap:
                    overlap_ratio = len(overlap) / min(len(role1_set), len(role2_set))
                    overlaps.append({
                        "role1": role1,
                        "role2": role2,
                        "overlap_ratio": overlap_ratio,
                        "overlap_count": len(overlap),
                    })

        # 推薦高重疊的角色整合
        for overlap in overlaps:
            if overlap["overlap_ratio"] > 0.7:  # 70% 重疊
                recommendations.append({
                    "type": "consolidation",
                    "roles": [overlap["role1"], overlap["role2"]],
                    "reason": f"High overlap ({overlap['overlap_ratio']:.1%})",
                    "suggestion": f"Consider consolidating {overlap['role1']} and {overlap['role2']}",
                })

        logger.info("role_consolidation_recommended", user_id=user_id, count=len(recommendations))
        return recommendations


def main():
    """測試範例"""
    from .permission_matrix import PermissionMatrix

    # 創建權限矩陣
    matrix = PermissionMatrix()

    # 添加角色和權限
    matrix.grant_permission("admin", "database", "read", AccessDecision.ALLOW)
    matrix.grant_permission("admin", "database", "write", AccessDecision.ALLOW)
    matrix.grant_permission("admin", "database", "delete", AccessDecision.ALLOW)

    matrix.grant_permission("user", "database", "read", AccessDecision.ALLOW)
    matrix.grant_permission("user", "database", "write", AccessDecision.CONDITIONAL, condition="user_id == owner_id")
    matrix.grant_permission("user", "database", "delete", AccessDecision.DENY)

    matrix.grant_permission("guest", "database", "read", AccessDecision.ALLOW)

    # 創建映射器
    mapper = AuthZMapper(matrix)

    # 分配角色
    mapper.assign_role_to_user("alice", "admin")
    mapper.assign_role_to_user("bob", "user")
    mapper.assign_role_to_user("bob", "guest")
    mapper.set_user_attribute("bob", "owner_id", 123)

    # 檢查權限
    print("=== Permission Checks ===")
    decision, roles = mapper.check_user_permission("alice", "database", "delete")
    print(f"Alice delete database: {decision} (via {roles})")

    decision, roles = mapper.check_user_permission("bob", "database", "write", {"owner_id": 123})
    print(f"Bob write database: {decision} (via {roles})")

    # 檢測衝突
    print("\n=== Conflicts ===")
    conflicts = mapper.detect_permission_conflicts("bob")
    for conflict in conflicts:
        print(f"Conflict: {conflict}")

    # 模擬移除角色
    print("\n=== Role Removal Simulation ===")
    impact = mapper.simulate_role_removal("bob", "user")
    print(f"Impact: {impact}")


if __name__ == "__main__":
    main()
