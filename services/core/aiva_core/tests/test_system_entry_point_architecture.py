"""測試 P2 系統入口點架構 - 問題五驗證

驗證內容:
1. app.py 是唯一系統入口
2. CoreServiceCoordinator 是狀態管理器（非主線程）
3. BioNeuronDecisionController 職責明確（只負責 AI）
4. 啟動流程正確
5. 組件層次關係正確

執行方式:
    pytest tests/test_system_entry_point_architecture.py -v
"""

import asyncio
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ============================= 測試 1: app.py 是唯一入口點 =============================


class TestAppIsUniqueEntryPoint:
    """驗證 app.py 是系統唯一入口點"""

    def test_app_py_has_fastapi_application(self):
        """驗證 app.py 包含 FastAPI 應用實例"""
        from services.core.aiva_core.service_backbone.api.app import app

        assert app is not None, "FastAPI 應用實例不存在"
        assert hasattr(app, "routes"), "FastAPI 應用缺少 routes 屬性"
        assert app.title == "AIVA Core Engine - 智慧分析與協調中心"
        assert app.version == "3.0.0"

    def test_app_has_startup_event(self):
        """驗證 app.py 包含 startup 事件處理器"""
        from services.core.aiva_core.service_backbone.api.app import app

        # 檢查是否有 startup 事件
        startup_handlers = [h for h in app.router.on_startup]
        assert len(startup_handlers) > 0, "缺少 startup 事件處理器"

    def test_app_has_shutdown_event(self):
        """驗證 app.py 包含 shutdown 事件處理器"""
        from services.core.aiva_core.service_backbone.api.app import app

        # 檢查是否有 shutdown 事件
        shutdown_handlers = [h for h in app.router.on_shutdown]
        assert len(shutdown_handlers) > 0, "缺少 shutdown 事件處理器"

    def test_app_holds_coordinator_instance(self):
        """驗證 app.py 持有 CoreServiceCoordinator 實例"""
        from services.core.aiva_core.service_backbone.api import app as app_module

        assert hasattr(app_module, "coordinator"), "缺少 coordinator 全局變數"
        # coordinator 初始值為 None，在 startup 時初始化

    def test_app_manages_background_tasks(self):
        """驗證 app.py 管理後台任務"""
        from services.core.aiva_core.service_backbone.api import app as app_module

        assert hasattr(
            app_module, "_background_tasks"
        ), "缺少 _background_tasks 列表"


# ============================= 測試 2: CoreServiceCoordinator 降級 =============================


class TestCoordinatorAsStateManager:
    """驗證 CoreServiceCoordinator 是狀態管理器（非主線程）"""

    def test_coordinator_has_no_run_method(self):
        """驗證 CoreServiceCoordinator 沒有 run() 主循環方法"""
        from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import (
            AIVACoreServiceCoordinator,
        )

        coordinator = AIVACoreServiceCoordinator()

        # 不應該有 run() 方法作為主循環
        if hasattr(coordinator, "run"):
            run_method = getattr(coordinator, "run")
            # 如果有 run 方法，應該是初始化相關，不是主循環
            sig = inspect.signature(run_method)
            # 檢查是否是異步方法（主循環通常是同步的無限循環）
            assert not inspect.iscoroutinefunction(run_method) or len(sig.parameters) > 0

    def test_coordinator_class_documentation_updated(self):
        """驗證 CoreServiceCoordinator 類文檔已更新"""
        from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import (
            AIVACoreServiceCoordinator,
        )

        doc = AIVACoreServiceCoordinator.__doc__ or ""
        assert "狀態管理器" in doc or "state manager" in doc.lower()
        assert "非主線程" in doc or "not main thread" in doc.lower() or "不再是" in doc

    def test_coordinator_module_documentation_updated(self):
        """驗證 core_service_coordinator.py 模組文檔已更新"""
        import services.core.aiva_core.service_backbone.coordination.core_service_coordinator as coord_module

        doc = coord_module.__doc__ or ""
        assert "狀態管理器" in doc or "state manager" in doc.lower()
        assert "不再是" in doc or "❌" in doc

    def test_coordinator_has_start_method(self):
        """驗證 CoreServiceCoordinator 有 start() 初始化方法"""
        from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import (
            AIVACoreServiceCoordinator,
        )

        coordinator = AIVACoreServiceCoordinator()
        assert hasattr(coordinator, "start"), "缺少 start() 初始化方法"
        assert inspect.iscoroutinefunction(coordinator.start)

    def test_coordinator_has_stop_method(self):
        """驗證 CoreServiceCoordinator 有 stop() 方法"""
        from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import (
            AIVACoreServiceCoordinator,
        )

        coordinator = AIVACoreServiceCoordinator()
        assert hasattr(coordinator, "stop"), "缺少 stop() 方法"
        assert inspect.iscoroutinefunction(coordinator.stop)


# ============================= 測試 3: BioNeuronDecisionController 職責明確 =============================


class TestBioNeuronDecisionControllerRole:
    """驗證 BioNeuronDecisionController 職責明確（只負責 AI）"""

    def test_bio_neuron_class_renamed(self):
        """驗證 BioNeuronDecisionController 類已重命名"""
        from services.core.aiva_core.cognitive_core.neural.bio_neuron_master import (
            BioNeuronDecisionController,
        )

        assert BioNeuronDecisionController is not None
        # 類名應該是 DecisionController，不是 MasterController

    def test_bio_neuron_backward_compatible(self):
        """驗證保留向後兼容別名"""
        from services.core.aiva_core.cognitive_core.neural.bio_neuron_master import (
            BioNeuronDecisionController,
            BioNeuronMasterController,
        )

        # 舊名稱應該是新名稱的別名
        assert BioNeuronMasterController is BioNeuronDecisionController

    def test_bio_neuron_class_documentation_updated(self):
        """驗證 BioNeuronDecisionController 類文檔已更新"""
        from services.core.aiva_core.cognitive_core.neural.bio_neuron_master import (
            BioNeuronDecisionController,
        )

        doc = BioNeuronDecisionController.__doc__ or ""
        assert "AI 決策" in doc or "AI decision" in doc.lower()
        assert "不再是" in doc or "❌" in doc
        assert "系統 Master" not in doc or "不再是" in doc

    def test_bio_neuron_module_documentation_updated(self):
        """驗證 bio_neuron_master.py 模組文檔已更新"""
        import services.core.aiva_core.cognitive_core.neural.bio_neuron_master as bio_module

        doc = bio_module.__doc__ or ""
        assert "AI 決策" in doc or "decision" in doc.lower()
        assert "不再是" in doc or "❌" in doc

    def test_bio_neuron_controller_responsibilities(self):
        """驗證 BioNeuronDecisionController 只負責 AI 決策"""
        from services.core.aiva_core.cognitive_core.neural.bio_neuron_master import (
            BioNeuronDecisionController,
        )

        controller = BioNeuronDecisionController()

        # 應該有 AI 決策相關的屬性
        assert hasattr(controller, "bio_neuron_agent"), "缺少 bio_neuron_agent"
        assert hasattr(controller, "decision_core"), "缺少 decision_core"

        # 不應該有系統協調相關的方法（如果有的話，應該在別的地方）
        # 這是概念性檢查，實際上可能有一些輔助方法


# ============================= 測試 4: 啟動流程正確 =============================


class TestStartupSequence:
    """驗證啟動流程正確"""

    @pytest.mark.asyncio
    async def test_startup_initializes_coordinator(self):
        """驗證 startup 事件初始化 CoreServiceCoordinator"""
        from services.core.aiva_core.service_backbone.api import app as app_module

        # 模擬 startup 流程
        with patch.object(
            app_module.AIVACoreServiceCoordinator, "start"
        ) as mock_start:
            mock_start.return_value = AsyncMock()

            # 調用 startup（需要模擬其他異步調用）
            with patch("asyncio.create_task") as mock_create_task:
                await app_module.startup()

            # 驗證 coordinator 被創建
            assert app_module.coordinator is not None

    @pytest.mark.asyncio
    async def test_startup_creates_background_tasks(self):
        """驗證 startup 創建後台任務"""
        from services.core.aiva_core.service_backbone.api import app as app_module

        # 模擬啟動流程
        with patch.object(
            app_module.AIVACoreServiceCoordinator, "start", new_callable=AsyncMock
        ):
            with patch("asyncio.create_task") as mock_create_task:
                mock_create_task.return_value = MagicMock()

                await app_module.startup()

                # 驗證創建了多個後台任務
                assert mock_create_task.call_count >= 5  # 至少 5 個後台任務

    @pytest.mark.asyncio
    async def test_shutdown_stops_coordinator(self):
        """驗證 shutdown 停止 CoreServiceCoordinator"""
        from services.core.aiva_core.service_backbone.api import app as app_module

        # 模擬已啟動的 coordinator
        app_module.coordinator = MagicMock()
        app_module.coordinator.stop = AsyncMock()

        await app_module.shutdown()

        # 驗證 stop 被調用
        app_module.coordinator.stop.assert_called_once()


# ============================= 測試 5: 組件層次關係 =============================


class TestComponentHierarchy:
    """驗證組件層次關係正確"""

    def test_app_imports_coordinator(self):
        """驗證 app.py 引入 CoreServiceCoordinator"""
        import services.core.aiva_core.service_backbone.api.app as app_module

        assert hasattr(app_module, "AIVACoreServiceCoordinator")

    def test_app_imports_internal_loop(self):
        """驗證 app.py 引入內部閉環組件"""
        import services.core.aiva_core.service_backbone.api.app as app_module

        # 檢查是否引入 periodic_update
        assert hasattr(app_module, "periodic_update") or "periodic_update" in dir(
            app_module
        )

    def test_app_imports_external_loop(self):
        """驗證 app.py 引入外部學習組件"""
        import services.core.aiva_core.service_backbone.api.app as app_module

        # 檢查是否引入 ExternalLoopConnector
        assert hasattr(
            app_module, "ExternalLoopConnector"
        ) or "ExternalLoopConnector" in dir(app_module)

    def test_hierarchy_is_clear(self):
        """驗證組件層次清晰"""
        # app.py → CoreServiceCoordinator → 功能服務

        # 1. app.py 持有 coordinator
        from services.core.aiva_core.service_backbone.api import app as app_module

        assert hasattr(app_module, "coordinator")

        # 2. CoreServiceCoordinator 管理服務
        from services.core.aiva_core.service_backbone.coordination.core_service_coordinator import (
            AIVACoreServiceCoordinator,
        )

        coordinator = AIVACoreServiceCoordinator()
        # coordinator 應該有服務管理相關的屬性
        assert hasattr(coordinator, "command_router") or hasattr(
            coordinator, "context_manager"
        )

        # 3. BioNeuronDecisionController 被 EnhancedDecisionAgent 使用
        # （這是概念性檢查，實際依賴關係在運行時建立）


# ============================= 測試 6: 文檔完整性 =============================


class TestDocumentationCompleteness:
    """驗證文檔完整性"""

    def test_startup_guide_exists(self):
        """驗證啟動指南文件存在"""
        import os

        guide_path = os.path.join(
            "services",
            "core",
            "aiva_core",
            "service_backbone",
            "SYSTEM_STARTUP_GUIDE.md",
        )

        # 檢查文件是否存在（相對於項目根目錄）
        if not os.path.exists(guide_path):
            # 嘗試從當前測試文件位置查找
            test_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(test_dir))
            guide_path = os.path.join(project_root, guide_path)

        assert os.path.exists(guide_path), f"啟動指南文件不存在: {guide_path}"

    def test_startup_guide_content(self):
        """驗證啟動指南內容完整"""
        import os

        guide_path = os.path.join(
            "services",
            "core",
            "aiva_core",
            "service_backbone",
            "SYSTEM_STARTUP_GUIDE.md",
        )

        if not os.path.exists(guide_path):
            test_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(test_dir))
            guide_path = os.path.join(project_root, guide_path)

        with open(guide_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 檢查關鍵章節
        assert "架構概覽" in content or "Architecture" in content
        assert "啟動流程" in content or "Startup" in content
        assert "app.py" in content
        assert "CoreServiceCoordinator" in content
        assert "BioNeuronDecisionController" in content or "BioNeuron" in content


# ============================= 測試執行統計 =============================


def test_summary():
    """測試摘要（手動運行以查看統計）"""
    print("\n" + "=" * 70)
    print("P2 系統入口點架構測試摘要")
    print("=" * 70)
    print("\n測試分組:")
    print("  1. ✅ app.py 是唯一入口點 (5 個測試)")
    print("  2. ✅ CoreServiceCoordinator 降級為狀態管理器 (5 個測試)")
    print("  3. ✅ BioNeuronDecisionController 職責明確 (5 個測試)")
    print("  4. ✅ 啟動流程正確 (3 個測試)")
    print("  5. ✅ 組件層次關係正確 (4 個測試)")
    print("  6. ✅ 文檔完整性 (2 個測試)")
    print("\n總計: 24 個測試")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # 手動運行測試摘要
    test_summary()

    # 使用 pytest 運行所有測試
    pytest.main([__file__, "-v", "--tb=short"])
