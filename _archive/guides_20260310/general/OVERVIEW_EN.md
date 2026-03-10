# AIVA Architecture Overview

> **Version**: EN-1.0 (對應 AIVA Core v4.4.0) | **Updated**: 2026-01-31  
> 📖 [繁體中文版](../README.md) | [技術詳細文檔](../../services/core/aiva_core/README.md)

## System Architecture (Five Core Modules)

The AIVA system is built upon a "Five Core Modules" architecture, designed for scalability, autonomy, and cross-platform capability.

### Core Modules

1.  **Cognitive Core (`services/core/aiva_core/cognitive_core`)**
    *   **Role**: The brain of the system.
    *   **Function**: Handles decision-making, learning, and knowledge retrieval. It integrates the 5M Neural Core and the RAG system. It has absorbed the functionality of the legacy "External Learning" module.
    *   **Key Components**: `enhanced_decision_agent.py`, `internal_loop_connector.py`.
    *   **Files**: 48 Python modules
    *   **Key Subsystems**: 
        - Neural Network (5M parameters, PyTorch)
        - Decision Support (CapabilityOrchestrator + EnhancedDecisionAgent)
        - RAG Vector Retrieval (384-dimensional semantic vectors)
        - Learning System (integrated from external_learning)

2.  **Task Planning (`services/core/aiva_core/task_planning`)**
    *   **Role**: The strategist.
    *   **Function**: Breaks down high-level intents into executable plans. It uses `PlanningDispatcher` (formerly `TaskDispatcher`) to route commands.
    *   **Key Components**: `PlanningDispatcher`, `UnifiedExecutor`, `CommandBuilder`.
    *   **Files**: 28 Python modules
    *   **Key Subsystems**:
        - Commander (AI strategy engine)
        - Planner (Task generation)
        - Executor (Plan execution)
        - Persistence (State management)

3.  **Internal Exploration (`services/core/aiva_core/internal_exploration`)**
    *   **Role**: The researcher.
    *   **Function**: Explores the codebase and system environment to understand available tools and capabilities. It uses a dual-layer architecture separating language processing tools from business logic executors.
    *   **Key Components**: `aiva_internal_executor.py`, `aiva_flow_classifier.py`.
    *   **Files**: 16 Python modules
    *   **Key Features**:
        - Multi-language AST parsing (Python, Go, Rust, TypeScript)
        - Capability auto-classification
        - Self-healing diagnostics

4.  **Core Capabilities (`services/core/aiva_core/core_capabilities`)**
    *   **Role**: The toolbox.
    *   **Function**: Contains the fundamental capabilities used by the system, such as dialog processing and basic analysis tools.
    *   **Key Components**: `AIVACommandProcessor` (replaces `AIVADialogAssistant`).
    *   **Files**: 21 Python modules
    *   **Key Subsystems**:
        - Analysis (AI-enhanced code analysis)
        - Attack (Vulnerability exploitation orchestrator)
        - CLI (AIVA CLI interface)
        - Dialog (Conversational assistant)
        - Orchestration (Two-phase scan orchestration)

5.  **Service Backbone (`services/core/aiva_core/service_backbone`)**
    *   **Role**: The infrastructure.
    *   **Function**: Manages messaging, resource allocation, and inter-module communication.
    *   **Key Components**: Messaging bus, Resource Manager.
    *   **Files**: 37 Python modules
    *   **Key Subsystems**:
        - API (RESTful services)
        - Coordination (Component coordination)
        - Performance (Monitoring, health checks)
        - Storage (Storage services)
        - Utils (System repair tools)

### Feature Modules (`services/features`)

AIVA extends its core capabilities with specialized feature modules located in `services/features/`. Each feature module (e.g., `function_xss`, `function_sqli`) is a self-contained unit with its own CLI interface (`__main__.py`).

**Key Feature Modules**:
- `function_sqli/` - SQL Injection detection (6 engines)
- `function_xss/` - XSS detection (Reflected, Stored, DOM)
- `function_ssrf/` - SSRF detection
- `function_idor/` - IDOR detection
- `function_authn_go/` - Authentication testing (Go)
- `function_crypto/` - Cryptography testing (Python + Rust)
- `function_postex/` - Post-exploitation

### CLI Usage

Modules can be executed directly via CLI from the project root:

```bash
# General Syntax
python3 -m services.features.<function_name> [OPTIONS]

# XSS Detection
python3 -m services.features.function_xss --url "http://localhost:3000/search" --type reflected --param q

# SQL Injection
python3 -m services.features.function_sqli --url "http://localhost:3000/login" --level 3

# SSRF Detection
python3 -m services.features.function_ssrf --url "http://localhost:3000/api" --param callback

# IDOR Testing
python3 -m services.features.function_idor --url "http://localhost:3000/user/123" --test-horizontal
```

## Architecture Principles

- ✅ **Single Source of Truth (SOT)**: Follow aiva_common specifications, avoid data duplication
- ✅ **Fail Fast**: Don't hide errors, no fallback logic
- ✅ **Event-Driven**: Use asyncio.Future instead of polling
- ✅ **Modular Design**: Five modules work independently but collaboratively
- ✅ **Real Execution**: All 840 capabilities are truly registered, no simulated data
- ✅ **Bug Bounty Optimized**: Four decision methods optimized for HackerOne practice

## Bug Bounty Decision Engine

AIVA v4.4.0 introduces a complete Bug Bounty decision engine, optimized for HackerOne/Bugcrowd scenarios.

### Four Decision Methods

1. **decide_scan_strategy()** - Smart scanning tool selection
2. **decide_phase1_strategy()** - Phase1 deep scan decision
3. **decide_phase2_targets()** - Attack target priority ranking
4. **evaluate_phase2_results()** - Result evaluation and next actions

## Additional Resources

- 📖 [中文技術文檔](../../services/core/aiva_core/README.md) - 完整技術細節
- 🌐 [Web UI 使用指南](../../web/README.md) - Web 管理介面
- 💻 [CLI 參考手冊](../technical/CLI_GUIDE.md) - 命令列詳細說明
- 🔧 [開發指南](../development/) - 開發相關文檔

## Archive

Obsolete documentation referencing the legacy "Six Modules" architecture or "External Learning" as a separate top-level module can be found in `docs/_archive/`.

---

**Last Updated**: 2026-01-31  
**Version**: EN-1.0  
**Corresponds to**: AIVA Core v4.4.0
