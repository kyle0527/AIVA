# AIVA General Documentation

> **Version**: 2.0 | **Updated**: 2026-05-24

## 1. System Architecture (Five Core Modules)

The AIVA system is built upon a "Five Core Modules" architecture, designed for scalability, autonomy, and cross-platform capability.

### 1.1 Core Modules

1.  **Cognitive Core (`services/core/aiva_core/cognitive_core`)**
    *   **Role**: The brain of the system.
    *   **Function**: Handles decision-making, learning, and knowledge retrieval. It integrates the 5M Neural Core and the RAG system. It has absorbed the functionality of the legacy "External Learning" module.

2.  **Task Planning (`services/core/aiva_core/task_planning`)**
    *   **Role**: The strategist.
    *   **Function**: Breaks down high-level intents into executable plans. It uses `PlanningDispatcher` (formerly `TaskDispatcher`) to route commands.

3.  **Internal Exploration (`services/core/aiva_core/internal_exploration`)**
    *   **Role**: The researcher.
    *   **Function**: Explores the codebase and system environment to understand available tools and capabilities. It uses `aiva_internal_executor.py` and `aiva_flow_classifier.py`.

4.  **Core Capabilities (`services/core/aiva_core/core_capabilities`)**
    *   **Role**: The toolbox.
    *   **Function**: Contains the fundamental capabilities used by the system, such as dialog processing (`AIVACommandProcessor`) and basic analysis tools.

5.  **Service Backbone (`services/core/aiva_core/service_backbone`)**
    *   **Role**: The infrastructure.
    *   **Function**: Manages messaging, resource allocation, and inter-module communication.

### 1.2 Feature Modules (`services/features`)

AIVA extends its core capabilities with specialized feature modules located in `services/features/`. Each feature module (e.g., `function_xss`, `function_sqli`) is a self-contained unit with its own CLI interface (`__main__.py`).

## 2. User Interface Guide

The AIVA Web UI (`web/index_v3.html`) provides a dashboard for monitoring the system and manually triggering security modules.

### 2.1 Dashboard Overview

*   **Status Panel**: Shows system health, active threads, and memory usage.
*   **Log Panel**: Displays real-time logs of system activities and execution results.
*   **Module List**: Lists available security testing modules (e.g., Price Manipulation, IDOR, XSS Detection).

### 2.2 Executing Modules

To execute a module:

1.  Locate the desired module in the list (e.g., **XSS Detection**).
2.  Click the **Execute** button (gear icon).
3.  A configuration modal will appear.
4.  Enter the required parameters:
    *   **Target URL**: The URL to test.
    *   **Parameters**: Specific params like `product_id`, `price`, `type` (Reflected/Stored/DOM), etc.
    *   **Options**: Checkboxes for toggles like "Full Scan".
5.  Click **Execute** in the modal.
6.  Monitor the **Log Panel** for the execution start message and results.

### 2.3 CLI Usage (Advanced)

Modules can also be executed directly via CLI from the project root:

```bash
# Example: XSS Detection
python3 -m services.features.function_xss --url "http://localhost:3000/search" --type reflected --param q
```

## 3. Troubleshooting

*   **UI Not Loading**: Ensure the web server is running.
*   **Module Execution Failed**: Check the console logs. Ensure the backend service handling the requests is active.
*   **Logs Not Updating**: Refresh the page.
