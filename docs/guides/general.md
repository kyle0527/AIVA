# AIVA General Documentation

> **Version**: 2.0 | **Updated**: 2026-01-31

## 1. System Architecture (Five Core Modules)

The AIVA system is built upon a "Five Core Modules" architecture, designed for scalability, autonomy, and cross-platform capability.

### 1.1 Core Modules

1.  **Cognitive Core (`services/core/aiva_core/cognitive_core`)**
    *   **Role**: The brain of the system.
    *   **Function**: Handles decision-making, learning, and knowledge retrieval. It integrates the 5M Neural Core and the RAG system. It has absorbed the functionality of the legacy "External Learning" module.
    *   **Key Components**: `enhanced_decision_agent.py`, `internal_loop_connector.py`.

2.  **Task Planning (`services/core/aiva_core/task_planning`)**
    *   **Role**: The strategist.
    *   **Function**: Breaks down high-level intents into executable plans. It uses `PlanningDispatcher` (formerly `TaskDispatcher`) to route commands.
    *   **Key Components**: `PlanningDispatcher`, `UnifiedExecutor`, `CommandBuilder`.

3.  **Internal Exploration (`services/core/aiva_core/internal_exploration`)**
    *   **Role**: The researcher.
    *   **Function**: Explores the codebase and system environment to understand available tools and capabilities. It uses a dual-layer architecture separating language processing tools from business logic executors.
    *   **Key Components**: `aiva_internal_executor.py`, `aiva_flow_classifier.py`.

4.  **Core Capabilities (`services/core/aiva_core/core_capabilities`)**
    *   **Role**: The toolbox.
    *   **Function**: Contains the fundamental capabilities used by the system, such as dialog processing and basic analysis tools.
    *   **Key Components**: `AIVACommandProcessor` (replaces `AIVADialogAssistant`).

5.  **Service Backbone (`services/core/aiva_core/service_backbone`)**
    *   **Role**: The infrastructure.
    *   **Function**: Manages messaging, resource allocation, and inter-module communication.
    *   **Key Components**: Messaging bus, Resource Manager.

### 1.2 Feature Modules (`services/features`)

AIVA extends its core capabilities with specialized feature modules located in `services/features/`. Each feature module (e.g., `function_xss`, `function_sqli`) is a self-contained unit with its own CLI interface (`__main__.py`).

## 2. User Interface Guide

The AIVA Web UI (`web/index_v3.html`) provides a dashboard for monitoring the system and manually triggering security modules.

### 2.1 Dashboard Overview

*   **Status Panel**: Shows system health, active threads, and memory usage.
*   **Log Panel**: Displays real-time logs of system activities and execution results.
*   **Module List**: Lists available security testing modules (e.g., Price Manipulation, IDOR, XSS Detection) dynamically loaded from the configuration.

### 2.2 Executing Modules

To execute a module:

1.  Locate the desired module in the list (e.g., **XSS Detection**).
2.  Click the **Execute** button (gear icon).
3.  A configuration modal will appear.
4.  Enter the required parameters:
    *   **Target URL**: The URL to test (e.g., `http://localhost:3000/search`).
    *   **Parameters**: Specific params like `product_id`, `price`, `type` (Reflected/Stored/DOM), etc.
    *   **Options**: Checkboxes for toggles like "Full Scan".
5.  Click **Execute** in the modal.
6.  Monitor the **Log Panel** for the execution start message (`[INFO] 開始執行...`) and results.

### 2.3 CLI Usage (Advanced)

Modules can also be executed directly via CLI from the project root. This is useful for automation or headless operation.

**General Syntax:**
```bash
python3 -m services.features.<function_name> [OPTIONS]
```

**Examples:**

*   **XSS Detection:**
    ```bash
    python3 -m services.features.function_xss --url "http://localhost:3000/search" --type reflected --param q
    ```

*   **SQL Injection:**
    ```bash
    python3 -m services.features.function_sqli --url "http://localhost:3000/login" --level 3
    ```

## 3. Extending AIVA

### 3.1 Adding a New Feature Module

1.  Create a new directory in `services/features/` (e.g., `function_new_test`).
2.  Implement `__main__.py` to handle CLI arguments (using `argparse` or `click`).
3.  Ensure the module outputs results in JSON format for the system to parse.
4.  (Optional) Add the module to the `moduleConfig` array in `web/index_v3.html` to make it accessible via the UI.

### 3.2 Updating the UI

The UI is data-driven. To add a new module to the dashboard:
1.  Open `web/index_v3.html`.
2.  Locate the `moduleConfig` constant in the JavaScript section.
3.  Add a new object to the array:
    ```javascript
    {
        id: 'new_test',
        name: 'New Test (Description)',
        desc: 'Short description of the test',
        params: [
            { name: 'url', label: 'Target URL', type: 'url', default: 'http://...' },
            { name: 'depth', label: 'Scan Depth', type: 'number', default: '1' }
        ]
    }
    ```

## 4. Troubleshooting

*   **UI Not Loading**:
    *   Ensure the web server is running.
    *   Check if `web/index_v3.html` is accessible.
*   **Module Execution Failed**:
    *   Check the browser console (F12) for JavaScript errors.
    *   Verify the backend service handling the requests is active.
    *   Ensure the Python environment has all dependencies installed (`pip install -r requirements.txt`).
*   **Logs Not Updating**:
    *   Refresh the page.
    *   Check the WebSocket connection (if applicable) or the log polling mechanism.
*   **CLI Errors**:
    *   Ensure `PYTHONPATH` includes the project root.
    *   Run from the root directory: `export PYTHONPATH=$PYTHONPATH:.`

## 5. Archive

Obsolete documentation referencing the legacy "Six Modules" architecture or "External Learning" as a separate top-level module can be found in `docs/_archive/`.
