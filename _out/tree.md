# 專案樹狀結構

生成時間: 2025-10-13 08:28:35

```
├── 📁 **_out**/
│   ├── 📄 ext_counts.csv
│   ├── 📄 loc_by_ext.csv
│   ├── 📄 PROJECT_REPORT.txt
│   ├── 📄 tree_ascii.txt
│   ├── 📄 tree_unicode.txt
│   ├── 📄 tree.html
│   └── 📄 tree.mmd
├── 📁 **docker**/
│   ├── 📁 **initdb**/
│   │   └── 🗄️ 001_schema.sql
│   ├── 🔧 docker-compose.production.yml
│   ├── 🔧 docker-compose.yml
│   ├── 📄 Dockerfile.integration
│   └── ⚡ entrypoint.integration.sh
├── 📁 **docs**/
│   └── 📁 **diagrams**/
│       ├── 📄 Function____init__.mmd
│       ├── 📄 Function___add_chunk.mmd
│       ├── 📄 Function___create_input_vector.mmd
│       ├── 📄 Function___extract_keywords.mmd
│       ├── 📄 Function___get_mode_display.mmd
│       ├── 📄 Function___index_file.mmd
│       ├── 📄 Function___init_ai_agent.mmd
│       ├── 📄 Function___softmax.mmd
│       ├── 📄 Function__analyze_code.mmd
│       ├── 📄 Function__check_confidence.mmd
│       ├── 📄 Function__create_scan_task.mmd
│       ├── 📄 Function__detect_vulnerability.mmd
│       ├── 📄 Function__forward.mmd
│       ├── 📄 Function__get_ai_history.mmd
│       ├── 📄 Function__get_chunk_count.mmd
│       ├── 📄 Function__get_detections.mmd
│       ├── 📄 Function__get_file_content.mmd
│       ├── 📄 Function__get_stats.mmd
│       ├── 📄 Function__get_tasks.mmd
│       ├── 📄 Function__index_codebase.mmd
│       ├── 📄 Function__invoke.mmd
│       ├── 📄 Function__read_code.mmd
│       ├── 📄 Function__search.mmd
│       └── 📄 Module.mmd
├── 📁 **emoji_backups**/
│   ├── 🐍 services__core__aiva_core__ai_engine__bio_neuron_core_v2.py
│   ├── 🐍 services__core__aiva_core__analysis__strategy_generator.py
│   ├── 🐍 services__core__aiva_core__app.py
│   ├── 🐍 services__core__aiva_core__ui_panel__server.py
│   ├── 🐍 services__function__function_idor__aiva_func_idor__worker.py
│   ├── 🐍 services__scan__aiva_scan__dynamic_engine__example_browser_pool.py
│   ├── 🐍 services__scan__aiva_scan__dynamic_engine__example_usage.py
│   ├── 🐍 services__scan__aiva_scan__info_gatherer__javascript_source_analyzer.py
│   ├── 🐍 tools__replace_emoji.py
│   └── 🐍 tools__update_imports.py
├── 📁 **emoji_backups2**/
│   ├── 🐍 services__core__aiva_core__ai_engine__bio_neuron_core_v2.py
│   └── 🐍 services__core__aiva_core__app.py
├── 📁 **services**/
│   ├── 📁 **aiva_common**/
│   │   ├── 📁 **utils**/
│   │   │   ├── 📁 **dedup**/
│   │   │   ├── 📁 **network**/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 ids.py
│   │   │   └── 🐍 logging.py
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config.py
│   │   ├── 🐍 enums.py
│   │   ├── 🐍 mq.py
│   │   ├── 📄 py.typed
│   │   └── 🐍 schemas.py
│   ├── 📁 **core**/
│   │   └── 📁 **aiva_core**/
│   │       ├── 📁 **ai_engine**/
│   │       ├── 📁 **analysis**/
│   │       ├── 📁 **execution**/
│   │       ├── 📁 **ingestion**/
│   │       ├── 📁 **output**/
│   │       ├── 📁 **state**/
│   │       ├── 📁 **ui_panel**/
│   │       ├── 🐍 __init__.py
│   │       ├── 🐍 ai_ui_schemas.py
│   │       ├── 🐍 app.py
│   │       └── 🐍 schemas.py
│   ├── 📁 **function**/
│   │   ├── 📁 **common**/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 detection_config.py
│   │   │   └── 🐍 unified_smart_detection_manager.py
│   │   ├── 📁 **function_idor**/
│   │   │   └── 📁 **aiva_func_idor**/
│   │   ├── 📁 **function_sqli**/
│   │   │   ├── 📁 **aiva_func_sqli**/
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 **function_ssrf**/
│   │   │   ├── 📁 **aiva_func_ssrf**/
│   │   │   └── 🐍 __init__.py
│   │   └── 📁 **function_xss**/
│   │       ├── 📁 **aiva_func_xss**/
│   │       └── 🐍 __init__.py
│   ├── 📁 **integration**/
│   │   ├── 📁 **aiva_integration**/
│   │   │   ├── 📁 **analysis**/
│   │   │   ├── 📁 **config_template**/
│   │   │   ├── 📁 **middlewares**/
│   │   │   ├── 📁 **observability**/
│   │   │   ├── 📁 **perf_feedback**/
│   │   │   ├── 📁 **reception**/
│   │   │   ├── 📁 **reporting**/
│   │   │   ├── 📁 **security**/
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 app.py
│   │   │   └── 🐍 settings.py
│   │   ├── 📁 **alembic**/
│   │   │   ├── 📁 **versions**/
│   │   │   └── 🐍 env.py
│   │   ├── 📁 **api_gateway**/
│   │   │   └── 📁 **api_gateway**/
│   │   └── 📄 alembic.ini
│   ├── 📁 **scan**/
│   │   └── 📁 **aiva_scan**/
│   │       ├── 📁 **core_crawling_engine**/
│   │       ├── 📁 **dynamic_engine**/
│   │       ├── 📁 **info_gatherer**/
│   │       ├── 🐍 __init__.py
│   │       ├── 🐍 authentication_manager.py
│   │       ├── 🐍 config_control_center.py
│   │       ├── 🐍 fingerprint_manager.py
│   │       ├── 🐍 header_configuration.py
│   │       ├── 🐍 scan_context.py
│   │       ├── 🐍 scan_orchestrator_new.py
│   │       ├── 🐍 scan_orchestrator_old.py
│   │       ├── 🐍 scan_orchestrator.py
│   │       ├── 🐍 schemas.py
│   │       ├── 🐍 scope_manager.py
│   │       ├── 🐍 strategy_controller.py
│   │       └── 🐍 worker.py
│   └── 🐍 __init__.py
├── 📁 **tools**/
│   ├── 🐍 find_non_cp950_filtered.py
│   ├── 🐍 find_non_cp950.py
│   ├── 📄 markdown_check_out.txt
│   ├── 🐍 markdown_check.py
│   ├── 📄 non_cp950_filtered_report.txt
│   ├── 📄 non_cp950_report.txt
│   ├── 🐍 py2mermaid.py
│   ├── 📄 replace_emoji_out.txt
│   ├── 🐍 replace_emoji.py
│   ├── 📄 replace_non_cp950_out.txt
│   ├── 🐍 replace_non_cp950.py
│   └── 🐍 update_imports.py
├── 🐍 __init__.py
├── 📝 ARCHITECTURE_REPORT.md
├── 📝 CORE_MODULE_ANALYSIS.md
├── 📝 DATA_CONTRACT_ANALYSIS.md
├── 📝 DATA_CONTRACT_UPDATE.md
├── 📝 DATA_CONTRACT.md
├── 🐍 demo_bio_neuron_agent.py
├── 🐍 demo_ui_panel.py
├── 📝 FINAL_FIX_REPORT.md
├── 📄 generate_clean_tree.ps1
├── 📄 generate_project_report.ps1
├── 📄 generate_stats.ps1
├── 📄 mypy.ini
├── 📄 pyproject.toml
├── ⚙️ pyrightconfig.json
├── 📝 QUICK_START.md
├── 📄 ruff.toml
├── 📝 SCAN_ENGINE_IMPROVEMENT_REPORT.md
├── ⚡ setup_env.bat
└── ⚡ start_dev.bat

```
