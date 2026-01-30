# AIVA 數據流分析摘要

生成時間：2026-01-13 10:18:05

## 基本統計

- 處理檔案數: 119
- 數據流鏈路: 150
- 腳本節點: 119
- 真實連接: 134

## 函數統計

- 總函數數: 638
- 入口函數: 601
- 平均每腳本函數數: 5.4

## 檔案和函數對應表

| 腳本名稱 | 檔案名稱 | 函數數量 | 入口函數 | 主要函數 |
|----------|----------|----------|----------|----------|
| business_schemas | business_schemas.py | 3 | validate_task_id, total_candidates, total_tasks | validate_task_id, total_candidates, total_tasks |
| command_handler | command_handler.py | 2 | __init__, handle_command | __init__, handle_command |
| finding_helper | finding_helper.py | 1 | create_bizlogic_finding | create_bizlogic_finding |
| price_manipulation_scanner | price_manipulation_scanner.py | 10 | __init__, _verify_actual_price_change, _verify_transaction_completed... | __init__, _verify_actual_price_change, _verify_transaction_completed... |
| race_condition_scanner | race_condition_scanner.py | 6 | __init__, test_concurrent_requests, test_balance_manipulation... | __init__, test_concurrent_requests, test_balance_manipulation... |
| workflow_bypass_scanner | workflow_bypass_scanner.py | 7 | __init__, test_step_skipping, test_direct_checkout... | __init__, test_step_skipping, test_direct_checkout... |
| __init__ | __init__.py | 0 |  |  |
| __main__ | __main__.py | 5 | main | mk_finding_dict, run_price_test, run_race_test... |
| __init__ | __init__.py | 0 |  |  |
| command_handler | command_handler.py | 3 | __init__, handle_command | test_idor_handler, __init__, handle_command |
| enhanced_worker | enhanced_worker.py | 6 | run, process_task, to_details... | run, process_task, to_details... |
| resource_id_extractor | resource_id_extractor.py | 3 | extract_from_url, generate_test_ids, replace_id_in_url | extract_from_url, generate_test_ids, replace_id_in_url |
| smart_idor_detector | smart_idor_detector.py | 13 | add_finding, add_error, increment_attempts... | add_finding, add_error, increment_attempts... |
| worker | worker.py | 17 | run, __init__, detect_idor... | _validated_http_url, run, __init__... |
| __init__ | __init__.py | 0 |  |  |
| __main__ | __main__.py | 0 |  |  |
| sensitive_info_detector | sensitive_info_detector.py | 0 |  |  |
| __init__ | __init__.py | 0 |  |  |
| backend_db_fingerprinter | backend_db_fingerprinter.py | 9 | __init__, fingerprint, _extract_version... | __init__, fingerprint, _extract_version... |
| command_handler | command_handler.py | 2 | __init__, handle_command | __init__, handle_command |
| config | config.py | 6 | validate, create_safe_config, create_aggressive_config... | validate, create_safe_config, create_aggressive_config... |
| detection_models | detection_models.py | 3 | __str__, create_detection_result, create_detection_error | __str__, create_detection_result, create_detection_error |
| exceptions | exceptions.py | 2 | __init__, __str__, __init__... | __init__, __str__ |
| hackingtool_config | hackingtool_config.py | 9 | __init__, check_tool_availability, get_available_tools... | __init__, check_tool_availability, get_available_tools... |
| hackingtool_manager | hackingtool_manager.py | 10 | __init__, check_all_tools_status, _check_tool_status... | __init__, check_all_tools_status, _check_tool_status... |
| hackingtool_sql_cli | hackingtool_sql_cli.py | 9 | main, __init__, show_status... | main, __init__, show_status... |
| payload_wrapper_encoder | payload_wrapper_encoder.py | 4 | build_request_dump, __init__, encode... | build_request_dump, __init__, encode... |
| result_binder_publisher | result_binder_publisher.py | 6 | __init__, worker_id, publish_status... | __init__, worker_id, publish_status... |
| smart_detection_manager | smart_detection_manager.py | 6 | __init__, register_detector, start_detection... | __init__, register_detector, start_detection... |
| task_queue | task_queue.py | 4 | __init__, put, get... | __init__, put, get... |
| telemetry | telemetry.py | 7 | to_details, record_engine_execution, record_payload_sent... | to_details, record_engine_execution, record_payload_sent... |
| worker | worker.py | 13 | run, process_task, detect... | run, _consume_queue, _execute_task... |
| __init__ | __init__.py | 0 |  |  |
| command_handler | command_handler.py | 3 | __init__, handle_command | test_ssrf_handler, __init__, handle_command |
| dns_rebinding_detector | dns_rebinding_detector.py | 9 | __init__, generate_vectors, _generate_rebind_it_domain... | __init__, generate_vectors, _generate_rebind_it_domain... |
| internal_address_detector | internal_address_detector.py | 12 | summary, analyze, _test_internal_services... | summary, analyze, _test_internal_services... |
| oast_dispatcher | oast_dispatcher.py | 6 | __init__, register, fetch_events... | __init__, register, fetch_events... |
| param_semantics_analyzer | param_semantics_analyzer.py | 13 | analyze, _get_base_payloads, _get_advanced_payloads... | analyze, _get_base_payloads, _get_advanced_payloads... |
| result_publisher | result_publisher.py | 6 | __init__, worker_id, publish_status... | __init__, worker_id, publish_status... |
| smart_ssrf_detector | smart_ssrf_detector.py | 28 | add_finding, add_error, increment_attempts... | add_finding, add_error, increment_attempts... |
| worker | worker.py | 17 | run, to_details, register... | run, _execute_task, process_task... |
| __init__ | __init__.py | 0 |  |  |
| __main__ | __main__.py | 0 |  |  |
| blind_xss_listener_validator | blind_xss_listener_validator.py | 6 | register_probe, fetch_events, __init__... | register_probe, fetch_events, __init__... |
| command_handler | command_handler.py | 5 | __init__, handle_command, _execute_xss_scan... | _test_xss_command_handler, __init__, handle_command... |
| dom_xss_detector | dom_xss_detector.py | 1 | analyze | analyze |
| hackingtool_config | hackingtool_config.py | 11 | get_xss_tools_config, __init__, _initialize_tools... | get_xss_tools_config, __init__, _initialize_tools... |
| payload_generator | payload_generator.py | 4 | generate, generate_basic_payloads, generate_advanced_payloads... | generate, generate_basic_payloads, generate_advanced_payloads... |
| result_publisher | result_publisher.py | 6 | __init__, worker_id, publish_status... | __init__, worker_id, publish_status... |
| stored_detector | stored_detector.py | 5 | __init__, execute, _submit_payload... | __init__, execute, _submit_payload... |
| task_queue | task_queue.py | 6 | __init__, put, get... | __init__, put, get... |
| traditional_detector | traditional_detector.py | 10 | to_detail, __init__, execute... | _inject_mapping, _inject_query, _payload_in_response... |
| worker | worker.py | 25 | run, _inject_query, to_details... | _validated_http_url, run, _consume_queue... |
| __init__ | __init__.py | 0 |  |  |
| __main__ | __main__.py | 4 | main | run_reflected_test, run_dom_test, run_stored_test... |
| hackingtool_engine | hackingtool_engine.py | 33 | detect_xss, detect, __init__... | get_xss_engine, detect_xss, detect... |
| xss_tools | xss_tools.py | 30 | __post_init__, __post_init__, __init__... | __post_init__, __init__, _find_dalfox_path... |
| __init__ | __init__.py | 0 |  |  |
| dorktara | dorktara.py | 2 | dorkFind | get_user_agent, dorkFind |
| entry | entry.py | 1 | entryy | entryy |
| payloader | payloader.py | 3 |  | pylds, islem, Menu |
| promm | promm.py | 1 |  | proxy_lister |
| xssScan | xssScan.py | 2 | xssFind | get_user_agent, xssFind |
| xsstrike | xsstrike.py | 0 |  |  |
| checker | checker.py | 1 | checker | checker |
| colors | colors.py | 0 |  |  |
| config | config.py | 0 |  |  |
| dom | dom.py | 1 | dom | dom |
| encoders | encoders.py | 1 | base64 | base64 |
| filterChecker | filterChecker.py | 1 | filterChecker | filterChecker |
| fuzzer | fuzzer.py | 1 | fuzzer | fuzzer |
| generator | generator.py | 1 | generator | generator |
| htmlParser | htmlParser.py | 1 | htmlParser | htmlParser |
| jsContexter | jsContexter.py | 1 | jsContexter | jsContexter |
| log | log.py | 12 | _vuln, _run, _good... | _vuln, _run, _good... |
| photon | photon.py | 2 | photon, rec | photon, rec |
| prompt | prompt.py | 1 | prompt | prompt |
| requester | requester.py | 1 | requester | requester |
| updater | updater.py | 1 | updater | updater |
| utils | utils.py | 23 | converter, counter, closest... | converter, counter, closest... |
| wafDetector | wafDetector.py | 1 | wafDetector | wafDetector |
| zetanize | zetanize.py | 3 | zetanize | zetanize, e, d |
| __init__ | __init__.py | 0 |  |  |
| bruteforcer | bruteforcer.py | 1 | bruteforcer | bruteforcer |
| crawl | crawl.py | 1 | crawl | crawl |
| scan | scan.py | 1 | scan | scan |
| singleFuzz | singleFuzz.py | 1 | singleFuzz | singleFuzz |
| __init__ | __init__.py | 0 |  |  |
| retireJs | retireJs.py | 16 | _simple_match, _replacement_match, unique... | is_defined, scan, _simple_match... |
| __init__ | __init__.py | 0 |  |  |
| ssrf_config | ssrf_config.py | 0 |  |  |
| ssrf_detector | ssrf_detector.py | 3 | __init__, analyze, _issue_to_finding | __init__, analyze, _issue_to_finding |
| ssrf_engine | ssrf_engine.py | 8 | __init__, close, _resolve_ips... | __init__, close, _resolve_ips... |
| ssrf_worker | ssrf_worker.py | 1 | run | run |
| sqli_config | sqli_config.py | 0 |  |  |
| sqli_detector | sqli_detector.py | 10 | detect, __init__, _try_import_engine... | detect, __init__, _try_import_engine... |
| boolean_detection_engine | boolean_detection_engine.py | 6 | __init__, detect, _get_baseline_response... | __init__, detect, _get_baseline_response... |
| error_detection_engine | error_detection_engine.py | 4 | __init__, detect, _analyze_error_response... | __init__, detect, _analyze_error_response... |
| hackingtool_engine | hackingtool_engine.py | 18 | __init__, _validate_tools_availability, _check_tool_availability... | __init__, _validate_tools_availability, _check_tool_availability... |
| oob_detection_engine | oob_detection_engine.py | 4 | __init__, detect, _check_oob_response... | __init__, detect, _check_oob_response... |
| time_detection_engine | time_detection_engine.py | 5 | __init__, detect, _measure_baseline_times... | __init__, detect, _measure_baseline_times... |
| union_detection_engine | union_detection_engine.py | 8 | __init__, detect, _get_baseline_response... | __init__, detect, _get_baseline_response... |
| __init__ | __init__.py | 0 |  |  |
| bounty_hunter | bounty_hunter.py | 27 | main, __post_init__, __post_init__... | main, __post_init__, __init__... |
| sql_tools | sql_tools.py | 30 | __init__, _find_sqlmap_path, install_sqlmap... | __init__, _find_sqlmap_path, install_sqlmap... |
| __init__ | __init__.py | 0 |  |  |
| exception | exception.py | 0 |  |  |
| nosqlmap | nosqlmap.py | 0 |  |  |
| nsmcouch | nsmcouch.py | 0 |  |  |
| nsmmongo | nsmmongo.py | 0 |  |  |
| nsmscan | nsmscan.py | 0 |  |  |
| nsmweb | nsmweb.py | 0 |  |  |
| setup | setup.py | 0 |  |  |
| idor_config | idor_config.py | 0 |  |  |
| idor_detector | idor_detector.py | 7 | __init__, analyze, _perform_horizontal_tests... | __init__, analyze, _perform_horizontal_tests... |
| idor_engine | idor_engine.py | 10 | __init__, close, extract_ids_from_url... | __init__, close, extract_ids_from_url... |
| idor_worker | idor_worker.py | 2 | run | _topic, run |
| bizlogic_tools | bizlogic_tools.py | 8 | __post_init__, __post_init__, __post_init__... | __post_init__, __init__, comprehensive_scan... |
| __init__ | __init__.py | 0 |  |  |
## 統計概況

- 處理文件數：119
- 腳本節點數：119
- 真實連接數：134
- 數據流鏈路：150

## 腳本與函數對應表

### `services\features\features_ready\function_bizlogic\business_schemas.py`

**入口函數：**
- validate_task_id
- total_candidates
- total_tasks

**導出函數：**
- validate_task_id
- total_candidates
- total_tasks

**外部調用：**
- ('startswith', 'v')
- ('now', 'datetime')
- ('now', 'datetime')
- ('now', 'datetime')
- ('now', 'datetime')
- ('now', 'datetime')
- ('now', 'datetime')
- ('now', 'datetime')
- ('now', 'datetime')


### `services\features\features_ready\function_bizlogic\command_handler.py`

**入口函數：**
- __init__
- handle_command

**導出函數：**
- __init__
- handle_command

**外部調用：**
- ('time', 'time')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'options_dict')
- ('get', 'options_dict')
- ('get', 'options_dict')
- ('get', 'options_dict')
- ('get', 'options_dict')
- ('get', 'options_dict')
- ('get', 'options_dict')
- ('get', 'options_dict')
- ('get', 'options_dict')
- ('get', 'options_dict')
- ('fromtimestamp', 'datetime')
- ('now', 'datetime')
- ('time', 'time')
- ('get', 'scan_result')
- ('get', 'scan_result')
- ('fromtimestamp', 'datetime')
- ('now', 'datetime')
- ('fromtimestamp', 'datetime')
- ('now', 'datetime')
- ('get', 'scan_result')
- ('get', 'scan_result')
- ('get', 'scan_result')
- ('time', 'time')
- ('time', 'time')
- ('get', 'scan_result')


### `services\features\features_ready\function_bizlogic\finding_helper.py`

**入口函數：**
- create_bizlogic_finding

**導出函數：**
- create_bizlogic_finding

**外部調用：**
- ('get', 'evidence_data')
- ('get', 'evidence_data')
- ('get', 'evidence_data')


### `services\features\features_ready\function_bizlogic\price_manipulation_scanner.py`

**入口函數：**
- __init__
- _verify_actual_price_change
- _verify_transaction_completed
- _verify_user_privilege
- _detect_business_limits
- test_negative_price
- test_zero_price
- test_price_tampering
- test_overflow_price
- run_all_tests

**導出函數：**
- __init__
- _verify_actual_price_change
- _verify_transaction_completed
- _verify_user_privilege
- _detect_business_limits
- test_negative_price
- test_zero_price
- test_price_tampering
- test_overflow_price
- run_all_tests

**外部調用：**
- ('getLogger', 'logging')
- ('get', 'permission_matrix')
- ('info', 'logger')
- ('info', 'logger')
- ('warning', 'logger')
- ('get', 'response_data')
- ('get', 'response_data')
- ('get', 'response_data')
- ('get', 'response_data')
- ('get', 'response_data')
- ('get', 'response_data')
- ('get', 'response_data')
- ('get', 'response_data')
- ('AsyncClient', 'httpx')
- ('AsyncClient', 'httpx')
- ('AsyncClient', 'httpx')
- ('AsyncClient', 'httpx')
- ('gather', 'asyncio')
- ('lower', 'status')
- ('test_negative_price', 'self')
- ('test_zero_price', 'self')
- ('test_price_tampering', 'self')
- ('test_overflow_price', 'self')
- ('extend', 'all_findings')
- ('lower', 'status')
- ('get', 'response_data')
- ('get', 'response_data')
- ('get', 'response_data')
- ('post', 'client')
- ('json', 'response')
- ('_verify_actual_price_change', 'self')
- ('_verify_user_privilege', 'self')
- ('get', 'privilege_check')
- ('append', 'findings')
- ('warning', 'logger')
- ('debug', 'logger')
- ('exception', 'logger')
- ('sleep', 'asyncio')
- ('post', 'client')
- ('json', 'response')
- ('_verify_actual_price_change', 'self')
- ('_verify_user_privilege', 'self')
- ('get', 'privilege_check')
- ('append', 'findings')
- ('warning', 'logger')
- ('debug', 'logger')
- ('exception', 'logger')
- ('get', 'price_verification')
- ('debug', 'logger')
- ('get', 'response_data')
- ('get', 'response_data')
- ('get', 'response_data')
- ('info', 'logger')
- ('info', 'logger')
- ('_verify_transaction_completed', 'self')
- ('get', 'transaction_verification')
- ('debug', 'logger')
- ('post', 'client')
- ('json', 'response')
- ('_verify_actual_price_change', 'self')
- ('_verify_user_privilege', 'self')
- ('get', 'privilege_check')
- ('get', 'price_verification')
- ('append', 'findings')
- ('warning', 'logger')
- ('debug', 'logger')
- ('exception', 'logger')
- ('post', 'client')
- ('append', 'findings')
- ('warning', 'logger')
- ('debug', 'logger')
- ('get', 'client')
- ('json', 'check_response')
- ('debug', 'logger')
- ('get', 'price_verification')
- ('debug', 'logger')
- ('get', 'response_data')
- ('get', 'response_data')
- ('info', 'logger')
- ('info', 'logger')
- ('_verify_transaction_completed', 'self')
- ('get', 'transaction_verification')
- ('debug', 'logger')
- ('get', 'response_data')
- ('get', 'transaction_verification')
- ('get', 'price_verification')
- ('debug', 'logger')
- ('info', 'logger')
- ('_verify_transaction_completed', 'self')
- ('get', 'transaction_verification')
- ('debug', 'logger')
- ('get', 'check_data')
- ('get', 'check_data')
- ('lower', 'transaction_type')
- ('get', 'price_verification')
- ('get', 'transaction_verification')
- ('dumps', 'json')
- ('get', 'transaction_verification')
- ('lower', 'final_status')
- ('dumps', 'json')
- ('get', 'price_verification')
- ('get', 'transaction_verification')
- ('dumps', 'json')
- ('get', 'price_verification')
- ('get', 'transaction_verification')
- ('get', 'price_verification')
- ('get', 'transaction_verification')


### `services\features\features_ready\function_bizlogic\race_condition_scanner.py`

**入口函數：**
- __init__
- test_concurrent_requests
- test_balance_manipulation
- test_coupon_reuse
- test_inventory_depletion
- run_all_tests

**導出函數：**
- __init__
- test_concurrent_requests
- test_balance_manipulation
- test_coupon_reuse
- test_inventory_depletion
- run_all_tests

**外部調用：**
- ('getLogger', 'logging')
- ('info', 'logger')
- ('info', 'logger')
- ('warning', 'logger')
- ('AsyncClient', 'httpx')
- ('now', 'datetime')
- ('AsyncClient', 'httpx')
- ('AsyncClient', 'httpx')
- ('AsyncClient', 'httpx')
- ('gather', 'asyncio')
- ('append', 'tasks')
- ('now', 'datetime')
- ('post', 'client')
- ('post', 'client')
- ('post', 'client')
- ('test_concurrent_requests', 'self')
- ('test_coupon_reuse', 'self')
- ('extend', 'all_findings')
- ('upper', 'method')
- ('post', 'client')
- ('gather', 'asyncio')
- ('append', 'findings')
- ('warning', 'logger')
- ('error', 'logger')
- ('get', 'client')
- ('warning', 'logger')
- ('error', 'logger')
- ('gather', 'asyncio')
- ('get', 'client')
- ('error', 'logger')
- ('gather', 'asyncio')
- ('append', 'findings')
- ('warning', 'logger')
- ('error', 'logger')
- ('gather', 'asyncio')
- ('append', 'findings')
- ('warning', 'logger')
- ('error', 'logger')
- ('get', 'test_endpoints')
- ('get', 'test_endpoints')
- ('upper', 'method')
- ('get', 'client')
- ('request', 'client')
- ('append', 'findings')
- ('warning', 'logger')
- ('json', 'balance_response')
- ('json', 'final_balance_response')


### `services\features\features_ready\function_bizlogic\workflow_bypass_scanner.py`

**入口函數：**
- __init__
- test_step_skipping
- test_direct_checkout
- test_payment_bypass
- test_verification_bypass
- test_admin_access_bypass
- run_all_tests

**導出函數：**
- __init__
- test_step_skipping
- test_direct_checkout
- test_payment_bypass
- test_verification_bypass
- test_admin_access_bypass
- run_all_tests

**外部調用：**
- ('getLogger', 'logging')
- ('info', 'logger')
- ('info', 'logger')
- ('warning', 'logger')
- ('AsyncClient', 'httpx')
- ('AsyncClient', 'httpx')
- ('AsyncClient', 'httpx')
- ('AsyncClient', 'httpx')
- ('AsyncClient', 'httpx')
- ('gather', 'asyncio')
- ('test_direct_checkout', 'self')
- ('test_payment_bypass', 'self')
- ('test_admin_access_bypass', 'self')
- ('extend', 'all_findings')
- ('error', 'logger')
- ('get', 'client')
- ('post', 'client')
- ('append', 'findings')
- ('warning', 'logger')
- ('error', 'logger')
- ('post', 'client')
- ('json', 'response')
- ('error', 'logger')
- ('post', 'client')
- ('json', 'register_response')
- ('error', 'logger')
- ('get', 'client')
- ('append', 'findings')
- ('warning', 'logger')
- ('append', 'findings')
- ('warning', 'logger')
- ('get', 'response_data')
- ('append', 'findings')
- ('warning', 'logger')
- ('get', 'response_data')
- ('append', 'findings')
- ('warning', 'logger')
- ('get', 'client')
- ('debug', 'logger')
- ('append', 'findings')
- ('warning', 'logger')
- ('get', 'response_data')
- ('get_event_loop', 'asyncio')


### `services\features\features_ready\function_bizlogic\__init__.py`


### `services\features\features_ready\function_bizlogic\__main__.py`

**入口函數：**
- main

**導出函數：**
- mk_finding_dict
- run_price_test
- run_race_test
- run_workflow_test
- main

**外部調用：**
- ('basicConfig', 'logging')
- ('getLogger', 'logging')
- ('model_dump', 'payload')
- ('info', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('ArgumentParser', 'argparse')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_subparsers', 'parser')
- ('add_parser', 'subparsers')
- ('add_argument', 'price_parser')
- ('add_parser', 'subparsers')
- ('add_argument', 'race_parser')
- ('add_parser', 'subparsers')
- ('add_argument', 'flow_parser')
- ('parse_args', 'parser')
- ('run', 'asyncio')
- ('dumps', 'json')
- ('run_all_tests', 'tester')
- ('run_all_tests', 'tester')
- ('run_all_tests', 'tester')
- ('dumps', 'json')
- ('get', 'result')
- ('loads', 'json')
- ('error', 'logger')
- ('exit', 'sys')
- ('get', 'result')
- ('get', 'result')
- ('error', 'logger')
- ('dumps', 'json')
- ('uuid4', 'uuid')
- ('uuid4', 'uuid')


### `services\features\features_ready\function_crypto\__init__.py`


### `services\features\features_ready\function_idor\command_handler.py`

**入口函數：**
- __init__
- handle_command

**導出函數：**
- test_idor_handler
- __init__
- handle_command

**外部調用：**
- ('run', 'asyncio')
- ('time', 'time')
- ('handle_command', 'handler')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'payload')
- ('AsyncClient', 'httpx')
- ('time', 'time')
- ('get', 'r')
- ('now', 'datetime')
- ('now', 'datetime')
- ('time', 'time')
- ('now', 'datetime')
- ('now', 'datetime')
- ('get', 'r')
- ('get', 'r')
- ('get', 'r')


### `services\features\features_ready\function_idor\enhanced_worker.py`

**入口函數：**
- run
- process_task
- to_details
- __init__
- run
- _execute_task
- process_task
- _convert_to_finding_payloads

**導出函數：**
- run
- process_task
- to_details
- __init__
- run
- _execute_task
- process_task
- _convert_to_finding_payloads

**外部調用：**
- ('info', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('_convert_to_finding_payloads', 'self')
- ('get_summary', 'stats_collector')
- ('get', 'detection_metrics')
- ('info', 'logger')
- ('run', 'worker')
- ('process_task', 'worker')
- ('AsyncClient', 'httpx')
- ('info', 'logger')
- ('record_request', 'stats_collector')
- ('record_request', 'stats_collector')
- ('get', 'detection_metrics')
- ('record_request', 'stats_collector')
- ('get', 'finding')
- ('get', 'vuln')
- ('append', 'findings')
- ('process_task', 'self')
- ('exception', 'logger')
- ('lower', 'escalation_type')
- ('subscribe', 'broker')
- ('model_validate_json', 'AivaMessage')
- ('exception', 'logger')
- ('publish', 'broker')
- ('lower', 'escalation_type')
- ('get', 'detection_metrics')
- ('get', 'detection_metrics')
- ('get', 'detection_metrics')
- ('get', 'finding_data')
- ('get', 'finding_data')
- ('get', 'finding_data')
- ('_execute_task', 'self')
- ('model_dump', 'finding')
- ('dumps', 'json')
- ('model_dump', 'out')


### `services\features\features_ready\function_idor\resource_id_extractor.py`

**入口函數：**
- extract_from_url
- generate_test_ids
- replace_id_in_url

**導出函數：**
- extract_from_url
- generate_test_ids
- replace_id_in_url

**外部調用：**
- ('split', 'path')
- ('split', 'url')
- ('split', 'query')
- ('fullmatch', 're')
- ('split', 'url')
- ('split', 'param')
- ('append', 'ids')
- ('fullmatch', 're')
- ('copy', 'parts')
- ('append', 'test_ids')
- ('append', 'ids')
- ('append', 'test_ids')
- ('append', 'test_ids')
- ('extend', 'test_ids')
- ('randint', 'random')
- ('randint', 'random')
- ('isdigit', 'char')
- ('add', 'unique_ids')
- ('md5', 'hashlib')
- ('sha256', 'hashlib')
- ('encode', 'random_str')
- ('encode', 'random_str')
- ('randint', 'random')


### `services\features\features_ready\function_idor\smart_idor_detector.py`

**入口函數：**
- add_finding
- add_error
- increment_attempts
- __init__
- detect_vulnerabilities
- _calculate_total_steps
- _extract_resource_ids
- _execute_horizontal_testing
- _execute_vertical_testing
- _test_horizontal_access
- _test_vertical_access
- _build_horizontal_finding
- _build_vertical_finding

**導出函數：**
- add_finding
- add_error
- increment_attempts
- __init__
- detect_vulnerabilities
- _calculate_total_steps
- _extract_resource_ids
- _execute_horizontal_testing
- _execute_vertical_testing
- _test_horizontal_access
- _test_vertical_access
- _build_horizontal_finding
- _build_vertical_finding

**外部調用：**
- ('info', 'logger')
- ('start_detection', 'smart_manager')
- ('debug', 'logger')
- ('increment_attempts', 'context')
- ('increment_attempts', 'context')
- ('info', 'logger')
- ('info', 'logger')
- ('sleep', 'asyncio')
- ('extend', 'resource_ids')
- ('extend', 'resource_ids')
- ('add_error', 'context')
- ('warning', 'logger')
- ('should_continue_testing', 'smart_manager')
- ('should_continue_testing', 'smart_manager')
- ('_build_horizontal_finding', 'self')
- ('add_finding', 'context')
- ('report_vulnerability_found', 'smart_manager')
- ('info', 'logger')
- ('add_error', 'context')
- ('warning', 'logger')
- ('_build_vertical_finding', 'self')
- ('add_finding', 'context')
- ('report_vulnerability_found', 'smart_manager')
- ('info', 'logger')
- ('add_error', 'context')
- ('warning', 'logger')
- ('_extract_resource_ids', 'self')
- ('debug', 'logger')
- ('exception', 'logger')
- ('add_error', 'context')
- ('update_progress', 'smart_manager')
- ('update_progress', 'smart_manager')
- ('_execute_horizontal_testing', 'self')
- ('_execute_vertical_testing', 'self')
- ('get_metrics', 'smart_manager')
- ('_test_horizontal_access', 'self')
- ('_test_vertical_access', 'self')
- ('get_metrics', 'smart_manager')
- ('get_metrics', 'smart_manager')


### `services\features\features_ready\function_idor\worker.py`

**入口函數：**
- run
- __init__
- detect_idor
- detect_vertical_escalation
- _infer_required_privilege
- _build_vertical_finding
- _build_finding
- _extract_auth
- _get_test_user_auth
- _extract_auth_config
- _build_auth_from_config
- _build_bearer_auth
- _build_cookie_auth
- _build_api_key_auth
- _build_basic_auth
- __init__
- process_task

**導出函數：**
- _validated_http_url
- run
- __init__
- detect_idor
- detect_vertical_escalation
- _infer_required_privilege
- _build_vertical_finding
- _build_finding
- _extract_auth
- _get_test_user_auth
- _extract_auth_config
- _build_auth_from_config
- _build_bearer_auth
- _build_cookie_auth
- _build_api_key_auth
- _build_basic_auth
- __init__
- process_task

**外部調用：**
- ('validate_python', '_HTTP_URL_VALIDATOR')
- ('subscribe', 'broker')
- ('AsyncClient', 'httpx')
- ('info', 'logger')
- ('_infer_required_privilege', 'self')
- ('_extract_auth', 'self')
- ('lower', 'url')
- ('_extract_auth_config', 'self')
- ('_build_auth_from_config', 'self')
- ('get', 'auth_config')
- ('get', 'config')
- ('get', 'config')
- ('get', 'config')
- ('get', 'config')
- ('get', 'config')
- ('get', 'config')
- ('model_validate_json', 'AivaMessage')
- ('info', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('_extract_auth', 'self')
- ('_get_test_user_auth', 'self')
- ('debug', 'logger')
- ('_build_bearer_auth', 'self')
- ('detect_idor', 'worker')
- ('detect_vertical_escalation', 'worker')
- ('exception', 'logger')
- ('debug', 'logger')
- ('warning', 'logger')
- ('warning', 'logger')
- ('warning', 'logger')
- ('append', 'findings')
- ('_build_cookie_auth', 'self')
- ('b64encode', 'base64')
- ('model_dump', 'f')
- ('publish', 'broker')
- ('warning', 'logger')
- ('warning', 'logger')
- ('append', 'findings')
- ('_build_vertical_finding', 'self')
- ('_build_api_key_auth', 'self')
- ('model_dump', 'finding')
- ('_build_finding', 'self')
- ('_build_basic_auth', 'self')
- ('dumps', 'json')
- ('model_dump', 'out')


### `services\features\features_ready\function_idor\__init__.py`


### `services\features\features_ready\function_idor\__main__.py`

**外部調用：**
- ('run', 'asyncio')


### `services\features\features_ready\function_info_leak\sensitive_info_detector.py`


### `services\features\features_ready\function_info_leak\__init__.py`


### `services\features\features_ready\function_sqli\backend_db_fingerprinter.py`

**入口函數：**
- __init__
- fingerprint
- _extract_version
- get_supported_databases
- analyze_response_characteristics
- _contains_sql_keywords
- _extract_error_signatures
- add_custom_pattern
- add_custom_version_pattern

**導出函數：**
- __init__
- fingerprint
- _extract_version
- get_supported_databases
- analyze_response_characteristics
- _contains_sql_keywords
- _extract_error_signatures
- add_custom_pattern
- add_custom_version_pattern

**外部調用：**
- ('upper', 'text')
- ('info', 'logger')
- ('info', 'logger')
- ('search', 're')
- ('_contains_sql_keywords', 'self')
- ('_extract_error_signatures', 'self')
- ('findall', 're')
- ('extend', 'error_signatures')
- ('search', 're')
- ('group', 'match')
- ('info', 'logger')
- ('append', 'found_keywords')
- ('info', 'logger')
- ('_extract_version', 'self')


### `services\features\features_ready\function_sqli\command_handler.py`

**入口函數：**
- __init__
- handle_command

**導出函數：**
- __init__
- handle_command

**外部調用：**
- ('time', 'time')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'payload')
- ('time', 'time')
- ('get', 'result')
- ('now', 'datetime')
- ('now', 'datetime')
- ('time', 'time')
- ('get', 'result')
- ('get', 'result')
- ('get', 'result')
- ('get', 'result')
- ('get', 'result')
- ('now', 'datetime')
- ('now', 'datetime')


### `services\features\features_ready\function_sqli\config.py`

**入口函數：**
- validate
- create_safe_config
- create_aggressive_config
- is_engine_enabled
- get_enabled_engines
- to_dict

**導出函數：**
- validate
- create_safe_config
- create_aggressive_config
- is_engine_enabled
- get_enabled_engines
- to_dict


### `services\features\features_ready\function_sqli\detection_models.py`

**入口函數：**
- __str__
- create_detection_result
- create_detection_error

**導出函數：**
- __str__
- create_detection_result
- create_detection_error


### `services\features\features_ready\function_sqli\exceptions.py`

**入口函數：**
- __init__
- __str__
- __init__
- __init__
- __init__
- __init__
- __str__
- __init__
- __init__
- __init__

**導出函數：**
- __init__
- __str__
- __init__
- __init__
- __init__
- __init__
- __str__
- __init__
- __init__
- __init__

**外部調用：**
- ('append', 'parts')
- ('append', 'parts')
- ('append', 'parts')
- ('append', 'parts')
- ('append', 'parts')


### `services\features\features_ready\function_sqli\hackingtool_config.py`

**入口函數：**
- __init__
- check_tool_availability
- get_available_tools
- get_enabled_tools
- get_tools_by_type
- get_tools_by_capability
- generate_capability_records
- install_tool
- run_tool

**導出函數：**
- __init__
- check_tool_availability
- get_available_tools
- get_enabled_tools
- get_tools_by_type
- get_tools_by_capability
- generate_capability_records
- install_tool
- run_tool

**外部調用：**
- ('exists', 'install_path')
- ('check_tool_availability', 'self')
- ('append', 'records')
- ('run', 'subprocess')
- ('model_dump', 'response')
- ('which', 'shutil')
- ('append', 'available')
- ('check_tool_availability', 'self')
- ('append', 'enabled')
- ('run', 'subprocess')
- ('model_dump', 'response')
- ('model_dump', 'response')


### `services\features\features_ready\function_sqli\hackingtool_manager.py`

**入口函數：**
- __init__
- check_all_tools_status
- _check_tool_status
- _test_tool_executable
- install_tool
- install_all_tools
- uninstall_tool
- get_tool_recommendations
- get_installation_script
- generate_status_report

**導出函數：**
- __init__
- check_all_tools_status
- _check_tool_status
- _test_tool_executable
- install_tool
- install_all_tools
- uninstall_tool
- get_tool_recommendations
- get_installation_script
- generate_status_report

**外部調用：**
- ('info', 'logger')
- ('now', 'datetime')
- ('info', 'logger')
- ('info', 'logger')
- ('sort', 'recommendations')
- ('append', 'script_lines')
- ('exists', 'tool_path')
- ('exists', 'install_path')
- ('mkdir', 'install_path')
- ('info', 'logger')
- ('exists', 'install_path')
- ('lower', 'target_type')
- ('extend', 'script_lines')
- ('extend', 'script_lines')
- ('check_all_tools_status', 'self')
- ('cwd', 'Path')
- ('_check_tool_status', 'self')
- ('error', 'logger')
- ('lower', 'tool_name')
- ('create_subprocess_shell', 'asyncio')
- ('wait_for', 'asyncio')
- ('warning', 'logger')
- ('warning', 'logger')
- ('exists', 'install_path')
- ('rmtree', 'shutil')
- ('info', 'logger')
- ('_check_tool_status', 'self')
- ('error', 'logger')
- ('install_tool', 'self')
- ('info', 'logger')
- ('error', 'logger')
- ('rmtree', 'shutil')
- ('info', 'logger')
- ('error', 'logger')
- ('lower', 'target_type')
- ('append', 'recommendations')
- ('append', 'script_lines')
- ('now', 'datetime')
- ('which', 'shutil')
- ('append', 'missing_deps')
- ('_test_tool_executable', 'self')
- ('endswith', 'test_cmd')
- ('endswith', 'test_cmd')
- ('communicate', 'process')
- ('create_subprocess_shell', 'asyncio')
- ('wait_for', 'asyncio')
- ('error', 'logger')
- ('values', 'results')
- ('_check_tool_status', 'self')
- ('values', 'status_data')
- ('values', 'status_data')
- ('now', 'datetime')
- ('communicate', 'process')
- ('now', 'datetime')
- ('decode', 'stderr')
- ('get', 'result')
- ('now', 'datetime')


### `services\features\features_ready\function_sqli\hackingtool_sql_cli.py`

**入口函數：**
- main
- __init__
- show_status
- install_tool
- install_all_tools
- test_tool
- generate_report
- list_tools
- get_recommendations

**導出函數：**
- main
- __init__
- show_status
- install_tool
- install_all_tools
- test_tool
- generate_report
- list_tools
- get_recommendations

**外部調用：**
- ('ArgumentParser', 'argparse')
- ('add_subparsers', 'parser')
- ('add_parser', 'subparsers')
- ('add_parser', 'subparsers')
- ('add_argument', 'install_parser')
- ('add_parser', 'subparsers')
- ('add_parser', 'subparsers')
- ('add_argument', 'test_parser')
- ('add_argument', 'test_parser')
- ('add_parser', 'subparsers')
- ('add_parser', 'subparsers')
- ('add_parser', 'subparsers')
- ('add_argument', 'rec_parser')
- ('parse_args', 'parser')
- ('run', 'asyncio')
- ('items', 'status_data')
- ('items', 'results')
- ('dumps', 'json')
- ('write_text', 'report_file')
- ('items', 'HACKINGTOOL_SQL_CONFIGS')
- ('print_help', 'parser')
- ('get', 'result')
- ('error', 'logger')
- ('append', 'successful')
- ('append', 'failed')
- ('show_status', 'cli')
- ('values', 'status_data')
- ('values', 'status_data')
- ('install_tool', 'cli')
- ('get', 'result')
- ('install_all_tools', 'cli')
- ('keys', 'HACKINGTOOL_SQL_CONFIGS')
- ('get', 'result')
- ('test_tool', 'cli')
- ('generate_report', 'cli')
- ('list_tools', 'cli')
- ('print_help', 'parser')
- ('get_recommendations', 'cli')


### `services\features\features_ready\function_sqli\payload_wrapper_encoder.py`

**入口函數：**
- build_request_dump
- __init__
- encode
- _inject_query

**導出函數：**
- build_request_dump
- __init__
- encode
- _inject_query

**外部調用：**
- ('append', 'body_parts')
- ('append', 'body_parts')
- ('append', 'body_parts')
- ('append', 'body_parts')
- ('append', 'lines')
- ('append', 'lines')
- ('items', 'request_kwargs')
- ('setdefault', 'request_kwargs')
- ('replace', 'body')


### `services\features\features_ready\function_sqli\result_binder_publisher.py`

**入口函數：**
- __init__
- worker_id
- publish_status
- publish_error
- publish_finding
- _publish

**導出函數：**
- __init__
- worker_id
- publish_status
- publish_error
- publish_finding
- _publish

**外部調用：**
- ('_publish', 'self')
- ('publish_status', 'self')
- ('_publish', 'self')
- ('model_dump', 'payload')
- ('dumps', 'json')
- ('uuid4', 'uuid')
- ('model_dump', 'message')


### `services\features\features_ready\function_sqli\smart_detection_manager.py`

**入口函數：**
- __init__
- register_detector
- start_detection
- get_detection_status
- stop_detection
- list_active_detections

**導出函數：**
- __init__
- register_detector
- start_detection
- get_detection_status
- stop_detection
- list_active_detections

**外部調用：**
- ('getLogger', 'logging')
- ('info', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('info', 'logger')


### `services\features\features_ready\function_sqli\task_queue.py`

**入口函數：**
- __init__
- put
- get
- close

**導出函數：**
- __init__
- put
- get
- close

**外部調用：**
- ('Queue', 'asyncio')


### `services\features\features_ready\function_sqli\telemetry.py`

**入口函數：**
- to_details
- record_engine_execution
- record_payload_sent
- record_detection
- record_error
- add_engine
- add_error

**導出函數：**
- to_details
- record_engine_execution
- record_payload_sent
- record_detection
- record_error
- add_engine
- add_error

**外部調用：**
- ('record_engine_execution', 'self')
- ('record_error', 'self')
- ('fromkeys', 'dict')


### `services\features\features_ready\function_sqli\worker.py`

**入口函數：**
- run
- process_task
- detect
- __init__
- register_engine
- unregister_engine
- _setup_default_engines
- execute_detection
- _build_finding
- __init__
- _create_config_from_strategy
- process_task
- process_task_dict

**導出函數：**
- run
- _consume_queue
- _execute_task
- process_task
- detect
- __init__
- register_engine
- unregister_engine
- _setup_default_engines
- execute_detection
- _build_finding
- __init__
- _create_config_from_strategy
- process_task
- process_task_dict

**外部調用：**
- ('create_task', 'asyncio')
- ('info', 'logger')
- ('_setup_default_engines', 'self')
- ('debug', 'logger')
- ('upper', 'strategy')
- ('_create_config_from_strategy', 'self')
- ('publish_status', 'publisher')
- ('process_task', 'service')
- ('debug', 'logger')
- ('register_engine', 'self')
- ('register_engine', 'self')
- ('register_engine', 'self')
- ('register_engine', 'self')
- ('register_engine', 'self')
- ('register_engine', 'self')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('process_task', 'self')
- ('subscribe', 'broker')
- ('model_validate_json', 'AivaMessage')
- ('close', 'queue')
- ('get', 'queue')
- ('process_task', 'service')
- ('info', 'logger')
- ('publish_status', 'publisher')
- ('exception', 'logger')
- ('debug', 'logger')
- ('AsyncClient', 'httpx')
- ('execute_detection', 'orchestrator')
- ('put', 'queue')
- ('publish_finding', 'publisher')
- ('publish_error', 'publisher')
- ('record_payload_test', 'stats')
- ('detect', 'engine')
- ('set_module_specific', 'stats')
- ('warning', 'logger')
- ('warning', 'logger')
- ('exception', 'logger')
- ('execute_detection', 'orchestrator')
- ('model_dump', 'f')
- ('record_request', 'stats')
- ('_build_finding', 'self')
- ('record_request', 'stats')
- ('record_error', 'stats')
- ('record_request', 'stats')
- ('record_error', 'stats')
- ('record_request', 'stats')
- ('record_error', 'stats')
- ('warning', 'logger')
- ('record_vulnerability', 'stats')
- ('record_payload_test', 'stats')


### `services\features\features_ready\function_sqli\__init__.py`


### `services\features\features_ready\function_ssrf\command_handler.py`

**入口函數：**
- __init__
- handle_command

**導出函數：**
- test_ssrf_handler
- __init__
- handle_command

**外部調用：**
- ('run', 'asyncio')
- ('time', 'time')
- ('handle_command', 'handler')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'payload')
- ('analyze', 'analyzer')
- ('time', 'time')
- ('AsyncClient', 'httpx')
- ('extend', 'results')
- ('now', 'datetime')
- ('now', 'datetime')
- ('time', 'time')
- ('now', 'datetime')
- ('now', 'datetime')
- ('get', 'r')
- ('get', 'r')
- ('get', 'r')
- ('get', 'r')
- ('get', 'r')
- ('keys', 'parameters')
- ('keys', 'parameters')


### `services\features\features_ready\function_ssrf\dns_rebinding_detector.py`

**入口函數：**
- __init__
- generate_vectors
- _generate_rebind_it_domain
- _generate_rbndr_domain
- test_rebinding
- _resolve_domain
- verify_internal_access
- generate_payloads

**導出函數：**
- __init__
- generate_vectors
- _generate_rebind_it_domain
- _generate_rbndr_domain
- test_rebinding
- _resolve_domain
- verify_internal_access
- generate_payloads
- ip_to_hex

**外部調用：**
- ('_generate_rebind_it_domain', 'self')
- ('append', 'vectors')
- ('_generate_rbndr_domain', 'self')
- ('append', 'vectors')
- ('split', 'first_ip')
- ('split', 'second_ip')
- ('append', 'vectors')
- ('AsyncClient', 'httpx')
- ('info', 'logger')
- ('_resolve_domain', 'self')
- ('debug', 'logger')
- ('debug', 'logger')
- ('_resolve_domain', 'self')
- ('debug', 'logger')
- ('getaddrinfo', 'socket')
- ('AsyncClient', 'httpx')
- ('rstrip', 'rebinding_url')
- ('debug', 'logger')
- ('generate_vectors', 'self')
- ('warning', 'logger')
- ('sleep', 'asyncio')
- ('warning', 'logger')
- ('info', 'logger')
- ('error', 'logger')
- ('split', 'domain')
- ('split', 'domain')
- ('debug', 'logger')
- ('get', 'client')
- ('info', 'logger')
- ('debug', 'logger')
- ('append', 'payloads')
- ('aclose', 'client')
- ('aclose', 'client')
- ('append', 'payloads')
- ('split', 'ip')


### `services\features\features_ready\function_ssrf\internal_address_detector.py`

**入口函數：**
- summary
- analyze
- _test_internal_services
- _test_protocol_support
- _is_successful_response
- _is_metadata_response
- _is_service_response
- _identify_service_type
- _is_protocol_supported
- _assess_risk_level
- _generate_evidence
- is_internal_address

**導出函數：**
- summary
- analyze
- _test_internal_services
- _test_protocol_support
- _is_successful_response
- _is_metadata_response
- _is_service_response
- _identify_service_type
- _is_protocol_supported
- _assess_risk_level
- _generate_evidence
- is_internal_address

**外部調用：**
- ('lower', 'response')
- ('get', 'metadata_indicators')
- ('lower', 'response')
- ('get', 'service_indicators')
- ('lower', 'response')
- ('get', 'service_map')
- ('get', 'protocol_indicators')
- ('lower', 'response')
- ('get', 'results')
- ('get', 'results')
- ('get', 'results')
- ('get', 'results')
- ('get', 'results')
- ('get', 'results')
- ('get', 'results')
- ('get', 'results')
- ('findall', 're')
- ('is_internal_address', 'self')
- ('_is_metadata_response', 'self')
- ('_is_protocol_supported', 'self')
- ('append', 'indicators')
- ('append', 'evidence')
- ('append', 'evidence')
- ('append', 'evidence')
- ('append', 'evidence')
- ('ip_address', 'ipaddress')
- ('append', 'indicators')
- ('append', 'indicators')
- ('append', 'indicators')
- ('_is_service_response', 'self')
- ('_is_protocol_supported', 'self')
- ('_identify_service_type', 'self')
- ('append', 'detected_services')
- ('info', 'logger')
- ('debug', 'logger')
- ('append', 'supported_protocols')
- ('info', 'logger')
- ('debug', 'logger')
- ('lower', 'indicator')
- ('ip_network', 'ipaddress')
- ('removesuffix', 'protocol')
- ('removesuffix', 'protocol')
- ('lower', 'address')


### `services\features\features_ready\function_ssrf\oast_dispatcher.py`

**入口函數：**
- __init__
- register
- fetch_events
- _validate_oast_event
- close
- _resolve_token

**導出函數：**
- __init__
- register
- fetch_events
- _validate_oast_event
- close
- _resolve_token

**外部調用：**
- ('get', 'payload')
- ('get', 'payload')
- ('_resolve_token', 'self')
- ('get', 'payload')
- ('rstrip', 'token')
- ('split', 'normalized')
- ('AsyncClient', 'httpx')
- ('raise_for_status', 'response')
- ('json', 'response')
- ('AsyncClient', 'httpx')
- ('raise_for_status', 'response')
- ('json', 'response')
- ('get', 'entry')
- ('_validate_oast_event', 'self')
- ('startswith', 'normalized')
- ('post', 'client')
- ('get', 'payload')
- ('get', 'client')
- ('dumps', 'json')
- ('append', 'events')
- ('getenv', 'os')
- ('aclose', 'client')
- ('aclose', 'client')
- ('get', 'entry')
- ('get', 'entry')


### `services\features\features_ready\function_ssrf\param_semantics_analyzer.py`

**入口函數：**
- analyze
- _get_base_payloads
- _get_advanced_payloads
- _add_standard_vectors
- _add_semantic_vectors
- _add_file_vectors
- _add_protocol_vectors
- _add_cross_protocol_vectors
- _get_selected_protocols
- _add_oast_vector
- _build_payloads
- _should_enable_oast
- _tokenize

**導出函數：**
- analyze
- _get_base_payloads
- _get_advanced_payloads
- _add_standard_vectors
- _add_semantic_vectors
- _add_file_vectors
- _add_protocol_vectors
- _add_cross_protocol_vectors
- _get_selected_protocols
- _add_oast_vector
- _build_payloads
- _should_enable_oast
- _tokenize

**外部調用：**
- ('_tokenize', 'self')
- ('_get_base_payloads', 'self')
- ('_add_standard_vectors', 'self')
- ('_add_semantic_vectors', 'self')
- ('_add_cross_protocol_vectors', 'self')
- ('_add_oast_vector', 'self')
- ('_get_selected_protocols', 'self')
- ('get', 'headers')
- ('_should_enable_oast', 'self')
- ('extend', 'payload_sources')
- ('split', 're')
- ('_build_payloads', 'self')
- ('extend', 'payloads')
- ('extend', 'advanced')
- ('generate_payloads', 'dns_detector')
- ('extend', 'advanced')
- ('info', 'logger')
- ('_add_file_vectors', 'self')
- ('_add_protocol_vectors', 'self')
- ('strip', 'payload')
- ('strip', 'payload')
- ('strip', 'payload')
- ('extend', 'payload_sources')
- ('_get_advanced_payloads', 'self')
- ('add', 'seen')
- ('add', 'seen')
- ('add', 'seen')
- ('add', 'seen')
- ('sub', 're')
- ('strip', 'p')
- ('split', 'protocols_hdr')
- ('strip', 'p')
- ('lower', 'normalized')
- ('strip', 'p')
- ('strip', 'p')


### `services\features\features_ready\function_ssrf\result_publisher.py`

**入口函數：**
- __init__
- worker_id
- publish_status
- publish_finding
- publish_error
- _publish

**導出函數：**
- __init__
- worker_id
- publish_status
- publish_finding
- publish_error
- _publish

**外部調用：**
- ('model_dump', 'message')
- ('getenv', 'os')
- ('_publish', 'self')
- ('_publish', 'self')
- ('publish_status', 'self')
- ('model_dump', 'payload')
- ('model_dump', 'finding')
- ('dumps', 'json')


### `services\features\features_ready\function_ssrf\smart_ssrf_detector.py`

**入口函數：**
- add_finding
- add_error
- increment_attempts
- add_oast_callbacks
- __init__
- detect_vulnerabilities
- _prioritize_vectors
- _execute_detection
- _test_vector
- _resolve_payload
- _issue_request
- _parse_target_config
- _process_parameter_injection
- _inject_query_parameter
- _inject_form_parameter
- _inject_json_parameter
- _inject_header_parameter
- _inject_cookie_parameter
- _inject_body_raw_parameter
- _execute_http_request
- _build_internal_finding
- _build_oast_finding
- _verify_internal_service_access
- _verify_service_content
- _verify_cloud_metadata
- _is_valid_json_api
- _is_admin_interface
- _extract_token

**導出函數：**
- add_finding
- add_error
- increment_attempts
- add_oast_callbacks
- __init__
- detect_vulnerabilities
- _prioritize_vectors
- _execute_detection
- _test_vector
- _resolve_payload
- _issue_request
- _parse_target_config
- _process_parameter_injection
- _inject_query_parameter
- _inject_form_parameter
- _inject_json_parameter
- _inject_header_parameter
- _inject_cookie_parameter
- _inject_body_raw_parameter
- _execute_http_request
- _build_internal_finding
- _build_oast_finding
- _verify_internal_service_access
- _verify_service_content
- _verify_cloud_metadata
- _is_valid_json_api
- _is_admin_interface
- _extract_token

**外部調用：**
- ('info', 'logger')
- ('analyze', 'analyzer')
- ('debug', 'logger')
- ('_parse_target_config', 'self')
- ('_process_parameter_injection', 'self')
- ('get', 'injection_handlers')
- ('_verify_service_content', 'self')
- ('_is_valid_json_api', 'self')
- ('_is_admin_interface', 'self')
- ('search', 're')
- ('search', 're')
- ('search', 're')
- ('search', 're')
- ('error', 'logger')
- ('start_detection', 'smart_manager')
- ('update_progress', 'smart_manager')
- ('increment_attempts', 'context')
- ('replace', 'payload')
- ('_execute_http_request', 'self')
- ('request', 'client')
- ('_verify_cloud_metadata', 'self')
- ('loads', 'json')
- ('group', 'match')
- ('group', 'match')
- ('group', 'match')
- ('group', 'match')
- ('split', 'domain')
- ('_prioritize_vectors', 'self')
- ('info', 'logger')
- ('append', 'cloud_vectors')
- ('append', 'other_vectors')
- ('should_continue_testing', 'smart_manager')
- ('info', 'logger')
- ('_test_vector', 'self')
- ('_resolve_payload', 'self')
- ('_issue_request', 'self')
- ('_verify_internal_service_access', 'self')
- ('add_error', 'context')
- ('warning', 'logger')
- ('register', 'dispatcher')
- ('request', 'client')
- ('request', 'client')
- ('request', 'client')
- ('loads', 'json')
- ('loads', 'json')
- ('_execute_detection', 'self')
- ('exception', 'logger')
- ('add_error', 'context')
- ('_build_internal_finding', 'self')
- ('add_finding', 'context')
- ('info', 'logger')
- ('debug', 'logger')
- ('sleep', 'asyncio')
- ('add_oast_callbacks', 'context')
- ('_build_oast_finding', 'self')
- ('add_finding', 'context')
- ('info', 'logger')
- ('md5', 'hashlib')
- ('get_metrics', 'smart_manager')
- ('encode', 'payload')
- ('get_metrics', 'smart_manager')
- ('summary', 'detection')
- ('_extract_token', 'self')
- ('summary', 'detection')


### `services\features\features_ready\function_ssrf\worker.py`

**入口函數：**
- run
- to_details
- register
- fetch_events
- close
- __init__

**導出函數：**
- run
- _execute_task
- process_task
- _resolve_payload
- _issue_request
- _build_internal_finding
- _build_oast_finding
- _format_request
- _format_response
- _safe_elapsed
- _severity_from_summary
- _extract_token
- to_details
- register
- fetch_events
- close
- __init__
- process_task

**外部調用：**
- ('analyze', 'analyzer')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('total_seconds', 'elapsed')
- ('AsyncClient', 'httpx')
- ('publish_status', 'publisher')
- ('info', 'logger')
- ('publish_status', 'publisher')
- ('error', 'logger')
- ('record_payload_test', 'stats_collector')
- ('analyze', 'detector')
- ('replace', 'payload')
- ('_replace', 'parsed')
- ('request', 'client')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('split', 'payload')
- ('warning', 'logger')
- ('exception', 'logger')
- ('publish_finding', 'publisher')
- ('record_request', 'stats_collector')
- ('record_vulnerability', 'stats_collector')
- ('record_payload_test', 'stats_collector')
- ('append', 'findings')
- ('get_summary', 'stats_collector')
- ('register', 'dispatcher')
- ('AsyncClient', 'httpx')
- ('subscribe', 'broker')
- ('model_validate_json', 'AivaMessage')
- ('close', 'dispatcher')
- ('publish_error', 'publisher')
- ('error', 'logger')
- ('record_request', 'stats_collector')
- ('record_error', 'stats_collector')
- ('error', 'logger')
- ('record_request', 'stats_collector')
- ('record_error', 'stats_collector')
- ('error', 'logger')
- ('record_request', 'stats_collector')
- ('record_error', 'stats_collector')
- ('fetch_events', 'dispatcher')
- ('record_vulnerability', 'stats_collector')
- ('record_payload_test', 'stats_collector')
- ('append', 'findings')
- ('dumps', 'json')
- ('fetch_events', 'dispatcher')
- ('summary', 'detection')
- ('replace', 'content')


### `services\features\features_ready\function_ssrf\__init__.py`


### `services\features\features_ready\function_ssrf\__main__.py`

**外部調用：**
- ('run', 'asyncio')


### `services\features\features_ready\function_xss\blind_xss_listener_validator.py`

**入口函數：**
- register_probe
- fetch_events
- __init__
- register_probe
- fetch_events
- _resolve_token
- __init__
- provision_payload
- collect_events

**導出函數：**
- register_probe
- fetch_events
- __init__
- register_probe
- fetch_events
- _resolve_token
- __init__
- provision_payload
- collect_events

**外部調用：**
- ('_resolve_token', 'self')
- ('AsyncClient', 'httpx')
- ('raise_for_status', 'response')
- ('get', 'payload')
- ('get', 'payload')
- ('AsyncClient', 'httpx')
- ('raise_for_status', 'response')
- ('get', 'payload')
- ('get', 'entry')
- ('append', 'events')
- ('getenv', 'os')
- ('post', 'client')
- ('json', 'response')
- ('get', 'payload')
- ('get', 'client')
- ('json', 'response')
- ('dumps', 'json')
- ('getenv', 'os')
- ('aclose', 'client')
- ('aclose', 'client')
- ('get', 'entry')
- ('get', 'entry')


### `services\features\features_ready\function_xss\command_handler.py`

**入口函數：**
- __init__
- handle_command
- _execute_xss_scan
- _build_scan_options

**導出函數：**
- _test_xss_command_handler
- __init__
- handle_command
- _execute_xss_scan
- _build_scan_options

**外部調用：**
- ('run', 'asyncio')
- ('time', 'time')
- ('_build_scan_options', 'self')
- ('update', 'default_options')
- ('handle_command', 'handler')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'payload')
- ('get', 'scan_options')
- ('_execute_xss_scan', 'self')
- ('update', 'default_options')
- ('fromtimestamp', 'datetime')
- ('now', 'datetime')
- ('update', 'default_options')
- ('time', 'time')
- ('fromtimestamp', 'datetime')
- ('now', 'datetime')
- ('fromtimestamp', 'datetime')
- ('now', 'datetime')
- ('fromtimestamp', 'datetime')
- ('now', 'datetime')
- ('update', 'default_options')
- ('time', 'time')
- ('time', 'time')
- ('time', 'time')
- ('time', 'time')
- ('update', 'default_options')
- ('now', 'datetime')
- ('update', 'default_options')


### `services\features\features_ready\function_xss\dom_xss_detector.py`

**入口函數：**
- analyze

**導出函數：**
- analyze

**外部調用：**
- ('find', 'document')
- ('lower', 'window')
- ('strip', 'window')


### `services\features\features_ready\function_xss\hackingtool_config.py`

**入口函數：**
- get_xss_tools_config
- __init__
- _initialize_tools
- _calculate_priority_order
- get_tool_config
- get_tools_by_language
- get_high_priority_tools
- get_tools_by_mode
- validate_tool_requirements
- export_config
- get_execution_plan

**導出函數：**
- get_xss_tools_config
- __init__
- _initialize_tools
- _calculate_priority_order
- get_tool_config
- get_tools_by_language
- get_high_priority_tools
- get_tools_by_mode
- validate_tool_requirements
- export_config
- get_execution_plan

**外部調用：**
- ('_initialize_tools', 'self')
- ('_calculate_priority_order', 'self')
- ('get_tool_config', 'self')
- ('get_high_priority_tools', 'self')
- ('append', 'execution_plan')
- ('dump', 'json')
- ('get_tools_by_mode', 'self')
- ('lower', 'language')


### `services\features\features_ready\function_xss\payload_generator.py`

**入口函數：**
- generate
- generate_basic_payloads
- generate_advanced_payloads
- generate_all_payloads

**導出函數：**
- generate
- generate_basic_payloads
- generate_advanced_payloads
- generate_all_payloads

**外部調用：**
- ('generate', 'self')
- ('generate', 'self')
- ('generate', 'self')
- ('setdefault', 'ordered')
- ('keys', 'ordered')
- ('setdefault', 'ordered')
- ('setdefault', 'ordered')


### `services\features\features_ready\function_xss\result_publisher.py`

**入口函數：**
- __init__
- worker_id
- publish_status
- publish_finding
- publish_error
- _publish

**導出函數：**
- __init__
- worker_id
- publish_status
- publish_finding
- publish_error
- _publish

**外部調用：**
- ('model_dump', 'message')
- ('getenv', 'os')
- ('_publish', 'self')
- ('_publish', 'self')
- ('publish_status', 'self')
- ('model_dump', 'payload')
- ('model_dump', 'finding')
- ('dumps', 'json')


### `services\features\features_ready\function_xss\stored_detector.py`

**入口函數：**
- __init__
- execute
- _submit_payload
- _verify_persistence
- _inject_query

**導出函數：**
- __init__
- execute
- _submit_payload
- _verify_persistence
- _inject_query

**外部調用：**
- ('escape', 'html')
- ('AsyncClient', 'httpx')
- ('_inject_query', 'self')
- ('request', 'client')
- ('_submit_payload', 'self')
- ('get', 'client')
- ('_verify_persistence', 'self')
- ('aclose', 'client')
- ('split', 'pair')
- ('append', 'results')
- ('encode', 'payload')


### `services\features\features_ready\function_xss\task_queue.py`

**入口函數：**
- __init__
- put
- get
- close
- __len__
- _discard_invalid_locked

**導出函數：**
- __init__
- put
- get
- close
- __len__
- _discard_invalid_locked

**外部調用：**
- ('Condition', 'asyncio')
- ('count', 'itertools')
- ('_clock', 'self')
- ('heappush', 'heapq')
- ('heappop', 'heapq')
- ('_clock', 'self')
- ('_discard_invalid_locked', 'self')
- ('_clock', 'self')
- ('heappop', 'heapq')
- ('heappush', 'heapq')
- ('append', 'ready_entries')
- ('wait_for', 'asyncio')


### `services\features\features_ready\function_xss\traditional_detector.py`

**入口函數：**
- to_detail
- __init__
- execute
- errors
- _build_request_parts

**導出函數：**
- _inject_mapping
- _inject_query
- _payload_in_response
- _verify_execution_context
- _detect_waf_interference
- to_detail
- __init__
- execute
- errors
- _build_request_parts

**外部調用：**
- ('keys', 'mapping')
- ('escape', 'html')
- ('lower', 'response_text')
- ('setdefault', 'query_items')
- ('search', 're')
- ('get', 'response_headers')
- ('get', 'response_headers')
- ('lower', 'csp')
- ('finditer', 're')
- ('AsyncClient', 'httpx')
- ('escape', 're')
- ('escape', 're')
- ('escape', 're')
- ('escape', 're')
- ('group', 'match')
- ('lower', 'payload')
- ('search', 're')
- ('split', 'pair')
- ('escape', 're')
- ('escape', 're')
- ('append', 'results')
- ('aclose', 'client')
- ('deepcopy', 'copy')
- ('lower', 'payload')
- ('_build_request_parts', 'self')
- ('deepcopy', 'copy')
- ('request', 'client')
- ('encode', 'payload')


### `services\features\features_ready\function_xss\worker.py`

**入口函數：**
- run
- _inject_query
- to_details
- __init__

**導出函數：**
- _validated_http_url
- run
- _consume_queue
- _execute_task
- process_task
- _setup_blind_xss
- _execute_traditional_detection
- _handle_detection_errors
- _get_dom_engine
- _process_detections
- _analyze_detection_with_dom
- _execute_stored_xss
- _collect_blind_callbacks
- _finalize_statistics
- _build_payloads
- _build_finding
- _build_blind_finding
- _build_impact
- _build_recommendation
- _proof_text
- _format_request
- _format_response
- _inject_query
- to_details
- __init__
- process_task

**外部調用：**
- ('validate_python', '_HTTP_URL_VALIDATOR')
- ('create_task', 'asyncio')
- ('analyze', 'dom_engine')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('set_module_specific', 'stats_collector')
- ('finalize', 'stats_collector')
- ('generate', 'generator')
- ('subscribe', 'broker')
- ('publish_status', 'publisher')
- ('info', 'logger')
- ('publish_status', 'publisher')
- ('finalize', 'stats_collector')
- ('record_payload_test', 'stats_collector')
- ('execute', 'detector')
- ('warning', 'logger')
- ('record_error', 'stats_collector')
- ('to_detail', 'error')
- ('append', 'findings')
- ('record_vulnerability', 'stats_collector')
- ('record_payload_test', 'stats_collector')
- ('collect_events', 'validator')
- ('append', 'findings')
- ('record_oast_callback', 'stats_collector')
- ('record_vulnerability', 'stats_collector')
- ('extend', 'combined_custom')
- ('extend', 'combined_custom')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('append', 'lines')
- ('model_validate_json', 'AivaMessage')
- ('close', 'queue')
- ('get', 'queue')
- ('exception', 'logger')
- ('publish_finding', 'publisher')
- ('get_summary', 'stats_collector')
- ('provision_payload', 'validator')
- ('record_oast_probe', 'stats_collector')
- ('exception', 'logger')
- ('record_error', 'stats_collector')
- ('execute', 'sdetector')
- ('append', 'findings')
- ('exception', 'logger')
- ('put', 'queue')
- ('publish_error', 'publisher')
- ('get_summary', 'stats_collector')
- ('lower', 'name')
- ('lower', 'name')


### `services\features\features_ready\function_xss\__init__.py`


### `services\features\features_ready\function_xss\__main__.py`

**入口函數：**
- main

**導出函數：**
- run_reflected_test
- run_dom_test
- run_stored_test
- main

**外部調用：**
- ('basicConfig', 'logging')
- ('getLogger', 'logging')
- ('info', 'logger')
- ('generate_basic_payloads', 'generator')
- ('info', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('ArgumentParser', 'argparse')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('parse_args', 'parser')
- ('run', 'asyncio')
- ('execute', 'detector')
- ('AsyncClient', 'httpx')
- ('generate_basic_payloads', 'generator')
- ('execute', 'detector')
- ('dumps', 'json')
- ('analyze', 'detector')
- ('error', 'logger')
- ('exit', 'sys')
- ('get', 'client')
- ('error', 'logger')
- ('dumps', 'json')
- ('uuid4', 'uuid')
- ('uuid4', 'uuid')
- ('uuid4', 'uuid')
- ('uuid4', 'uuid')


### `services\features\features_ready\function_xss\engines\hackingtool_engine.py`

**入口函數：**
- detect_xss
- detect
- __init__
- initialize
- _validate_tool_availability
- _check_dalfox_availability
- _check_xspear_availability
- _check_xsser_availability
- _detect_language_environments
- _check_go_environment
- _check_ruby_environment
- _check_python_environment
- _check_rust_environment
- detect
- _get_available_execution_plans
- _execute_parallel_detection
- _execute_tool_detection
- _execute_go_tool
- _execute_ruby_tool
- _execute_python_tool
- _execute_rust_tool
- _parse_tool_output
- _parse_json_output
- _create_result_from_json
- _parse_regex_output
- _process_regex_matches
- _create_result_from_regex
- _run_command
- _is_language_available
- get_available_tools
- get_language_status
- cleanup
- __del__

**導出函數：**
- get_xss_engine
- detect_xss
- detect
- __init__
- initialize
- _validate_tool_availability
- _check_dalfox_availability
- _check_xspear_availability
- _check_xsser_availability
- _detect_language_environments
- _check_go_environment
- _check_ruby_environment
- _check_python_environment
- _check_rust_environment
- detect
- _get_available_execution_plans
- _execute_parallel_detection
- _execute_tool_detection
- _execute_go_tool
- _execute_ruby_tool
- _execute_python_tool
- _execute_rust_tool
- _parse_tool_output
- _parse_json_output
- _create_result_from_json
- _parse_regex_output
- _process_regex_matches
- _create_result_from_regex
- _run_command
- _is_language_available
- get_available_tools
- get_language_status
- cleanup
- __del__

**外部調用：**
- ('getLogger', 'logging')
- ('items', 'language_checkers')
- ('_get_available_execution_plans', 'self')
- ('Semaphore', 'asyncio')
- ('_parse_json_output', 'self')
- ('_parse_regex_output', 'self')
- ('_create_result_from_regex', 'self')
- ('lower', 'pattern')
- ('cleanup', 'self')
- ('detect', 'engine')
- ('mkdtemp', 'tempfile')
- ('_check_dalfox_availability', 'self')
- ('append', 'available_tools')
- ('_check_xspear_availability', 'self')
- ('append', 'available_tools')
- ('_check_xsser_availability', 'self')
- ('append', 'available_tools')
- ('_execute_parallel_detection', 'self')
- ('get', 'plan')
- ('_execute_tool_detection', 'self')
- ('append', 'tasks')
- ('gather', 'asyncio')
- ('time', 'time')
- ('exists', 'output_file')
- ('exists', 'output_file')
- ('format', 'run_pattern')
- ('_run_command', 'self')
- ('_create_result_from_json', 'self')
- ('get', 'json_data')
- ('get', 'json_data')
- ('get', 'json_data')
- ('findall', 're')
- ('create_subprocess_exec', 'asyncio')
- ('CompletedProcess', 'subprocess')
- ('lower', 'language')
- ('_is_language_available', 'self')
- ('initialize', '_xss_engine_instance')
- ('_detect_language_environments', 'self')
- ('_validate_tool_availability', 'self')
- ('which', 'shutil')
- ('_run_command', 'self')
- ('which', 'shutil')
- ('_run_command', 'self')
- ('_run_command', 'self')
- ('timeout', 'asyncio')
- ('timeout', 'asyncio')
- ('_run_command', 'self')
- ('which', 'shutil')
- ('_run_command', 'self')
- ('which', 'shutil')
- ('_is_language_available', 'self')
- ('append', 'available_tools')
- ('get', 'execution_plan')
- ('_parse_tool_output', 'self')
- ('_run_command', 'self')
- ('_run_command', 'self')
- ('_run_command', 'self')
- ('loads', 'json')
- ('get', 'json_data')
- ('_process_regex_matches', 'self')
- ('wait_for', 'asyncio')
- ('kill', 'process')
- ('TimeoutError', 'asyncio')
- ('append', 'available_tools')
- ('rmtree', 'shutil')
- ('_run_command', 'self')
- ('which', 'shutil')
- ('_run_command', 'self')
- ('which', 'shutil')
- ('append', 'detection_results')
- ('lower', 'tool_name')
- ('timeout', 'asyncio')
- ('time', 'time')
- ('read_text', 'output_file')
- ('unlink', 'output_file')
- ('read_text', 'output_file')
- ('unlink', 'output_file')
- ('get', 'json_data')
- ('get', 'item')
- ('communicate', 'process')
- ('decode', 'stdout')
- ('decode', 'stderr')
- ('wait', 'process')
- ('time', 'time')
- ('time', 'time')
- ('strip', 'stdout')
- ('strip', 'stdout')
- ('get', 'json_data')
- ('get', 'item')
- ('get', 'item')
- ('upper', 'language')
- ('lower', 't')
- ('_execute_go_tool', 'self')
- ('time', 'time')
- ('time', 'time')
- ('time', 'time')
- ('_execute_ruby_tool', 'self')
- ('_execute_python_tool', 'self')
- ('_execute_rust_tool', 'self')


### `services\features\features_ready\function_xss\integration_tools\xss_tools.py`

**入口函數：**
- __post_init__
- __post_init__
- __init__
- _find_dalfox_path
- install_dalfox
- scan_target
- _parse_dalfox_output
- __init__
- _load_payloads
- _load_context_specific_payloads
- generate_payloads
- generate_custom_payload
- __init__
- scan_dom_xss
- _analyze_javascript
- _test_dom_payloads
- _check_xss_execution
- __init__
- scan_stored_xss
- _submit_payloads
- _check_stored_execution
- _detect_stored_xss
- __init__
- _generate_blind_payloads
- scan_blind_xss
- _submit_blind_payloads
- _submit_via_forms
- _submit_via_parameters
- _submit_via_headers
- _submit_via_user_agent
- __init__
- _parse_target
- comprehensive_scan
- _custom_xss_scan
- _check_xss_reflection
- _generate_summary

**導出函數：**
- __post_init__
- __post_init__
- __init__
- _find_dalfox_path
- install_dalfox
- scan_target
- _parse_dalfox_output
- __init__
- _load_payloads
- _load_context_specific_payloads
- generate_payloads
- generate_custom_payload
- __init__
- scan_dom_xss
- _analyze_javascript
- _test_dom_payloads
- _check_xss_execution
- __init__
- scan_stored_xss
- _submit_payloads
- _check_stored_execution
- _detect_stored_xss
- __init__
- _generate_blind_payloads
- scan_blind_xss
- _submit_blind_payloads
- _submit_via_forms
- _submit_via_parameters
- _submit_via_headers
- _submit_via_user_agent
- __init__
- _parse_target
- comprehensive_scan
- _custom_xss_scan
- _check_xss_reflection
- _generate_summary

**外部調用：**
- ('getLogger', 'logging')
- ('_find_dalfox_path', 'self')
- ('print', 'console')
- ('_load_payloads', 'self')
- ('_load_context_specific_payloads', 'self')
- ('find_all', 'soup')
- ('_generate_blind_payloads', 'self')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('_parse_target', 'self')
- ('print', 'console')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('_generate_summary', 'self')
- ('print', 'console')
- ('print', 'console')
- ('run', 'subprocess')
- ('print', 'console')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('extend', 'cmd')
- ('print', 'console')
- ('extend', 'payloads')
- ('extend', 'payloads')
- ('extend', 'payloads')
- ('replace', 'payload')
- ('update', 'test_headers')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('extend', 'all_vulns')
- ('run', 'subprocess')
- ('print', 'console')
- ('create_subprocess_exec', 'asyncio')
- ('communicate', 'process')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('extend', 'cmd')
- ('extend', 'cmd')
- ('extend', 'cmd')
- ('extend', 'cmd')
- ('create_subprocess_exec', 'asyncio')
- ('communicate', 'process')
- ('_parse_dalfox_output', 'self')
- ('warning', 'logger')
- ('error', 'logger')
- ('startswith', 'line')
- ('error', 'logger')
- ('ClientSession', 'aiohttp')
- ('error', 'logger')
- ('ClientSession', 'aiohttp')
- ('extend', 'vulnerabilities')
- ('error', 'logger')
- ('error', 'logger')
- ('ClientSession', 'aiohttp')
- ('print', 'console')
- ('print', 'console')
- ('append', 'vulnerabilities')
- ('error', 'logger')
- ('get', 'session')
- ('warning', 'logger')
- ('items', 'parameters')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('ClientSession', 'aiohttp')
- ('error', 'logger')
- ('replace', 'payload')
- ('get', 'results')
- ('now', 'datetime')
- ('install_dalfox', 'self')
- ('extend', 'cmd')
- ('decode', 'stdout')
- ('strip', 'output')
- ('get', 'session')
- ('_analyze_javascript', 'self')
- ('extend', 'vulnerabilities')
- ('extend', 'vulnerabilities')
- ('get', 'session')
- ('_check_xss_execution', 'self')
- ('warning', 'logger')
- ('time', 'time')
- ('_submit_payloads', 'self')
- ('sleep', 'asyncio')
- ('_check_stored_execution', 'self')
- ('time', 'time')
- ('_submit_blind_payloads', 'self')
- ('warning', 'logger')
- ('get', 'session')
- ('warning', 'logger')
- ('now', 'datetime')
- ('error', 'logger')
- ('error', 'logger')
- ('error', 'logger')
- ('error', 'logger')
- ('_custom_xss_scan', 'self')
- ('error', 'logger')
- ('get', 'results')
- ('get', 'results')
- ('get', 'results')
- ('get', 'results')
- ('get', 'results')
- ('loads', 'json')
- ('text', 'response')
- ('_test_dom_payloads', 'self')
- ('text', 'response')
- ('append', 'vulnerabilities')
- ('warning', 'logger')
- ('get', 'session')
- ('warning', 'logger')
- ('warning', 'logger')
- ('get', 'session')
- ('warning', 'logger')
- ('get', 'options')
- ('decode', 'stderr')
- ('decode', 'stderr')
- ('get', 'result')
- ('append', 'vulnerabilities')
- ('post', 'session')
- ('get', 'session')
- ('text', 'response')
- ('_detect_stored_xss', 'self')
- ('post', 'session')
- ('get', 'session')
- ('_check_xss_reflection', 'self')
- ('get', 'v')
- ('get', 'v')
- ('get', 'v')
- ('get', 'v')
- ('search', 're')
- ('append', 'vulnerabilities')
- ('append', 'vulnerabilities')
- ('warning', 'logger')
- ('get', 'v')
- ('get', 'v')
- ('get', 'v')
- ('get', 'v')
- ('get', 'result')
- ('get', 'result')
- ('append', 'vulnerabilities')
- ('post', 'session')
- ('get', 'session')
- ('escape', 're')
- ('escape', 're')
- ('text', 'response')
- ('text', 'response')
- ('get', 'result')
- ('get', 'result')
- ('get', 'result')


### `services\features\features_ready\function_xss\integration_tools\__init__.py`


### `services\features\features_ready\function_xss\external_tools\XSS-LOADER\dorktara.py`

**入口函數：**
- dorkFind

**導出函數：**
- get_user_agent
- dorkFind

**外部調用：**
- ('rstrip', 'line')
- ('exit', 'sys')
- ('get', 'requests')
- ('findAll', 'soup')
- ('choice', 'random')
- ('write', 'f')


### `services\features\features_ready\function_xss\external_tools\XSS-LOADER\entry.py`

**入口函數：**
- entryy

**導出函數：**
- entryy

**外部調用：**
- ('shuffle', 'random')


### `services\features\features_ready\function_xss\external_tools\XSS-LOADER\payloader.py`

**導出函數：**
- pylds
- islem
- Menu

**外部調用：**
- ('entryy', 'entry')
- ('read', 'f')
- ('read', 'f')
- ('read', 'f')
- ('read', 'f')
- ('upper', 'secim')
- ('escape', 'html')
- ('replace', 'secim')
- ('replace', 'secim')
- ('encode', 'secim')
- ('encode', 'secim')
- ('encode', 'secim')
- ('xssFind', 'xssScan')
- ('dorkFind', 'dorktara')
- ('exit', 'sys')
- ('replace', 'secim')
- ('encode', 'secim')
- ('b64encode', 'base64')
- ('decode', 'b')
- ('replace', 'secim')
- ('replace', 'secim')
- ('replace', 'secim')
- ('replace', 'secim')
- ('replace', 'secim')
- ('exit', 'sys')


### `services\features\features_ready\function_xss\external_tools\XSS-LOADER\promm.py`

**導出函數：**
- proxy_lister

**外部調用：**
- ('get', 'requests')
- ('BeautifulSoup', 'bs4')
- ('find_all', 'row')
- ('append', 'data')
- ('find_all', 'soup')
- ('writelines', 'file')


### `services\features\features_ready\function_xss\external_tools\XSS-LOADER\xssScan.py`

**入口函數：**
- xssFind

**導出函數：**
- get_user_agent
- xssFind

**外部調用：**
- ('replace', 'choose')
- ('rstrip', 'line')
- ('exit', 'sys')
- ('get', 'requests')
- ('choice', 'random')
- ('write', 'ss')
- ('exit', 'sys')


### `services\features\features_ready\function_xss\external_tools\XSStrike\xsstrike.py`

**外部調用：**
- ('ArgumentParser', 'argparse')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('add_argument', 'parser')
- ('parse_args', 'parser')
- ('loads', 'json')
- ('no_format', 'logger')
- ('system', 'os')
- ('append', 'seedList')
- ('run', 'logger')
- ('debug', 'logger')
- ('no_format', 'logger')
- ('submit', 'threadpool')
- ('format_help', 'parser')
- ('append', 'forms')
- ('info', 'logger')
- ('append', 'domURLs')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\checker.py`

**入口函數：**
- checker

**導出函數：**
- checker

**外部調用：**
- ('finditer', 're')
- ('append', 'reflectedPositions')
- ('start', 'match')
- ('partial_ratio', 'fuzz')
- ('append', 'allEfficiencies')
- ('partial_ratio', 'fuzz')
- ('append', 'allEfficiencies')
- ('append', 'efficiencies')
- ('append', 'efficiencies')
- ('lower', 'checkString')
- ('lower', 'checkString')
- ('replace', 'checkString')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\colors.py`

**外部調用：**
- ('platform', 'platform')
- ('startswith', 'checkplatform')
- ('system', 'os')
- ('lower', 'machine')
- ('version', 'platform')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\config.py`


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\dom.py`

**入口函數：**
- dom

**導出函數：**
- dom

**外部調用：**
- ('findall', 're')
- ('split', 'script')
- ('split', 'line')
- ('finditer', 're')
- ('finditer', 're')
- ('add', 'allControlledVariables')
- ('append', 'highlighted')
- ('sub', 're')
- ('replace', 'line')
- ('findall', 're')
- ('replace', 'line')
- ('add', 'controlledVariables')
- ('lstrip', 'line')
- ('start', 'grp')
- ('end', 'grp')
- ('add', 'controlledVariables')
- ('start', 'grp')
- ('end', 'grp')
- ('search', 're')
- ('search', 're')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\encoders.py`

**入口函數：**
- base64

**導出函數：**
- base64

**外部調用：**
- ('match', 're')
- ('b64decode', 'b64')
- ('b64encode', 'b64')
- ('encode', 'string')
- ('encode', 'string')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\filterChecker.py`

**入口函數：**
- filterChecker

**導出函數：**
- filterChecker

**外部調用：**
- ('keys', 'occurences')
- ('add', 'environments')
- ('extend', 'efficiencies')
- ('add', 'environments')
- ('add', 'environments')
- ('add', 'environments')
- ('add', 'environments')
- ('add', 'environments')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\fuzzer.py`

**入口函數：**
- fuzzer

**導出函數：**
- fuzzer

**外部調用：**
- ('info', 'logger')
- ('lower', 'fuzz')
- ('error', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('good', 'logger')
- ('error', 'logger')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\generator.py`

**入口函數：**
- generator

**導出函數：**
- generator

**外部調用：**
- ('append', 'ends')
- ('append', 'ends')
- ('append', 'ends')
- ('startswith', 'attributeName')
- ('append', 'ends')
- ('append', 'ends')
- ('replace', 'payload')
- ('split', 'attributeValue')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\htmlParser.py`

**入口函數：**
- htmlParser

**導出函數：**
- htmlParser

**外部調用：**
- ('count', 'response')
- ('sub', 're')
- ('finditer', 're')
- ('replace', 'response')
- ('finditer', 're')
- ('finditer', 're')
- ('finditer', 're')
- ('finditer', 're')
- ('append', 'non_executable_contexts')
- ('keys', 'database')
- ('group', 'occurence')
- ('start', 'occurence')
- ('split', 're')
- ('start', 'occurence')
- ('start', 'occurence')
- ('start', 'occurence')
- ('replace', 'script_checkable')
- ('start', 'each')
- ('end', 'each')
- ('group', 'each')
- ('start', 'occurence')
- ('group', 'occurence')
- ('group', 'occurence')
- ('group', 'occurence')
- ('search', 're')
- ('split', 'part')
- ('group', 'occurence')
- ('split', 'part')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\jsContexter.py`

**入口函數：**
- jsContexter

**導出函數：**
- jsContexter

**外部調用：**
- ('split', 'script')
- ('sub', 're')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\log.py`

**入口函數：**
- _vuln
- _run
- _good
- log_red_line
- log_no_format
- log_debug_json
- setup_logger
- format
- emit

**導出函數：**
- _vuln
- _run
- _good
- _switch_to_no_format_loggers
- _switch_to_default_loggers
- _get_level_and_log
- log_red_line
- log_no_format
- log_debug_json
- setup_logger
- format
- emit

**外部調用：**
- ('addLevelName', 'logging')
- ('addLevelName', 'logging')
- ('addLevelName', 'logging')
- ('isEnabledFor', 'self')
- ('isEnabledFor', 'self')
- ('isEnabledFor', 'self')
- ('removeHandler', 'self')
- ('addHandler', 'self')
- ('removeHandler', 'self')
- ('addHandler', 'self')
- ('isEnabledFor', 'self')
- ('getLogger', 'logging')
- ('setLevel', 'logger')
- ('setLevel', 'console_handler')
- ('setFormatter', 'console_handler')
- ('addHandler', 'logger')
- ('setLevel', 'no_format_console_handler')
- ('setFormatter', 'no_format_console_handler')
- ('_log', 'self')
- ('_log', 'self')
- ('_log', 'self')
- ('removeHandler', 'self')
- ('addHandler', 'self')
- ('removeHandler', 'self')
- ('addHandler', 'self')
- ('upper', 'level')
- ('keys', 'log_config')
- ('info', 'self')
- ('Formatter', 'logging')
- ('Formatter', 'logging')
- ('FileHandler', 'logging')
- ('setLevel', 'file_handler')
- ('setFormatter', 'file_handler')
- ('addHandler', 'logger')
- ('FileHandler', 'logging')
- ('setLevel', 'no_format_file_handler')
- ('setFormatter', 'no_format_file_handler')
- ('keys', 'log_config')
- ('lower', 'level')
- ('debug', 'self')
- ('Formatter', 'logging')
- ('debug', 'self')
- ('debug', 'self')
- ('dumps', 'json')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\photon.py`

**入口函數：**
- photon
- rec

**導出函數：**
- photon
- rec

**外部調用：**
- ('add', 'storage')
- ('add', 'processed')
- ('run', 'logger')
- ('append', 'forms')
- ('findall', 're')
- ('items', 'params')
- ('append', 'forms')
- ('endswith', 'link')
- ('split', 'target')
- ('append', 'inps')
- ('append', 'checkedDOMs')
- ('good', 'logger')
- ('red_line', 'logger')
- ('red_line', 'logger')
- ('split', 'link')
- ('submit', 'threadpool')
- ('sub', 're')
- ('no_format', 'logger')
- ('startswith', 'link')
- ('add', 'storage')
- ('add', 'storage')
- ('add', 'storage')
- ('add', 'storage')
- ('split', 'link')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\prompt.py`

**入口函數：**
- prompt

**導出函數：**
- prompt

**外部調用：**
- ('NamedTemporaryFile', 'tempfile')
- ('fork', 'os')
- ('write', 'tmpfile')
- ('flush', 'tmpfile')
- ('waitpid', 'os')
- ('seek', 'tmpfile')
- ('execvp', 'os')
- ('error', 'logger')
- ('info', 'logger')
- ('read', 'tmpfile')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\requester.py`

**入口函數：**
- requester

**導出函數：**
- requester

**外部調用：**
- ('filterwarnings', 'warnings')
- ('sleep', 'time')
- ('debug', 'logger')
- ('debug', 'logger')
- ('debug_json', 'logger')
- ('debug_json', 'logger')
- ('choice', 'random')
- ('choice', 'random')
- ('get', 'requests')
- ('warning', 'logger')
- ('warning', 'logger')
- ('sleep', 'time')
- ('warning', 'logger')
- ('Response', 'requests')
- ('post', 'requests')
- ('post', 'requests')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\updater.py`

**入口函數：**
- updater

**導出函數：**
- updater

**外部調用：**
- ('run', 'logger')
- ('search', 're')
- ('good', 'logger')
- ('info', 'logger')
- ('good', 'logger')
- ('run', 'logger')
- ('system', 'os')
- ('system', 'os')
- ('good', 'logger')
- ('group', 'changelog')
- ('getcwd', 'os')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\utils.py`

**入口函數：**
- converter
- counter
- closest
- fillHoles
- stripper
- extractHeaders
- replaceValue
- getUrl
- extractScripts
- randomUpper
- flattenParams
- genGen
- getParams
- writer
- reader
- js_extractor
- handle_anchor
- deJSON
- updateVar
- isBadContext
- equalize
- escaped

**導出函數：**
- converter
- counter
- closest
- fillHoles
- stripper
- extractHeaders
- replaceValue
- getUrl
- extractScripts
- randomUpper
- flattenParams
- genGen
- getParams
- writer
- reader
- js_extractor
- handle_anchor
- deJSON
- getVar
- updateVar
- isBadContext
- equalize
- escaped

**外部調用：**
- ('sub', 're')
- ('items', 'numbers')
- ('replace', 'headers')
- ('findall', 're')
- ('findall', 're')
- ('items', 'params')
- ('write', 'savefile')
- ('close', 'savefile')
- ('findall', 're')
- ('replace', 'data')
- ('search', 're')
- ('values', 'anotherMap')
- ('keys', 'anotherMap')
- ('lower', 'response')
- ('append', 'flatted')
- ('split', 'data')
- ('append', 'scripts')
- ('append', 'array')
- ('group', 'match')
- ('loads', 'json')
- ('dumps', 'json')
- ('append', 'filled')
- ('extend', 'filled')
- ('split', 'url')
- ('append', 'scripts')
- ('choice', 'random')
- ('split', 'url')
- ('split', 'part')
- ('dumps', 'json')
- ('encode', 'obj')
- ('startswith', 'url')
- ('split', 'data')
- ('values', 'data')
- ('append', 'each')
- ('endswith', 'parent_url')
- ('values', 'numbers')
- ('upper', 'string')
- ('lower', 'string')
- ('loads', 'json')
- ('replace', 'data')
- ('replace', 'match')
- ('rstrip', 'line')
- ('append', 'vectors')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\wafDetector.py`

**入口函數：**
- wafDetector

**導出函數：**
- wafDetector

**外部調用：**
- ('debug', 'logger')
- ('debug_json', 'logger')
- ('load', 'json')
- ('items', 'wafSignatures')
- ('search', 're')
- ('search', 're')
- ('search', 're')
- ('extend', 'bestMatch')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\zetanize.py`

**入口函數：**
- zetanize

**導出函數：**
- zetanize
- e
- d

**外部調用：**
- ('sub', 're')
- ('findall', 're')
- ('encode', 'string')
- ('decode', 'string')
- ('search', 're')
- ('search', 're')
- ('findall', 're')
- ('search', 're')
- ('search', 're')
- ('search', 're')
- ('group', 'page')
- ('group', 'inpName')
- ('lower', 'inpType')
- ('group', 'method')
- ('group', 'inpType')
- ('group', 'inpValue')


### `services\features\features_ready\function_xss\external_tools\XSStrike\core\__init__.py`


### `services\features\features_ready\function_xss\external_tools\XSStrike\modes\bruteforcer.py`

**入口函數：**
- bruteforcer

**導出函數：**
- bruteforcer

**外部調用：**
- ('debug', 'logger')
- ('debug', 'logger')
- ('debug_json', 'logger')
- ('keys', 'params')
- ('no_format', 'logger')
- ('error', 'logger')
- ('deepcopy', 'copy')
- ('run', 'logger')
- ('info', 'logger')


### `services\features\features_ready\function_xss\external_tools\XSStrike\modes\crawl.py`

**入口函數：**
- crawl

**導出函數：**
- crawl

**外部調用：**
- ('values', 'form')
- ('startswith', 'url')
- ('keys', 'paramData')
- ('startswith', 'url')
- ('startswith', 'url')
- ('match', 're')
- ('deepcopy', 'copy')
- ('keys', 'occurences')
- ('items', 'vectors')
- ('vuln', 'logger')
- ('vuln', 'logger')


### `services\features\features_ready\function_xss\external_tools\XSStrike\modes\scan.py`

**入口函數：**
- scan

**導出函數：**
- scan

**外部調用：**
- ('debug', 'logger')
- ('debug', 'logger')
- ('debug', 'logger')
- ('debug_json', 'logger')
- ('keys', 'params')
- ('startswith', 'target')
- ('run', 'logger')
- ('error', 'logger')
- ('error', 'logger')
- ('good', 'logger')
- ('deepcopy', 'copy')
- ('info', 'logger')
- ('keys', 'occurences')
- ('debug', 'logger')
- ('run', 'logger')
- ('debug', 'logger')
- ('run', 'logger')
- ('values', 'vectors')
- ('info', 'logger')
- ('items', 'vectors')
- ('no_format', 'logger')
- ('good', 'logger')
- ('red_line', 'logger')
- ('red_line', 'logger')
- ('error', 'logger')
- ('info', 'logger')
- ('error', 'logger')
- ('no_format', 'logger')
- ('run', 'logger')
- ('keys', 'params')
- ('replace', 'vect')
- ('red_line', 'logger')
- ('good', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('append', 'efficiencies')
- ('red_line', 'logger')
- ('good', 'logger')
- ('info', 'logger')
- ('info', 'logger')


### `services\features\features_ready\function_xss\external_tools\XSStrike\modes\singleFuzz.py`

**入口函數：**
- singleFuzz

**導出函數：**
- singleFuzz

**外部調用：**
- ('debug', 'logger')
- ('debug', 'logger')
- ('debug', 'logger')
- ('debug_json', 'logger')
- ('keys', 'params')
- ('startswith', 'target')
- ('error', 'logger')
- ('error', 'logger')
- ('good', 'logger')
- ('info', 'logger')
- ('deepcopy', 'copy')
- ('keys', 'params')


### `services\features\features_ready\function_xss\external_tools\XSStrike\modes\__init__.py`


### `services\features\features_ready\function_xss\external_tools\XSStrike\plugins\retireJs.py`

**入口函數：**
- _simple_match
- _replacement_match
- unique
- _replace_version
- is_vulnerable
- scan_filename
- retireJs

**導出函數：**
- is_defined
- scan
- _simple_match
- _replacement_match
- _scanhash
- check
- unique
- _is_at_or_above
- _to_comparable
- _replace_version
- is_vulnerable
- scan_uri
- scan_filename
- scan_file_content
- main_scanner
- retireJs

**外部調用：**
- ('search', 're')
- ('split', 're')
- ('split', 're')
- ('search', 're')
- ('sub', 're')
- ('extend', 'uri_scan_result')
- ('group', 'match')
- ('search', 're')
- ('search', 're')
- ('sub', 're')
- ('append', 'detected')
- ('group', 'ar')
- ('group', 'ar')
- ('group', 'ar')
- ('group', 'match')
- ('loads', 'json')
- ('red_line', 'logger')
- ('good', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('red_line', 'logger')
- ('get', 'result')
- ('get', 'result')
- ('get', 'result')
- ('get', 'result')
- ('sha1', 'hashlib')
- ('add', 'vulnerabilities')
- ('replace', 'vulnerability')
- ('info', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('encode', 'content')
- ('get', 'result')


### `services\features\features_ready\function_xss\external_tools\XSStrike\plugins\__init__.py`


### `services\features\features_ready\function_ssrf\config\ssrf_config.py`


### `services\features\features_ready\function_ssrf\detector\ssrf_detector.py`

**入口函數：**
- __init__
- analyze
- _issue_to_finding

**導出函數：**
- __init__
- analyze
- _issue_to_finding

**外部調用：**
- ('run', 'engine')
- ('_issue_to_finding', 'self')
- ('close', 'engine')


### `services\features\features_ready\function_ssrf\engine\ssrf_engine.py`

**入口函數：**
- __init__
- close
- _resolve_ips
- _is_internal_ip
- check_internal_access
- check_cloud_metadata
- check_file_protocol
- run

**導出函數：**
- __init__
- close
- _resolve_ips
- _is_internal_ip
- check_internal_access
- check_cloud_metadata
- check_file_protocol
- run

**外部調用：**
- ('Limits', 'httpx')
- ('AsyncClient', 'httpx')
- ('getaddrinfo', 'socket')
- ('ip_address', 'ipaddress')
- ('_resolve_ips', 'self')
- ('append', 'issues')
- ('append', 'tasks')
- ('append', 'tasks')
- ('append', 'tasks')
- ('_is_internal_ip', 'self')
- ('append', 'issues')
- ('append', 'issues')
- ('lower', 'url')
- ('check_internal_access', 'self')
- ('check_cloud_metadata', 'self')
- ('check_file_protocol', 'self')
- ('gather', 'asyncio')
- ('append', 'ips')
- ('append', 'issues')
- ('extend', 'issues')
- ('append', 'issues')
- ('append', 'issues')
- ('get', 'headers_map')
- ('split', 'url')


### `services\features\features_ready\function_ssrf\worker\ssrf_worker.py`

**入口函數：**
- run

**導出函數：**
- run

**外部調用：**
- ('subscribe', 'broker')
- ('model_validate_json', 'AivaMessage')
- ('getenv', 'os')
- ('getenv', 'os')
- ('publish', 'broker')
- ('analyze', 'detector')
- ('publish', 'broker')
- ('exception', 'logger')
- ('publish', 'broker')
- ('publish', 'broker')
- ('getenv', 'os')
- ('getenv', 'os')
- ('getenv', 'os')
- ('getenv', 'os')
- ('getenv', 'os')
- ('model_dump', 'f')
- ('dumps', 'json')
- ('dumps', 'json')
- ('dumps', 'json')
- ('dumps', 'json')
- ('model_dump', 'out')


### `services\features\features_ready\function_sqli\config\sqli_config.py`


### `services\features\features_ready\function_sqli\detector\sqli_detector.py`

**入口函數：**
- detect
- __init__
- _try_import_engine
- detect_sqli
- _execute_parallel_detection
- _process_and_merge_results
- _deduplicate_and_normalize
- _order_engines
- idx
- _async_wrapper

**導出函數：**
- detect
- __init__
- _try_import_engine
- detect_sqli
- _execute_parallel_detection
- _process_and_merge_results
- _deduplicate_and_normalize
- _order_engines
- idx
- _async_wrapper

**外部調用：**
- ('_order_engines', 'self')
- ('_process_and_merge_results', 'self')
- ('_deduplicate_and_normalize', 'self')
- ('_try_import_engine', 'self')
- ('import_module', 'importlib')
- ('get', 'params')
- ('get', 'params')
- ('_execute_parallel_detection', 'self')
- ('detect', 'engine')
- ('gather', 'asyncio')
- ('extend', 'flat_results')
- ('add', 'seen')
- ('append', 'merged')
- ('iscoroutinefunction', 'inspect')
- ('index', 'order')
- ('to_thread', 'asyncio')


### `services\features\features_ready\function_sqli\engines\boolean_detection_engine.py`

**入口函數：**
- __init__
- detect
- _get_baseline_response
- _send_payload_request
- _analyze_boolean_responses
- _build_detection_result

**導出函數：**
- __init__
- detect
- _get_baseline_response
- _send_payload_request
- _analyze_boolean_responses
- _build_detection_result

**外部調用：**
- ('debug', 'logger')
- ('debug', 'logger')
- ('encode', 'encoder')
- ('_get_baseline_response', 'self')
- ('warning', 'logger')
- ('encode', 'encoder')
- ('request', 'client')
- ('_analyze_boolean_responses', 'self')
- ('request', 'client')
- ('error', 'logger')
- ('gather', 'asyncio')
- ('warning', 'logger')
- ('_build_detection_result', 'self')
- ('append', 'results')
- ('info', 'logger')
- ('warning', 'logger')
- ('total_seconds', 'true_time')
- ('total_seconds', 'false_time')
- ('_send_payload_request', 'self')
- ('_send_payload_request', 'self')


### `services\features\features_ready\function_sqli\engines\error_detection_engine.py`

**入口函數：**
- __init__
- detect
- _analyze_error_response
- _build_detection_result

**導出函數：**
- __init__
- detect
- _analyze_error_response
- _build_detection_result

**外部調用：**
- ('debug', 'logger')
- ('debug', 'logger')
- ('encode', 'encoder')
- ('_analyze_error_response', 'self')
- ('search', 're')
- ('request', 'client')
- ('_build_detection_result', 'self')
- ('append', 'results')
- ('info', 'logger')
- ('warning', 'logger')


### `services\features\features_ready\function_sqli\engines\hackingtool_engine.py`

**入口函數：**
- __init__
- _validate_tools_availability
- _check_tool_availability
- _is_tool_installed
- _check_tool_version
- initialize
- detect
- _run_tool_detection
- _execute_tool
- _parse_tool_output
- _create_detection_result
- _determine_severity
- _extract_payload
- _extract_db_fingerprint
- _extract_parameter
- get_tool_status
- install_missing_tools
- _convert_to_detection_result

**導出函數：**
- __init__
- _validate_tools_availability
- _check_tool_availability
- _is_tool_installed
- _check_tool_version
- initialize
- detect
- _run_tool_detection
- _execute_tool
- _parse_tool_output
- _create_detection_result
- _determine_severity
- _extract_payload
- _extract_db_fingerprint
- _extract_parameter
- get_tool_status
- install_missing_tools
- _convert_to_detection_result

**外部調用：**
- ('info', 'logger')
- ('get', 'HACKINGTOOL_SQL_CONFIGS')
- ('lower', 'tool_name')
- ('_check_tool_version', 'self')
- ('_validate_tools_availability', 'self')
- ('info', 'logger')
- ('info', 'logger')
- ('info', 'logger')
- ('get', 'HACKINGTOOL_SQL_CONFIGS')
- ('debug', 'logger')
- ('_parse_tool_output', 'self')
- ('get', 'execution_result')
- ('get', 'execution_result')
- ('get', 'severity_map')
- ('items', 'HACKINGTOOL_SQL_CONFIGS')
- ('_check_tool_availability', 'self')
- ('warning', 'logger')
- ('_is_tool_installed', 'self')
- ('warning', 'logger')
- ('debug', 'logger')
- ('run', 'subprocess')
- ('warning', 'logger')
- ('create_task', 'asyncio')
- ('append', 'detection_tasks')
- ('gather', 'asyncio')
- ('_execute_tool', 'self')
- ('get', 'execution_result')
- ('finditer', 're')
- ('_create_detection_result', 'self')
- ('_determine_severity', 'self')
- ('_extract_payload', 'self')
- ('_extract_db_fingerprint', 'self')
- ('search', 're')
- ('search', 're')
- ('append', 'available_tools')
- ('run', 'subprocess')
- ('warning', 'logger')
- ('warning', 'logger')
- ('warning', 'logger')
- ('_run_tool_detection', 'self')
- ('error', 'logger')
- ('create_subprocess_shell', 'asyncio')
- ('decode', 'stderr')
- ('decode', 'stdout')
- ('decode', 'stderr')
- ('append', 'vulnerabilities_found')
- ('append', 'results')
- ('error', 'logger')
- ('info', 'logger')
- ('error', 'logger')
- ('wait_for', 'asyncio')
- ('kill', 'process')
- ('group', 'match')
- ('groups', 'match')
- ('span', 'match')
- ('_extract_parameter', 'self')
- ('info', 'logger')
- ('error', 'logger')
- ('get', 'execution_result')
- ('cwd', 'Path')
- ('communicate', 'process')
- ('wait', 'process')
- ('group', 'match')
- ('group', 'match')
- ('append', 'results')
- ('_convert_to_detection_result', 'self')
- ('keys', 'HACKINGTOOL_SQL_CONFIGS')
- ('append', 'results')
- ('now', 'datetime')


### `services\features\features_ready\function_sqli\engines\oob_detection_engine.py`

**入口函數：**
- __init__
- detect
- _check_oob_response
- _build_detection_result

**導出函數：**
- __init__
- detect
- _check_oob_response
- _build_detection_result

**外部調用：**
- ('debug', 'logger')
- ('debug', 'logger')
- ('search', 're')
- ('search', 're')
- ('format', 'payload_template')
- ('encode', 'encoder')
- ('_check_oob_response', 'self')
- ('request', 'client')
- ('_build_detection_result', 'self')
- ('append', 'results')
- ('info', 'logger')
- ('warning', 'logger')
- ('uuid4', 'uuid')


### `services\features\features_ready\function_sqli\engines\time_detection_engine.py`

**入口函數：**
- __init__
- detect
- _measure_baseline_times
- _measure_payload_time
- _build_detection_result

**導出函數：**
- __init__
- detect
- _measure_baseline_times
- _measure_payload_time
- _build_detection_result

**外部調用：**
- ('debug', 'logger')
- ('debug', 'logger')
- ('debug', 'logger')
- ('_measure_baseline_times', 'self')
- ('warning', 'logger')
- ('warning', 'logger')
- ('time', 'time')
- ('encode', 'encoder')
- ('time', 'time')
- ('time', 'time')
- ('encode', 'encoder')
- ('time', 'time')
- ('request', 'client')
- ('warning', 'logger')
- ('request', 'client')
- ('append', 'times')
- ('sleep', 'asyncio')
- ('warning', 'logger')
- ('_measure_payload_time', 'self')
- ('_build_detection_result', 'self')
- ('append', 'results')
- ('info', 'logger')
- ('warning', 'logger')


### `services\features\features_ready\function_sqli\engines\union_detection_engine.py`

**入口函數：**
- __init__
- detect
- _get_baseline_response
- _check_union_success
- _check_column_count_error
- _check_content_change
- _get_detection_type
- _build_detection_result

**導出函數：**
- __init__
- detect
- _get_baseline_response
- _check_union_success
- _check_column_count_error
- _check_content_change
- _get_detection_type
- _build_detection_result

**外部調用：**
- ('debug', 'logger')
- ('debug', 'logger')
- ('lower', 'content')
- ('get', 'confidence_map')
- ('_get_baseline_response', 'self')
- ('warning', 'logger')
- ('encode', 'encoder')
- ('search', 're')
- ('search', 're')
- ('findall', 're')
- ('findall', 're')
- ('encode', 'encoder')
- ('_check_union_success', 'self')
- ('_check_column_count_error', 'self')
- ('_check_content_change', 'self')
- ('request', 'client')
- ('error', 'logger')
- ('request', 'client')
- ('_build_detection_result', 'self')
- ('append', 'results')
- ('info', 'logger')
- ('warning', 'logger')
- ('isdigit', 'n')
- ('_get_detection_type', 'self')


### `services\features\features_ready\function_sqli\engines\__init__.py`


### `services\features\features_ready\function_sqli\integration_tools\bounty_hunter.py`

**入口函數：**
- main
- __post_init__
- __post_init__
- __init__
- _load_bounty_payloads
- _load_fp_filters
- scan_high_value_target
- _test_payload_type
- _test_single_payload
- _analyze_bounty_response
- _is_false_positive
- _verify_vulnerability
- _get_injection_type
- _generate_poc
- _get_baseline_response
- __init__
- add_high_value_target
- hunt_vulnerabilities
- generate_bounty_report
- __init__
- run
- _show_main_menu
- _add_targets
- _start_hunting
- _show_vulnerabilities
- _generate_report
- _show_statistics
- __init__
- initialize
- execute
- cleanup

**導出函數：**
- main
- __post_init__
- __post_init__
- __init__
- _load_bounty_payloads
- _load_fp_filters
- scan_high_value_target
- _test_payload_type
- _test_single_payload
- _analyze_bounty_response
- _is_false_positive
- _verify_vulnerability
- _get_injection_type
- _generate_poc
- _get_baseline_response
- __init__
- add_high_value_target
- hunt_vulnerabilities
- generate_bounty_report
- __init__
- run
- _show_main_menu
- _add_targets
- _start_hunting
- _show_vulnerabilities
- _generate_report
- _show_statistics
- __init__
- initialize
- execute
- cleanup

**外部調用：**
- ('getLogger', 'logging')
- ('register', 'CapabilityRegistry')
- ('run', 'asyncio')
- ('_load_bounty_payloads', 'self')
- ('_load_fp_filters', 'self')
- ('print', 'console')
- ('print', 'console')
- ('_is_false_positive', 'self')
- ('lower', 'content')
- ('get', 'type_mapping')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('ask', 'Prompt')
- ('print', 'console')
- ('print', 'console')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_row', 'table')
- ('add_row', 'table')
- ('add_row', 'table')
- ('add_row', 'table')
- ('print', 'console')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('run', 'cli')
- ('ClientSession', 'aiohttp')
- ('_get_baseline_response', 'self')
- ('items', 'params')
- ('get', 'verification_payloads')
- ('time', 'time')
- ('time', 'time')
- ('add_task', 'progress')
- ('fit', 'Panel')
- ('_show_main_menu', 'self')
- ('ask', 'Prompt')
- ('ask', 'Prompt')
- ('print', 'console')
- ('print', 'console')
- ('add_row', 'table')
- ('print', 'console')
- ('write', 'f')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('extend', 'vulnerabilities')
- ('_test_single_payload', 'self')
- ('copy', 'params')
- ('time', 'time')
- ('warning', 'logger')
- ('warning', 'logger')
- ('error', 'logger')
- ('items', 'parameters')
- ('update', 'progress')
- ('advance', 'progress')
- ('ask', 'Prompt')
- ('ask', 'Confirm')
- ('now', 'datetime')
- ('error', 'logger')
- ('get', 'parameters')
- ('get', 'parameters')
- ('get', 'parameters')
- ('error', 'logger')
- ('error', 'logger')
- ('now', 'datetime')
- ('ClientTimeout', 'aiohttp')
- ('TCPConnector', 'aiohttp')
- ('_test_payload_type', 'self')
- ('print', 'console')
- ('_verify_vulnerability', 'self')
- ('append', 'vulnerabilities')
- ('print', 'console')
- ('_analyze_bounty_response', 'self')
- ('lower', 'error')
- ('lower', 'content')
- ('get', 'baseline')
- ('_get_injection_type', 'self')
- ('_generate_poc', 'self')
- ('text', 'response')
- ('time', 'time')
- ('text', 'response')
- ('time', 'time')
- ('_add_targets', 'self')
- ('text', 'response')
- ('time', 'time')
- ('lower', 'indicator')
- ('lower', 'content')
- ('get', 'baseline')
- ('now', 'datetime')
- ('_start_hunting', 'self')
- ('_show_vulnerabilities', 'self')
- ('_generate_report', 'self')
- ('_show_statistics', 'self')
- ('lower', 'content')
- ('print', 'console')
- ('print', 'console')
- ('lower', 'content')
- ('lower', 'content')


### `services\features\features_ready\function_sqli\integration_tools\sql_tools.py`

**入口函數：**
- __init__
- _find_sqlmap_path
- install_sqlmap
- scan_target
- _parse_sqlmap_output
- __init__
- _ensure_session
- close
- _load_payloads
- scan_target
- _test_injection_type
- _get_baseline_response
- _test_payload
- _analyze_response
- __init__
- _load_nosql_payloads
- scan_target
- _test_nosql_payload
- __init__
- _ensure_session
- scan_blind_injection
- _test_time_blind_injection
- _test_boolean_blind_injection
- __init__
- comprehensive_scan
- _parse_target
- _result_to_dict
- __init__
- show_main_menu
- run_interactive
- _comprehensive_scan
- _sqlmap_scan
- _custom_payload_test
- _nosql_scan
- _blind_injection_scan
- _show_scan_history
- _export_report
- _display_scan_results

**導出函數：**
- __init__
- _find_sqlmap_path
- install_sqlmap
- scan_target
- _parse_sqlmap_output
- __init__
- _ensure_session
- close
- _load_payloads
- scan_target
- _test_injection_type
- _get_baseline_response
- _test_payload
- _analyze_response
- __init__
- _load_nosql_payloads
- scan_target
- _test_nosql_payload
- __init__
- _ensure_session
- scan_blind_injection
- _test_time_blind_injection
- _test_boolean_blind_injection
- __init__
- comprehensive_scan
- _parse_target
- _result_to_dict
- __init__
- show_main_menu
- run_interactive
- _comprehensive_scan
- _sqlmap_scan
- _custom_payload_test
- _nosql_scan
- _blind_injection_scan
- _show_scan_history
- _export_report
- _display_scan_results

**外部調用：**
- ('_find_sqlmap_path', 'self')
- ('split', 'output')
- ('_load_payloads', 'self')
- ('_load_nosql_payloads', 'self')
- ('_parse_target', 'self')
- ('print', 'console')
- ('print', 'console')
- ('fit', 'Panel')
- ('print', 'console')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('print', 'console')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('print', 'console')
- ('mkdir', 'output_dir')
- ('add_column', 'summary_table')
- ('add_column', 'summary_table')
- ('add_row', 'summary_table')
- ('add_row', 'summary_table')
- ('add_row', 'summary_table')
- ('add_row', 'summary_table')
- ('add_row', 'summary_table')
- ('print', 'console')
- ('add_column', 'method_table')
- ('add_column', 'method_table')
- ('print', 'console')
- ('run', 'subprocess')
- ('print', 'console')
- ('print', 'console')
- ('extend', 'cmd')
- ('extend', 'cmd')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('mkdtemp', 'tempfile')
- ('extend', 'cmd')
- ('print', 'console')
- ('strip', 'line')
- ('items', 'patterns')
- ('ClientSession', 'aiohttp')
- ('sleep', 'asyncio')
- ('ClientSession', 'aiohttp')
- ('_get_baseline_response', 'self')
- ('time', 'time')
- ('ClientSession', 'aiohttp')
- ('ClientSession', 'aiohttp')
- ('sleep', 'asyncio')
- ('ClientSession', 'aiohttp')
- ('extend', 'results')
- ('extend', 'results')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('add_row', 'table')
- ('input', 'console')
- ('strip', 'choice')
- ('print', 'console')
- ('_display_scan_results', 'self')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('add_row', 'table')
- ('print', 'console')
- ('print', 'console')
- ('add_row', 'method_table')
- ('create_subprocess_exec', 'asyncio')
- ('communicate', 'process')
- ('print', 'console')
- ('error', 'logger')
- ('error', 'logger')
- ('extend', 'cmd')
- ('extend', 'cmd')
- ('extend', 'cmd')
- ('get', 'options')
- ('extend', 'cmd')
- ('extend', 'cmd')
- ('extend', 'cmd')
- ('create_subprocess_exec', 'asyncio')
- ('communicate', 'process')
- ('_parse_sqlmap_output', 'self')
- ('error', 'logger')
- ('error', 'logger')
- ('search', 're')
- ('append', 'results')
- ('extend', 'results')
- ('error', 'logger')
- ('time', 'time')
- ('debug', 'logger')
- ('search', 're')
- ('replace', 'test_data')
- ('debug', 'logger')
- ('_test_time_blind_injection', 'self')
- ('_test_boolean_blind_injection', 'self')
- ('time', 'time')
- ('time', 'time')
- ('_ensure_session', 'self')
- ('get', 'session')
- ('get', 'session')
- ('append', 'results')
- ('debug', 'logger')
- ('add_task', 'progress')
- ('add_task', 'progress')
- ('add_task', 'progress')
- ('add_task', 'progress')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('show_main_menu', 'self')
- ('input', 'console')
- ('print', 'console')
- ('input', 'console')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('input', 'console')
- ('input', 'console')
- ('ClientSession', 'aiohttp')
- ('time', 'time')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('input', 'console')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('input', 'console')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('add_column', 'table')
- ('print', 'console')
- ('print', 'console')
- ('print', 'console')
- ('sleep', 'asyncio')
- ('now', 'datetime')
- ('sleep', 'asyncio')
- ('dump', 'json')
- ('print', 'console')
- ('title', 'method')
- ('install_sqlmap', 'self')
- ('decode', 'stdout')
- ('ClientTimeout', 'aiohttp')
- ('TCPConnector', 'aiohttp')
- ('ClientTimeout', 'aiohttp')
- ('TCPConnector', 'aiohttp')
- ('_test_injection_type', 'self')
- ('_test_payload', 'self')
- ('append', 'results')
- ('sleep', 'asyncio')
- ('debug', 'logger')
- ('_ensure_session', 'self')
- ('ClientTimeout', 'aiohttp')
- ('text', 'response')
- ('ClientTimeout', 'aiohttp')
- ('ClientTimeout', 'aiohttp')
- ('_ensure_session', 'self')
- ('get', 'session')
- ('debug', 'logger')
- ('text', 'true_response')
- ('text', 'false_response')
- ('now', 'datetime')
- ('update', 'progress')
- ('remove_task', 'progress')
- ('update', 'progress')
- ('remove_task', 'progress')
- ('update', 'progress')
- ('remove_task', 'progress')
- ('update', 'progress')
- ('remove_task', 'progress')
- ('print', 'console')
- ('print', 'console')
- ('add_row', 'table')
- ('get', 'session')
- ('print', 'console')
- ('print', 'console')
- ('add_row', 'table')
- ('add_row', 'table')
- ('get', 'current_vuln')
- ('get', 'current_vuln')
- ('get', 'current_vuln')
- ('get', 'current_vuln')
- ('text', 'response')
- ('time', 'time')
- ('text', 'response')
- ('time', 'time')
- ('_replace', 'parsed_url')
- ('replace', 'test_data')
- ('get', 'session')
- ('_analyze_response', 'self')
- ('post', 'session')
- ('_analyze_response', 'self')
- ('_test_nosql_payload', 'self')
- ('append', 'results')
- ('sleep', 'asyncio')
- ('debug', 'logger')
- ('time', 'time')
- ('append', 'results')
- ('_result_to_dict', 'self')
- ('error', 'logger')
- ('update', 'progress')
- ('_result_to_dict', 'self')
- ('error', 'logger')
- ('update', 'progress')
- ('_result_to_dict', 'self')
- ('error', 'logger')
- ('update', 'progress')
- ('_result_to_dict', 'self')
- ('error', 'logger')
- ('update', 'progress')
- ('_comprehensive_scan', 'self')
- ('text', 'response')
- ('time', 'time')
- ('now', 'datetime')
- ('decode', 'stderr')
- ('decode', 'stderr')
- ('group', 'match')
- ('text', 'response')
- ('time', 'time')
- ('text', 'response')
- ('time', 'time')
- ('lower', 'content')
- ('get', 'options')
- ('get', 'r')
- ('get', 'r')
- ('get', 'r')
- ('get', 'r')
- ('_sqlmap_scan', 'self')
- ('lower', 'content')
- ('_custom_payload_test', 'self')
- ('_nosql_scan', 'self')
- ('_blind_injection_scan', 'self')
- ('_show_scan_history', 'self')
- ('_export_report', 'self')
- ('print', 'console')
- ('print', 'console')


### `services\features\features_ready\function_sqli\integration_tools\__init__.py`


### `services\features\features_ready\function_sqli\external_tools\NoSQLMap\exception.py`


### `services\features\features_ready\function_sqli\external_tools\NoSQLMap\nosqlmap.py`


### `services\features\features_ready\function_sqli\external_tools\NoSQLMap\nsmcouch.py`


### `services\features\features_ready\function_sqli\external_tools\NoSQLMap\nsmmongo.py`


### `services\features\features_ready\function_sqli\external_tools\NoSQLMap\nsmscan.py`


### `services\features\features_ready\function_sqli\external_tools\NoSQLMap\nsmweb.py`


### `services\features\features_ready\function_sqli\external_tools\NoSQLMap\setup.py`

**外部調用：**
- ('read', 'f')


### `services\features\features_ready\function_idor\config\idor_config.py`


### `services\features\features_ready\function_idor\detector\idor_detector.py`

**入口函數：**
- __init__
- analyze
- _perform_horizontal_tests
- _perform_vertical_tests
- _to_finding
- _determine_severity
- _create_vulnerability

**導出函數：**
- __init__
- analyze
- _perform_horizontal_tests
- _perform_vertical_tests
- _to_finding
- _determine_severity
- _create_vulnerability

**外部調用：**
- ('_determine_severity', 'self')
- ('_create_vulnerability', 'self')
- ('extract_ids_from_url', 'engine')
- ('generate_variants', 'engine')
- ('extend', 'findings')
- ('extend', 'findings')
- ('close', 'engine')
- ('replace_id_in_url', 'engine')
- ('test_vertical', 'engine')
- ('append', 'findings')
- ('_perform_horizontal_tests', 'self')
- ('_perform_vertical_tests', 'self')
- ('test_horizontal', 'engine')
- ('append', 'findings')
- ('_to_finding', 'self')
- ('_to_finding', 'self')


### `services\features\features_ready\function_idor\engine\idor_engine.py`

**入口函數：**
- __init__
- close
- extract_ids_from_url
- generate_variants
- replace_id_in_url
- test_horizontal
- test_vertical
- _is_public_resource
- _has_shared_access
- _calculate_sensitivity

**導出函數：**
- __init__
- close
- extract_ids_from_url
- generate_variants
- replace_id_in_url
- test_horizontal
- test_vertical
- _is_public_resource
- _has_shared_access
- _calculate_sensitivity

**外部調用：**
- ('AsyncClient', 'httpx')
- ('finditer', 're')
- ('replace', 'url')
- ('lower', 'response_text')
- ('lower', 'url')
- ('lower', 'user_b_text')
- ('lower', 'response_text')
- ('items', 'sensitive_fields')
- ('append', 'ids')
- ('fromkeys', 'dict')
- ('_is_public_resource', 'self')
- ('_has_shared_access', 'self')
- ('_calculate_sensitivity', 'self')
- ('_calculate_sensitivity', 'self')
- ('lower', 'indicator')
- ('loads', 'json')
- ('loads', 'json')
- ('get', 'data_a')
- ('get', 'data_a')
- ('get', 'data_a')
- ('get', 'data_b')
- ('get', 'data_b')
- ('get', 'data_b')
- ('group', 'm')
- ('span', 'm')
- ('strip', 'user_a_text')
- ('strip', 'user_b_text')


### `services\features\features_ready\function_idor\worker\idor_worker.py`

**入口函數：**
- run

**導出函數：**
- _topic
- run

**外部調用：**
- ('getenv', 'os')
- ('subscribe', 'broker')
- ('model_validate_json', 'AivaMessage')
- ('getenv', 'os')
- ('getenv', 'os')
- ('publish', 'broker')
- ('analyze', 'detector')
- ('publish', 'broker')
- ('exception', 'logger')
- ('publish', 'broker')
- ('publish', 'broker')
- ('getenv', 'os')
- ('getenv', 'os')
- ('getenv', 'os')
- ('getenv', 'os')
- ('model_dump', 'f')
- ('dumps', 'json')
- ('dumps', 'json')
- ('dumps', 'json')
- ('dumps', 'json')
- ('model_dump', 'out')


### `services\features\features_ready\function_bizlogic\integration_tools\bizlogic_tools.py`

**入口函數：**
- __post_init__
- __post_init__
- __post_init__
- __init__
- comprehensive_scan
- scan_sync
- scan
- _wrap_race_condition_test
- _wrap_price_test
- _wrap_workflow_test

**導出函數：**
- __post_init__
- __post_init__
- __post_init__
- __init__
- comprehensive_scan
- scan_sync
- scan
- _wrap_race_condition_test
- _wrap_price_test
- _wrap_workflow_test

**外部調用：**
- ('getLogger', 'logging')
- ('now', 'datetime')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('now', 'datetime')
- ('isoformat', 'end_time')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('get', 'options')
- ('append', 'tasks')
- ('run', 'asyncio')
- ('comprehensive_scan', 'self')
- ('isoformat', 'start_time')
- ('append', 'tasks')
- ('append', 'tasks')
- ('append', 'tasks')
- ('append', 'tasks')
- ('append', 'tasks')
- ('_wrap_workflow_test', 'self')
- ('gather', 'asyncio')
- ('comprehensive_scan', 'self')
- ('test_concurrent_requests', 'scanner')
- ('now', 'datetime')
- ('_wrap_race_condition_test', 'self')
- ('_wrap_price_test', 'self')
- ('_wrap_price_test', 'self')
- ('_wrap_price_test', 'self')
- ('_wrap_workflow_test', 'self')
- ('get_event_loop', 'asyncio')
- ('run_until_complete', 'loop')
- ('comprehensive_scan', 'self')
- ('get', 'finding')


### `services\features\features_ready\function_bizlogic\integration_tools\__init__.py`


## 真實腳本連接（含調用函數）

- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_bizlogic\command_handler.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_bizlogic\finding_helper.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_bizlogic\price_manipulation_scanner.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_bizlogic\race_condition_scanner.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_bizlogic\workflow_bypass_scanner.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_bizlogic\__main__.py`
- `services\features\features_ready\function_bizlogic\price_manipulation_scanner.py` → `services\features\features_ready\function_bizlogic\__main__.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_bizlogic\__main__.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_idor\command_handler.py`
- `services\features\features_ready\function_bizlogic\command_handler.py` → `services\features\features_ready\function_idor\command_handler.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_idor\command_handler.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_idor\enhanced_worker.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_idor\resource_id_extractor.py`
- `services\features\features_ready\function_sqli\smart_detection_manager.py` → `services\features\features_ready\function_idor\smart_idor_detector.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_idor\worker.py`
- `services\features\features_ready\function_idor\smart_idor_detector.py` → `services\features\features_ready\function_idor\worker.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_idor\__main__.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_sqli\command_handler.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_sqli\hackingtool_config.py`
- `services\features\features_ready\function_sqli\hackingtool_config.py` → `services\features\features_ready\function_sqli\hackingtool_manager.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_sqli\hackingtool_manager.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_sqli\hackingtool_sql_cli.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_sqli\hackingtool_sql_cli.py`
- `services\features\features_ready\function_sqli\result_binder_publisher.py` → `services\features\features_ready\function_sqli\worker.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_sqli\worker.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_sqli\worker.py`
- `services\features\features_ready\function_xss\engines\hackingtool_engine.py` → `services\features\features_ready\function_sqli\worker.py`
- `services\features\features_ready\function_idor\worker.py` → `services\features\features_ready\function_sqli\worker.py`
- `services\features\features_ready\function_sqli\telemetry.py` → `services\features\features_ready\function_sqli\worker.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_ssrf\command_handler.py`
- `services\features\features_ready\function_bizlogic\command_handler.py` → `services\features\features_ready\function_ssrf\command_handler.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_ssrf\command_handler.py`
- `services\features\features_ready\function_ssrf\param_semantics_analyzer.py` → `services\features\features_ready\function_ssrf\command_handler.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_ssrf\dns_rebinding_detector.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_ssrf\internal_address_detector.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_ssrf\oast_dispatcher.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_ssrf\param_semantics_analyzer.py`
- `services\features\features_ready\function_ssrf\dns_rebinding_detector.py` → `services\features\features_ready\function_ssrf\param_semantics_analyzer.py`
- `services\features\features_ready\function_sqli\result_binder_publisher.py` → `services\features\features_ready\function_ssrf\result_publisher.py`
- `services\features\features_ready\function_ssrf\param_semantics_analyzer.py` → `services\features\features_ready\function_ssrf\smart_ssrf_detector.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_ssrf\smart_ssrf_detector.py`
- `services\features\features_ready\function_sqli\smart_detection_manager.py` → `services\features\features_ready\function_ssrf\smart_ssrf_detector.py`
- `services\features\features_ready\function_idor\smart_idor_detector.py` → `services\features\features_ready\function_ssrf\smart_ssrf_detector.py`
- `services\features\features_ready\function_ssrf\oast_dispatcher.py` → `services\features\features_ready\function_ssrf\smart_ssrf_detector.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_ssrf\smart_ssrf_detector.py`
- `services\features\features_ready\function_ssrf\internal_address_detector.py` → `services\features\features_ready\function_ssrf\smart_ssrf_detector.py`
- `services\features\features_ready\function_ssrf\param_semantics_analyzer.py` → `services\features\features_ready\function_ssrf\worker.py`
- `services\features\features_ready\function_sqli\result_binder_publisher.py` → `services\features\features_ready\function_ssrf\worker.py`
- `services\features\features_ready\function_ssrf\internal_address_detector.py` → `services\features\features_ready\function_ssrf\worker.py`
- `services\features\features_ready\function_ssrf\oast_dispatcher.py` → `services\features\features_ready\function_ssrf\worker.py`
- `services\features\features_ready\function_sqli\telemetry.py` → `services\features\features_ready\function_ssrf\worker.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_ssrf\__main__.py`
- `services\features\features_ready\function_ssrf\oast_dispatcher.py` → `services\features\features_ready\function_xss\blind_xss_listener_validator.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\blind_xss_listener_validator.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_xss\command_handler.py`
- `services\features\features_ready\function_bizlogic\command_handler.py` → `services\features\features_ready\function_xss\command_handler.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\command_handler.py`
- `services\features\features_ready\function_sqli\result_binder_publisher.py` → `services\features\features_ready\function_xss\result_publisher.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_xss\stored_detector.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\stored_detector.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\traditional_detector.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_xss\traditional_detector.py`
- `services\features\features_ready\function_ssrf\internal_address_detector.py` → `services\features\features_ready\function_xss\worker.py`
- `services\features\features_ready\function_xss\payload_generator.py` → `services\features\features_ready\function_xss\worker.py`
- `services\features\features_ready\function_sqli\result_binder_publisher.py` → `services\features\features_ready\function_xss\worker.py`
- `services\features\features_ready\function_xss\stored_detector.py` → `services\features\features_ready\function_xss\worker.py`
- `services\features\features_ready\function_sqli\telemetry.py` → `services\features\features_ready\function_xss\worker.py`
- `services\features\features_ready\function_xss\traditional_detector.py` → `services\features\features_ready\function_xss\worker.py`
- `services\features\features_ready\function_xss\blind_xss_listener_validator.py` → `services\features\features_ready\function_xss\worker.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\worker.py`
- `services\features\features_ready\function_xss\payload_generator.py` → `services\features\features_ready\function_xss\__main__.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_xss\__main__.py`
- `services\features\features_ready\function_xss\stored_detector.py` → `services\features\features_ready\function_xss\__main__.py`
- `services\features\features_ready\function_ssrf\internal_address_detector.py` → `services\features\features_ready\function_xss\__main__.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\__main__.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\engines\hackingtool_engine.py`
- `services\features\features_ready\function_xss\external_tools\XSStrike\core\log.py` → `services\features\features_ready\function_xss\engines\hackingtool_engine.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\integration_tools\xss_tools.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_xss\integration_tools\xss_tools.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\external_tools\XSS-LOADER\dorktara.py`
- `services\features\features_ready\function_xss\external_tools\XSS-LOADER\entry.py` → `services\features\features_ready\function_xss\external_tools\XSS-LOADER\payloader.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_xss\external_tools\XSS-LOADER\payloader.py`
- `services\features\features_ready\function_xss\external_tools\XSS-LOADER\xssScan.py` → `services\features\features_ready\function_xss\external_tools\XSS-LOADER\payloader.py`
- `services\features\features_ready\function_xss\external_tools\XSS-LOADER\dorktara.py` → `services\features\features_ready\function_xss\external_tools\XSS-LOADER\payloader.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\external_tools\XSS-LOADER\promm.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\external_tools\XSS-LOADER\xssScan.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\xsstrike.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\core\encoders.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\core\photon.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\core\requester.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\core\updater.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\core\utils.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\core\utils.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\core\zetanize.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\modes\bruteforcer.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\modes\scan.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\plugins\retireJs.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_xss\external_tools\XSStrike\plugins\retireJs.py`
- `services\features\features_ready\function_ssrf\engine\ssrf_engine.py` → `services\features\features_ready\function_ssrf\detector\ssrf_detector.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_ssrf\engine\ssrf_engine.py`
- `services\features\features_ready\function_ssrf\internal_address_detector.py` → `services\features\features_ready\function_ssrf\worker\ssrf_worker.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_sqli\detector\sqli_detector.py`
- `services\features\features_ready\function_xss\engines\hackingtool_engine.py` → `services\features\features_ready\function_sqli\detector\sqli_detector.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_sqli\engines\boolean_detection_engine.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_sqli\engines\error_detection_engine.py`
- `services\features\features_ready\function_sqli\engines\boolean_detection_engine.py` → `services\features\features_ready\function_sqli\engines\error_detection_engine.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_sqli\engines\hackingtool_engine.py`
- `services\features\features_ready\function_xss\engines\hackingtool_engine.py` → `services\features\features_ready\function_sqli\engines\hackingtool_engine.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_sqli\engines\hackingtool_engine.py`
- `services\features\features_ready\function_xss\external_tools\XSStrike\core\log.py` → `services\features\features_ready\function_sqli\engines\oob_detection_engine.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_sqli\engines\oob_detection_engine.py`
- `services\features\features_ready\function_sqli\engines\boolean_detection_engine.py` → `services\features\features_ready\function_sqli\engines\oob_detection_engine.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_sqli\engines\time_detection_engine.py`
- `services\features\features_ready\function_sqli\engines\boolean_detection_engine.py` → `services\features\features_ready\function_sqli\engines\time_detection_engine.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_sqli\engines\union_detection_engine.py`
- `services\features\features_ready\function_sqli\engines\boolean_detection_engine.py` → `services\features\features_ready\function_sqli\engines\union_detection_engine.py`
- `services\features\features_ready\function_sqli\payload_wrapper_encoder.py` → `services\features\features_ready\function_sqli\engines\union_detection_engine.py`
- `services\features\features_ready\function_ssrf\oast_dispatcher.py` → `services\features\features_ready\function_sqli\integration_tools\bounty_hunter.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_sqli\integration_tools\bounty_hunter.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_sqli\integration_tools\bounty_hunter.py`
- `services\features\features_ready\function_sqli\engines\boolean_detection_engine.py` → `services\features\features_ready\function_sqli\integration_tools\bounty_hunter.py`
- `services\features\features_ready\function_xss\integration_tools\xss_tools.py` → `services\features\features_ready\function_sqli\integration_tools\sql_tools.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_sqli\integration_tools\sql_tools.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_sqli\integration_tools\sql_tools.py`
- `services\features\features_ready\function_sqli\engines\boolean_detection_engine.py` → `services\features\features_ready\function_sqli\integration_tools\sql_tools.py`
- `services\features\features_ready\function_sqli\engines\hackingtool_engine.py` → `services\features\features_ready\function_idor\detector\idor_detector.py`
- `services\features\features_ready\function_idor\engine\idor_engine.py` → `services\features\features_ready\function_idor\detector\idor_detector.py`
- `services\features\features_ready\function_ssrf\engine\ssrf_engine.py` → `services\features\features_ready\function_idor\detector\idor_detector.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_idor\engine\idor_engine.py`
- `services\features\features_ready\function_ssrf\internal_address_detector.py` → `services\features\features_ready\function_idor\worker\idor_worker.py`
- `services\features\features_ready\function_sqli\task_queue.py` → `services\features\features_ready\function_bizlogic\integration_tools\bizlogic_tools.py`
- `services\features\features_ready\function_idor\enhanced_worker.py` → `services\features\features_ready\function_bizlogic\integration_tools\bizlogic_tools.py`
- `services\features\features_ready\function_xss\integration_tools\xss_tools.py` → `services\features\features_ready\function_bizlogic\integration_tools\bizlogic_tools.py`
- `services\features\features_ready\function_bizlogic\race_condition_scanner.py` → `services\features\features_ready\function_bizlogic\integration_tools\bizlogic_tools.py`
