# AIVA Core CLI 指令參考手冊

> 生成時間: 2026-01-20 16:01:57
> 來源設定檔: classification_data.json
> 總流程數: 212

## 快速指令索引

此表格列出所有可用流程及其對應的 CLI 執行指令。AI 代理可根據需求檢索此表。

| ID | 任務路徑 (Path) | 主要模組 | CLI 指令 |
|:---:|---|---|---|
| 1 | scan_authentication -> _find_go_binary | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 1` |
| 2 | get_engine_info -> _find_go_binary | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 2` |
| 3 | scan_target -> AuthnManager | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 3` |
| 4 | _check_go_availability -> _find_go_binary | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 4` |
| 5 | BizLogicManager.comprehensive_scan -> RaceConditionScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 5` |
| 6 | BizLogicManager.comprehensive_scan -> PriceManipulationScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 6` |
| 7 | BizLogicManager.comprehensive_scan -> WorkflowBypassScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 7` |
| 8 | main -> run_price_test | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 8` |
| 9 | main -> run_race_test | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 9` |
| 10 | main -> run_workflow_test | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 10` |
| 11 | main -> mk_finding_dict -> create_bizlogic_finding | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 11` |
| 12 | IDORDetector.analyze -> IDOREngine | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 12` |
| 13 | run -> EnhancedIDORWorker | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 13` |
| 14 | run -> _topic | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 14` |
| 15 | run -> IdorConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 15` |
| 16 | run -> IDORDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 16` |
| 17 | EnhancedIDORWorker.run -> ResourceIdExtractor | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 17` |
| 18 | process_task -> ResourceIdExtractor | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 18` |
| 19 | process_task -> EnhancedIDORWorker | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 19` |
| 20 | SmartIDORDetector.detect_vulnerabilities -> IDORDetectionContext | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 20` |
| 21 | SmartIDORDetector.__init__ -> IdorConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 21` |
| 22 | EnhancedIDORWorker.process_task -> EnhancedIdorTelemetry | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 22` |
| 23 | EnhancedIDORWorker.__init__ -> IdorConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 23` |
| 24 | EnhancedIDORWorker.__init__ -> SmartIDORDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 24` |
| 25 | PostExManager.__init__ -> PostExDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 25` |
| 26 | run -> _process_task | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 26` |
| 27 | PostExDetector.scan_full -> PostExResult | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 27` |
| 28 | main -> LateralMovementTester | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 28` |
| 29 | main -> PersistenceChecker | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 29` |
| 30 | main -> PrivilegeEscalator | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 30` |
| 31 | PostExDetector.__init__ -> PrivilegeEscalationEngine | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 31` |
| 32 | PostExDetector.__init__ -> LateralMovementEngine | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 32` |
| 33 | PostExDetector.__init__ -> PersistenceEngine | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 33` |
| 34 | scan_target -> PostExManager | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 34` |
| 35 | BountyHunterCLI.__init__ -> BountyHunterManager | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 35` |
| 36 | ErrorDetectionEngine.detect -> PayloadWrapperEncoder | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 36` |
| 37 | BountyHunterManager.__init__ -> BountyHunterScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 37` |
| 38 | _consume_queue -> _execute_task | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 38` |
| 39 | HackingToolDetectionEngine._convert_to_detection_result -> DetectionResult | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 39` |
| 40 | UnionDetectionEngine.detect -> PayloadWrapperEncoder | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 40` |
| 41 | SqliWorkerService.process_task -> SqliOrchestrator | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 41` |
| 42 | SqliWorkerService.process_task -> SqliContext | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 42` |
| 43 | HackingToolDetectionEngine._create_detection_result -> DetectionResult | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 43` |
| 44 | SqliOrchestrator.__init__ -> SqliEngineConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 44` |
| 45 | SQLInjectionManager.__init__ -> SqlmapIntegration | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 45` |
| 46 | SQLInjectionManager.__init__ -> CustomSQLInjectionScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 46` |
| 47 | SQLInjectionManager.__init__ -> NoSQLInjectionScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 47` |
| 48 | SQLInjectionManager.__init__ -> BlindSQLInjectionScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 48` |
| 49 | SqliWorkerService.__init__ -> SqliOrchestrator | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 49` |
| 50 | SqliWorkerService.__init__ -> SqliEngineConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 50` |
| 51 | run -> SqliResultBinderPublisher | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 51` |
| 52 | run -> SqliTaskQueue | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 52` |
| 53 | run -> SqliWorkerService | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 53` |
| 54 | main -> HackingToolSQLCLI | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 54` |
| 55 | main -> BountyHunterCLI | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 55` |
| 56 | BountyHunterScanner._analyze_bounty_response -> BountyVulnerability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 56` |
| 57 | process_task -> SqliWorkerService | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 57` |
| 58 | BooleanDetectionEngine.detect -> PayloadWrapperEncoder | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 58` |
| 59 | SqlmapIntegration._parse_sqlmap_output -> SQLInjectionResult | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 59` |
| 60 | SqliOrchestrator._setup_default_engines -> SqliConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 60` |
| 61 | BountyHunterManager.add_high_value_target -> HighValueTarget | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 61` |
| 62 | TimeDetectionEngine.detect -> PayloadWrapperEncoder | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 62` |
| 63 | SQLInjectionBountyCapability.__init__ -> BountyHunterManager | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 63` |
| 64 | OOBDetectionEngine.detect -> PayloadWrapperEncoder | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 64` |
| 65 | _build_internal_finding -> _severity_from_summary | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 65` |
| 66 | OastDispatcher.register -> OastProbe | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 66` |
| 67 | SSRFDetector.analyze -> SSRFEngine | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 67` |
| 68 | OastDispatcher.fetch_events -> OastEvent | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 68` |
| 69 | SsrfWorkerService.__init__ -> OastDispatcher | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 69` |
| 70 | SsrfWorkerService.__init__ -> InternalAddressDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 70` |
| 71 | SsrfWorkerService.__init__ -> ParamSemanticsAnalyzer | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 71` |
| 72 | run -> SsrfResultPublisher | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 72` |
| 73 | run -> ParamSemanticsAnalyzer | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 73` |
| 74 | run -> InternalAddressDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 74` |
| 75 | run -> OastDispatcher | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 75` |
| 76 | run -> _execute_task -> process_task -> ParamSemanticsAnalyzer | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 76` |
| 77 | run -> _execute_task -> process_task -> InternalAddressDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 77` |
| 78 | run -> _execute_task -> process_task -> OastDispatcher | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 78` |
| 79 | run -> _execute_task -> process_task -> SsrfTelemetry | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 79` |
| 80 | run -> _execute_task -> process_task -> _resolve_payload | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 80` |
| 81 | run -> _execute_task -> process_task -> _issue_request | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 81` |
| 82 | run -> SsrfConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 82` |
| 83 | run -> SSRFDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 83` |
| 84 | ParamSemanticsAnalyzer._get_advanced_payloads -> DnsRebindingDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 84` |
| 85 | SsrfWorkerService.process_task -> SsrfResultPublisher | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 85` |
| 86 | SsrfWorkerService.process_task -> _execute_task -> process_task -> ParamSemanticsAnalyzer | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 86` |
| 87 | SsrfWorkerService.process_task -> _execute_task -> process_task -> InternalAddressDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 87` |
| 88 | SsrfWorkerService.process_task -> _execute_task -> process_task -> OastDispatcher | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 88` |
| 89 | SsrfWorkerService.process_task -> _execute_task -> process_task -> SsrfTelemetry | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 89` |
| 90 | SsrfWorkerService.process_task -> _execute_task -> process_task -> _resolve_payload | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 90` |
| 91 | SsrfWorkerService.process_task -> _execute_task -> process_task -> _issue_request | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 91` |
| 92 | ParamSemanticsAnalyzer.analyze -> AnalysisPlan | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 92` |
| 93 | SmartSSRFDetector.__init__ -> SsrfConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 93` |
| 94 | SmartSSRFDetector.detect_vulnerabilities -> SSRFDetectionContext | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 94` |
| 95 | WebAttackManager.comprehensive_scan -> WebTarget | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 95` |
| 96 | WebAttackManager.comprehensive_scan -> ScanResult | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 96` |
| 97 | WebAttackManager.__init__ -> SubdomainEnumerator | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 97` |
| 98 | WebAttackManager.__init__ -> DirectoryScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 98` |
| 99 | WebAttackManager.__init__ -> VulnerabilityScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 99` |
| 100 | WebAttackManager.__init__ -> TechnologyDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 100` |
| 101 | WebAttackCapability.__init__ -> WebAttackManager | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 101` |
| 102 | WebAttackCapability.__init__ -> WebAttackCLI | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 102` |
| 103 | register_capability -> WebAttackCapability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 103` |
| 104 | scan_target -> WebScannerManager | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 104` |
| 105 | bruteforcer -> getUrl | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 105` |
| 106 | bruteforcer -> getParams | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 106` |
| 107 | bruteforcer -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 107` |
| 108 | DOMXSSDetector._test_dom_payloads -> XSSVulnerability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 108` |
| 109 | CrossLanguageXSSEngine._detect_language_environments -> checker -> fillHoles | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 109` |
| 110 | CrossLanguageXSSEngine._detect_language_environments -> LanguageEnvironment | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 110` |
| 111 | run -> XssResultPublisher | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 111` |
| 112 | run -> XssTaskQueue | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 112` |
| 113 | CrossLanguageXSSEngine.__init__ -> get_xss_tools_config | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 113` |
| 114 | _consume_queue -> _execute_task -> process_task -> XssPayloadGenerator | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 114` |
| 115 | _consume_queue -> _execute_task -> process_task -> _setup_blind_xss -> BlindXssListenerValidator | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 115` |
| 116 | _consume_queue -> _execute_task -> process_task -> _build_payloads | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 116` |
| 117 | _consume_queue -> _execute_task -> process_task -> XssExecutionTelemetry | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 117` |
| 118 | _consume_queue -> _execute_task -> process_task -> TraditionalXssDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 118` |
| 119 | _consume_queue -> _execute_task -> process_task -> _execute_traditional_detection -> _handle_detection_errors | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 119` |
| 120 | _consume_queue -> _execute_task -> process_task -> _get_dom_engine | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 120` |
| 121 | _consume_queue -> _execute_task -> process_task -> _process_detections -> _analyze_detection_with_dom | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 121` |
| 122 | _consume_queue -> _execute_task -> process_task -> _execute_stored_xss -> StoredXssDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 122` |
| 123 | _consume_queue -> _execute_task -> process_task -> _collect_blind_callbacks | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 123` |
| 124 | _consume_queue -> _execute_task -> process_task -> _finalize_statistics | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 124` |
| 125 | StoredXSSDetector._check_stored_execution -> XSSVulnerability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 125` |
| 126 | scan_filename -> scan -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 126` |
| 127 | scan_filename -> scan -> dom | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 127` |
| 128 | scan_filename -> scan -> getUrl | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 128` |
| 129 | scan_filename -> scan -> getParams | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 129` |
| 130 | scan_filename -> scan -> wafDetector -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 130` |
| 131 | scan_filename -> scan -> htmlParser -> isBadContext | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 131` |
| 132 | scan_filename -> scan -> filterChecker -> checker -> fillHoles | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 132` |
| 133 | scan_filename -> scan -> generator -> extractScripts | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 133` |
| 134 | scan_filename -> scan -> generator -> genGen | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 134` |
| 135 | scan_filename -> scan -> generator -> jsContexter -> stripper | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 135` |
| 136 | scan_filename -> scan -> checker -> fillHoles | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 136` |
| 137 | detect_xss -> get_xss_engine -> CrossLanguageXSSEngine | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 137` |
| 138 | _replacement_match -> deJSON | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 138` |
| 139 | dorkFind -> get_user_agent | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 139` |
| 140 | main -> run_reflected_test -> XssPayloadGenerator | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 140` |
| 141 | main -> run_reflected_test -> TraditionalXssDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 141` |
| 142 | main -> run_dom_test -> DomXssDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 142` |
| 143 | main -> run_stored_test -> XssPayloadGenerator | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 143` |
| 144 | main -> run_stored_test -> StoredXssDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 144` |
| 145 | main -> run_xss_test | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 145` |
| 146 | _simple_match -> deJSON | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 146` |
| 147 | singleFuzz -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 147` |
| 148 | singleFuzz -> getUrl | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 148` |
| 149 | singleFuzz -> getParams | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 149` |
| 150 | singleFuzz -> wafDetector -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 150` |
| 151 | singleFuzz -> fuzzer -> counter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 151` |
| 152 | singleFuzz -> fuzzer -> replaceValue | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 152` |
| 153 | singleFuzz -> fuzzer -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 153` |
| 154 | log_red_line -> _switch_to_no_format_loggers | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 154` |
| 155 | log_red_line -> _get_level_and_log | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 155` |
| 156 | log_red_line -> _switch_to_default_loggers | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 156` |
| 157 | _is_at_or_above -> _to_comparable | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 157` |
| 158 | retireJs -> js_extractor | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 158` |
| 159 | retireJs -> updateVar | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 159` |
| 160 | retireJs -> handle_anchor | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 160` |
| 161 | retireJs -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 161` |
| 162 | retireJs -> main_scanner -> getVar | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 162` |
| 163 | retireJs -> main_scanner -> scan_uri -> scan -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 163` |
| 164 | retireJs -> main_scanner -> scan_uri -> scan -> dom | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 164` |
| 165 | retireJs -> main_scanner -> scan_uri -> scan -> getUrl | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 165` |
| 166 | retireJs -> main_scanner -> scan_uri -> scan -> getParams | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 166` |
| 167 | retireJs -> main_scanner -> scan_uri -> scan -> wafDetector -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 167` |
| 168 | retireJs -> main_scanner -> scan_uri -> scan -> htmlParser -> isBadContext | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 168` |
| 169 | retireJs -> main_scanner -> scan_uri -> scan -> filterChecker -> checker -> fillHoles | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 169` |
| 170 | retireJs -> main_scanner -> scan_uri -> scan -> generator -> extractScripts | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 170` |
| 171 | retireJs -> main_scanner -> scan_uri -> scan -> generator -> genGen | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 171` |
| 172 | retireJs -> main_scanner -> scan_uri -> scan -> generator -> jsContexter -> stripper | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 172` |
| 173 | retireJs -> main_scanner -> scan_uri -> scan -> checker -> fillHoles | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 173` |
| 174 | retireJs -> main_scanner -> scan_file_content -> scan -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 174` |
| 175 | retireJs -> main_scanner -> scan_file_content -> scan -> dom | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 175` |
| 176 | retireJs -> main_scanner -> scan_file_content -> scan -> getUrl | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 176` |
| 177 | retireJs -> main_scanner -> scan_file_content -> scan -> getParams | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 177` |
| 178 | retireJs -> main_scanner -> scan_file_content -> scan -> wafDetector -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 178` |
| 179 | retireJs -> main_scanner -> scan_file_content -> scan -> htmlParser -> isBadContext | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 179` |
| 180 | retireJs -> main_scanner -> scan_file_content -> scan -> filterChecker -> checker -> fillHoles | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 180` |
| 181 | retireJs -> main_scanner -> scan_file_content -> scan -> generator -> extractScripts | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 181` |
| 182 | retireJs -> main_scanner -> scan_file_content -> scan -> generator -> genGen | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 182` |
| 183 | retireJs -> main_scanner -> scan_file_content -> scan -> generator -> jsContexter -> stripper | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 183` |
| 184 | retireJs -> main_scanner -> scan_file_content -> scan -> checker -> fillHoles | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 184` |
| 185 | retireJs -> main_scanner -> scan_file_content -> _scanhash | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 185` |
| 186 | XssTaskQueue.put -> QueuedTask | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 186` |
| 187 | XssTaskQueue.put -> _QueueEntry | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 187` |
| 188 | log_no_format -> _switch_to_no_format_loggers | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 188` |
| 189 | log_no_format -> _get_level_and_log | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 189` |
| 190 | log_no_format -> _switch_to_default_loggers | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 190` |
| 191 | DOMXSSDetector._analyze_javascript -> XSSVulnerability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 191` |
| 192 | XSSManager._custom_xss_scan -> XSSVulnerability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 192` |
| 193 | DalfoxIntegration._parse_dalfox_output -> XSSVulnerability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 193` |
| 194 | _build_finding -> _validated_http_url | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 194` |
| 195 | BlindXssListenerValidator.__init__ -> OastHttpCallbackStore | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 195` |
| 196 | BlindXSSDetector.scan_blind_xss -> XSSVulnerability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 196` |
| 197 | TraditionalXssDetector._build_request_parts -> _inject_query | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 197` |
| 198 | TraditionalXssDetector._build_request_parts -> _inject_mapping | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 198` |
| 199 | XSSManager.__init__ -> DalfoxIntegration | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 199` |
| 200 | XSSManager.__init__ -> XSSPayloadGenerator | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 200` |
| 201 | XSSManager.__init__ -> DOMXSSDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 201` |
| 202 | XSSManager.__init__ -> StoredXSSDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 202` |
| 203 | XSSManager.__init__ -> BlindXSSDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 203` |
| 204 | crawl -> requester -> converter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 204` |
| 205 | crawl -> htmlParser -> isBadContext | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 205` |
| 206 | crawl -> filterChecker -> checker -> fillHoles | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 206` |
| 207 | crawl -> generator -> extractScripts | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 207` |
| 208 | crawl -> generator -> genGen | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 208` |
| 209 | crawl -> generator -> jsContexter -> stripper | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 209` |
| 210 | xssFind -> get_user_agent | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 210` |
| 211 | setup_logger -> CustomStreamHandler | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 211` |
| 212 | main -> PathsConfig::new | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 212` |
