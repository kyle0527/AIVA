# AIVA Core CLI 指令參考手冊

> 生成時間: 2026-02-05 06:24:09
> 來源設定檔: external_classification.json
> 總流程數: 525

## 快速指令索引

此表格列出所有可用流程及其對應的 CLI 執行指令。AI 代理可根據需求檢索此表。

| ID | 任務路徑 (Path) | 主要模組 | CLI 指令 |
|:---:|---|---|---|
| 1 | RiskFactor -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 1` |
| 2 | RiskAssessment -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 2` |
| 3 | AttackPathNode -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 3` |
| 4 | AttackPath -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 4` |
| 5 | TaskDependency -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 5` |
| 6 | TaskExecution -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 6` |
| 7 | TaskExecution.validate_task_id -> AIVAError | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 7` |
| 8 | TaskQueue -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 8` |
| 9 | GeneralTestStrategy -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 9` |
| 10 | ModuleStatus -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 10` |
| 11 | SystemOrchestration -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 11` |
| 12 | VulnerabilityCorrelation -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 12` |
| 13 | AssetAnalysis -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 13` |
| 14 | XssCandidate -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 14` |
| 15 | SqliCandidate -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 15` |
| 16 | SsrfCandidate -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 16` |
| 17 | IdorCandidate -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 17` |
| 18 | AttackSurfaceAnalysis -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 18` |
| 19 | TestTask -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 19` |
| 20 | StrategyGenerationConfig -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 20` |
| 21 | VulnerabilityTestStrategy -> Field -> Field -> Field -> Field -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 21` |
| 22 | create_bizlogic_finding -> Vulnerability -> FindingTarget -> FindingEvidence | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 22` |
| 23 | PriceManipulationScanner._verify_actual_price_change -> response_data.get -> response_data.get -> response_data.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 23` |
| 24 | PriceManipulationScanner._verify_transaction_completed -> response_data.get -> response_data.get -> response_data.get -> response_data.get -> response_data.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 24` |
| 25 | PriceManipulationScanner._verify_user_privilege -> permission_matrix.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 25` |
| 26 | PriceManipulationScanner._detect_business_limits -> response_data.get -> response_data.get -> response_data.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 26` |
| 27 | PriceManipulationScanner.test_negative_price -> client.post -> response.json -> self._verify_actual_price_change -> response_data.get -> response_data.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 27` |
| 28 | PriceManipulationScanner.test_zero_price -> client.post -> response.json -> self._verify_actual_price_change -> response_data.get -> response_data.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 28` |
| 29 | PriceManipulationScanner.test_price_tampering -> client.post -> response.json -> self._verify_actual_price_change -> self._verify_user_privilege -> self._verify_transaction_completed | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 29` |
| 30 | PriceManipulationScanner.test_overflow_price -> float -> client.post -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 30` |
| 31 | PriceManipulationScanner.run_all_tests -> all_findings.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 31` |
| 32 | RaceConditionScanner.test_concurrent_requests -> datetime.now -> client.post -> client.get -> client.request -> tasks.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 32` |
| 33 | RaceConditionScanner.test_balance_manipulation -> client.get -> client.post -> range -> client.get -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 33` |
| 34 | RaceConditionScanner.test_coupon_reuse -> client.post -> range -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 34` |
| 35 | RaceConditionScanner.test_inventory_depletion -> client.post -> range -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 35` |
| 36 | RaceConditionScanner.run_all_tests -> all_findings.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 36` |
| 37 | WorkflowBypassScanner.test_step_skipping -> client.get -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 37` |
| 38 | WorkflowBypassScanner.test_direct_checkout -> client.get -> findings.append -> client.post -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 38` |
| 39 | WorkflowBypassScanner.test_payment_bypass -> client.post -> response.json -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 39` |
| 40 | WorkflowBypassScanner.test_verification_bypass -> client.post -> register_response.json -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 40` |
| 41 | WorkflowBypassScanner.test_admin_access_bypass -> client.get -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 41` |
| 42 | WorkflowBypassScanner.run_all_tests -> all_findings.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 42` |
| 43 | mk_finding_dict -> create_bizlogic_finding | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 43` |
| 44 | run_price_test -> PriceManipulationTester -> tester.run_all_tests | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 44` |
| 45 | run_race_test -> RaceConditionTester -> tester.run_all_tests | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 45` |
| 46 | run_workflow_test -> WorkflowBypassTester -> args.admin_paths.split -> tester.run_all_tests | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 46` |
| 47 | main -> argparse.ArgumentParser -> parser.add_argument -> parser.add_argument -> parser.add_argument -> parser.add_argument | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 47` |
| 48 | BizLogicManager.__init__ -> self.logger.info | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 48` |
| 49 | BizLogicManager.comprehensive_scan -> datetime.now -> options.get -> options.get -> start_time.isoformat -> RaceConditionScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 49` |
| 50 | BizLogicManager.scan -> task.metadata.get -> task.metadata.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 50` |
| 51 | BizLogicManager._wrap_race_condition_test -> self.logger.error | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 51` |
| 52 | BizLogicManager._wrap_price_test -> self.logger.error | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 52` |
| 53 | BizLogicManager._wrap_workflow_test -> self.logger.error | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 53` |
| 54 | ResourceIdExtractor.extract_from_url -> url.split -> path.split -> ids.append -> url.split -> param.split | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 54` |
| 55 | ResourceIdExtractor.generate_test_ids -> test_ids.append -> original_id.value.split -> parts.copy -> random.randint -> test_ids.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 55` |
| 56 | ResourceIdExtractor.replace_id_in_url -> parsed.path.split -> parse_qsl -> urlencode -> urlunparse | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 56` |
| 57 | IDORDetectionContext -> field -> field -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 57` |
| 58 | IDORDetectionContext.add_finding -> self.findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 58` |
| 59 | IDORDetectionContext.add_error -> self.errors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 59` |
| 60 | SmartIDORDetector.__init__ -> IdorConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 60` |
| 61 | SmartIDORDetector.detect_vulnerabilities -> UnifiedSmartDetectionManager -> IDORDetectionContext -> smart_manager.start_detection -> smart_manager.metrics.start_phase -> self._extract_resource_ids | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 61` |
| 62 | SmartIDORDetector._extract_resource_ids -> context.id_extractor.extract_from_url -> context.id_extractor.extract_from_url -> resource_ids.extend -> context.id_extractor.extract_from_url -> resource_ids.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 62` |
| 63 | SmartIDORDetector._execute_horizontal_testing -> self._test_horizontal_access -> smart_manager.update_progress | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 63` |
| 64 | SmartIDORDetector._execute_vertical_testing -> self._test_vertical_access -> smart_manager.update_progress | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 64` |
| 65 | SmartIDORDetector._test_horizontal_access -> context.cross_user_tester.test_horizontal_idor -> self._build_horizontal_finding -> context.add_finding -> smart_manager.report_vulnerability_found -> context.increment_attempts | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 65` |
| 66 | SmartIDORDetector._test_vertical_access -> context.vertical_tester.test_vertical_escalation -> self._build_vertical_finding -> context.add_finding -> smart_manager.report_vulnerability_found -> context.increment_attempts | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 66` |
| 67 | IdorConfig -> Field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 67` |
| 68 | IDORDetector.analyze -> IDOREngine -> engine.extract_ids_from_url -> self._perform_horizontal_tests -> findings.extend -> self._perform_vertical_tests | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 68` |
| 69 | IDORDetector._perform_horizontal_tests -> engine.replace_id_in_url -> engine.test_horizontal -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 69` |
| 70 | IDORDetector._perform_vertical_tests -> engine.test_vertical -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 70` |
| 71 | IDORDetector._to_finding -> self._determine_severity -> self._create_vulnerability -> FindingEvidence -> FindingImpact -> FindingRecommendation | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 71` |
| 72 | IDOREngine.close -> self.client.aclose | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 72` |
| 73 | IDOREngine.extract_ids_from_url -> ids.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 73` |
| 74 | IDOREngine.test_horizontal -> self.client.get -> self.client.get -> self._calculate_sensitivity | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 74` |
| 75 | IDOREngine.test_vertical -> self.client.get -> self._calculate_sensitivity | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 75` |
| 76 | IDOREngine._is_public_resource -> response_text.lower -> url.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 76` |
| 77 | IDOREngine._has_shared_access -> data_a.get -> data_a.get -> data_a.get -> data_b.get -> data_b.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 77` |
| 78 | IDOREngine._calculate_sensitivity -> response_text.lower -> max | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 78` |
| 79 | RiskScore.calculate -> min | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 79` |
| 80 | DetectionResult -> field -> field -> field -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 80` |
| 81 | DetectionResult.__post_init__ -> self.risk_score.calculate | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 81` |
| 82 | DetectionResult.to_sarif -> self._sarif_level -> self._sarif_level -> results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 82` |
| 83 | SensitiveInfoDetector.__init__ -> self._build_patterns -> self._patterns.update -> set | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 83` |
| 84 | SensitiveInfoDetector.calculate_entropy -> Counter -> math.log2 | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 84` |
| 85 | SensitiveInfoDetector._is_false_positive -> value.lower -> context.lower -> self.calculate_entropy | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 85` |
| 86 | SensitiveInfoDetector.detect_in_html -> DetectionResult -> result.matches.extend -> result.matches.extend -> result.matches.extend -> result.matches.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 86` |
| 87 | SensitiveInfoDetector.detect_in_headers -> DetectionResult -> result.matches.append -> self._detect_in_text -> result.matches.extend -> self._filter_by_severity | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 87` |
| 88 | SensitiveInfoDetector.detect_in_response -> DetectionResult -> self.detect_in_html -> result.matches.extend -> self._detect_in_text -> result.matches.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 88` |
| 89 | SensitiveInfoDetector._detect_html_comments -> comment_match.group -> matches.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 89` |
| 90 | SensitiveInfoDetector._detect_script_blocks -> script_match.group -> matches.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 90` |
| 91 | SensitiveInfoDetector._detect_meta_tags -> meta_match.group -> matches.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 91` |
| 92 | SensitiveInfoDetector._detect_in_text -> text.split -> regex_match.group -> max -> min -> self._is_false_positive | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 92` |
| 93 | SensitiveInfoDetector._detect_high_entropy_strings -> self.calculate_entropy -> self._is_false_positive -> SensitiveMatch -> self._seen_hashes.add -> matches.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 93` |
| 94 | SensitiveInfoDetector.format_report -> sorted -> severity_icons.get -> lines.append -> lines.append -> lines.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 94` |
| 95 | SensitiveInfoDetector.export_report -> self.format_report -> open -> f.write | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 95` |
| 96 | SensitiveInfoDetector.get_statistics -> Counter | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 96` |
| 97 | quick_scan -> SensitiveInfoDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 97` |
| 98 | batch_scan -> SensitiveInfoDetector -> detector.detect_in_response -> results.append -> detector.export_report -> detector.get_statistics | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 98` |
| 99 | PostExManager.__init__ -> PostExDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 99` |
| 100 | PostExManager.scan -> options.get -> options.get -> options.get -> options.get -> self._detector.analyze | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 100` |
| 101 | PostExManager._generate_summary -> severity_counts.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 101` |
| 102 | scan_target -> PostExManager | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 102` |
| 103 | PostExDetector.analyze -> findings.append -> findings.append -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 103` |
| 104 | PostExDetector._mk_finding -> Vulnerability -> FindingEvidence -> FindingImpact -> FindingRecommendation -> FindingTarget | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 104` |
| 105 | PostExDetector.__init__ -> PrivilegeEscalationEngine -> LateralMovementEngine -> PersistenceEngine | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 105` |
| 106 | PostExDetector.scan_full -> self.privesc_engine.scan -> self.lateral_engine.scan_network -> self.persistence_engine.scan -> self._generate_summary -> PostExResult | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 106` |
| 107 | LateralMovementTester.scan_network -> self._log_action -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 107` |
| 108 | LateralMovementTester.enumerate_services -> self._log_action -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 108` |
| 109 | LateralMovementTester.test_credential_reuse -> self._log_action -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 109` |
| 110 | LateralMovementTester.simulate_pass_the_hash -> self._log_action -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 110` |
| 111 | LateralMovementTester.test_remote_access -> self._log_action -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 111` |
| 112 | main -> print -> print -> PrivilegeEscalator -> escalator.run_full_assessment -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 112` |
| 113 | LateralMovementEngine.scan_network -> IPv4Network -> self._discover_hosts -> vectors.extend -> vectors.extend -> vectors.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 113` |
| 114 | LateralMovementEngine._discover_hosts -> alive_hosts.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 114` |
| 115 | LateralMovementEngine._is_host_alive -> socket.socket -> sock.settimeout -> sock.connect_ex -> sock.close | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 115` |
| 116 | LateralMovementEngine._check_smb_access -> socket.socket -> sock.settimeout -> sock.connect_ex -> sock.close -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 116` |
| 117 | LateralMovementEngine._check_ssh_access -> socket.socket -> sock.settimeout -> sock.connect_ex -> sock.close -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 117` |
| 118 | LateralMovementEngine._check_rdp_access -> socket.socket -> sock.settimeout -> sock.connect_ex -> sock.close -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 118` |
| 119 | LateralMovementEngine._check_winrm_access -> socket.socket -> sock.settimeout -> sock.connect_ex -> sock.close -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 119` |
| 120 | PersistenceEngine.scan -> vectors.extend -> vectors.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 120` |
| 121 | PersistenceEngine._check_linux_persistence -> vectors.extend -> vectors.extend -> vectors.extend -> vectors.extend -> vectors.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 121` |
| 122 | PersistenceEngine._check_cron_persistence -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 122` |
| 123 | PersistenceEngine._check_systemd_persistence -> Path.home -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 123` |
| 124 | PersistenceEngine._check_shell_rc_persistence -> Path.home -> Path.home -> Path.home -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 124` |
| 125 | PersistenceEngine._check_ssh_persistence -> Path.home -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 125` |
| 126 | PersistenceEngine._check_ld_preload_persistence -> Path.home -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 126` |
| 127 | PersistenceChecker.__init__ -> platform.system | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 127` |
| 128 | PersistenceChecker.check_startup_items -> self._log_action -> winreg.OpenKey -> winreg.EnumValue -> winreg.CloseKey -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 128` |
| 129 | PersistenceChecker.check_scheduled_tasks -> self._log_action -> task_result.stdout.split -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 129` |
| 130 | PersistenceChecker.check_services -> self._log_action -> service_result.stdout.split -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 130` |
| 131 | PersistenceChecker.check_registry_persistence -> self._log_action -> self.test_results.append -> winreg.OpenKey -> winreg.EnumValue -> winreg.CloseKey | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 131` |
| 132 | PersistenceChecker.check_cron_jobs -> self._log_action -> self.test_results.append -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 132` |
| 133 | EnhancedPrivilegeAnalyzer.analyze_system_permissions -> platform.system -> platform.version -> platform.architecture -> hasattr -> hasattr | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 133` |
| 134 | PrivilegeEscalator.check_suid_binaries -> self._log_action -> platform.system -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 134` |
| 135 | PrivilegeEscalator.check_sudo_misconfiguration -> self._log_action -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 135` |
| 136 | PrivilegeEscalator.check_kernel_exploits -> self._log_action -> platform.release -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 136` |
| 137 | PrivilegeEscalator.check_writable_services -> self._log_action -> self.test_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 137` |
| 138 | PrivilegeEscalator.run_full_assessment -> platform.system | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 138` |
| 139 | PrivilegeEscalator.clear_results -> self.test_results.clear | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 139` |
| 140 | PrivilegeEscalationEngine.scan -> vectors.extend -> vectors.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 140` |
| 141 | PrivilegeEscalationEngine._check_linux_privesc -> vectors.extend -> vectors.extend -> vectors.extend -> vectors.extend -> vectors.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 141` |
| 142 | PrivilegeEscalationEngine._check_suid_binaries -> Path -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 142` |
| 143 | PrivilegeEscalationEngine._check_sudo_config -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 143` |
| 144 | PrivilegeEscalationEngine._check_writable_paths -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 144` |
| 145 | PrivilegeEscalationEngine._check_cron_jobs -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 145` |
| 146 | PrivilegeEscalationEngine._check_docker_socket -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 146` |
| 147 | PrivilegeEscalationEngine._check_kernel_version -> platform.release -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 147` |
| 148 | BackendDbFingerprinter.fingerprint -> self._extract_version | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 148` |
| 149 | BackendDbFingerprinter._extract_version -> self._version_patterns.get -> match.group | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 149` |
| 150 | BackendDbFingerprinter.analyze_response_characteristics -> response.headers.get -> response.headers.get -> self._contains_sql_keywords -> self._extract_error_signatures | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 150` |
| 151 | BackendDbFingerprinter._contains_sql_keywords -> text.upper -> found_keywords.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 151` |
| 152 | BackendDbFingerprinter._extract_error_signatures -> error_signatures.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 152` |
| 153 | SqliConfig.validate -> ValueError -> ValueError -> ValueError -> ValueError -> ValueError | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 153` |
| 154 | SqliError.__str__ -> parts.append -> parts.append -> parts.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 154` |
| 155 | NetworkError.__str__ -> parts.append -> parts.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 155` |
| 156 | HackingToolSQLConfig -> field -> field -> field -> field -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 156` |
| 157 | HackingToolSQLIntegrator.check_tool_availability -> Path | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 157` |
| 158 | HackingToolSQLIntegrator.get_available_tools -> available.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 158` |
| 159 | HackingToolSQLIntegrator.get_enabled_tools -> enabled.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 159` |
| 160 | HackingToolSQLIntegrator.generate_capability_records -> CapabilityRecord -> records.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 160` |
| 161 | HackingToolSQLIntegrator.install_tool -> print -> print -> print -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 161` |
| 162 | HackingToolSQLIntegrator.run_tool -> APIResponse -> APIResponse -> APIResponse | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 162` |
| 163 | HackingToolSQLManager.__init__ -> Path.cwd -> new_id -> self.tools_dir.mkdir | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 163` |
| 164 | HackingToolSQLManager.check_all_tools_status -> self._check_tool_status | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 164` |
| 165 | HackingToolSQLManager._check_tool_status -> missing_deps.append -> tool_path.exists -> self._test_tool_executable | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 165` |
| 166 | HackingToolSQLManager.install_tool -> datetime.now -> shutil.rmtree -> install_path.mkdir -> stderr.decode -> self._check_tool_status | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 166` |
| 167 | HackingToolSQLManager.install_all_tools -> self.install_tool | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 167` |
| 168 | HackingToolSQLManager.uninstall_tool -> shutil.rmtree | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 168` |
| 169 | HackingToolSQLManager.get_tool_recommendations -> self._check_tool_status -> recommendations.append -> recommendations.sort | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 169` |
| 170 | HackingToolSQLManager.get_installation_script -> script_lines.extend -> script_lines.append -> script_lines.extend -> script_lines.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 170` |
| 171 | HackingToolSQLManager.generate_status_report -> self.check_all_tools_status | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 171` |
| 172 | HackingToolSQLCLI.show_status -> print -> self.manager.check_all_tools_status -> print -> print -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 172` |
| 173 | HackingToolSQLCLI.install_tool -> print -> print -> print -> self.manager.install_tool -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 173` |
| 174 | HackingToolSQLCLI.install_all_tools -> print -> self.manager.install_all_tools -> successful.append -> print -> failed.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 174` |
| 175 | HackingToolSQLCLI.test_tool -> print -> print -> self.manager._check_tool_status -> print -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 175` |
| 176 | HackingToolSQLCLI.generate_report -> print -> self.manager.generate_status_report -> print -> print -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 176` |
| 177 | HackingToolSQLCLI.list_tools -> print -> print -> print -> print -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 177` |
| 178 | HackingToolSQLCLI.get_recommendations -> print -> self.manager.get_tool_recommendations -> print -> print -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 178` |
| 179 | main -> argparse.ArgumentParser -> parser.add_subparsers -> subparsers.add_parser -> subparsers.add_parser -> install_parser.add_argument | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 179` |
| 180 | EncodedPayload.build_request_dump -> lines.append -> body_parts.append -> body_parts.append -> body_parts.append -> body_parts.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 180` |
| 181 | PayloadWrapperEncoder.encode -> target.method.upper -> request_kwargs.setdefault -> body.replace -> ValueError -> request_kwargs.items | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 181` |
| 182 | PayloadWrapperEncoder._inject_query -> urlencode | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 182` |
| 183 | SqliResultBinderPublisher.__init__ -> uuid.uuid4 | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 183` |
| 184 | SqliResultBinderPublisher.publish_status -> TaskUpdatePayload -> self._publish | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 184` |
| 185 | SqliResultBinderPublisher.publish_error -> type -> self.publish_status | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 185` |
| 186 | SqliResultBinderPublisher.publish_finding -> self._publish | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 186` |
| 187 | SqliResultBinderPublisher._publish -> AivaMessage -> self._broker.publish | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 187` |
| 188 | SqliTaskQueue.put -> RuntimeError -> self._queue.put | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 188` |
| 189 | SqliTaskQueue.get -> self._queue.get -> self._queue.task_done | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 189` |
| 190 | SqliTaskQueue.close -> self._queue.put | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 190` |
| 191 | SqliExecutionTelemetry -> field -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 191` |
| 192 | SqliExecutionTelemetry.record_engine_execution -> self.engines_run.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 192` |
| 193 | SqliExecutionTelemetry.record_error -> self.errors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 193` |
| 194 | SqliExecutionTelemetry.add_engine -> self.record_engine_execution | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 194` |
| 195 | SqliExecutionTelemetry.add_error -> self.record_error | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 195` |
| 196 | SqliDetector.__init__ -> self._try_import_engine -> self.engines.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 196` |
| 197 | SqliDetector._try_import_engine -> importlib.import_module -> getattr -> cls | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 197` |
| 198 | SqliDetector.detect_sqli -> params.get -> params.get -> self._order_engines -> self._execute_parallel_detection | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 198` |
| 199 | SqliDetector._execute_parallel_detection -> engine.detect | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 199` |
| 200 | SqliDetector._process_and_merge_results -> flat_results.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 200` |
| 201 | SqliDetector._deduplicate_and_normalize -> set -> seen.add -> merged.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 201` |
| 202 | BooleanDetectionEngine.detect -> PayloadWrapperEncoder -> self._get_baseline_response -> isinstance -> isinstance -> cast | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 202` |
| 203 | BooleanDetectionEngine._get_baseline_response -> encoder.encode | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 203` |
| 204 | BooleanDetectionEngine._send_payload_request -> encoder.encode | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 204` |
| 205 | BooleanDetectionEngine._analyze_boolean_responses -> abs -> abs -> getattr -> getattr -> abs | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 205` |
| 206 | BooleanDetectionEngine._build_detection_result -> Vulnerability -> FindingEvidence -> FindingImpact -> FindingRecommendation -> FindingTarget | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 206` |
| 207 | ErrorDetectionEngine.detect -> PayloadWrapperEncoder -> encoder.encode -> client.request -> self._analyze_error_response -> self._build_detection_result | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 207` |
| 208 | ErrorDetectionEngine._build_detection_result -> Vulnerability -> FindingEvidence -> FindingImpact -> FindingRecommendation -> FindingTarget | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 208` |
| 209 | HackingToolDetectionEngine.__init__ -> new_id | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 209` |
| 210 | HackingToolDetectionEngine._validate_tools_availability -> available_tools.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 210` |
| 211 | HackingToolDetectionEngine._check_tool_availability -> HACKINGTOOL_SQL_CONFIGS.get -> tool_name.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 211` |
| 212 | HackingToolDetectionEngine.initialize -> self.integrator.get_enabled_tools -> RuntimeError -> self._validate_tools_availability -> RuntimeError | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 212` |
| 213 | HackingToolDetectionEngine.detect -> self.integrator.get_enabled_tools -> detection_tasks.append -> results.append -> self._convert_to_detection_result -> results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 213` |
| 214 | HackingToolDetectionEngine._run_tool_detection -> HACKINGTOOL_SQL_CONFIGS.get -> RuntimeError -> self._execute_tool -> RuntimeError -> self._parse_tool_output | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 214` |
| 215 | HackingToolDetectionEngine._execute_tool -> RuntimeError -> process.kill -> process.wait -> RuntimeError -> stderr.decode | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 215` |
| 216 | HackingToolDetectionEngine._parse_tool_output -> execution_result.get -> execution_result.get -> match.group -> match.groups -> match.span | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 216` |
| 217 | HackingToolDetectionEngine._create_detection_result -> self._determine_severity -> config.confidence_mapping.get -> self._extract_payload -> self._extract_db_fingerprint -> Vulnerability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 217` |
| 218 | HackingToolDetectionEngine.get_tool_status -> self.integrator.check_tool_availability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 218` |
| 219 | HackingToolDetectionEngine.install_missing_tools -> self.integrator.install_tool | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 219` |
| 220 | HackingToolDetectionEngine._convert_to_detection_result -> DetectionResult | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 220` |
| 221 | OOBDetectionEngine.detect -> PayloadWrapperEncoder -> uuid.uuid4 -> payload_template.format -> encoder.encode -> client.request | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 221` |
| 222 | OOBDetectionEngine._build_detection_result -> Vulnerability -> FindingEvidence -> FindingImpact -> FindingRecommendation -> FindingTarget | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 222` |
| 223 | TimeDetectionEngine.detect -> PayloadWrapperEncoder -> self._measure_baseline_times -> self._measure_payload_time -> self._build_detection_result -> results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 223` |
| 224 | TimeDetectionEngine._measure_baseline_times -> encoder.encode -> client.request -> times.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 224` |
| 225 | TimeDetectionEngine._measure_payload_time -> encoder.encode -> client.request | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 225` |
| 226 | TimeDetectionEngine._build_detection_result -> min -> Vulnerability -> FindingEvidence -> FindingImpact -> FindingRecommendation | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 226` |
| 227 | UnionDetectionEngine.detect -> PayloadWrapperEncoder -> self._get_baseline_response -> encoder.encode -> client.request -> self._check_union_success | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 227` |
| 228 | UnionDetectionEngine._get_baseline_response -> encoder.encode | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 228` |
| 229 | UnionDetectionEngine._check_union_success -> content.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 229` |
| 230 | UnionDetectionEngine._check_content_change -> abs -> set -> set -> sorted | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 230` |
| 231 | UnionDetectionEngine._build_detection_result -> confidence_map.get -> Vulnerability -> FindingEvidence -> FindingImpact -> FindingRecommendation | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 231` |
| 232 | SQLTarget -> field -> field -> field -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 232` |
| 233 | SqlmapIntegration.__init__ -> self._find_sqlmap_path | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 233` |
| 234 | SqlmapIntegration.install_sqlmap -> process.communicate | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 234` |
| 235 | SqlmapIntegration.scan_target -> cmd.extend -> cmd.extend -> cmd.extend -> cmd.extend -> cmd.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 235` |
| 236 | SqlmapIntegration._parse_sqlmap_output -> output.split -> line.strip -> SQLInjectionResult -> results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 236` |
| 237 | CustomSQLInjectionScanner.__init__ -> self._load_payloads | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 237` |
| 238 | CustomSQLInjectionScanner.close -> self.session.close | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 238` |
| 239 | CustomSQLInjectionScanner.scan_target -> self._test_injection_type -> results.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 239` |
| 240 | CustomSQLInjectionScanner._test_injection_type -> self._get_baseline_response -> self._test_payload -> results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 240` |
| 241 | CustomSQLInjectionScanner._get_baseline_response -> self.session.get -> response.text -> self.session.post -> response.text | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 241` |
| 242 | CustomSQLInjectionScanner._test_payload -> urllib.parse.urlparse -> urllib.parse.parse_qs -> urllib.parse.urlencode -> urllib.parse.urlunparse -> test_data.replace | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 242` |
| 243 | CustomSQLInjectionScanner._analyze_response -> abs | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 243` |
| 244 | NoSQLInjectionScanner.__init__ -> self._load_nosql_payloads | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 244` |
| 245 | NoSQLInjectionScanner.scan_target -> self._test_nosql_payload -> results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 245` |
| 246 | NoSQLInjectionScanner._test_nosql_payload -> test_data.replace -> self.session.post -> response.text | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 246` |
| 247 | BlindSQLInjectionScanner.scan_blind_injection -> self._test_time_blind_injection -> results.extend -> self._test_boolean_blind_injection -> results.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 247` |
| 248 | BlindSQLInjectionScanner._test_time_blind_injection -> urllib.parse.quote -> self._ensure_session -> session.get -> results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 248` |
| 249 | BlindSQLInjectionScanner._test_boolean_blind_injection -> urllib.parse.quote -> self._ensure_session -> session.get -> true_response.text -> urllib.parse.quote | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 249` |
| 250 | SQLInjectionManager.__init__ -> SqlmapIntegration -> CustomSQLInjectionScanner -> NoSQLInjectionScanner -> BlindSQLInjectionScanner | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 250` |
| 251 | SQLInjectionManager.comprehensive_scan -> self._parse_target -> Progress -> progress.add_task -> self.sqlmap.scan_target -> self._result_to_dict | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 251` |
| 252 | SQLInjectionManager._parse_target -> urllib.parse.urlparse | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 252` |
| 253 | SQLInjectionCLI.show_main_menu -> Panel.fit -> Table -> table.add_column -> table.add_column -> table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 253` |
| 254 | SQLInjectionCLI.run_interactive -> self.show_main_menu -> self._comprehensive_scan -> self._sqlmap_scan -> self._custom_payload_test -> self._nosql_scan | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 254` |
| 255 | SQLInjectionCLI._comprehensive_scan -> self.manager.comprehensive_scan -> self._display_scan_results | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 255` |
| 256 | SQLInjectionCLI._sqlmap_scan -> self.manager._parse_target -> self.manager.sqlmap.scan_target -> Table -> table.add_column -> table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 256` |
| 257 | SQLInjectionCLI._custom_payload_test -> urllib.parse.quote -> session.get -> response.text -> content.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 257` |
| 258 | SQLInjectionCLI._nosql_scan -> self.manager._parse_target -> self.manager.nosql_scanner.scan_target -> Table -> table.add_column -> table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 258` |
| 259 | SQLInjectionCLI._blind_injection_scan -> self.manager._parse_target -> self.manager.blind_scanner.scan_blind_injection -> Table -> table.add_column -> table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 259` |
| 260 | SQLInjectionCLI._show_scan_history -> Table -> table.add_column -> table.add_column -> table.add_column -> table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 260` |
| 261 | SQLInjectionCLI._export_report -> Path -> output_dir.mkdir -> self.manager._result_to_dict -> open | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 261` |
| 262 | SQLInjectionCLI._display_scan_results -> Table -> Table -> method_table.add_column -> method_table.add_column -> method_table.add_row | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 262` |
| 263 | DnsRebindingDetector.generate_vectors -> self._generate_rebind_it_domain -> vectors.append -> self._generate_rbndr_domain -> vectors.append -> vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 263` |
| 264 | DnsRebindingDetector._generate_rebind_it_domain -> ip_to_hex -> ip_to_hex | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 264` |
| 265 | DnsRebindingDetector._generate_rbndr_domain -> first_ip.split -> second_ip.split | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 265` |
| 266 | DnsRebindingDetector.test_rebinding -> self._resolve_domain -> self._resolve_domain -> set -> set -> client.aclose | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 266` |
| 267 | DnsRebindingDetector._resolve_domain -> domain.split -> domain.split -> socket.getaddrinfo | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 267` |
| 268 | DnsRebindingDetector.verify_internal_access -> rebinding_url.rstrip -> client.get -> client.aclose | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 268` |
| 269 | DnsRebindingDetector.generate_payloads -> self.generate_vectors -> payloads.append -> payloads.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 269` |
| 270 | InternalAddressDetection -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 270` |
| 271 | InternalAddressDetector -> field -> field -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 271` |
| 272 | InternalAddressDetector.analyze -> isinstance -> set -> indicators.append -> indicators.append -> indicators.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 272` |
| 273 | InternalAddressDetector._test_internal_services -> test_function -> self._identify_service_type -> detected_services.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 273` |
| 274 | InternalAddressDetector._test_protocol_support -> test_function -> supported_protocols.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 274` |
| 275 | InternalAddressDetector._is_successful_response -> response.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 275` |
| 276 | InternalAddressDetector._is_metadata_response -> metadata_indicators.get -> response.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 276` |
| 277 | InternalAddressDetector._is_service_response -> service_indicators.get -> response.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 277` |
| 278 | InternalAddressDetector._is_protocol_supported -> protocol_indicators.get -> response.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 278` |
| 279 | InternalAddressDetector._generate_evidence -> evidence.append -> evidence.append -> evidence.append -> evidence.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 279` |
| 280 | InternalAddressDetector.is_internal_address -> ipaddress.ip_address | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 280` |
| 281 | OastDispatcher.register -> client.post -> response.raise_for_status -> response.json -> RuntimeError -> client.aclose | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 281` |
| 282 | OastDispatcher.fetch_events -> self._resolve_token -> client.get -> response.raise_for_status -> response.json -> RuntimeError | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 282` |
| 283 | OastDispatcher.close -> self._client.aclose | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 283` |
| 284 | OastDispatcher._resolve_token -> token.rstrip -> normalized.split | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 284` |
| 285 | AnalysisPlan -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 285` |
| 286 | ParamSemanticsAnalyzer.analyze -> self._tokenize -> AnalysisPlan -> self._get_base_payloads -> self._add_standard_vectors -> self._add_semantic_vectors | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 286` |
| 287 | ParamSemanticsAnalyzer._get_base_payloads -> payloads.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 287` |
| 288 | ParamSemanticsAnalyzer._get_advanced_payloads -> advanced.extend -> DnsRebindingDetector -> dns_detector.generate_payloads -> advanced.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 288` |
| 289 | ParamSemanticsAnalyzer._add_standard_vectors -> plan.vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 289` |
| 290 | ParamSemanticsAnalyzer._add_semantic_vectors -> plan.vectors.append -> self._add_file_vectors -> self._add_protocol_vectors | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 290` |
| 291 | ParamSemanticsAnalyzer._add_file_vectors -> plan.vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 291` |
| 292 | ParamSemanticsAnalyzer._add_protocol_vectors -> plan.vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 292` |
| 293 | ParamSemanticsAnalyzer._add_cross_protocol_vectors -> self._get_selected_protocols -> plan.vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 293` |
| 294 | ParamSemanticsAnalyzer._get_selected_protocols -> headers.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 294` |
| 295 | ParamSemanticsAnalyzer._add_oast_vector -> plan.vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 295` |
| 296 | ParamSemanticsAnalyzer._build_payloads -> set -> payload.strip -> seen.add -> payload.strip -> seen.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 296` |
| 297 | ParamSemanticsAnalyzer._should_enable_oast -> set -> payload_sources.extend -> payload_sources.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 297` |
| 298 | SsrfResultPublisher.__init__ -> new_id | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 298` |
| 299 | SsrfResultPublisher.publish_status -> TaskUpdatePayload -> AivaMessage -> self._publish | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 299` |
| 300 | SsrfResultPublisher.publish_finding -> AivaMessage -> self._publish | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 300` |
| 301 | SsrfResultPublisher.publish_error -> self.publish_status | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 301` |
| 302 | SsrfResultPublisher._publish -> message.model_dump -> self._broker.publish | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 302` |
| 303 | SSRFDetectionContext -> field -> field -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 303` |
| 304 | SSRFDetectionContext.add_finding -> self.findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 304` |
| 305 | SSRFDetectionContext.add_error -> self.errors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 305` |
| 306 | SmartSSRFDetector.__init__ -> SsrfConfig | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 306` |
| 307 | SmartSSRFDetector.detect_vulnerabilities -> UnifiedSmartDetectionManager -> SSRFDetectionContext -> analyzer.analyze -> ValueError -> smart_manager.start_detection | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 307` |
| 308 | SmartSSRFDetector._prioritize_vectors -> cloud_vectors.append -> other_vectors.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 308` |
| 309 | SmartSSRFDetector._execute_detection -> smart_manager.update_progress -> self._test_vector -> context.increment_attempts | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 309` |
| 310 | SmartSSRFDetector._test_vector -> self._resolve_payload -> self._issue_request -> context.detector.analyze -> self._build_internal_finding -> context.add_finding | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 310` |
| 311 | SmartSSRFDetector._resolve_payload -> dispatcher.register -> payload.replace | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 311` |
| 312 | SmartSSRFDetector._issue_request -> self._parse_target_config -> self._process_parameter_injection | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 312` |
| 313 | SmartSSRFDetector._process_parameter_injection -> injection_handlers.get -> handler | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 313` |
| 314 | SmartSSRFDetector._execute_http_request -> urlunparse | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 314` |
| 315 | SmartSSRFDetector._verify_internal_service_access -> self._verify_service_content | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 315` |
| 316 | SmartSSRFDetector._verify_service_content -> response.text.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 316` |
| 317 | SmartSSRFDetector._extract_token -> match.group -> domain.split | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 317` |
| 318 | SSRFDetector.analyze -> SSRFEngine -> engine.run -> engine.close | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 318` |
| 319 | SSRFDetector._issue_to_finding -> Vulnerability -> FindingEvidence -> FindingImpact -> FindingRecommendation -> FindingTarget | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 319` |
| 320 | SSRFEngine.close -> self.client.aclose | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 320` |
| 321 | SSRFEngine._resolve_ips -> socket.getaddrinfo -> ips.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 321` |
| 322 | SSRFEngine._is_internal_ip -> ipaddress.ip_address | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 322` |
| 323 | SSRFEngine.check_internal_access -> self._resolve_ips -> self.client.get -> issues.append -> issues.append -> issues.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 323` |
| 324 | SSRFEngine.check_cloud_metadata -> issues.append -> self.client.get -> issues.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 324` |
| 325 | SSRFEngine.check_file_protocol -> issues.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 325` |
| 326 | SSRFEngine.run -> tasks.append -> tasks.append -> tasks.append -> issues.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 326` |
| 327 | WebScannerManager.scan -> self._scan_subdomains_sync -> findings.extend -> self._scan_directories_sync -> findings.extend -> self._detect_technologies_sync | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 327` |
| 328 | WebScannerManager._scan_subdomains_sync -> urllib.parse.urlparse -> socket.gethostbyname -> subdomains.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 328` |
| 329 | WebScannerManager._scan_directories_sync -> target.rstrip -> requests.head -> directories.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 329` |
| 330 | WebScannerManager._detect_technologies_sync -> requests.get -> technologies.append -> technologies.append -> response.text.lower -> technologies.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 330` |
| 331 | scan_target -> WebScannerManager | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 331` |
| 332 | WebTarget -> field -> field -> field -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 332` |
| 333 | WebTarget.__post_init__ -> urllib.parse.urlparse | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 333` |
| 334 | SubdomainEnumerator.__init__ -> set | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 334` |
| 335 | SubdomainEnumerator.enumerate_subdomains -> self.found_subdomains.clear -> self._enumerate_crt_sh -> self._enumerate_dns_brute -> self._enumerate_search_engines -> self._enumerate_common_subdomains | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 335` |
| 336 | SubdomainEnumerator._enumerate_crt_sh -> self.session.get -> response.json -> isinstance -> entry.get -> self.found_subdomains.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 336` |
| 337 | SubdomainEnumerator._enumerate_dns_brute -> dns.resolver.Resolver -> self.found_subdomains.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 337` |
| 338 | SubdomainEnumerator._enumerate_common_subdomains -> self.session.get -> self.found_subdomains.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 338` |
| 339 | DirectoryScanner.scan_directories -> self.found_directories.clear -> self._get_default_wordlist -> urllib.parse.urljoin -> tasks.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 339` |
| 340 | DirectoryScanner._check_path -> self.session.get -> self.found_directories.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 340` |
| 341 | VulnerabilityScanner.scan_vulnerabilities -> self.vulnerabilities.clear -> self._scan_xss -> self._scan_sql_injection -> self._scan_directory_traversal -> self._scan_security_headers | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 341` |
| 342 | VulnerabilityScanner._scan_xss -> urllib.parse.quote -> self.session.get -> response.text -> self.vulnerabilities.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 342` |
| 343 | VulnerabilityScanner._scan_sql_injection -> urllib.parse.quote -> self.session.get -> response.text -> content.lower -> self.vulnerabilities.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 343` |
| 344 | VulnerabilityScanner._scan_directory_traversal -> urllib.parse.quote -> self.session.get -> response.text -> self.vulnerabilities.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 344` |
| 345 | VulnerabilityScanner._scan_security_headers -> self.session.get -> missing_headers.append -> self.vulnerabilities.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 345` |
| 346 | VulnerabilityScanner._scan_clickjacking -> self.session.get -> self.vulnerabilities.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 346` |
| 347 | TechnologyDetector.detect_technologies -> self.technologies.clear -> session.get -> response.text -> headers.get -> self.technologies.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 347` |
| 348 | TechnologyDetector._detect_frameworks -> content.lower -> self.technologies.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 348` |
| 349 | TechnologyDetector._detect_js_libraries -> self.technologies.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 349` |
| 350 | TechnologyDetector._detect_css_frameworks -> self.technologies.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 350` |
| 351 | WebAttackManager.__init__ -> SubdomainEnumerator -> DirectoryScanner -> VulnerabilityScanner -> TechnologyDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 351` |
| 352 | WebAttackManager.comprehensive_scan -> WebTarget -> Progress -> progress.add_task -> self.subdomain_enumerator.enumerate_subdomains -> progress.update | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 352` |
| 353 | WebAttackCLI.show_main_menu -> Panel.fit -> Table -> table.add_column -> table.add_column -> table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 353` |
| 354 | WebAttackCLI.run_interactive -> self.show_main_menu -> self._comprehensive_scan -> self._subdomain_enumeration -> self._directory_scan -> self._vulnerability_scan | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 354` |
| 355 | WebAttackCLI._comprehensive_scan -> self.manager.comprehensive_scan -> self._display_scan_results | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 355` |
| 356 | WebAttackCLI._subdomain_enumeration -> self.manager.subdomain_enumerator.enumerate_subdomains -> Table -> table.add_column -> table.add_row | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 356` |
| 357 | WebAttackCLI._directory_scan -> self.manager.directory_scanner.scan_directories -> Table -> table.add_column -> table.add_column -> table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 357` |
| 358 | WebAttackCLI._vulnerability_scan -> self.manager.vulnerability_scanner.scan_vulnerabilities -> Table -> table.add_column -> table.add_column -> table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 358` |
| 359 | WebAttackCLI._technology_detection -> self.manager.technology_detector.detect_technologies -> Table -> table.add_column -> table.add_row | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 359` |
| 360 | WebAttackCLI._show_scan_history -> Table -> table.add_column -> table.add_column -> table.add_column -> table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 360` |
| 361 | WebAttackCLI._export_results -> Path -> output_dir.mkdir -> open | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 361` |
| 362 | WebAttackCLI._display_scan_results -> Table -> Table -> vuln_table.add_column -> vuln_table.add_column -> vuln_table.add_column | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 362` |
| 363 | WebAttackCapability.__init__ -> WebAttackManager -> WebAttackCLI | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 363` |
| 364 | WebAttackCapability.initialize -> __import__ | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 364` |
| 365 | WebAttackCapability.execute -> command_handlers.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 365` |
| 366 | WebAttackCapability._execute_comprehensive_scan -> parameters.get -> parameters.get -> self.manager.comprehensive_scan | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 366` |
| 367 | WebAttackCapability._execute_subdomain_scan -> parameters.get -> self.manager.subdomain_enumerator.enumerate_subdomains | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 367` |
| 368 | WebAttackCapability._execute_directory_scan -> parameters.get -> self.manager.directory_scanner.scan_directories | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 368` |
| 369 | WebAttackCapability._execute_vulnerability_scan -> parameters.get -> self.manager.vulnerability_scanner.scan_vulnerabilities | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 369` |
| 370 | WebAttackCapability._execute_technology_detection -> parameters.get -> self.manager.technology_detector.detect_technologies | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 370` |
| 371 | WebAttackCapability._execute_interactive -> self.cli.run_interactive | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 371` |
| 372 | WebAttackCapability.cleanup -> self.manager.scan_results.clear | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 372` |
| 373 | register_capability -> WebAttackCapability -> RealCapabilityRegistry.register_capability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 373` |
| 374 | DirectoryBruteforcer.__init__ -> self._get_default_wordlist -> requests.Session -> self.session.headers.update | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 374` |
| 375 | DirectoryBruteforcer.scan -> open -> line.strip -> line.strip -> urls_to_test.append -> urls_to_test.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 375` |
| 376 | DirectoryBruteforcer._test_url -> self.session.get -> self._determine_severity | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 376` |
| 377 | DirectoryBruteforcer._determine_severity -> response.url.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 377` |
| 378 | PortScanner.scan -> concurrent.futures.ThreadPoolExecutor -> executor.submit -> future.result -> results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 378` |
| 379 | PortScanner._scan_port -> socket.socket -> sock.settimeout -> sock.connect_ex -> self._grab_banner -> self.COMMON_PORTS.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 379` |
| 380 | PortScanner._grab_banner -> sock.settimeout | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 380` |
| 381 | SubdomainScanner.__init__ -> self._get_default_wordlist -> dns.resolver.Resolver | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 381` |
| 382 | SubdomainScanner.scan -> set -> subdomains.update -> subdomains.update -> subdomains.update | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 382` |
| 383 | SubdomainScanner._passive_discovery -> set -> subdomains.update | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 383` |
| 384 | SubdomainScanner._search_crtsh -> set -> requests.get -> response.json -> cert.get -> subdomain.strip | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 384` |
| 385 | SubdomainScanner._bruteforce_discovery -> set -> open -> line.strip -> line.strip -> self._resolve_domain | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 385` |
| 386 | SubdomainScanner._dns_zone_transfer -> set -> dns.resolver.resolve -> dns.zone.from_xfr -> self._resolve_domain -> subdomains.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 386` |
| 387 | SubdomainScanner._resolve_domain -> self.resolver.resolve | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 387` |
| 388 | TechDetector.__init__ -> requests.Session -> self.session.headers.update -> self._load_fingerprints | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 388` |
| 389 | TechDetector.detect -> set -> self.session.get -> technologies.update -> technologies.update -> technologies.update | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 389` |
| 390 | TechDetector._analyze_headers -> set -> version_match.group -> technologies.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 390` |
| 391 | TechDetector._analyze_html -> set -> evidence.append -> technologies.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 391` |
| 392 | TechDetector._analyze_cookies -> set -> technologies.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 392` |
| 393 | TechDetector._analyze_meta_tags -> set -> generator_match.group -> tech_match.group -> tech_match.group -> technologies.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 393` |
| 394 | TechDetector._analyze_scripts -> set -> src.lower -> technologies.add -> technologies.add -> technologies.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 394` |
| 395 | WebCrawler.__init__ -> set -> requests.Session -> self.session.headers.update | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 395` |
| 396 | WebCrawler.crawl -> to_visit.pop -> self.visited.add -> self._crawl_page -> results.append -> to_visit.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 396` |
| 397 | WebCrawler._crawl_page -> self.session.get -> BeautifulSoup -> self._extract_forms -> self._extract_links -> self._extract_parameters | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 397` |
| 398 | WebCrawler._extract_forms -> urljoin -> forms.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 398` |
| 399 | WebCrawler._extract_links -> tag.get -> tag.get -> urljoin -> links.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 399` |
| 400 | WebCrawler._extract_parameters -> set -> parse_qs -> parameters.update -> input_tag.get -> parameters.add | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 400` |
| 401 | OastHttpCallbackStore.register_probe -> client.post -> response.raise_for_status -> cast -> RuntimeError -> client.aclose | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 401` |
| 402 | OastHttpCallbackStore.fetch_events -> self._resolve_token -> client.get -> response.raise_for_status -> cast -> RuntimeError | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 402` |
| 403 | BlindXssListenerValidator.__init__ -> RuntimeError -> OastHttpCallbackStore | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 403` |
| 404 | BlindXssListenerValidator.provision_payload -> self._store.register_probe | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 404` |
| 405 | BlindXssListenerValidator.collect_events -> self._store.fetch_events | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 405` |
| 406 | XSSCommandHandler.__init__ -> XSSManager -> self.logger.info | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 406` |
| 407 | XSSCommandHandler.handle_command -> ValueError -> payload.get -> ValueError -> payload.get -> payload.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 407` |
| 408 | XSSCommandHandler._execute_xss_scan -> self._build_scan_options -> context.custom_config.get -> self.logger.warning -> self.xss_manager.comprehensive_scan | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 408` |
| 409 | XSSCommandHandler._build_scan_options -> default_options.update -> default_options.update -> default_options.update -> default_options.update -> default_options.update | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 409` |
| 410 | DomXssDetector.analyze -> document.find -> max -> min -> window.lower -> window.strip | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 410` |
| 411 | HackingToolXSSConfig.__init__ -> self._initialize_tools -> self._calculate_priority_order | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 411` |
| 412 | HackingToolXSSConfig._calculate_priority_order -> sorted | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 412` |
| 413 | HackingToolXSSConfig.validate_tool_requirements -> self.get_tool_config | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 413` |
| 414 | HackingToolXSSConfig.export_config -> open -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 414` |
| 415 | HackingToolXSSConfig.get_execution_plan -> self.get_high_priority_tools -> self.get_tools_by_mode -> execution_plan.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 415` |
| 416 | XssPayloadGenerator.generate -> OrderedDict -> ordered.setdefault -> ordered.setdefault -> ordered.setdefault | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 416` |
| 417 | XssResultPublisher.__init__ -> new_id | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 417` |
| 418 | XssResultPublisher.publish_status -> TaskUpdatePayload -> AivaMessage -> self._publish | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 418` |
| 419 | XssResultPublisher.publish_finding -> AivaMessage -> self._publish | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 419` |
| 420 | XssResultPublisher.publish_error -> self.publish_status | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 420` |
| 421 | XssResultPublisher._publish -> message.model_dump -> self._broker.publish | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 421` |
| 422 | StoredXssDetector.execute -> self._submit_payload -> ValueError -> client.get -> results.append -> client.aclose | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 422` |
| 423 | StoredXssDetector._submit_payload -> self._inject_query -> payload.encode | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 423` |
| 424 | StoredXssDetector._verify_persistence -> html.escape | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 424` |
| 425 | StoredXssDetector._inject_query -> pair.split -> urlencode | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 425` |
| 426 | _QueueEntry -> field -> field | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 426` |
| 427 | XssTaskQueue.__init__ -> itertools.count | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 427` |
| 428 | XssTaskQueue.put -> self._clock -> max -> QueuedTask -> RuntimeError -> _QueueEntry | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 428` |
| 429 | XssTaskQueue.get -> self._discard_invalid_locked -> self._condition.wait -> self._clock -> suppress -> heapq.heappop | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 429` |
| 430 | XssTaskQueue.close -> self._condition.notify_all | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 430` |
| 431 | XssTaskQueue._discard_invalid_locked -> heapq.heappop | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 431` |
| 432 | TraditionalXssDetector.__init__ -> max | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 432` |
| 433 | TraditionalXssDetector.execute -> self._build_request_parts -> client.request -> self._errors.append -> results.append -> client.aclose | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 433` |
| 434 | TraditionalXssDetector._build_request_parts -> _inject_query -> copy.deepcopy -> _inject_mapping -> copy.deepcopy -> _inject_mapping | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 434` |
| 435 | _inject_query -> pair.split -> query_items.setdefault -> urlencode | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 435` |
| 436 | _payload_in_response -> unquote_plus -> unescape | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 436` |
| 437 | _verify_execution_context -> html.escape -> response_headers.get -> response_headers.get -> csp.lower -> match.group | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 437` |
| 438 | _detect_waf_interference -> response_text.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 438` |
| 439 | run_reflected_test -> FunctionTaskPayload -> XssPayloadGenerator -> generator.generate_basic_payloads -> TraditionalXssDetector -> detector.execute | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 439` |
| 440 | run_dom_test -> DomXssDetector -> client.get -> detector.analyze | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 440` |
| 441 | run_stored_test -> FunctionTaskPayload -> XssPayloadGenerator -> generator.generate_basic_payloads -> StoredXssDetector -> detector.execute | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 441` |
| 442 | main -> argparse.ArgumentParser -> parser.add_argument -> parser.add_argument -> parser.add_argument -> parser.add_argument | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 442` |
| 443 | run_xss_test -> requests.get -> requests.post -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 443` |
| 444 | CrossLanguageXSSEngine.__init__ -> get_xss_tools_config -> logging.getLogger -> Path | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 444` |
| 445 | CrossLanguageXSSEngine.initialize -> self._detect_language_environments -> self._validate_tool_availability -> RuntimeError -> self.logger.info -> self.logger.error | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 445` |
| 446 | CrossLanguageXSSEngine._validate_tool_availability -> self._check_dalfox_availability -> available_tools.append -> self._check_xspear_availability -> available_tools.append -> self._check_xsser_availability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 446` |
| 447 | CrossLanguageXSSEngine._check_dalfox_availability -> self.language_environments.get -> self._run_command -> self.logger.debug -> self.logger.warning | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 447` |
| 448 | CrossLanguageXSSEngine._check_xspear_availability -> self.language_environments.get -> self._run_command -> self.logger.debug -> self.logger.warning | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 448` |
| 449 | CrossLanguageXSSEngine._check_xsser_availability -> self.language_environments.get -> self._run_command -> self.logger.debug -> self.logger.warning | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 449` |
| 450 | CrossLanguageXSSEngine._detect_language_environments -> checker -> self.logger.info -> self.logger.error -> LanguageEnvironment | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 450` |
| 451 | CrossLanguageXSSEngine._check_go_environment -> self._run_command -> result.stdout.strip -> shutil.which | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 451` |
| 452 | CrossLanguageXSSEngine._check_ruby_environment -> self._run_command -> result.stdout.strip -> shutil.which | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 452` |
| 453 | CrossLanguageXSSEngine._check_python_environment -> self._run_command -> result.stdout.strip -> shutil.which | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 453` |
| 454 | CrossLanguageXSSEngine._check_rust_environment -> self._run_command -> result.stdout.strip -> shutil.which | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 454` |
| 455 | CrossLanguageXSSEngine.detect -> ValueError -> self._get_available_execution_plans -> self.logger.warning | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 455` |
| 456 | CrossLanguageXSSEngine._get_available_execution_plans -> self.config.get_execution_plan -> t.lower -> self.config.get_tool_config -> available_tools.append -> self.logger.warning | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 456` |
| 457 | CrossLanguageXSSEngine._execute_parallel_detection -> plan.get -> self._execute_tool_detection -> tasks.append -> self.logger.error -> detection_results.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 457` |
| 458 | CrossLanguageXSSEngine._execute_tool_detection -> self.logger.info -> self.config.get_tool_config -> ValueError -> execution_plan.get -> self._execute_go_tool | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 458` |
| 459 | CrossLanguageXSSEngine._execute_go_tool -> self._run_command -> output_file.read_text -> self.logger.warning -> output_file.unlink -> ValueError | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 459` |
| 460 | CrossLanguageXSSEngine._execute_ruby_tool -> self._run_command -> output_file.read_text -> self.logger.warning -> output_file.unlink -> ValueError | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 460` |
| 461 | CrossLanguageXSSEngine._execute_python_tool -> tool_config.name.lower -> run_pattern.format | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 461` |
| 462 | CrossLanguageXSSEngine._execute_rust_tool -> ValueError | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 462` |
| 463 | CrossLanguageXSSEngine._parse_tool_output -> self._parse_json_output -> self._parse_regex_output | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 463` |
| 464 | CrossLanguageXSSEngine._create_result_from_json -> json_data.get -> json_data.get -> json_data.get -> item.get -> item.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 464` |
| 465 | CrossLanguageXSSEngine._parse_regex_output -> self._process_regex_matches | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 465` |
| 466 | CrossLanguageXSSEngine._process_regex_matches -> pattern.lower -> float | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 466` |
| 467 | CrossLanguageXSSEngine._run_command -> process.kill -> process.wait | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 467` |
| 468 | CrossLanguageXSSEngine._is_language_available -> self.language_environments.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 468` |
| 469 | CrossLanguageXSSEngine.get_available_tools -> available_tools.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 469` |
| 470 | CrossLanguageXSSEngine.cleanup -> shutil.rmtree -> self.logger.info -> self.logger.error | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 470` |
| 471 | CrossLanguageXSSEngine.__del__ -> self.cleanup | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 471` |
| 472 | get_xss_engine -> CrossLanguageXSSEngine -> _xss_engine_instance.initialize | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 472` |
| 473 | detect_xss -> get_xss_engine | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 473` |
| 474 | DalfoxIntegration.__init__ -> self._find_dalfox_path | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 474` |
| 475 | DalfoxIntegration.install_dalfox -> process.communicate | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 475` |
| 476 | DalfoxIntegration.scan_target -> cmd.extend -> cmd.extend -> cmd.extend -> cmd.extend -> cmd.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 476` |
| 477 | DalfoxIntegration._parse_dalfox_output -> XSSVulnerability -> vulnerabilities.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 477` |
| 478 | XSSPayloadGenerator.__init__ -> self._load_payloads -> self._load_context_specific_payloads | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 478` |
| 479 | XSSPayloadGenerator.generate_payloads -> payloads.extend -> payloads.extend -> payloads.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 479` |
| 480 | DOMXSSDetector.scan_dom_xss -> session.get -> response.text -> self._analyze_javascript -> vulnerabilities.extend -> self._test_dom_payloads | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 480` |
| 481 | DOMXSSDetector._analyze_javascript -> BeautifulSoup -> soup.find_all -> XSSVulnerability -> vulnerabilities.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 481` |
| 482 | DOMXSSDetector._test_dom_payloads -> session.get -> response.text -> XSSVulnerability -> vulnerabilities.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 482` |
| 483 | StoredXSSDetector.scan_stored_xss -> self._submit_payloads -> self._check_stored_execution -> vulnerabilities.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 483` |
| 484 | StoredXSSDetector._submit_payloads -> payload.replace -> target.parameters.copy -> urlencode -> session.post -> urlencode | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 484` |
| 485 | StoredXSSDetector._check_stored_execution -> target.url.replace -> target.url.replace -> session.get -> response.text -> XSSVulnerability | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 485` |
| 486 | BlindXSSDetector.__init__ -> self._generate_blind_payloads | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 486` |
| 487 | BlindXSSDetector.scan_blind_xss -> self._submit_blind_payloads -> XSSVulnerability -> vulnerabilities.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 487` |
| 488 | BlindXSSDetector._submit_blind_payloads -> method | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 488` |
| 489 | BlindXSSDetector._submit_via_forms -> target.parameters.copy -> urlencode -> session.post -> urlencode -> session.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 489` |
| 490 | BlindXSSDetector._submit_via_parameters -> quote -> session.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 490` |
| 491 | BlindXSSDetector._submit_via_headers -> target.headers.copy -> test_headers.update -> session.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 491` |
| 492 | BlindXSSDetector._submit_via_user_agent -> target.headers.copy -> session.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 492` |
| 493 | XSSManager.__init__ -> DalfoxIntegration -> XSSPayloadGenerator -> DOMXSSDetector -> StoredXSSDetector -> BlindXSSDetector | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 493` |
| 494 | XSSManager._parse_target -> parse_qs -> parameters.items | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 494` |
| 495 | XSSManager.comprehensive_scan -> self._parse_target -> self.dalfox.scan_target -> asdict -> self.scan_results.extend -> self.dom_detector.scan_dom_xss | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 495` |
| 496 | XSSManager._custom_xss_scan -> self.payload_generator.generate_payloads -> target.parameters.copy -> urlencode -> session.post -> response.text | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 496` |
| 497 | XSSManager._generate_summary -> all_vulns.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 497` |
| 498 | DeserializationDetector.generate_payloads -> self._generate_python_pickle -> self._generate_python_yaml -> self._generate_java_payload -> self._generate_jackson_payload -> self._generate_php_payload | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 498` |
| 499 | DeserializationDetector.test_deserialization -> self.generate_detection_payloads -> self._measure_baseline -> self._test_single_payload -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 499` |
| 500 | DeserializationDetector._measure_response_time -> requests.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 500` |
| 501 | DeserializationDetector._check_deserialization_error -> error_patterns.get -> response_text.lower | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 501` |
| 502 | DeserializationDetector.test_cookie_deserialization -> self.generate_payloads -> self._measure_response_time -> requests.get -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 502` |
| 503 | DeserializationDetector.generate_java_payload_with_ysoserial -> print | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 503` |
| 504 | DeserializationDetector.generate_detection_payloads -> self._create_java_detection_payload -> self._create_jackson_payload -> self._create_fastjson_payload -> self._create_xstream_payload -> self._create_kryo_detection_payload | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 504` |
| 505 | DeserializationDetector._create_python_pickle_payload -> pickle.dumps | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 505` |
| 506 | DeserializationDetector._measure_baseline -> requests.post -> requests.get -> times.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 506` |
| 507 | DeserializationDetector._test_single_payload -> requests.post -> requests.get -> requests.get -> requests.get | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 507` |
| 508 | PassiveAnalyzer.__init__ -> self._setup_patterns | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 508` |
| 509 | PassiveAnalyzer.analyze_har -> open -> entry.get -> entry.get -> request.get -> findings.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 509` |
| 510 | PassiveAnalyzer._analyze_request -> findings.extend -> parse_qs -> findings.extend -> request.get -> findings.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 510` |
| 511 | PassiveAnalyzer._analyze_response -> response.get -> findings.extend -> findings.extend -> response.get -> findings.extend | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 511` |
| 512 | PassiveAnalyzer._check_sensitive_data_in_url -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 512` |
| 513 | PassiveAnalyzer._check_sensitive_params -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 513` |
| 514 | PassiveAnalyzer._check_sensitive_data_in_body -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 514` |
| 515 | PassiveAnalyzer._check_security_headers -> headers.get -> findings.append -> findings.append -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 515` |
| 516 | PassiveAnalyzer._analyze_cookies -> cookie.strip -> cookie.split -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 516` |
| 517 | PassiveAnalyzer._analyze_set_cookie -> set_cookie.split -> set_cookie.lower -> findings.append -> findings.append -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 517` |
| 518 | PassiveAnalyzer._check_error_disclosure -> response.get -> findings.append -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 518` |
| 519 | XXEDetector.__init__ -> self._generate_payloads | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 519` |
| 520 | XXEDetector.test_xxe -> requests.post -> requests.get -> self._analyze_response -> findings.append -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 520` |
| 521 | XXEDetector.test_with_soap -> requests.post -> self._analyze_response -> findings.append | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 521` |
| 522 | unknown_stdin | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 522` |
| 523 | unknown_stdin | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 523` |
| 524 | main -> PathsConfig::new | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 524` |
| 525 | main -> PathsConfig::new | unknown | `python -m aiva_core.internal_exploration.aiva_cli_implementation --flow 525` |
