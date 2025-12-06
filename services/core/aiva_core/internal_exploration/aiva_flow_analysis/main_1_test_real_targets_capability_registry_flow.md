# main 拼接序列 2

```mermaid
flowchart TD
    graph_0_test_real_targets[test_real_targets\nfull_penetration_test, test_individual_capabilities, main]
    graph_0_test_real_targets -->|get_capability_registry| graph_1_capability_registry
    graph_1_capability_registry[capability_registry\nget_capability_registry, initialize_capability_registry]
```

## 圖序列

1. test_real_targets
2. capability_registry

## 檔案路徑

- C:\D\fold7\AIVA-git\test_real_targets.py
- C:\D\fold7\AIVA-git\services\core\aiva_core\core_capabilities\capability_registry.py
