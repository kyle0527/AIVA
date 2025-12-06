# main 拼接序列 4

```mermaid
flowchart TD
    graph_0_test_real_targets[test_real_targets\nfull_penetration_test, test_individual_capabilities, main]
    graph_0_test_real_targets -->|get_capability_registry| graph_1_capabilities
    graph_1_capabilities[capabilities\nget_capability_registry, get_function_caller, get_internal_loop_connector]
```

## 圖序列

1. test_real_targets
2. capabilities

## 檔案路徑

- C:\D\fold7\AIVA-git\test_real_targets.py
- C:\D\fold7\AIVA-git\api\routers\capabilities.py
