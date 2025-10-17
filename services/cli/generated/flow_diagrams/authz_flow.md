# 權限檢測流程

```mermaid
sequenceDiagram
    participant CLI as 🖥️ CLI
    participant Core as 🤖 Core
    participant MQ as 📨 MQ

    CLI->>Core: authz check
    Core->>MQ: Send Task
    MQ->>AuthZ: Process
    AuthZ->>MQ: Result
    MQ->>Core: Complete
    Core->>CLI: Return Result
```


