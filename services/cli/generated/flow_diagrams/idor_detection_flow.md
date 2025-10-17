# IDOR 檢測流程

```mermaid
sequenceDiagram
    participant CLI as 🖥️ CLI
    participant Core as 🤖 Core
    participant Func as ⚡ Function
    participant MQ as 📨 MQ

    CLI->>Core: detect idor
    Core->>MQ: Send Task
    MQ->>Function: Process
    Function->>MQ: Result
    MQ->>Core: Complete
    Core->>CLI: Return Result
```


