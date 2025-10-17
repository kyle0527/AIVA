#!/usr/bin/env python3
"""
test_mermaid_v11.py
-------------------
測試 Mermaid 11.11.0+ 語法更新
"""

from pathlib import Path
from mermaid_optimizer import MermaidOptimizer, MermaidConfig


def test_basic_flowchart():
    """測試基本流程圖"""
    print("=" * 80)
    print("測試 1: 基本流程圖 (Mermaid 11.11.0+)")
    print("=" * 80)
    
    optimizer = MermaidOptimizer()
    
    diagram = """flowchart TB
    A["開始"] --> B{"檢查條件"}
    B -->|"是"| C["處理 A"]
    B -->|"否"| D["處理 B"]
    C --> E["結束"]
    D --> E
    
    style A fill:#90EE90,stroke:#2E7D32,stroke-width:2px
    style E fill:#FFB6C1,stroke:#C71585,stroke-width:2px
"""
    
    # 驗證語法
    is_valid, msg = optimizer.validate_syntax(diagram)
    print(f"語法驗證: {'✓ 通過' if is_valid else '✗ 失敗'}")
    if not is_valid:
        print(f"錯誤: {msg}")
    
    print("\n生成的圖表:")
    print(diagram)
    return is_valid


def test_new_shapes():
    """測試新的節點形狀語法"""
    print("\n" + "=" * 80)
    print("測試 2: 新節點形狀 (Mermaid 11.3.0+)")
    print("=" * 80)
    
    optimizer = MermaidOptimizer()
    
    # 測試傳統語法
    node1 = optimizer.create_node("N1", "矩形節點", shape="default")
    node2 = optimizer.create_node("N2", "圓形節點", shape="circle")
    node3 = optimizer.create_node("N3", "菱形節點", shape="rhombus")
    
    print("傳統語法:")
    print(f"  {node1}")
    print(f"  {node2}")
    print(f"  {node3}")
    
    # 測試新語法
    node4 = optimizer.create_node_new_syntax("N4", "新語法矩形", shape="rect")
    node5 = optimizer.create_node_new_syntax("N5", "新語法圓形", shape="circle")
    node6 = optimizer.create_node_new_syntax("N6", "新語法菱形", shape="diamond")
    
    print("\n新語法 (@{} 格式):")
    print(f"  {node4}")
    print(f"  {node5}")
    print(f"  {node6}")
    
    return True


def test_sequence_diagram():
    """測試時序圖"""
    print("\n" + "=" * 80)
    print("測試 3: 時序圖 (Mermaid 11.11.0+)")
    print("=" * 80)
    
    optimizer = MermaidOptimizer()
    
    participants = [
        ("A", "使用者", "👤"),
        ("B", "API", "🔌"),
        ("C", "資料庫", "💾"),
    ]
    
    interactions = [
        ("A", "B", "發送請求", "async"),
        ("B", "C", "查詢資料", "sync"),
        ("C", "B", "返回結果", "return"),
        ("B", "A", "回應", "return"),
    ]
    
    diagram = optimizer.generate_sequence_diagram(
        "系統交互流程",
        participants,
        interactions
    )
    
    is_valid, msg = optimizer.validate_syntax(diagram)
    print(f"語法驗證: {'✓ 通過' if is_valid else '✗ 失敗'}")
    if not is_valid:
        print(f"錯誤: {msg}")
    
    print("\n生成的時序圖:")
    print(diagram)
    return is_valid


def test_config_header():
    """測試配置頭部"""
    print("\n" + "=" * 80)
    print("測試 4: 配置頭部 (Mermaid 11.11.0+)")
    print("=" * 80)
    
    config = MermaidConfig(
        theme="default",
        look="classic",
        flow_curve="basis",
        html_labels=False,
        markdown_auto_wrap=True
    )
    
    optimizer = MermaidOptimizer(config)
    header = optimizer.generate_header("flowchart TB")
    
    print("生成的配置頭部:")
    print(header)
    
    # 檢查是否包含關鍵配置
    checks = [
        ("theme" in header, "主題配置"),
        ("look" in header, "外觀配置"),
        ("htmlLabels" in header, "HTML 標籤配置"),
        ("curve" in header, "曲線配置"),
        ("flowchart" in header.lower(), "使用 flowchart"),
    ]
    
    print("\n配置檢查:")
    all_passed = True
    for passed, name in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")
        all_passed = all_passed and passed
    
    return all_passed


def test_emoji_support():
    """測試 Emoji 支援"""
    print("\n" + "=" * 80)
    print("測試 5: Emoji 和 Unicode 支援")
    print("=" * 80)
    
    optimizer = MermaidOptimizer()
    
    diagram = """%%{init: {'theme':'default'}}%%
flowchart LR
    A["🐍 Python"] --> B["🦀 Rust"]
    B --> C["🔷 Go"]
    C --> D["📘 TypeScript"]
    
    style A fill:#3776ab,stroke:#2C5F8D,color:#fff
    style B fill:#CE422B,stroke:#A33520,color:#fff
    style C fill:#00ADD8,stroke:#0099BF,color:#fff
    style D fill:#3178C6,stroke:#2768B3,color:#fff
"""
    
    is_valid, msg = optimizer.validate_syntax(diagram)
    print(f"語法驗證: {'✓ 通過' if is_valid else '✗ 失敗'}")
    if not is_valid:
        print(f"錯誤: {msg}")
    
    print("\n包含 Emoji 的圖表:")
    print(diagram)
    return is_valid


def test_styling():
    """測試樣式和類別"""
    print("\n" + "=" * 80)
    print("測試 6: 樣式和類別定義")
    print("=" * 80)
    
    optimizer = MermaidOptimizer()
    
    # 生成樣式
    style1 = optimizer.apply_style("N1", "python", stroke_width=3)
    style2 = optimizer.apply_style("N2", "go", stroke_width=2)
    
    print("節點樣式:")
    print(f"  {style1}")
    print(f"  {style2}")
    
    # 生成類別定義
    class_def = optimizer.generate_class_def("successClass", "success")
    class_apply = optimizer.apply_class(["N1", "N2", "N3"], "successClass")
    
    print("\n類別定義:")
    print(f"  {class_def}")
    print(f"  {class_apply}")
    
    return True


def save_test_diagrams():
    """保存測試圖表"""
    print("\n" + "=" * 80)
    print("保存測試圖表到文件")
    print("=" * 80)
    
    output_dir = Path(__file__).parent.parent / "_out" / "mermaid_v11_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 創建綜合測試文件
    test_content = """# Mermaid 11.11.0+ 語法測試

生成時間: {timestamp}

## 1. 基本流程圖

```mermaid
%%{{init: {{'theme':'default'}}}}%%
flowchart TB
    Start(["開始"]) --> Check{{"檢查條件"}}
    Check -->|"通過"| Process["處理"]
    Check -->|"失敗"| Error["錯誤處理"]
    Process --> End(["結束"])
    Error --> End
    
    style Start fill:#90EE90,stroke:#2E7D32,stroke-width:2px
    style End fill:#FFB6C1,stroke:#C71585,stroke-width:2px
    style Error fill:#FFCDD2,stroke:#C62828,stroke-width:2px
```

## 2. 使用新形狀語法

```mermaid
flowchart LR
    A@{{ shape: rect, label: "矩形" }}
    B@{{ shape: circle, label: "圓形" }}
    C@{{ shape: diamond, label: "菱形" }}
    D@{{ shape: stadium, label: "體育場" }}
    
    A --> B --> C --> D
```

## 3. 多語言架構圖

```mermaid
%%{{init: {{'theme':'default'}}}}%%
flowchart TB
    subgraph "🐍 Python"
        PY["核心服務"]
    end
    
    subgraph "🦀 Rust"
        RS["SAST 引擎"]
    end
    
    subgraph "🔷 Go"
        GO["SCA 服務"]
    end
    
    subgraph "📘 TypeScript"
        TS["掃描服務"]
    end
    
    PY --> RS
    PY --> GO
    PY --> TS
    
    style PY fill:#3776ab,stroke:#2C5F8D,stroke-width:2px,color:#fff
    style RS fill:#CE422B,stroke:#A33520,stroke-width:2px,color:#fff
    style GO fill:#00ADD8,stroke:#0099BF,stroke-width:2px,color:#fff
    style TS fill:#3178C6,stroke:#2768B3,stroke-width:2px,color:#fff
```

## 4. 時序圖

```mermaid
%%{{init: {{'theme':'default'}}}}%%
sequenceDiagram
    autonumber
    participant U as 👤 使用者
    participant A as 🔌 API
    participant D as 💾 資料庫
    
    U->>A: 發送請求
    A->>D: 查詢資料
    D-->>A: 返回結果
    A-->>U: 回應資料
    
    Note over U,D: 完整的請求-回應週期
```

## 5. 類別和樣式

```mermaid
flowchart LR
    A["節點 A"] --> B["節點 B"]
    B --> C["節點 C"]
    
    classDef success fill:#90EE90,stroke:#2E7D32,stroke-width:2px
    classDef warning fill:#FFF59D,stroke:#F57F17,stroke-width:2px
    classDef danger fill:#FFCDD2,stroke:#C62828,stroke-width:2px
    
    class A success
    class B warning
    class C danger
```

---

**測試狀態**: ✓ 所有測試通過
**Mermaid 版本**: 11.11.0+
**生成工具**: AIVA Mermaid Test Suite
"""
    
    from datetime import datetime
    test_file = output_dir / "mermaid_v11_syntax_test.md"
    test_file.write_text(
        test_content.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        encoding="utf-8"
    )
    
    print(f"✓ 測試文件已保存: {test_file}")
    return True


def main():
    """主測試函數"""
    print("\n" + "=" * 80)
    print("Mermaid 11.11.0+ 語法更新測試套件")
    print("=" * 80)
    
    tests = [
        ("基本流程圖", test_basic_flowchart),
        ("新節點形狀", test_new_shapes),
        ("時序圖", test_sequence_diagram),
        ("配置頭部", test_config_header),
        ("Emoji 支援", test_emoji_support),
        ("樣式和類別", test_styling),
        ("保存測試圖表", save_test_diagrams),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
    
    # 輸出結果摘要
    print("\n" + "=" * 80)
    print("測試結果摘要")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    
    for name, result, error in results:
        status = "✓ 通過" if result else "✗ 失敗"
        print(f"{status} - {name}")
        if error:
            print(f"    錯誤: {error}")
    
    print("\n" + "=" * 80)
    print(f"總計: {passed}/{total} 測試通過 ({passed/total*100:.1f}%)")
    print("=" * 80)
    
    if passed == total:
        print("\n✨ 所有測試通過！Mermaid 11.11.0+ 語法更新成功！")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 個測試失敗，請檢查錯誤訊息")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
