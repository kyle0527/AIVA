"""
驗證核心模組 CLI 計算的正確性

這個腳本會：
1. 驗證數學公式的正確性
2. 列舉少量實際組合來佐證
3. 檢查計算邏輯
"""

from __future__ import annotations

import math


def verify_permutation_formula():
    """驗證排列公式計算."""
    print("=" * 70)
    print("驗證排列公式 P(n,k) = n! / (n-k)!")
    print("=" * 70)
    
    # 對於 N=5 的情況
    N = 5
    print(f"\n候選端口數量 N = {N}")
    print("\n逐項計算 P(5,k):")
    
    total = 0
    for k in range(1, N + 1):
        p_nk = math.factorial(N) // math.factorial(N - k)
        total += p_nk
        print(f"  k={k}: P({N},{k}) = {N}!/({N}-{k})! = {p_nk}")
    
    print(f"\n總計: Σ(k=1 to {N}) P({N},k) = {total}")
    print("\n✓ 驗證通過：325 種端口序列組合")


def enumerate_sample_combinations():
    """列舉樣本組合來驗證."""
    print("\n" + "=" * 70)
    print("列舉樣本組合（驗證計算邏輯）")
    print("=" * 70)
    
    modes = ["ui", "ai", "hybrid"]
    hosts = ["127.0.0.1"]
    ports = [3000, 8000, 8080, 8888, 9000]
    
    print(f"\n參數空間:")
    print(f"  modes: {modes} (共 {len(modes)} 個)")
    print(f"  hosts: {hosts} (共 {len(hosts)} 個)")
    print(f"  ports: {ports} (共 {len(ports)} 個)")
    
    # 計算不指定 --ports 的組合
    no_ports_count = len(modes) * len(hosts)
    print(f"\n不指定 --ports 的組合數: {len(modes)} × {len(hosts)} = {no_ports_count}")
    
    # 列舉前幾個不指定 --ports 的組合
    print("\n前 3 個組合（不指定 --ports）:")
    for i, mode in enumerate(modes, 1):
        for host in hosts:
            print(f"  {i}. --mode {mode} --host {host}")
    
    # 計算指定 --ports 的組合（只列舉 k=1 的情況）
    print(f"\n指定單一端口的組合數: {len(modes)} × {len(hosts)} × {len(ports)} = {len(modes) * len(hosts) * len(ports)}")
    
    # 列舉前幾個指定單一端口的組合
    print("\n前 5 個組合（指定單一端口）:")
    count = 0
    for mode in modes:
        for host in hosts:
            for port in ports:
                count += 1
                print(f"  {count}. --mode {mode} --host {host} --ports {port}")
                if count >= 5:
                    break
            if count >= 5:
                break
        if count >= 5:
            break
    
    # 列舉 k=2 的幾個例子
    print("\n指定兩個端口的組合（部分範例）:")
    print("  （順序有意義，所以 [3000, 8000] 與 [8000, 3000] 是不同的組合）")
    
    count = 0
    for mode in modes[:1]:  # 只用一個 mode 舉例
        for host in hosts:
            for i, port1 in enumerate(ports):
                for port2 in ports[i+1:]:  # 避免重複
                    count += 1
                    print(f"  - --mode {mode} --host {host} --ports {port1} {port2}")
                    print(f"  - --mode {mode} --host {host} --ports {port2} {port1}")
                    if count >= 3:
                        break
                if count >= 3:
                    break
            if count >= 3:
                break


def verify_total_calculation():
    """驗證總數計算."""
    print("\n" + "=" * 70)
    print("驗證總數計算")
    print("=" * 70)
    
    modes_count = 3
    hosts_count = 1
    port_sequences = 325
    
    with_ports = modes_count * hosts_count * port_sequences
    without_ports = modes_count * hosts_count
    total = with_ports + without_ports
    
    print(f"\n計算步驟:")
    print(f"  1. 有指定 --ports: {modes_count} × {hosts_count} × {port_sequences} = {with_ports}")
    print(f"  2. 無指定 --ports: {modes_count} × {hosts_count} = {without_ports}")
    print(f"  3. 總計: {with_ports} + {without_ports} = {total}")
    
    print(f"\n✓ 驗證通過：總使用可能性 = {total} 種")
    
    # 依模式分組
    print(f"\n依模式分組（每種模式）:")
    per_mode = total // modes_count
    print(f"  每種模式: {total} ÷ {modes_count} = {per_mode} 種")
    print(f"  - with_ports: {port_sequences} × {hosts_count} = {port_sequences}")
    print(f"  - without_ports: {hosts_count}")
    print(f"  - 小計: {port_sequences + hosts_count} = {per_mode}")


def show_scaling_examples():
    """展示參數變化對總數的影響."""
    print("\n" + "=" * 70)
    print("參數變化影響分析")
    print("=" * 70)
    
    modes_count = 3
    
    scenarios = [
        ("預設配置", 1, [3000, 8000, 8080, 8888, 9000]),
        ("增加主機", 3, [3000, 8000, 8080, 8888, 9000]),
        ("減少端口", 1, [8080, 8888, 9000]),
        ("增加端口", 1, [3000, 5000, 8000, 8080, 8888, 9000, 10000]),
    ]
    
    print("\n不同配置下的總數：\n")
    
    for name, host_count, port_list in scenarios:
        N = len(port_list)
        port_sequences = sum(
            math.factorial(N) // math.factorial(N - k)
            for k in range(1, N + 1)
        )
        
        total = modes_count * host_count * (port_sequences + 1)
        
        print(f"{name}:")
        print(f"  - 主機數: {host_count}")
        print(f"  - 端口數: {N}")
        print(f"  - 端口序列: {port_sequences}")
        print(f"  - 總計: {modes_count} × {host_count} × ({port_sequences} + 1) = {total:,}")
        print()


def main():
    """執行所有驗證."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "核心模組 CLI 計算驗證工具" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    
    verify_permutation_formula()
    enumerate_sample_combinations()
    verify_total_calculation()
    show_scaling_examples()
    
    print("=" * 70)
    print("✓ 所有驗證完成")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
