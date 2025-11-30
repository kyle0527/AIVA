from __future__ import annotations
import argparse, asyncio, os, json, sys
from services.core.aiva_core_v1 import AivaCore

def cmd_list_caps():
    core = AivaCore()
    caps = core.list_caps()
    for k, v in sorted(caps.items()):
        print(f"{k:20s}  {v.get('desc','')}")

async def cmd_scan(target: str):
    core = AivaCore()
    flow = os.path.join("config", "flows", "scan_minimal.yaml")
    plan = core.plan(flow, target=target)
    print(f"[plan] run_id={plan.run_id}, nodes={[n.id for n in plan.nodes]}")
    summary = await core.exec(plan)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def main():
    ap = argparse.ArgumentParser(prog="aiva", description="AIVA Core v1 CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-caps", help="列出註冊能力")

    pscan = sub.add_parser("scan", help="執行最小掃描流程（index→ast→graph→report）")
    pscan.add_argument("--target", default=".", help="掃描根目錄")

    args = ap.parse_args()
    if args.cmd == "list-caps":
        cmd_list_caps()
    elif args.cmd == "scan":
        asyncio.run(cmd_scan(args.target))

if __name__ == "__main__":
    main()
