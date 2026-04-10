#!/usr/bin/env python3
"""
gRPC Protocol Buffers 編譯腳本
自動編譯 .proto 檔案為各語言的 gRPC 存根代碼
"""

import subprocess
import sys
from pathlib import Path

def compile_protos():
    """編譯 Protocol Buffers 檔案"""
    proto_dir = Path(__file__).parent
    proto_file = proto_dir / "aiva.proto"
    
    if not proto_file.exists():
        print(f"❌ Proto 檔案不存在: {proto_file}")
        return False
    
    # Python 編譯
    print("🔄 編譯 Python gRPC 存根...")
    python_out = proto_dir / "python"
    python_out.mkdir(exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--python_out={python_out}",
        f"--grpc_python_out={python_out}",
        str(proto_file)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Python gRPC 存根編譯完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ Python 編譯失敗: {e}")
        return False
    
    # Go 編譯
    print("🔄 編譯 Go gRPC 存根...")
    go_out = proto_dir / "go"
    go_out.mkdir(exist_ok=True)
    
    cmd = [
        "protoc",
        f"--proto_path={proto_dir}",
        f"--go_out={go_out}",
        f"--go-grpc_out={go_out}",
        str(proto_file)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Go gRPC 存根編譯完成")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"⚠️  Go 編譯跳過 (protoc-gen-go 未安裝): {e}")
    
    print("🎉 gRPC 編譯完成!")
    return True

if __name__ == "__main__":
    success = compile_protos()
    sys.exit(0 if success else 1)
