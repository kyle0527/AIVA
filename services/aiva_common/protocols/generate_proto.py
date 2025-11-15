#!/usr/bin/env python3
"""
Protocol Buffers 生成腳本

從 .proto 文件生成 Python gRPC 代碼
"""

import subprocess
import sys
from pathlib import Path


def generate_proto_files():
    """生成 protobuf Python 文件"""
    proto_dir = Path(__file__).parent
    output_dir = proto_dir
    
    proto_files = [
        "aiva_enums.proto",
        "aiva_errors.proto",
        "aiva_services.proto",
    ]
    
    for proto_file in proto_files:
        proto_path = proto_dir / proto_file
        if not proto_path.exists():
            print(f"❌ Proto 文件不存在: {proto_file}")
            continue
            
        print(f"🔨 生成 {proto_file}...")
        
        try:
            # 生成 Python代碼
            cmd = [
                sys.executable,
                "-m",
                "grpc_tools.protoc",
                f"--proto_path={proto_dir}",
                f"--python_out={output_dir}",
                f"--grpc_python_out={output_dir}",
                str(proto_path),
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                print(f"✅ {proto_file} 生成成功")
            else:
                print(f"❌ {proto_file} 生成失敗: {result.stderr}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 執行錯誤: {e}")
            print(f"   stdout: {e.stdout}")
            print(f"   stderr: {e.stderr}")
        except Exception as e:
            print(f"❌ 未預期錯誤: {e}")
    
    print("\n✅ Proto 生成完成！")
    print(f"   輸出目錄: {output_dir}")


if __name__ == "__main__":
    print("AIVA Protocol Buffers 生成器")
    print("=" * 50)
    generate_proto_files()
