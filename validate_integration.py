#!/usr/bin/env python3
"""
AIVA Features Supplement v2 Integration Validation
驗證補充包功能整合的基本導入和配置
"""

import sys
from pathlib import Path
import traceback

# 添加專案根路徑到Python路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_ssrf_integration():
    """測試SSRF模組整合"""
    try:
        from services.features.function_ssrf.config.ssrf_config import SsrfConfig
        config = SsrfConfig()
        print("✅ SSRF Config: Import and initialization successful")
        print(f"   - Safe mode: {config.safe_mode}")
        print(f"   - Timeout: {config.request_timeout}s")
        return True
    except Exception as e:
        print("❌ SSRF Config: Import failed")
        print(f"   Error: {e}")
        return False

def test_idor_integration():
    """測試IDOR模組整合"""
    try:
        from services.features.function_idor.config.idor_config import IdorConfig
        config = IdorConfig()
        print("✅ IDOR Config: Import and initialization successful")
        print(f"   - Horizontal enabled: {config.horizontal_enabled}")
        print(f"   - Vertical enabled: {config.vertical_enabled}")
        return True
    except Exception as e:
        print("❌ IDOR Config: Import failed")
        print(f"   Error: {e}")
        return False

def test_sqli_integration():
    """測試SQLI模組整合"""
    try:
        from services.features.function_sqli.config.sqli_config import SqliConfig
        config = SqliConfig()
        print("✅ SQLI Config: Import and initialization successful")
        print(f"   - Error detection: {config.enable_error_detection}")
        print(f"   - Timeout: {config.timeout_seconds}s")
        return True
    except Exception as e:
        print("❌ SQLI Config: Import failed")
        print(f"   Error: {e}")
        return False

def test_engine_imports():
    """測試核心引擎導入"""
    engines = [
        ("SSRF Engine", "services.features.function_ssrf.engine.ssrf_engine", "SSRFEngine"),
        ("IDOR Engine", "services.features.function_idor.engine.idor_engine", "IDOREngine"),
    ]
    
    results = []
    for name, module_path, class_name in engines:
        try:
            module = __import__(module_path, fromlist=[class_name])
            engine_class = getattr(module, class_name)
            print(f"✅ {name}: Import successful")
            results.append(True)
        except Exception as e:
            print(f"❌ {name}: Import failed - {e}")
            results.append(False)
    
    return all(results)

def main():
    """主驗證函數"""
    print("🧪 AIVA Features Supplement v2 - Integration Validation")
    print("=" * 60)
    print(f"📍 Project root: {project_root}")
    print(f"🐍 Python version: {sys.version.split()[0]}")
    print()
    
    tests = [
        ("Configuration Modules", [
            test_ssrf_integration,
            test_idor_integration,  
            test_sqli_integration
        ]),
        ("Core Engines", [test_engine_imports])
    ]
    
    overall_success = True
    
    for category, test_functions in tests:
        print(f"🔍 Testing {category}:")
        print("-" * 40)
        
        category_results = []
        for test_func in test_functions:
            try:
                result = test_func()
                category_results.append(result)
            except Exception as e:
                print(f"❌ Test function failed: {e}")
                traceback.print_exc()
                category_results.append(False)
        
        category_success = all(category_results)
        overall_success = overall_success and category_success
        
        print()
    
    # 總結
    print("📊 Validation Summary:")
    print("=" * 60)
    
    if overall_success:
        print("🎉 All integration tests passed!")
        print("✅ Features Supplement v2 integration is ready for deployment")
    else:
        print("⚠️  Some integration tests failed")
        print("🔧 Please check the error messages above and fix the issues")
    
    print()
    print("🔗 Next steps:")
    print("1. Build Docker images: .\\scripts\\features\\build_docker_images.ps1")
    print("2. Start services: docker-compose -f docker-compose.features.yml up -d")
    print("3. Test workers: .\\scripts\\features\\test_workers.ps1")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())