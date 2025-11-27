#!/usr/bin/env python3
"""Test script to verify OSWorld integration and system setup"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
            print(f"    GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    except Exception as e:
        print(f"  ✗ PyTorch: {e}")
        return False
    
    try:
        from desktop_env.desktop_env import DesktopEnv
        print("  ✓ OSWorld desktop_env")
    except Exception as e:
        print(f"  ✗ OSWorld: {e}")
        return False
    
    try:
        from src.api.config import settings
        print("  ✓ Configuration loaded")
        print(f"    OSWorld path: {settings.OSWORLD_REPO_PATH}")
        print(f"    Provider: {settings.OSWORLD_PROVIDER}")
        print(f"    Device: {settings.DEVICE}")
    except Exception as e:
        print(f"  ✗ Configuration: {e}")
        return False
    
    try:
        from src.utils import SimpleMetricsTracker
        print("  ✓ SimpleMetricsTracker")
    except Exception as e:
        print(f"  ✗ SimpleMetricsTracker: {e}")
        return False
    
    try:
        from src.utils.visualizer import TrainingVisualizer
        print("  ✓ TrainingVisualizer")
    except Exception as e:
        print(f"  ✗ TrainingVisualizer: {e}")
        return False
    
    return True

def test_osworld_integration():
    """Test OSWorld integration"""
    print("\n🔍 Testing OSWorld integration...")
    
    try:
        from src.environment.osworld_integration import OSWorldManager
        print("  ✓ OSWorldManager imported")
        
        # Just check initialization, don't start environment
        manager = OSWorldManager()
        print("  ✓ OSWorldManager initialized")
        print(f"    OSWorld path: {manager.osworld_path}")
        
        return True
    except Exception as e:
        print(f"  ✗ OSWorld integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_docker():
    """Test Docker availability"""
    print("\n🔍 Testing Docker...")
    
    try:
        import docker
        client = docker.from_env()
        info = client.info()
        print("  ✓ Docker is running")
        print(f"    Version: {info.get('ServerVersion', 'unknown')}")
        print(f"    Containers: {info.get('Containers', 0)}")
        return True
    except Exception as e:
        print(f"  ✗ Docker: {e}")
        print("    Make sure Docker is running: sudo systemctl start docker")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🚀 OSWorld Integration Test")
    print("=" * 60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test Docker
    results.append(("Docker", test_docker()))
    
    # Test OSWorld integration
    results.append(("OSWorld Integration", test_osworld_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✅ All tests passed! Your system is ready.")
        print("\n📝 Next steps:")
        print("  1. Configure your .env file if needed (especially OPENAI_API_KEY)")
        print("  2. Start training: python src/training/train.py --visualize")
        print("  3. Or run the API: uvicorn src.api.main:app --reload")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
