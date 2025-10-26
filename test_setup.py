#!/usr/bin/env python3
"""
Test script to verify TAVIS.SciLens setup
"""
import sys
import importlib

def test_imports():
    """Test if all required packages are installed."""
    required_packages = {
        'flask': 'Flask',
        'flask_socketio': 'Flask-SocketIO',
        'cv2': 'OpenCV',
        'google.generativeai': 'Google Generative AI',
        'dotenv': 'python-dotenv',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'eventlet': 'eventlet'
    }
    
    print("Testing package imports...")
    missing = []
    
    for module, name in required_packages.items():
        try:
            importlib.import_module(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed successfully!")
    return True

def test_environment():
    """Test if environment variables are set."""
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or api_key == 'your_api_key_here':
        print("⚠ Warning: GEMINI_API_KEY not set in .env file")
        return False
    
    print("✓ GEMINI_API_KEY is configured")
    return True

def test_opencv():
    """Test OpenCV ArUco functionality."""
    try:
        import cv2
        
        # Check if opencv-contrib is installed (needed for ArUco)
        try:
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            print("✓ OpenCV ArUco module working")
            return True
        except AttributeError:
            print("✗ OpenCV ArUco not available")
            print("  Install: pip install opencv-contrib-python")
            return False
            
    except ImportError as e:
        print(f"✗ OpenCV error: {e}")
        return False

def main():
    print("=" * 50)
    print("TAVIS.SciLens Setup Verification")
    print("=" * 50)
    print()
    
    success = True
    
    # Test imports
    if not test_imports():
        success = False
    
    print()
    
    # Test environment
    if not test_environment():
        success = False
    
    print()
    
    # Test OpenCV
    if not test_opencv():
        success = False
    
    print()
    print("=" * 50)
    
    if success:
        print("✓ All tests passed! Ready to run the app.")
        print("\nTo start the app, run: python app.py")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())

