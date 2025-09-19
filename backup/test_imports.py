import sys
import os

print("Testing imports...")

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Current directory: {current_dir}")
sys.path.insert(0, current_dir)

try:
    import extract_audio as local_audio
    print("✅ Successfully imported extract_audio")
    print(f"Module location: {os.path.abspath(local_audio.__file__)}")
    
    # Test the function exists
    if hasattr(local_audio, 'extract_audio'):
        print("✅ extract_audio function found")
    else:
        print("❌ extract_audio function not found")
        print("Available functions:", dir(local_audio))
        
except Exception as e:
    print(f"❌ Error importing extract_audio: {str(e)}")
    print("Current sys.path:")
    for p in sys.path:
        print(f"  - {p}")
        if not os.path.exists(p):
            print("    (Path does not exist!)")
