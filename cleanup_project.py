import os
import shutil
from pathlib import Path

def cleanup_project():
    print("Cleaning up project...")
    
    # Create backup directory if it doesn't exist
    backup_dir = Path("backup")
    backup_dir.mkdir(exist_ok=True)
    
    # Files to keep
    keep_files = {
        'final_deepfake_predictor.py',
        'optimized_model_trainer.py',
        'test_predictor.py',
        'cleanup_project.py',
        'requirements.txt',
        'setup_environment.py',
        'run_enhanced_detection.py',
        'enhanced_lip_sync.py',
        'config.py'
    }
    
    # Move unnecessary files to backup
    for item in Path('.').glob('*'):
        if item.is_file() and item.name not in keep_files and item.suffix in ['.py', '.csv', '.pkl', '.npy', '.json']:
            print(f"Moving {item.name} to backup/")
            shutil.move(str(item), str(backup_dir / item.name))
    
    # Clean up directories
    for dir_name in ['__pycache__', 'lip_frames']:
        if Path(dir_name).exists():
            print(f"Removing directory: {dir_name}")
            shutil.rmtree(dir_name)
    
    print("\nCleanup complete!")
    print("Key files have been kept, and unnecessary files moved to backup/")
    print("You can now run: python setup_environment.py")

if __name__ == "__main__":
    cleanup_project()
