#!/usr/bin/env python3
"""
Enhanced Environment Setup Script for DeepTrust Model
Handles all dependencies, environment checks, and system requirements
"""

import os
import sys
import subprocess
import platform
import importlib
import pkg_resources
from pathlib import Path
import json
import shutil

class DeepTrustEnvironmentSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.requirements_file = self.project_root / "requirements.txt"
        self.system_info = {
            'platform': platform.system(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0]
        }
        
    def print_status(self, message, status="INFO"):
        """Print colored status messages"""
        colors = {
            'INFO': '',     # Blue
            'SUCCESS': '',  # Green
            'WARNING': '',  # Yellow
            'ERROR': '',    # Red
            'RESET': ''      # Reset
        }
        print(f"{colors.get(status, '')}{message}{colors['RESET']}")
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        self.print_status("Checking Python version...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.print_status("ERROR: Python 3.8+ required. Current version: {}.{}.{}".format(
                version.major, version.minor, version.micro), "ERROR")
            return False
        
        self.print_status("SUCCESS: Python {}.{}.{} - Compatible".format(version.major, version.minor, version.micro), "SUCCESS")
        return True
    
    def check_system_requirements(self):
        """Check system-specific requirements"""
        self.print_status("Checking system requirements...")
        
        # Check for required system libraries
        system_checks = {
            'Windows': self.check_windows_requirements,
            'Linux': self.check_linux_requirements,
            'Darwin': self.check_macos_requirements
        }
        
        checker = system_checks.get(self.system_info['platform'])
        if checker:
            return checker()
        else:
            self.print_status(f"WARNING: Unsupported platform: {self.system_info['platform']}", "WARNING")
            return True  # Continue anyway
    
    def check_windows_requirements(self):
        """Windows-specific checks"""
        # Check for Visual C++ redistributables (needed for some packages)
        try:
            import ctypes
            ctypes.windll.msvcrt
            self.print_status("SUCCESS: Visual C++ runtime available", "SUCCESS")
        except:
            self.print_status("WARNING: Visual C++ runtime not detected", "WARNING")
        
        # Check for ffmpeg
        if not self.check_ffmpeg():
            self.print_status("WARNING: FFmpeg not found - audio extraction may fail", "WARNING")
            self.install_ffmpeg_windows()
        
        return True
    
    def check_linux_requirements(self):
        """Linux-specific checks"""
        # Check for essential libraries
        required_libs = ['libsndfile1', 'ffmpeg']
        for lib in required_libs:
            if not shutil.which(lib):
                self.print_status(f"WARNING: {lib} not found - please install: sudo apt-get install {lib}", "WARNING")
        return True
    
    def check_macos_requirements(self):
        """macOS-specific checks"""
        # Check for Homebrew and ffmpeg
        if not shutil.which('brew'):
            self.print_status("WARNING: Homebrew not found - install from https://brew.sh/", "WARNING")
        
        if not self.check_ffmpeg():
            self.print_status("WARNING: FFmpeg not found - install with: brew install ffmpeg", "WARNING")
        
        return True
    
    def check_ffmpeg(self):
        """Check if FFmpeg is available"""
        return shutil.which('ffmpeg') is not None
    
    def install_ffmpeg_windows(self):
        """Attempt to install FFmpeg on Windows"""
        self.print_status("Attempting to install FFmpeg...", "INFO")
        try:
            # Try to install via pip
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'ffmpeg-python'], 
                         check=True, capture_output=True)
            self.print_status("SUCCESS: FFmpeg-python installed", "SUCCESS")
        except subprocess.CalledProcessError:
            self.print_status("ERROR: Failed to install FFmpeg automatically", "ERROR")
            self.print_status("INFO: Please download FFmpeg from https://ffmpeg.org/download.html", "INFO")
    
    def setup_virtual_environment(self):
        """Create and setup virtual environment"""
        self.print_status("Setting up virtual environment...")
        
        if not self.venv_path.exists():
            try:
                subprocess.run([sys.executable, '-m', 'venv', str(self.venv_path)], 
                             check=True, capture_output=True)
                self.print_status("SUCCESS: Virtual environment created successfully", "SUCCESS")
            except subprocess.CalledProcessError as e:
                self.print_status(f"ERROR: Failed to create virtual environment: {e}", "ERROR")
                return False
        else:
            self.print_status("SUCCESS: Virtual environment already exists", "SUCCESS")
        
        return True
    
    def get_venv_python(self):
        """Get path to virtual environment Python executable"""
        if self.system_info['platform'] == 'Windows':
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def get_venv_pip(self):
        """Get path to virtual environment pip executable"""
        if self.system_info['platform'] == 'Windows':
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def upgrade_pip(self):
        """Upgrade pip in virtual environment"""
        self.print_status("Upgrading pip...", "INFO")
        try:
            venv_python = self.get_venv_python()
            subprocess.run([str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            self.print_status("SUCCESS: Pip upgraded successfully", "SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"WARNING: Failed to upgrade pip: {e}", "WARNING")
            return False
    
    def create_enhanced_requirements(self):
        """Create enhanced requirements.txt with version pinning"""
        enhanced_requirements = """# Core ML and Data Science
numpy>=1.21.0,<1.25.0
pandas>=1.3.0,<2.1.0
scikit-learn>=1.0.0,<1.4.0
scipy>=1.7.0,<1.12.0

# Audio Processing
librosa>=0.9.0,<0.11.0
soundfile>=0.10.0
audioread>=2.1.0

# Computer Vision
opencv-python>=4.5.0,<4.9.0
Pillow>=8.0.0,<10.1.0

# Machine Learning Models
xgboost>=1.5.0,<2.1.0
lightgbm>=3.2.0,<4.1.0
catboost>=1.0.0,<1.3.0

# Visualization
matplotlib>=3.5.0,<3.8.0
seaborn>=0.11.0,<0.13.0

# Utilities
joblib>=1.1.0,<1.4.0
tqdm>=4.62.0,<4.67.0
psutil>=5.8.0

# Video Processing
ffmpeg-python>=0.2.0
moviepy>=1.0.0,<1.1.0

# Imbalanced Learning
imbalanced-learn>=0.8.0,<0.12.0

# Additional ML Tools
optuna>=2.10.0,<3.5.0  # For hyperparameter optimization
shap>=0.40.0,<0.43.0   # For model interpretability
"""
        
        with open(self.requirements_file, 'w') as f:
            f.write(enhanced_requirements)
        
        self.print_status("SUCCESS: Enhanced requirements.txt created", "SUCCESS")
    
    def install_dependencies(self):
        """Install all required dependencies"""
        self.print_status("Installing dependencies...", "INFO")
        
        if not self.requirements_file.exists():
            self.create_enhanced_requirements()
        
        try:
            venv_python = self.get_venv_python()
            
            # Install wheel first for better compilation
            subprocess.run([str(venv_python), '-m', 'pip', 'install', 'wheel'], 
                         check=True, capture_output=True)
            
            # Install requirements with retry mechanism
            for attempt in range(3):
                try:
                    result = subprocess.run([
                        str(venv_python), '-m', 'pip', 'install', 
                        '-r', str(self.requirements_file),
                        '--timeout', '300'
                    ], check=True, capture_output=True, text=True)
                    
                    self.print_status("SUCCESS: All dependencies installed successfully", "SUCCESS")
                    return True
                    
                except subprocess.CalledProcessError as e:
                    if attempt < 2:
                        self.print_status(f"WARNING: Installation attempt {attempt + 1} failed, retrying...", "WARNING")
                        # Try installing problematic packages individually
                        self.install_problematic_packages()
                    else:
                        self.print_status(f"ERROR: Failed to install dependencies after 3 attempts: {e}", "ERROR")
                        self.print_status(f"Error output: {e.stderr}", "ERROR")
                        return False
        
        except Exception as e:
            self.print_status(f"ERROR: Unexpected error during installation: {e}", "ERROR")
            return False
    
    def install_problematic_packages(self):
        """Install commonly problematic packages individually"""
        problematic_packages = [
            'numpy',
            'scipy', 
            'librosa',
            'opencv-python',
            'catboost'
        ]
        
        venv_python = self.get_venv_python()
        
        for package in problematic_packages:
            try:
                subprocess.run([str(venv_python), '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                self.print_status(f"SUCCESS: {package} installed individually", "SUCCESS")
            except subprocess.CalledProcessError:
                self.print_status(f"WARNING: Failed to install {package} individually", "WARNING")
    
    def verify_installation(self):
        """Verify that all critical packages are properly installed"""
        self.print_status("Verifying installation...", "INFO")
        
        critical_packages = [
            'numpy', 'pandas', 'sklearn', 'librosa', 'cv2', 
            'xgboost', 'lightgbm', 'catboost', 'matplotlib'
        ]
        
        venv_python = self.get_venv_python()
        failed_imports = []
        
        for package in critical_packages:
            try:
                result = subprocess.run([
                    str(venv_python), '-c', f'import {package}; print("{package} OK")'
                ], check=True, capture_output=True, text=True)
                self.print_status(f"SUCCESS: {package} - OK", "SUCCESS")
            except subprocess.CalledProcessError:
                failed_imports.append(package)
                self.print_status(f"ERROR: {package} - FAILED", "ERROR")
        
        if failed_imports:
            self.print_status(f"ERROR: Failed to import: {', '.join(failed_imports)}", "ERROR")
            return False
        
        self.print_status("SUCCESS: All critical packages verified", "SUCCESS")
        return True
    
    def create_activation_script(self):
        """Create easy activation script"""
        if self.system_info['platform'] == 'Windows':
            script_content = f"""@echo off
echo Activating DeepTrust Environment...
call "{self.venv_path}\\Scripts\\activate.bat"
echo Environment activated! You can now run:
echo   python optimized_model_trainer.py
echo   python ultimate_deepfake_predictor.py
cmd /k
"""
            script_path = self.project_root / "activate_deeptrust.bat"
        else:
            script_content = f"""#!/bin/bash
echo "Activating DeepTrust Environment..."
source "{self.venv_path}/bin/activate"
echo "Environment activated! You can now run:"
echo "  python optimized_model_trainer.py"
echo "  python ultimate_deepfake_predictor.py"
bash
"""
            script_path = self.project_root / "activate_deeptrust.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        if self.system_info['platform'] != 'Windows':
            os.chmod(script_path, 0o755)
        
        self.print_status(f"SUCCESS: Activation script created: {script_path.name}", "SUCCESS")
    
    def create_system_info_file(self):
        """Create system information file for debugging"""
        info = {
            'setup_timestamp': str(Path(__file__).stat().st_mtime),
            'system_info': self.system_info,
            'python_executable': str(self.get_venv_python()),
            'project_root': str(self.project_root),
            'ffmpeg_available': self.check_ffmpeg()
        }
        
        with open(self.project_root / 'system_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        self.print_status("SUCCESS: System info saved", "SUCCESS")
    
    def run_setup(self):
        """Run complete setup process"""
        self.print_status("Starting DeepTrust Environment Setup", "INFO")
        self.print_status("=" * 50, "INFO")
        
        # Step 1: Check Python version
        if not self.check_python_version():
            return False
        
        # Step 2: Check system requirements
        self.check_system_requirements()
        
        # Step 3: Setup virtual environment
        if not self.setup_virtual_environment():
            return False
        
        # Step 4: Upgrade pip
        self.upgrade_pip()
        
        # Step 5: Install dependencies
        if not self.install_dependencies():
            return False
        
        # Step 6: Verify installation
        if not self.verify_installation():
            return False
        
        # Step 7: Create helper scripts
        self.create_activation_script()
        self.create_system_info_file()
        
        self.print_status("=" * 50, "SUCCESS")
        self.print_status("Setup completed successfully!", "SUCCESS")
        self.print_status("Use 'activate_deeptrust.bat' (Windows) or './activate_deeptrust.sh' (Linux/Mac) to activate", "INFO")
        
        return True

def main():
    """Main setup function"""
    setup = DeepTrustEnvironmentSetup()
    success = setup.run_setup()
    
    if not success:
        print("\nSetup failed. Please check the error messages above.")
        sys.exit(1)
    
    print("\nSetup completed successfully!")
    print("You can now run the DeepTrust detection system!")

if __name__ == "__main__":
    main()
