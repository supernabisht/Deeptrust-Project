#!/usr/bin/env python3
"""
Unicode Fix Script for DeepTrust Project
Systematically removes all Unicode characters causing Windows encoding errors
"""

import os
import re
import codecs

def fix_unicode_in_file(filepath):
    """Fix Unicode characters in a single file"""
    try:
        # Read the file with UTF-8 encoding
        with codecs.open(filepath, 'r', 'utf-8') as f:
            content = f.read()
        
        # Dictionary of Unicode characters to replace
        unicode_replacements = {
            # Emojis and symbols
            '🚀': 'Starting',
            '✅': 'SUCCESS:',
            '❌': 'ERROR:',
            '⚠️': 'WARNING:',
            '🔍': 'Checking',
            '🎬': 'Video',
            '🎵': 'Audio',
            '🔧': 'Processing',
            '📊': 'Report',
            '🎉': 'Completed',
            '🐍': 'Python',
            '🖥️': 'System',
            '🏗️': 'Building',
            '📦': 'Package',
            '💡': 'Info',
            '🤖': 'AI',
            '📋': 'Summary',
            '📹': 'Video',
            '👄': 'Lip',
            '🎯': 'Target',
            '🌟': 'Feature',
            '🔥': 'Hot',
            '💻': 'Computer',
            '⭐': 'Star',
            '🎪': 'Show',
            '🎭': 'Performance',
            '🎨': 'Art',
            '🎤': 'Microphone',
            '🎧': 'Audio',
            '🎼': 'Music',
            '🎹': 'Piano',
            '🥁': 'Drums',
            '🎺': 'Trumpet',
            '🎸': 'Guitar',
            '🎻': 'Violin',
            '🎷': 'Saxophone',
            '📁': 'Folder',
            
            # Unicode escape sequences
            '\\U0001f680': 'Starting',
            '\\U0001f3ac': 'Video',
            '\\u2705': 'SUCCESS:',
            '\\u274c': 'ERROR:',
            '\\u26a0\\ufe0f': 'WARNING:',
            '\\u26a0': 'WARNING:',
            '\\ufe0f': '',
            '\\u2705': 'SUCCESS:',
            '\\u274c': 'ERROR:',
            '\\u1f680': 'Starting',
            '\\u1f3ac': 'Video',
        }
        
        # Apply replacements
        modified = False
        for unicode_char, replacement in unicode_replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, replacement)
                modified = True
                print(f"  Replaced '{unicode_char}' with '{replacement}'")
        
        # Remove any remaining non-ASCII characters in print statements
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'print(' in line or 'print "' in line:
                # Check for non-ASCII characters
                if any(ord(c) > 127 for c in line):
                    # Replace non-ASCII characters with safe alternatives
                    new_line = ''.join(c if ord(c) <= 127 else '?' for c in line)
                    if new_line != line:
                        lines[i] = new_line
                        modified = True
                        print(f"  Fixed non-ASCII in line {i+1}")
        
        if modified:
            content = '\n'.join(lines)
            # Write back with UTF-8 encoding
            with codecs.open(filepath, 'w', 'utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function to fix Unicode in all Python files"""
    print("Starting Unicode fix for DeepTrust project...")
    
    # List of Python files to fix
    python_files = [
        'lip_sync.py',
        'ultimate_deepfake_predictor.py', 
        'setup_environment.py',
        'run_enhanced_detection.py',
        'optimized_model_trainer.py',
        'enhanced_predictor.py',
        'extract_frames.py',
        'extract_audio.py',
        'extract_mfcc.py',
        'advanced_threshold_calibrator.py'
    ]
    
    fixed_files = []
    
    for filename in python_files:
        if os.path.exists(filename):
            print(f"\nProcessing {filename}...")
            if fix_unicode_in_file(filename):
                fixed_files.append(filename)
                print(f"  ✓ Fixed Unicode issues in {filename}")
            else:
                print(f"  - No Unicode issues found in {filename}")
        else:
            print(f"  ! File not found: {filename}")
    
    print(f"\nUnicode fix completed!")
    print(f"Fixed {len(fixed_files)} files: {', '.join(fixed_files)}")
    
    if fixed_files:
        print("\nAll Unicode characters have been replaced with ASCII-safe alternatives.")
        print("The pipeline should now run without Windows encoding errors.")
    else:
        print("\nNo Unicode issues were found in the specified files.")

if __name__ == "__main__":
    main()
