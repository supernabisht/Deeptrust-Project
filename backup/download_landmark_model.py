import os
import urllib.request
import bz2

def download_file(url, filename):
    """Download a file from URL with progress bar"""
    print(f"Downloading {filename}...")
    try:
        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(int(downloaded * 100 / total_size), 100)
            print(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end="")
            
        urllib.request.urlretrieve(url, filename, progress)
        print("\nDownload completed!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def extract_bz2(filename):
    """Extract a .bz2 file"""
    print(f"Extracting {filename}...")
    try:
        with bz2.BZ2File(filename) as fr, open(filename[:-4], 'wb') as fw:
            fw.write(fr.read())
        print("Extraction completed!")
        return True
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False

def main():
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    bz2_filename = "shape_predictor_68_face_landmarks.dat.bz2"
    dat_filename = "shape_predictor_68_face_landmarks.dat"
    
    # Check if file already exists
    if os.path.exists(dat_filename):
        print(f"{dat_filename} already exists!")
        return
        
    # Download the file
    if not os.path.exists(bz2_filename):
        if not download_file(model_url, bz2_filename):
            return
    
    # Extract the file
    if not os.path.exists(dat_filename):
        if not extract_bz2(bz2_filename):
            return
    
    # Clean up the .bz2 file
    try:
        os.remove(bz2_filename)
        print(f"Removed {bz2_filename}")
    except Exception as e:
        print(f"Warning: Could not remove {bz2_filename}: {e}")
    
    print("\nSetup complete! You can now run the enhanced detection.")

if __name__ == "__main__":
    main()
