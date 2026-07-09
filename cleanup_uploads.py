import os
import hashlib
import re

def get_file_hash(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def cleanup_uploads():
    uploads_dir = "uploads"
    if not os.path.exists(uploads_dir):
        print("Uploads directory not found.")
        return

    # Dictionary to keep track of seen hashes: hash -> file_path
    seen_hashes = {}
    deleted_files = 0

    # Sort files so that we process original files before duplicates (e.g. file.pdf before file_1.pdf)
    files = sorted(os.listdir(uploads_dir))
    
    for filename in files:
        if not filename.endswith(".pdf"):
            continue
            
        file_path = os.path.join(uploads_dir, filename)
        if not os.path.isfile(file_path):
            continue
            
        file_hash = get_file_hash(file_path)
        
        if file_hash in seen_hashes:
            print(f"Deleting duplicate file: {filename} (Duplicate of {os.path.basename(seen_hashes[file_hash])})")
            os.remove(file_path)
            deleted_files += 1
        else:
            seen_hashes[file_hash] = file_path

    print(f"\nCleanup complete. Deleted {deleted_files} duplicate physical files.")

if __name__ == "__main__":
    cleanup_uploads()
