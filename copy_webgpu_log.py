import os
import shutil

def find_windows_downloads():
    candidates = [
        '/mnt/c/Users',
        '/mnt/c/Users/Public'
    ]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        for user in os.listdir(base):
            d = os.path.join(base, user, 'Downloads')
            if os.path.isdir(d):
                p = os.path.join(d, 'webgpu-log.txt')
                if os.path.isfile(p):
                    return p
    return None

def main():
    src = find_windows_downloads()
    if not src:
        print('webgpu-log.txt not found in any Windows Downloads directory')
        return
    dst = os.path.join(os.path.dirname(__file__), 'webgpu-log.txt')
    shutil.copy2(src, dst)
    print(f'Copied to {dst}')

if __name__ == '__main__':
    main()


