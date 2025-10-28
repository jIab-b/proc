#!/usr/bin/env python3

import paramiko
import sys
import select
import argparse
import os
import hashlib

def compute_file_hash(file_path):
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error computing hash for {file_path}: {e}")
        return None

def get_remote_file_hash(sftp, remote_path):
    try:
        with sftp.file(remote_path, 'r') as f:
            hash_md5 = hashlib.md5()
            while True:
                chunk = f.read(4096)
                if not chunk:
                    break
                hash_md5.update(chunk)
            return hash_md5.hexdigest()
    except Exception:
        return None

def should_exclude_file(file_path, exclusions, check_global=True):
    file_name = os.path.basename(file_path)

    # Check global exclusions first
    if check_global and 'global' in exclusions:
        global_exclusions = exclusions['global']
        if file_name in global_exclusions.get('files', []):
            return True
        for dir_pattern in global_exclusions.get('dirs', []):
            if dir_pattern in file_path:
                return True
        file_ext = os.path.splitext(file_name)[1]
        if file_ext in global_exclusions.get('extensions', []):
            return True
        for pattern in global_exclusions.get('patterns', []):
            if pattern in file_name:
                return True

    # Check specific exclusions
    if file_name in exclusions.get('files', []):
        return True

    for dir_pattern in exclusions.get('dirs', []):
        if dir_pattern in file_path:
            return True

    file_ext = os.path.splitext(file_name)[1]
    if file_ext in exclusions.get('extensions', []):
        return True

    for pattern in exclusions.get('patterns', []):
        if pattern in file_name:
            return True

    return False

def ensure_remote_dir_exists(sftp, remote_path):
    try:
        sftp.stat(remote_path)
    except IOError:
        parent = remote_path.rsplit('/', 1)[0]
        if parent and parent != '/':
            ensure_remote_dir_exists(sftp, parent)
        try:
            sftp.mkdir(remote_path)
        except IOError:
            pass

def sync_files_to_remote(host, key_path, dirs_to_sync, remote_base_dir):
    exclusions = {
        'webgpu/': {
            'dirs': ['node_modules', 'dist', '.git', '__pycache__', '.vscode'],
            'files': ['package-lock.json', '.DS_Store', 'yarn.lock', 'pnpm-lock.yaml'],
            'extensions': ['.log', '.tmp', '.swp', '.bak'],
            'patterns': ['.min.', 'build.', 'dist.']
        },
        'global': {
            'files': ['.env', '.env.example']
        }
    }

    try:
        print(f"Syncing files to {host}:{remote_base_dir}")

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        private_key = paramiko.RSAKey.from_private_key_file(key_path)

        client.connect(hostname=host, username='root', pkey=private_key)

        sftp = client.open_sftp()

        ensure_remote_dir_exists(sftp, remote_base_dir)
        print(f"Remote base directory ready: {remote_base_dir}")

        def upload_dir(local_path, remote_path, dir_key):
            try:
                items = os.listdir(local_path)
            except PermissionError:
                print(f"Skipping {local_path} (permission denied)")
                return

            ensure_remote_dir_exists(sftp, remote_path)

            for item in items:
                local_item = os.path.join(local_path, item)
                remote_item = os.path.join(remote_path, item)

                if should_exclude_file(local_item, exclusions.get(dir_key, {})):
                    print(f"Excluding: {local_item}")
                    continue

                try:
                    if os.path.isdir(local_item):
                        upload_dir(local_item, remote_item, dir_key)
                    else:
                        local_hash = compute_file_hash(local_item)
                        remote_hash = get_remote_file_hash(sftp, remote_item)
                        
                        if local_hash != remote_hash:
                            print(f"Uploading (hash mismatch): {local_item} -> {remote_item}")
                            sftp.put(local_item, remote_item)
                        else:
                            print(f"Skipping (hash match): {local_item}")
                except Exception as e:
                    print(f"Error processing {local_item}: {e}")

        for item_to_sync in dirs_to_sync:
            local_item_path = os.path.join(os.getcwd(), item_to_sync)
            remote_item_path = os.path.join(remote_base_dir, item_to_sync.rstrip('/'))

            if os.path.isfile(local_item_path):
                if should_exclude_file(local_item_path, exclusions):
                    print(f"Excluding: {local_item_path}")
                    continue

                print(f"\nSyncing file: {item_to_sync}")
                print(f"Local: {local_item_path}")
                print(f"Remote: {remote_item_path}")
                try:
                    local_hash = compute_file_hash(local_item_path)
                    remote_hash = get_remote_file_hash(sftp, remote_item_path)
                    if local_hash != remote_hash:
                        print(f"Uploading (hash mismatch): {local_item_path} -> {remote_item_path}")
                        sftp.put(local_item_path, remote_item_path)
                    else:
                        print(f"Skipping (hash match): {local_item_path}")
                except Exception as e:
                    print(f"Error processing {local_item_path}: {e}")
            elif os.path.isdir(local_item_path):
                print(f"\nSyncing directory: {item_to_sync}")
                print(f"Local: {local_item_path}")
                print(f"Remote: {remote_item_path}")
                upload_dir(local_item_path, remote_item_path, item_to_sync)
            else:
                print(f"Warning: {local_item_path} does not exist, skipping")

        sftp.close()
        client.close()

        print("\nFile sync completed successfully!")

    except Exception as e:
        print(f"Sync failed: {e}")
        sys.exit(1)

def connect_to_server(host, key_path):
    try:
        client = paramiko.SSHClient()

        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        private_key = paramiko.RSAKey.from_private_key_file(key_path)

        print(f"Connecting to {host}...")
        client.connect(hostname=host, username='root', pkey=private_key)

        shell = client.invoke_shell()
        print("Interactive shell opened. Type 'exit' to close connection.")

        while True:
            if shell.recv_ready():
                output = shell.recv(1024).decode()
                print(output, end='')

            if select.select([sys.stdin], [], [], 0.1)[0]:
                user_input = sys.stdin.readline()
                if user_input.strip().lower() == 'exit':
                    break
                shell.send(user_input)

        client.close()
        print("Connection closed.")

    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    dirs_to_sync = ['webgpu/', 'main.py', 'aio.py', 'requirements_remote.txt', 'vps/']

    parser = argparse.ArgumentParser(description='SSH connection and file sync tool')
    parser.add_argument('--sync_proc', action='store_true',
                       help='Sync specified items to remote ~/proc/ directory')
    parser.add_argument('--host', default="209.38.93.185",
                       help='Remote host IP address (default: 209.38.93.185)')
    parser.add_argument('--key', default="/home/beed/.ssh/runpodprivate",
                       help='Path to SSH private key')

    args = parser.parse_args()

    if args.sync_proc:
        remote_proc_dir = "/root/proc/"
        sync_files_to_remote(args.host, args.key, dirs_to_sync, remote_proc_dir)
    else:
        connect_to_server(args.host, args.key)
