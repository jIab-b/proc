import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-ssh', action='store_true')
parser.add_argument('--sync_workspace', action='store_true')
args = parser.parse_args()

host = 'root@134.199.152.240'
key = '~/.ssh/runpodprivate'
port = '22'
remote_dir = '/root/proc'

if args.ssh:
    subprocess.run(['ssh', '-i', key, '-p', port, host])

if args.sync_workspace:
    subprocess.run(['ssh', '-i', key, '-p', port, host, f'mkdir -p {remote_dir}'])
    rsync_cmd = [
        'rsync', '-avz',
        '-e', f'ssh -i {key} -p {port}',
        '--exclude=node_modules',
        '--exclude=dist',
        '--exclude=connect.py',
        '--exclude=copy_webgpu_log.py',
        '.',
        f'{host}:{remote_dir}/'
    ]
    subprocess.run(rsync_cmd)
