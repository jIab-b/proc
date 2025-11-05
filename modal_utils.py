import os, io, zipfile, subprocess, shutil, hashlib, fnmatch
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Iterable
from concurrent.futures import ThreadPoolExecutor
import modal
from modal import Image, Volume

DEFAULT_VOLUME_NAME = os.environ.get("MODAL_VOLUME", "workspace")
DEFAULT_REMOTE_ROOT = os.environ.get("MODAL_REMOTE_ROOT", "/workspace")
DEFAULT_HF_HOME = os.environ.get("HF_HOME", f"{DEFAULT_REMOTE_ROOT}/hf")
class _Defer:
    def __init__(self, name: str):
        self.name = name

class _AppProxy:
    def __init__(self):
        self._app = None
    def bind(self, app_obj):
        self._app = app_obj
        return self
    def function(self, *dargs, **dkwargs):
        def decorator(fn):
            if self._app is None:
                return fn
            resolved = {}
            for k, v in dkwargs.items():
                if isinstance(v, _Defer):
                    v = globals().get(v.name)
                if k == "volumes" and isinstance(v, dict):
                    nv = {}
                    for mp, vol in v.items():
                        if isinstance(vol, _Defer):
                            vol = globals().get(vol.name)
                        nv[mp] = vol
                    if any(val is None for val in nv.values()):
                        v = None
                    else:
                        v = nv
                if v is None:
                    continue
                resolved[k] = v
            return self._app.function(*dargs, **resolved)(fn)
        return decorator

app = _AppProxy()
_image = None
_vol = None
DEFER_IMAGE = _Defer("_image")
DEFER_VOLUME = _Defer("_vol")


def _iter_files(base_path: str, exclude: Optional[List[str]]) -> Iterable[Tuple[str, str]]:
    base_path = os.path.abspath(base_path)
    if os.path.isfile(base_path):
        rel = os.path.basename(base_path)
        if not _excluded(rel, exclude):
            yield rel.replace("\\", "/"), base_path
        return
    for root, _, files in os.walk(base_path):
        for name in files:
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, base_path).replace("\\", "/")
            if _excluded(rel_path, exclude):
                continue
            yield rel_path, abs_path


def _excluded(rel_path: str, exclude: Optional[List[str]]) -> bool:
    if not exclude:
        return False
    for pat in exclude:
        if fnmatch.fnmatch(rel_path, pat):
            return True
    return False


def _hash_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def compute_hashes(path: str, exclude: Optional[List[str]] = None) -> Dict[str, str]:
    pairs = list(_iter_files(path, exclude))
    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futs = {ex.submit(_hash_file, ap): rp for rp, ap in pairs}
        for fut in futs:
            results[futs[fut]] = fut.result()
    return results


@app.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME})
def get_remote_hashes_remote(remote_path: str) -> Dict[str, str]:
    if not os.path.exists(remote_path):
        return {}
    if os.path.isfile(remote_path):
        return {os.path.basename(remote_path): _hash_file(remote_path)}
    out: Dict[str, str] = {}
    for root, _, files in os.walk(remote_path):
        for name in files:
            p = os.path.join(root, name)
            rel = os.path.relpath(p, remote_path).replace("\\", "/")
            out[rel] = _hash_file(p)
    return out


@app.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME})
def zip_remote_dir_remote(remote_dir: str) -> Optional[bytes]:
    if not os.path.exists(remote_dir):
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(remote_dir):
            for name in files:
                p = os.path.join(root, name)
                arc = os.path.relpath(p, remote_dir)
                zf.write(p, arc)
    buf.seek(0)
    return buf.read()


@app.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME})
def run_script_remote(script_path: str, venv: Optional[str] = None, workdir: str = DEFAULT_REMOTE_ROOT, gpu: Optional[str] = None) -> str:
    import subprocess as sp
    os.chdir(workdir)
    if venv:
        act = venv if venv.endswith("/bin/activate") else os.path.join(venv, "bin/activate")
        cmd = f"source {act} && python {script_path}"
    else:
        cmd = f"python {script_path}"
    sp.run(cmd, shell=True, check=True, executable="/bin/bash")
    return workdir


@app.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME})
def prefetch_hf_remote(repos: List, hf_home: str = DEFAULT_HF_HOME) -> str:
    from pathlib import Path as _P
    from huggingface_hub import snapshot_download
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_home)
    _P(hf_home).mkdir(parents=True, exist_ok=True)
    for item in repos:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            repo_id = item[0]
            alias = item[1] if len(item) > 1 else None
        else:
            repo_id = str(item)
            alias = None
        name = alias or repo_id.replace("/", "--")
        dest = os.path.join(hf_home, name)
        _P(dest).mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=dest, local_dir_use_symlinks=False)
    return hf_home


def _volume_cli(args: List[str]) -> None:
    subprocess.run(["modal", "volume", *args], check=True)


def sync_workspace(paths: List[str], exclude: Optional[List[str]] = None, volume_name: str = DEFAULT_VOLUME_NAME, remote_root: str = DEFAULT_REMOTE_ROOT) -> None:
    for src in paths:
        src_abs = os.path.abspath(src)
        if not os.path.exists(src_abs):
            continue
        is_dir = os.path.isdir(src_abs)
        base = os.path.basename(src_abs.rstrip(os.sep)) if is_dir else os.path.basename(src_abs)
        local_hashes = compute_hashes(src_abs, exclude)
        remote_target = os.path.join(remote_root, base) if is_dir else os.path.join(remote_root, base)
        remote_hashes = get_remote_hashes_remote.remote(remote_target)
        files_to_upload: List[Tuple[str, str]] = []
        for rel, h in local_hashes.items():
            if rel not in remote_hashes or remote_hashes[rel] != h:
                local_file = os.path.join(src_abs, rel) if is_dir else src_abs
                remote_file = os.path.join(base, rel) if is_dir else base
                files_to_upload.append((local_file, remote_file.replace("\\", "/")))
        for local_file, remote_file in files_to_upload:
            subprocess.run(["modal", "volume", "rm", volume_name, remote_file], check=False)
            _volume_cli(["put", volume_name, local_file, remote_file])


def sync_outputs(local_dir: str, remote_dir: str = f"{DEFAULT_REMOTE_ROOT}/out_local") -> None:
    local_path = Path(local_dir).expanduser().resolve()
    remote_hashes = get_remote_hashes_remote.remote(remote_dir)
    if not remote_hashes:
        if local_path.exists():
            shutil.rmtree(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        return
    local_hashes = compute_hashes(str(local_path)) if local_path.exists() else {}
    if local_hashes == remote_hashes:
        return
    zip_bytes = zip_remote_dir_remote.remote(remote_dir)
    if local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(local_path)


def run_scripts(script_path: str, venv: Optional[str] = None, workdir: str = DEFAULT_REMOTE_ROOT, gpu: Optional[str] = None) -> str:
    fn = run_script_remote
    if gpu and hasattr(fn, "options"):
        try:
            return fn.options(gpu=gpu).remote(script_path, venv, workdir, gpu)
        except Exception:
            pass
    return fn.remote(script_path, venv, workdir, gpu)


def prefetch_hf_models(repos: List, hf_home: str = DEFAULT_HF_HOME) -> str:
    return prefetch_hf_remote.remote(repos, hf_home)


