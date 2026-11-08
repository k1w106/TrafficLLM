"""Tạo GitHub Release và upload các artifact nặng (checkpoint, feature cache).

Ví dụ:

    python3 scripts/publish_release.py            # hỏi token, dùng mặc định bên dưới
    GITHUB_TOKEN=ghp_xxx python3 scripts/publish_release.py --dry-run

Vì sao cần script: GitHub giới hạn 100 MB/file trong git nhưng cho tới 2 GB/file
ở Release. Hai file dưới đây (~1 GB) chỉ đi qua Release, không bao giờ vào git.

Token cần scope `repo` (classic) hoặc quyền `Contents: write` (fine-grained).
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import sys
from pathlib import Path

import requests

DEFAULT_REPO = "k1w106/TrafficLLM"
DEFAULT_TAG = "p3-artifacts"
DEFAULT_TITLE = "P3 artifacts: checkpoint + feature cache"
DEFAULT_FILES = (
    "outputs/checkpoints/ustc_gpt2_seed42.pt",
    "outputs/features/ustc_seed42.npz",
)
API = "https://api.github.com"
CHUNK = 8 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    parser.add_argument("--replace", action="store_true", help="upload đè asset đã có")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in việc sẽ làm")
    return parser.parse_args()


def get_token() -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        return token.strip()
    return getpass.getpass("GitHub token (scope repo, không hiện ra màn hình): ").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_release(session: requests.Session, repo: str, tag: str, title: str) -> dict:
    """Trả về release có tag `tag`; tạo mới nếu chưa có."""
    response = session.get(f"{API}/repos/{repo}/releases/tags/{tag}", timeout=30)
    if response.status_code == 200:
        release = response.json()
        print(f"Release đã tồn tại: {release['html_url']}")
        return release

    response = session.post(
        f"{API}/repos/{repo}/releases",
        json={"tag_name": tag, "name": title, "body": "", "draft": False, "prerelease": False},
        timeout=60,
    )
    if response.status_code not in (200, 201):
        raise SystemExit(f"Tạo release thất bại ({response.status_code}): {response.text}")
    release = response.json()
    print(f"Đã tạo release: {release['html_url']}")
    return release


def upload_asset(
    session: requests.Session, repo: str, release: dict, path: Path, replace: bool
) -> str:
    name = path.name
    for asset in release.get("assets", []):
        if asset["name"] != name:
            continue
        if not replace:
            print(f"  bỏ qua {name} (đã có). Dùng --replace để upload đè.")
            return asset["browser_download_url"]
        session.delete(f"{API}/repos/{repo}/releases/assets/{asset['id']}", timeout=30)
        print(f"  đã xoá asset cũ {name}")

    total = path.stat().st_size
    url = release["upload_url"].split("{")[0]
    session.headers.update({"Content-Type": "application/octet-stream"})

    def stream():
        sent = 0
        last = -10
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(CHUNK), b""):
                sent += len(block)
                percent = int(sent * 100 / total)
                if percent >= last + 10 or sent == total:
                    print(f"  {name}: {percent}% ({sent / 2**20:.0f}/{total / 2**20:.0f} MB)", flush=True)
                    last = percent
                yield block

    response = session.post(f"{url}?name={name}", data=stream(), timeout=(30, 3600))
    if response.status_code not in (200, 201):
        raise SystemExit(f"Upload {name} thất bại ({response.status_code}): {response.text}")
    print(f"  xong {name}")
    return response.json()["browser_download_url"]


def main() -> None:
    args = parse_args()
    paths = [Path(item) for item in args.files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("Không thấy file: " + ", ".join(missing))

    print("Sẽ upload:")
    for path in paths:
        print(f"  {path}  ({path.stat().st_size / 2**20:.0f} MB)  sha256={sha256(path)}")
    if args.dry_run:
        return

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {get_token()}", "Accept": "application/vnd.github+json"})
    release = ensure_release(session, args.repo, args.tag, args.title)
    for path in paths:
        url = upload_asset(session, args.repo, release, path, replace=args.replace)
        print(f"  {url}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as error:
        sys.exit(f"Lỗi mạng: {error}")
