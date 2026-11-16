"""One-shot root-only volume initialization; application services run as UID10001."""

import os
from pathlib import Path


def main() -> None:
    root = Path(os.environ.get("LOCAL_STORAGE_DIR", "/app/data/uploads"))
    if not root.is_absolute() or root == Path("/"):
        raise ValueError("Storage directory must be an absolute, dedicated directory")
    root.mkdir(parents=True, exist_ok=True)
    os.chown(root, 10001, 10001)
    os.chmod(root, 0o700)


if __name__ == "__main__":
    main()
