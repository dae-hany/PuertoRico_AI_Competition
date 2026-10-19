"""Draw the private seeds of a course's final run, and keep them out of the repository.

    python tools/make_private_seeds.py --count 475 --exclude 1-25 \
        --out C:/course_2p/private_seeds.txt

The seeds are drawn with ``secrets`` (not ``random``, whose sequence can be
reconstructed) from a wide range, 1 to 2**31 - 1, without repeats and without
the ``--exclude`` seeds (the public ladder seeds). The file holds one seed per
line, which ``tools/course_ladder.py --seeds-file`` reads.

The script refuses to write inside this repository: a seed list that ends up in
a commit is public for good. It prints the SHA-256 of the file; noting it down
before the run lets the organizer show later that the list was fixed in advance.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

from course_ladder import inside_repo, parse_seeds  # noqa: E402

LOW, HIGH = 1, 2**31 - 1


def draw(count, exclude=(), low=LOW, high=HIGH):
    """``count`` distinct seeds in ``[low, high]``, none of them in ``exclude``, sorted."""
    banned = set(exclude)
    room = high - low + 1 - sum(1 for s in banned if low <= s <= high)
    if count > room:
        raise ValueError(f"cannot draw {count} distinct seeds from {room} candidates")
    chosen = set()
    while len(chosen) < count:
        s = low + secrets.randbelow(high - low + 1)
        if s not in banned:
            chosen.add(s)
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, required=True, help="how many seeds to draw")
    ap.add_argument("--exclude", default="1-25",
                    help="seeds never to draw, e.g. the public ladder seeds (default 1-25)")
    ap.add_argument("--out", required=True, help="file to write, outside this repository")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()

    if inside_repo(args.out):
        sys.exit(f"refusing to write {args.out}: it is inside the repository, and a "
                 "private seed list must never be committed")
    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} exists; pass --force to replace it")
    try:
        seeds = draw(args.count, parse_seeds(args.exclude))
    except ValueError as e:
        sys.exit(str(e))

    data = "".join(f"{s}\n" for s in seeds).encode("ascii")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"wrote {len(seeds)} seeds to {args.out}")
    print(f"sha256 {hashlib.sha256(data).hexdigest()}")


if __name__ == "__main__":
    main()
