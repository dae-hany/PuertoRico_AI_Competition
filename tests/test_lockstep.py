"""Lockstep comparison of the engine with the independent reference model
(verification/README.md). Set LOCKSTEP_GAMES to play more games per player
count than the quick default."""
import hashlib
import os

import pytest

from verification.lockstep import run

GAMES = int(os.environ.get("LOCKSTEP_GAMES", "30"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The files copied unchanged from the research engine (PuertoRico_ESWA 11d6c11).
VENDORED = {
    "verification/oracle.py":
        "85b5a93a9e438454aabf4f67296eb16e8b4b745acdf5c4f2bb5613c295c4d96f",
    "verification/rule/RULES.md":
        "532ef1be44e25becabd49b7a37aada70e61f86a54d8791976b91127e365b816a",
    "verification/rule/CONFORMANCE.md":
        "cf79f22c4a62c3a4e897e30b22c8f5379840d0a6f8b42b0e264f04527868c742",
}


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_engine_agrees_with_the_reference_model(n):
    stats = run([n], GAMES, seed=0)[n]
    assert stats["games"] == GAMES


@pytest.mark.parametrize("path,digest", sorted(VENDORED.items()))
def test_the_reference_model_is_the_vendored_copy(path, digest):
    # Line endings are normalized, so a Windows checkout with autocrlf matches.
    with open(os.path.join(ROOT, path), "rb") as f:
        data = f.read().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == digest, f"{path} was edited"
