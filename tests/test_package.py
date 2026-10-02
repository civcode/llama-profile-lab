"""Package-level bootstrap tests."""

import llama_profile_lab


def test_version_is_exposed() -> None:
    assert llama_profile_lab.__version__ == "1.0.0"
