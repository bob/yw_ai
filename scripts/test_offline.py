"""Run mocked unit tests without inherited credentials or Python network egress.

Pre-cache o200k_base and cl100k_base separately if necessary. This runner never
downloads tokenizer files. Linux CI also uses an OS network namespace; this
Python audit hook is a local guard for the repository's trusted test code.
"""

import argparse
import os
from pathlib import Path
import sys
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="test_*.py")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    # Preserve system paths and any explicitly selected static-tokenizer cache.
    preserved = {"PATH", "HOME", "TMPDIR", "TEMP", "TMP", "SYSTEMROOT", "TIKTOKEN_CACHE_DIR"}
    for key in list(os.environ):
        if key not in preserved:
            del os.environ[key]
    os.environ.update(
        LANGSMITH_TRACING="false", LANGCHAIN_TRACING_V2="false",
        OPENAI_API_KEY="offline-test-key", TYPESAFE_API_KEY="offline-test-key",
    )

    def deny_egress(event, arguments):
        if event in {
            "socket.connect", "socket.getaddrinfo", "socket.sendto", "socket.sendmsg",
            "subprocess.Popen", "os.system",
        }:
            raise RuntimeError(f"Offline tests blocked {event}")

    sys.addaudithook(deny_egress)
    import dotenv
    dotenv.load_dotenv = lambda *args, **kwargs: False

    from streamlit.testing.v1 import AppTest
    # Presentation delays are unrelated to the result. The pacing-specific
    # test installs its own fake clock and sleep and still verifies that logic.
    import langchain_app.ui.response_progress as progress
    progress.sleep = lambda seconds: None
    for factory in (AppTest.from_file, AppTest.from_function, AppTest.from_string):
        factory.__func__.__kwdefaults__["default_timeout"] = 15

    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=args.pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
