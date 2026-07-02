"""Training job entrypoint — the machine-readable face of detector.train.

The copilot dispatches training as a SUBPROCESS (a stand-in for a real
sandboxed Cloud Run Job): heavy ML deps stay out of the agent's process, and
the job's only contract with the caller is one JSON line on stdout. Everything
else it prints (progress, warnings) goes to stderr so the contract stays clean.

Run:  uv run --group detector python -m detector.job POS_DIR NEG_DIR OUT_PATH
"""

import contextlib
import json
import sys

from detector.train import train


def main() -> int:
    pos_dir, neg_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        # train() prints progress to stdout; shove it to stderr so stdout
        # carries exactly one JSON line.
        with contextlib.redirect_stdout(sys.stderr):
            metrics = train(pos_dir, neg_dir, out_path)
    except Exception as e:  # noqa: BLE001 — the caller gets errors as data
        print(json.dumps({"ok": False, "error": str(e)}))
        return 1
    print(json.dumps({"ok": True, **metrics}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
