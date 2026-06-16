"""Optional lightweight web interface for querying the corpus and downloading subset manifests (Task 45).

A thin HTTP wrapper over the tested query core in ``utils/subset.py`` - it reads existing JSONL
metadata only and never copies images. Flask is imported lazily so this module is optional and does
not affect the rest of the package when Flask is not installed.

Run:
    pip install flask
    python -m utils.subset_web data/**/*.jsonl    # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import glob
import sys

from utils.subset import build_manifest, load_records, query_subset

_FORM = """
<!doctype html><title>UMIE subset query</title>
<h1>UMIE subset query</h1>
<form action="/manifest">
  Modality: <input name="modality"> (e.g. CT)<br>
  RadLex label: <input name="label"> (matches hierarchy descendants)<br>
  Licence: <input name="license"><br>
  <input type="submit" value="Download manifest">
</form>
"""


def create_app(jsonl_globs: list[str]):  # type: ignore[no-untyped-def]  # Flask app, imported lazily
    """Build the Flask app serving the query form and the manifest endpoint."""
    from flask import Flask, jsonify, request  # lazy: Flask is an optional extra

    paths = [p for pattern in jsonl_globs for p in glob.glob(pattern, recursive=True)]
    records = load_records(paths)
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return _FORM

    @app.route("/manifest")
    def manifest():  # type: ignore[no-untyped-def]
        subset = query_subset(
            records,
            modality=request.args.get("modality") or None,
            label=request.args.get("label") or None,
            license=request.args.get("license") or None,
        )
        return jsonify(build_manifest(subset))

    return app


def main() -> None:
    """CLI entry point: ``python -m utils.subset_web <jsonl globs>``."""
    if len(sys.argv) < 2:
        print("usage: python -m utils.subset_web <jsonl globs...>")
        raise SystemExit(1)
    create_app(sys.argv[1:]).run()


if __name__ == "__main__":
    main()
