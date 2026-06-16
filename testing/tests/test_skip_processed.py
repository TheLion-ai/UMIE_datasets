"""Unit tests for SkipProcessed: resume filtering and the force_reprocess override."""

import os
import tempfile

from base.pipeline import PathArgs, PipelineArgs, PipelineContext
from config.dataset_config import DatasetArgs
from src.steps.skip_processed import SkipProcessed


def _make_ctx(tmp: str) -> PipelineContext:
    """Build a minimal PipelineContext for the step."""
    pa = PipelineArgs()
    identity, dicom, file_selection, output = pa.to_configs()
    return PipelineContext(
        paths=PathArgs(source_path=tmp, target_path=tmp),
        dataset=DatasetArgs(dataset_uid="99", dataset_name="synthetic", phases={"0": "CT"}),
        identity=identity,
        dicom=dicom,
        file_selection=file_selection,
        output=output,
    )


def _make_source(tmp: str, name: str) -> str:
    """Create a synthetic source file and return its path."""
    src = os.path.join(tmp, "source", name)
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src, "wb") as handle:
        handle.write(b"src")
    return src


def _precreate_output(step: SkipProcessed, src: str, data: bytes = b"out") -> str:
    """Pre-create the UMIE output for ``src`` exactly where the step expects it."""
    out = step.get_umie_img_path(src)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as handle:
        handle.write(data)
    return out


def test_only_unprocessed_sources_returned():
    """Sources whose UMIE output already exists are dropped; the rest pass through."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        step = SkipProcessed(ctx)

        done = _make_source(tmp, "done.png")
        todo = _make_source(tmp, "todo.png")

        # Pre-create the output only for `done` (verifying the path the step expects).
        _precreate_output(step, done)

        result = step.transform([done, todo])
        assert result == [todo]


def test_zero_byte_output_treated_as_unprocessed():
    """A zero-byte output (truncated write) does not count as processed."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        step = SkipProcessed(ctx)

        src = _make_source(tmp, "trunc.png")
        _precreate_output(step, src, data=b"")  # empty file

        assert step.transform([src]) == [src]


def test_force_reprocess_returns_all():
    """With force_reprocess set, every source is returned even if outputs exist."""
    with tempfile.TemporaryDirectory() as tmp:
        ctx = _make_ctx(tmp)
        step = SkipProcessed(ctx)

        done = _make_source(tmp, "done.png")
        todo = _make_source(tmp, "todo.png")
        _precreate_output(step, done)

        ctx.export.force_reprocess = True
        assert SkipProcessed(ctx).transform([done, todo]) == [done, todo]
