"""Filter out source files that have already been converted, to resume an interrupted run.

This optional, opt-in step (Theme H, Task 28) implements incremental processing / resume.
Given the list ``X`` of SOURCE file paths, it drops any source whose expected UMIE output
(``self.get_umie_img_path(src)``) already exists on disk, returning only the not-yet-processed
sources. An interrupted run can therefore be re-run and will only process the remainder.

Because it skips a source only when that source's *correct* output already exists, a resumed
run is byte-identical to a clean run: nothing already-correct is touched, and everything else
is processed exactly as before.

If :attr:`~base.pipeline.ExportConfig.force_reprocess` is ``True``, the step returns ``X``
unchanged so the whole dataset is reprocessed.

Placement / interaction with ``DeleteOldPreprocessedData``:
    ``DeleteOldPreprocessedData`` wipes the existing outputs; once it has run there are no
    outputs to skip, so a full reprocess naturally follows. Place ``SkipProcessed`` AFTER
    ``get_file_paths`` (it needs the full source list) and AFTER any delete-old step (so it
    never skips files whose outputs were just deleted). With both steps active, the
    delete-old step always wins and the resume optimisation simply becomes a no-op.

Robustness:
    ``get_umie_img_path`` can raise (e.g. a source whose phase id is not in the configured
    phases). Such sources are treated as not-yet-processed and kept in ``X`` so they flow on
    to the normal conversion steps rather than being silently dropped.
"""

import os

from base.step import BaseStep


class SkipProcessed(BaseStep):
    """Drop sources whose UMIE output already exists, so an interrupted run resumes cleanly."""

    #: Treat a zero-byte output as not-yet-processed (a truncated / interrupted write).
    _REQUIRE_NONZERO_SIZE = True

    def transform(self, X: list) -> list:
        """Return only the sources in ``X`` that still need processing.

        Args:
            X (list): List of SOURCE file paths.

        Returns:
            list: The subset of ``X`` whose UMIE output does not yet exist (or all of
            ``X`` when ``force_reprocess`` is set).
        """
        if self.export_config.force_reprocess:
            return X
        return [src for src in X if not self._is_already_processed(src)]

    def _is_already_processed(self, src: str) -> bool:
        """Return whether ``src`` already has a valid UMIE output on disk.

        A source counts as processed only when its mapped output path exists and (when
        :attr:`_REQUIRE_NONZERO_SIZE`) is non-empty. If the path cannot be mapped (e.g. an
        unmapped phase makes ``get_umie_img_path`` raise) the source is treated as
        not-yet-processed so it is kept for normal processing.

        Args:
            src (str): A SOURCE file path.

        Returns:
            bool: ``True`` if a valid output already exists, else ``False``.
        """
        try:
            output_path = self.get_umie_img_path(src)
        except Exception:
            return False
        if not os.path.exists(output_path):
            return False
        if self._REQUIRE_NONZERO_SIZE and os.path.getsize(output_path) == 0:
            return False
        return True
