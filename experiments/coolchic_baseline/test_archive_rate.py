import io
import math
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.coolchic_baseline.archive_rate import (  # noqa: E402
    RESULTS_HEADER,
    append_results,
    build_temporary_zip,
    calculate_archive_score,
)


class BuildTemporaryZipTest(unittest.TestCase):
    def test_builds_deterministic_deflate_archive_without_writing_to_run(self):
        payload = (b"Cool-Chic bitstream payload\n" * 1024)

        with TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            bitstream_path = run_dir / "bitstream.cool"
            bitstream_path.write_bytes(payload)

            archive_bytes = build_temporary_zip(bitstream_path)
            repeated_archive_bytes = build_temporary_zip(bitstream_path)

            self.assertEqual(archive_bytes, repeated_archive_bytes)
            self.assertLess(len(archive_bytes), len(payload))
            self.assertFalse((run_dir / "archive.zip").exists())

            with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as archive:
                self.assertEqual(archive.namelist(), ["bitstream.cool"])
                self.assertEqual(
                    archive.getinfo("bitstream.cool").compress_type,
                    zipfile.ZIP_DEFLATED,
                )
                self.assertEqual(archive.read("bitstream.cool"), payload)

    def test_archive_bytes_drive_rate_and_score(self):
        payload = b"A" * 16_384

        with TemporaryDirectory() as tmp_dir:
            bitstream_path = Path(tmp_dir) / "bitstream.cool"
            bitstream_path.write_bytes(payload)

            result = calculate_archive_score(
                bitstream_path=bitstream_path,
                n_frames=20,
                total_frames=1200,
                uncompressed_bytes=1_000_000,
                seg=0.01,
                pose=0.004,
            )

        raw_rate_score = 25 * (len(payload) * 1200 / 20) / 1_000_000
        self.assertEqual(result.raw_bytes, len(payload))
        self.assertLess(result.archive_bytes, result.raw_bytes)
        self.assertAlmostEqual(
            result.score,
            100 * 0.01 + math.sqrt(10 * 0.004) + 25 * result.rate_extrap,
        )
        self.assertLess(result.score, 100 * 0.01 + math.sqrt(10 * 0.004) + raw_rate_score)

    def test_results_v2_header_is_written_once_without_touching_legacy_results(self):
        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            legacy_path = output_dir / "results.csv"
            results_path = output_dir / "results_v2.csv"
            legacy_path.write_text("legacy row\n", encoding="utf-8")

            row = ("run_r10", 32, 23_100, 18_777, 0.001, 0.002, 0.018, 0.5)
            append_results(results_path, *row)
            append_results(results_path, *row)

            lines = results_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], RESULTS_HEADER)
            self.assertEqual(lines.count(RESULTS_HEADER), 1)
            self.assertEqual(len(lines), 3)
            self.assertEqual(legacy_path.read_text(encoding="utf-8"), "legacy row\n")


if __name__ == "__main__":
    unittest.main()
