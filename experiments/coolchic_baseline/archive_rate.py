"""Build the temporary archive used for Cool-Chic rate estimation."""
import io
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path


ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
RESULTS_HEADER = (
    "run,n_frames,raw_bytes,zip_bytes,seg,pose,zip_rate_extrap,score"
)


@dataclass(frozen=True)
class ArchiveScore:
    raw_bytes: int
    archive_bytes: int
    archive_bytes_extrap: float
    rate_actual: float
    rate_extrap: float
    score: float


def build_temporary_zip(bitstream_path: Path) -> bytes:
    """Return a deterministic Deflate-9 ZIP containing one bitstream."""
    member = zipfile.ZipInfo(bitstream_path.name, date_time=ZIP_TIMESTAMP)
    member.compress_type = zipfile.ZIP_DEFLATED
    member.create_system = 3
    member.external_attr = 0o600 << 16

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(
        archive_buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        archive.writestr(
            member,
            bitstream_path.read_bytes(),
            compress_type=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        )
    return archive_buffer.getvalue()


def calculate_archive_score(
    bitstream_path: Path,
    n_frames: int,
    total_frames: int,
    uncompressed_bytes: int,
    seg: float,
    pose: float,
) -> ArchiveScore:
    """Calculate the challenge score using temporary ZIP bytes for rate."""
    raw_bytes = bitstream_path.stat().st_size
    archive_bytes = len(build_temporary_zip(bitstream_path))
    archive_bytes_extrap = archive_bytes * total_frames / n_frames
    rate_actual = archive_bytes / uncompressed_bytes
    rate_extrap = archive_bytes_extrap / uncompressed_bytes
    score = 100 * seg + math.sqrt(10 * pose) + 25 * rate_extrap
    return ArchiveScore(
        raw_bytes=raw_bytes,
        archive_bytes=archive_bytes,
        archive_bytes_extrap=archive_bytes_extrap,
        rate_actual=rate_actual,
        rate_extrap=rate_extrap,
        score=score,
    )


def append_results(
    results_path: Path,
    run_name: str,
    n_frames: int,
    raw_bytes: int,
    archive_bytes: int,
    seg: float,
    pose: float,
    rate_extrap: float,
    score: float,
) -> None:
    """Append one ZIP-scored result without mixing it with legacy rows."""
    write_header = not results_path.exists()
    with open(results_path, "a", encoding="utf-8", newline="") as results_file:
        if write_header:
            results_file.write(f"{RESULTS_HEADER}\n")
        results_file.write(
            f"{run_name},{n_frames},{raw_bytes},{archive_bytes},"
            f"{seg:.8f},{pose:.8f},{rate_extrap:.8f},{score:.6f}\n"
        )
