#!/usr/bin/env python3

"""Boundary policy checks for the multiplatform Phase 0 guardrails.

This script enforces no-regression backend layering rules by comparing current
findings against a checked-in allowlist baseline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALLOWLIST = REPO_ROOT / "scripts/policy/backend_boundary_allowlist.txt"
SCAN_EXTENSIONS = {".h", ".hpp", ".cuh", ".cpp", ".cu"}

INCLUDE_RE = re.compile(r"^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]", re.MULTILINE)

TOKEN_PATTERNS = [
    ("CUstream_st", re.compile(r"\bCUstream_st\b")),
    ("cudaStream_t", re.compile(r"\bcudaStream_t\b")),
    ("cudaEvent_t", re.compile(r"\bcudaEvent_t\b")),
    ("cudaError_t", re.compile(r"\bcudaError_t\b")),
    ("cudaGraph_t", re.compile(r"\bcudaGraph_t\b")),
    ("cudaGraphExec_t", re.compile(r"\bcudaGraphExec_t\b")),
    ("cudaGraphNode_t", re.compile(r"\bcudaGraphNode_t\b")),
    ("thrust::", re.compile(r"\bthrust::")),
    ("hipStream_t", re.compile(r"\bhipStream_t\b")),
    ("hipEvent_t", re.compile(r"\bhipEvent_t\b")),
    ("hipError_t", re.compile(r"\bhipError_t\b")),
    ("MTL*", re.compile(r"\bMTL[A-Za-z0-9_]*\b")),
]


@dataclass(frozen=True)
class FindingKey:
    rule_id: str
    path: str
    signature: str


def relpath(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def iter_repo_files() -> list[Path]:
    paths: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in SCAN_EXTENSIONS:
            continue
        if ".git" in path.parts or "build" in path.parts:
            continue
        paths.append(path)
    return paths


def is_vendor_include(include_target: str) -> bool:
    target = include_target.strip()
    lower = target.lower()
    if target.startswith("tinygs/cuda/"):
        return True
    if lower.startswith("cuda"):
        return True
    if lower.startswith("thrust/"):
        return True
    if lower.startswith("hip"):
        return True
    if lower.startswith("metal"):
        return True
    return False


def is_public_header(path: Path) -> bool:
    rp = relpath(path)
    if not rp.startswith("tinygs/include/tinygs/"):
        return False
    if rp.startswith("tinygs/include/tinygs/cuda/"):
        return False
    return path.suffix in {".h", ".hpp", ".cuh"}


def is_platform_file(path: Path) -> bool:
    rp = relpath(path)
    if rp.startswith("tinygs/include/tinygs/platform/"):
        return True
    if rp.startswith("tinygs/src/platform/"):
        return True
    return False


def is_runtime_contract_header(path: Path) -> bool:
    rp = relpath(path)
    if not rp.startswith("tinygs/include/tinygs/platform/"):
        return False
    return path.name in {
        "backend_error.hpp",
        "runtime.hpp",
        "runtime_factory.hpp",
    }


def add_count(
    findings: dict[FindingKey, int],
    rule_id: str,
    path: Path,
    signature: str,
    count: int = 1,
) -> None:
    key = FindingKey(rule_id=rule_id, path=relpath(path), signature=signature)
    findings[key] += count


def scan_findings() -> dict[FindingKey, int]:
    findings: dict[FindingKey, int] = defaultdict(int)

    for path in iter_repo_files():
        text = path.read_text(encoding="utf-8", errors="ignore")

        if is_public_header(path):
            for include_target in INCLUDE_RE.findall(text):
                if is_vendor_include(include_target):
                    add_count(
                        findings,
                        "public_header_forbidden_include",
                        path,
                        f"include:{include_target}",
                    )

            for token_name, token_re in TOKEN_PATTERNS:
                token_count = len(token_re.findall(text))
                if token_count > 0:
                    add_count(
                        findings,
                        "public_header_forbidden_vendor_type",
                        path,
                        f"token:{token_name}",
                        token_count,
                    )

        if is_platform_file(path):
            for include_target in INCLUDE_RE.findall(text):
                if is_vendor_include(include_target):
                    add_count(
                        findings,
                        "platform_layer_forbidden_vendor_dep",
                        path,
                        f"include:{include_target}",
                    )

            for token_name, token_re in TOKEN_PATTERNS:
                token_count = len(token_re.findall(text))
                if token_count > 0:
                    add_count(
                        findings,
                        "platform_layer_forbidden_vendor_dep",
                        path,
                        f"token:{token_name}",
                        token_count,
                    )

        if is_runtime_contract_header(path):
            for include_target in INCLUDE_RE.findall(text):
                if is_vendor_include(include_target):
                    add_count(
                        findings,
                        "runtime_contract_forbidden_vendor_dep",
                        path,
                        f"include:{include_target}",
                    )

            for token_name, token_re in TOKEN_PATTERNS:
                token_count = len(token_re.findall(text))
                if token_count > 0:
                    add_count(
                        findings,
                        "runtime_contract_forbidden_vendor_dep",
                        path,
                        f"token:{token_name}",
                        token_count,
                    )

    return dict(findings)


def encode_entry(key: FindingKey, count: int) -> str:
    return f"{key.rule_id}|{key.path}|{key.signature}|{count}"


def write_allowlist(path: Path, findings: dict[FindingKey, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# tinygs backend boundary no-regression baseline",
        "# Format: rule_id|path|signature|count",
        "",
    ]
    for key, count in sorted(findings.items(), key=lambda item: encode_entry(item[0], item[1])):
        lines.append(encode_entry(key, count))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_allowlist(path: Path) -> dict[FindingKey, int]:
    if not path.exists():
        raise FileNotFoundError(f"Allowlist file not found: {path}")

    parsed: dict[FindingKey, int] = {}
    for idx, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("|")
        if len(fields) != 4:
            raise ValueError(f"{path}:{idx}: expected 4 fields separated by '|', got: {line}")
        rule_id, rel_path, signature, count_str = fields
        try:
            count = int(count_str)
        except ValueError as exc:
            raise ValueError(f"{path}:{idx}: invalid count: {count_str}") from exc
        parsed[FindingKey(rule_id=rule_id, path=rel_path, signature=signature)] = count
    return parsed


def print_scan(findings: dict[FindingKey, int]) -> None:
    if not findings:
        print("No boundary findings.")
        return

    totals: dict[str, int] = defaultdict(int)
    for key, count in findings.items():
        totals[key.rule_id] += count

    print("Current boundary findings:")
    for rule_id in sorted(totals.keys()):
        print(f"  {rule_id}: {totals[rule_id]}")
    print("")

    for key, count in sorted(findings.items(), key=lambda item: encode_entry(item[0], item[1])):
        print(encode_entry(key, count))


def check_against_allowlist(
    findings: dict[FindingKey, int],
    allowlist: dict[FindingKey, int],
    strict_public_zero: bool = False,
) -> int:
    regressions: list[str] = []
    resolved: list[str] = []

    for key, current_count in findings.items():
        baseline_count = allowlist.get(key, 0)
        if current_count > baseline_count:
            regressions.append(
                f"{encode_entry(key, current_count)} (baseline={baseline_count})"
            )

    for key, baseline_count in allowlist.items():
        current_count = findings.get(key, 0)
        if current_count < baseline_count:
            resolved.append(
                f"{encode_entry(key, current_count)} (baseline={baseline_count})"
            )

    if regressions:
        print("Boundary policy check failed: new or increased violations detected.")
        for line in sorted(regressions):
            print(f"  + {line}")
        if resolved:
            print("Resolved/decreased violations (non-blocking):")
            for line in sorted(resolved):
                print(f"  - {line}")
        return 1

    if strict_public_zero:
        strict_lines: list[str] = []
        for key, count in findings.items():
            if count == 0:
                continue
            if key.rule_id in {
                "public_header_forbidden_include",
                "public_header_forbidden_vendor_type",
            }:
                strict_lines.append(encode_entry(key, count))
        if strict_lines:
            print(
                "Boundary policy strict check failed: "
                "public non-CUDA headers still contain vendor dependencies."
            )
            for line in sorted(strict_lines):
                print(f"  * {line}")
            return 1

    print("Boundary policy check passed: no new violations.")
    if resolved:
        print("Resolved/decreased violations (non-blocking):")
        for line in sorted(resolved):
            print(f"  - {line}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Check tinygs backend boundary guardrails.")
    parser.add_argument(
        "mode",
        choices=["scan", "check", "write-baseline"],
        help="scan: print findings, check: enforce no-regression, write-baseline: overwrite allowlist",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help=f"Allowlist path (default: {DEFAULT_ALLOWLIST.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--strict-public-zero",
        action="store_true",
        help=(
            "For check mode only: require zero vendor includes/types in non-CUDA public headers."
        ),
    )
    args = parser.parse_args()

    findings = scan_findings()

    if args.mode == "scan":
        print_scan(findings)
        return 0

    if args.mode == "write-baseline":
        write_allowlist(args.allowlist, findings)
        print(f"Wrote baseline: {args.allowlist}")
        return 0

    try:
        allowlist = parse_allowlist(args.allowlist)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return check_against_allowlist(
        findings, allowlist, strict_public_zero=args.strict_public_zero
    )


if __name__ == "__main__":
    sys.exit(main())
