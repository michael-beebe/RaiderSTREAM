#!/usr/bin/env python3

from dataclasses import dataclass
from typing import TypeVar

todo = lambda: Exception("not yet implemented")

base_args = {
    "-s": 4194304,
    "-k": "seqCopy",
}

@dataclass
class BenchResult:
    kernel: str
    runtime: float
    mbps: float
    flops: float

flatten = lambda xs: [x for xsP in xs
                        for x in xsP]

def flatten_dict[T, U](xs: list[dict[T, U]]) -> dict[T, list[U]]:
    acc = dict()
    for x in xs:
        for k, v in x.items():
            if k in acc:
                acc[k].append(v)
            else:
                acc[k] = [v]
    return acc

def run_rstream(rstream_bin: str, **append_args) -> dict[str, BenchResult]:
    import subprocess

    args = {**base_args, **append_args}

    out = subprocess.check_output([
        rstream_bin,
        *flatten([[k, str(v)] for k, v in args])
    ], encoding='UTF-8')

    return parse_output(out.splitlines())

def run_rstream_avg(rstream_bin: str, n_reps: int, **append_args) -> dict[str, BenchResult]:
    runs = [run_rstream(rstream_bin, **append_args) for _ in range(n_reps)]

    runs_flat = flatten_dict(runs)

    return {k: average(rs) for k, rs in runs_flat.items()}

def average(rs: list[BenchResult]) -> BenchResult:
    n = len(rs)
    return BenchResult(
        kernel=rs[0].kernel,
        runtime=sum([b.runtime for b in rs]) / n,
        mbps=sum([b.mbps for b in rs]) / n,
        flops=sum([b.flops for b in rs]) / n
    )

def parse_output(lines: list[str]) -> dict[str, BenchResult]:
    import re
    def parse_line(line: str) -> BenchResult:
        blocks = re.split(r"\s{2,}", line)
        [name, rt_raw, mb_raw, flops_raw] = blocks
        return BenchResult(kernel=name, runtime=float(rt_raw), mbps=int(mb_raw), flops=int(flops_raw))

    # find the benchmark kernel line
    start = next(filter(lambda l: l[1].startswith("Benchmark Kernel"), enumerate(lines)))[0]
    # adjust to first line of data
    start = start + 2
    # seek
    lines = lines[start:]

    return dict([(r.kernel, r) for r in map(parse_line, lines)])

print(run_rstream("./raiderstream"))
