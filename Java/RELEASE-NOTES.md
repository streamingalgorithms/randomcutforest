# Release notes — Java

Tags for this implementation carry a `-java` suffix (`5.1.0-java`); the Rust
implementation releases independently and is not covered here.

This file starts at 5.1.0; the earlier entry is summarised from the issue
tracker.

---

## 5.1.0 (unreleased)

### Performance

Scoring allocation drops roughly 9x below 5.0.1, to about 1.4x the size of the
input point, and no longer varies with `boundingBoxCacheFraction`. The
throughput gains are concentrated where they matter most: the uncached
configuration, and the small dynamic cache that `ThresholdedRandomCutForest`
uses internally.

`ScoringBenchmark`, `D2`, `DEFAULT`/`SCALAR`/`EXACT`:

| cacheFraction | 5.0.1 ops/s | 5.1.0 ops/s | 5.0.1 B/op | 5.1.0 B/op |
| --- | --- | --- | --- | --- |
| 0.0 | 3,634.9 | 4,597.5 | 5,217.3 | 708.3 |
| 0.001 | 24,275.0 | 23,699.4 | 5,095.2 | 557.7 |
| 1.0 | 53,902.5 | 57,811.3 | 5,078.0 | 551.7 |

The `D1` dataset at `cacheFraction=0.001` was the worst outlier in 5.0.1 — the
only configuration that allocated four times the table average while gaining
almost nothing in throughput. It is also the configuration `TRCF.process`
selects when the cache is otherwise disabled, so it is the path most callers
actually take:

| `D1`, cacheFraction 0.001 | 5.0.1 | 5.1.0 |
| --- | --- | --- |
| throughput | 3,572.1 ops/s | 6,897.8 ops/s |
| allocation | 19,217.7 B/op | 974.2 B/op |

Reproduce:

```bash
java --add-modules jdk.incubator.vector \
     -jar benchmark/target/benchmarks.jar ScoringBenchmark \
     -p kind=SCALAR -p mode=EXACT -p func=DEFAULT -prof gc
```

**On reading these numbers.** At sub-kilobyte levels JMH's
`gc.alloc.rate.norm` is quantised by TLAB sampling and the error bars widen
past the mean — 708.290 ± 1615.257 B/op at `cacheFraction=0.0`, for instance.
Read the allocation figures as "roughly half a kilobyte, at the resolution
floor of the profiler" rather than as three significant figures. Uncached
throughput is likewise noisier than it was (±19% against ±2.7% in 5.0.1).

### Documentation

- `README`: the bounding box cache, `setMultiRead`, and parallel traversal are
  documented for the first time. These are the answers to several questions
  that previously looked like limitations of the approach.
- `README`: `ScoringStrategy` and `autoAdjust` are explained, with guidance on
  when the defaults are the wrong choice.
- New example `Thresholded_RCF_movie`, rendering anomaly detection as a
  phase portrait: blame attribution, expected values, detection lag, and every
  `CorrectionMode` suppression token.
- This file.

### Not yet measured

Stated so nobody infers more than the numbers support: `ProcessBenchmark` has
no baseline, and the figures recorded in issue #9 predate this round of work.
The end-to-end effect on `process`, `TRCF` and `RCFCaster` is therefore not
quantified here.

---

## 5.0.0 / 5.0.1

First release under the `streamingalgorithms` organization. Fork of
[random-cut-forest-by-aws](https://github.com/aws/random-cut-forest-by-aws).
Shipped without notes. Recorded here because it contains the largest positive
performance change in the project's history, and is a critical reason for this
fork to exist.

- Maven coordinates moved to the `org.streamingalgorithms` namespace.
- Package root moved from `com.amazon.randomcutforest` to
  `org.streamingalgorithms.randomcutforest`.
- Requires JDK 21 or later.

Issue [#3](https://github.com/streamingalgorithms/randomcutforest/issues/3)
observed that scoring allocated 4.5 MB to produce one 4-byte score from a
400-byte input. A representation change to the bounding boxes, SIMD via the
incubating Vector API, and a small data-dependent dynamic cache together
reduced that by roughly three orders of magnitude.

`ScoringBenchmark`, `D2`, `DEFAULT`/`SCALAR`/`EXACT`, against upstream:

| cacheFraction | upstream ops/s | 5.0.1 ops/s | upstream B/op | 5.0.1 B/op |
| --- | --- | --- | --- | --- |
| 0.0 | 977.8 | 3,634.9 | 4,498,782 | 5,217.3 |
| 1.0 | 19,263.6 | 53,902.5 | 54,742.6 | 5,078.0 |

Allocation fell about 860x uncached and 11x fully cached; throughput rose 3.7x
and 2.8x respectively. The newly viable `cacheFraction=0.001` reached 24,275
ops/s.
