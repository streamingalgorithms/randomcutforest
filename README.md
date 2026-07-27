# Random Cut Forest

[![Java CI](https://github.com/streamingalgorithms/randomcutforest/actions/workflows/maven.yml/badge.svg)](https://github.com/streamingalgorithms/randomcutforest/actions/workflows/maven.yml)
[![Rust CI](https://github.com/streamingalgorithms/randomcutforest/actions/workflows/rust.yml/badge.svg)](https://github.com/streamingalgorithms/randomcutforest/actions/workflows/rust.yml)
[![Maven Central](https://img.shields.io/maven-central/v/org.streamingalgorithms/randomcutforest-core.svg?label=Maven%20Central)](https://central.sonatype.com/namespace/org.streamingalgorithms)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**A sketch of an evolving data stream.**

Random Cut Forest (RCF) is a probabilistic data structure that maintains a
summary of a stream in one pass. It was originally conceived for anomaly detection. But a summary of a stream can 
answer questions much broader than a specific scoring method. The same forest can answer questions about multiple quantities, such as density, nearest
neighbors, missing values, and forecasting. Since the data structure continues to maintain itself, 
one gets automatic, dynamic estimators of these quantities. This library provides a framework that 
allows one to define other dynamic estimators and inference algorithms. For example, see the forecasting example in Figure 1 below.

<p align="center">
  <img src="docs/images/rcf_cast.gif" alt="RCFCaster producing a calibrated forecast over a drifting seasonal stream" width="820">
</p>

<p align="center">
  <em>Figure 1. <code>RCFCaster</code> Forecasting a stream -- the algorithm self-adapts. <br>
  Produced by <a href="Java/examples/src/main/java/org/streamingalgorithms/randomcutforest/examples/RCFCastExample.java">RCFCastExample</a>; see <a href="#reading-that-forecast">Reading that forecast</a>.</em>
</p>

Periodicity
is contextual and the (local) periodicity itself changes in Figure 1. GappedRCFCastExample below, in Figure 2, shows the same example when
segments of the original input (corresponding to vertical yellow and grey stripes indicating occlusion) are never fed to the algorithm, which is about 40% of the input. The input seen by the algorithm 
is tessalated and the algorithm stops predicting if the occluded gap exceeds the horizon. Note that a separate step to impute the missing data will lead to reconciliation issues with the drift. 

<p align="center">
  <img src="docs/images/gapped_rcf_cast.gif" alt="GappedRCFCastExample producing a calibrated forecast over a stream with segments of missing data" width="820">
</p>
<p align="center">
  <em> Figure 2. <code>GappedRCFCastExample</code> The stream imputes the missing data segments (vertical stripes) on the fly.  <br>
  Produced by <a href="Java/examples/src/main/java/org/streamingalgorithms/randomcutforest/examples/GappedRCFCastExample.java">GappedRCFCastExample</a></em>
</p>

RCFs are naturally multidimensional. The examples above plot 1 dimension even though the predictions were 
performed in the code for both. Multidimensional forecasting is fascinating in its connection to 
<a href=https://en.wikipedia.org/wiki/Granger_causality>Granger Causality</a> -- but note that a streaming algorithm typically does not make 
any assumptions about stationarity and hence the algorithm can adapt.

One can have dynamic multidimensional inference such as (multi-centroid) clustering such as in Figure 3. 

<p align="center">
  <img src="docs/images/dynamic_summarization.gif" alt="Dynamic multicentroid clustering over a stream" width="400">
</p>
<p align="center">
  <em> Figure 3. <code>Summarization</code> The time decay is set high to expire the previously input points. <br>
  Produced by <a href="Java/examples/src/main/java/org/streamingalgorithms/randomcutforest/examples/summarization/Summarization.java">Summarization</a></em>
</p>


## The connection to random forests

Historically decision trees employed complicated partitioning rule in <a href=https://en.wikipedia.org/wiki/Decision_tree_learning>Classification and Regression Trees (CART)</a>, chosen to
separate the training data optimally with a simpler inference rule. There has been
continued effort in determination of partitioning rules, including small space projections as in <a href=https://en.wikipedia.org/wiki/Random_forest>Random Forests</a> and continued to <a href=https://en.wikipedia.org/wiki/Isolation_forest>Isolation Forests</a>. Randomization has been 
seen as a vehicle for generalization and stochastic (batch) discrimination. But most such analysis would require the trees 
to be rebuilt or have a deliberate discrepancy between stated construction and use. The latter is a common failure mode. 

RCF inverts the thinking -- the partitioning is simple **(recursive) random cuts**: pick a dimension with
probability proportional to its extent, pick a split point uniformly. The goal is to preserve arbitrary
but natural (for example distances) properties over a collection of trees -- drawing upon online algorithms that 
it is easier to solve optimization problem if the underlying graph is a tree. RCF shows that the specific 
recursive partitioning can be maintained under insertion *and deletion* using stochastic coupling over the 
time dependent (streaming) input. A collection of such trees is a provable embedding of distances. This makes 
continuous learning over a stream of unknown length possible provided the sketch 
can be decoded at inference time -- and if the decoding can be averaged across models then we can use 
algorithms designed for trees. Complicated data pipelines are eliminated.
The best piece of data is that which does not have to be collected!

- **The sketch is reusable.** One forest, many scoring functions -- multimodality 
   of inference, which also implies efficiency because quantities only need to 
   be computed if they are required for the specific analysis trajectory. We do not need 
    to over optimize at build time and throughput is an easy guarantee. 
- **The arrow of time is preserved.** Adaptive re-sampling and batch rebuilds
  break causality; the future influences the past. RCF's sampler doesn't violate 
  the arrow of time.

Anomaly detection is usually the *beginning* of an
investigation, not the end. A single bit saying "anomalous" is rarely actionable.
The interesting follow-ups — *which dimensions mattered? what should the value
have been? did the local density move?* — are all questions about the normal, and
a structure that quantifies unusuality should be able to describe usual (though 
not necessarily by the same action or algorithm). 


For more, please consider:

| | |
| --- | --- |
| [**Random Cut Forests**](https://opensearch.org/blog/random-cut-forests/) | The design tenet, RCFs as sketches, and why simple cuts plus rich inference beats the alternative. Start here. |
| [**Streaming analytics**](https://opensearch.org/blog/streaming-analytics/) | What you build on top: thresholding, grades, forecasting, and the practicalities of a stream that never stops. |
| [**One million entities in one minute**](https://opensearch.org/blog/one-million-enitities-in-one-minute/) | What it costs at scale. |
| [Guha, Mishra, Roy, Schrijvers, ICML 2016](https://proceedings.mlr.press/v48/guha16.pdf) | The paper. |

---

## Quick start

Requires **JDK 21 or later**.

```xml
<dependency>
  <groupId>org.streamingalgorithms</groupId>
  <artifactId>randomcutforest-parkservices</artifactId>
  <version>5.1.0</version>
</dependency>
```

```groovy
implementation 'org.streamingalgorithms:randomcutforest-parkservices:5.1.0'
```

`parkservices` pulls in `randomcutforest-core` transitively. Take
`randomcutforest-core` alone if you want raw scores and intend to do your own
thresholding.

> **Vector API.** The core uses the incubating jdk.incubator.vector module for SIMD.
> Add --add-modules jdk.incubator.vector to your JVM arguments. Without it the library
> still runs and produces identical results, but falls back to a scalar path and gives
> up most of the throughput in the 5.1.0 numbers. VectorSupport.isVectorized() reports 
> which path you are on. This is an incubator module and may move
> between JDK releases.

### Detect anomalies

`ThresholdedRandomCutForest` turns raw scores into a graded determination, so you
are not left calibrating a threshold by hand.

```java
int baseDimensions = 3;   // your actual variables
int shingleSize = 8;      // how much context defines "normal here"

ThresholdedRandomCutForest forest = ThresholdedRandomCutForest.builder()
        .dimensions(baseDimensions * shingleSize)   // note: the product
        .shingleSize(shingleSize)
        .internalShinglingEnabled(true)
        .anomalyRate(0.01)
        .build();

for (double[] point : stream) {
    AnomalyDescriptor result = forest.process(point, timestamp);
    if (result.getAnomalyGrade() > 0) {
        System.out.printf("t=%d grade=%.2f expected=%s%n",
                result.getInternalTimeStamp(),
                result.getAnomalyGrade(),
                Arrays.toString(result.getExpectedValuesList()[0]));
    }
}
```

Note alongside the grade, one gets the *expected* value, the relative
attribution across dimensions, and the start of the deviation. Note that it may be impossible to detect expectation immediately -- suppose a shop has either a low volume week or a high volume week. If we see a high monday sales and a low tuesday sales -- we detected an anomalous pattern. But was monday the issue or tuesday? These pieces of information such as relative start time, expected values, etc., allow 
one to initiate a root-cause process — it comes from the same trees that made the judgement about anomaly/otherwise. It is possible that a powerful algorithm/agent reverse engineers an algorithm and explains it, but would
it not be easier if the algorithm willingly provided information: "here is what the decision was based on"? In fact such information immediately makes it feasible to use decades old 
predictor-corrector paradigms. Existing ThresholdedRandomCutForest employs such a predictor-corrector paradigm and more than 
one specific score in its corrector step to suppress anomalies.

### Forecast

`RCFCaster` extrapolates over a horizon and, more usefully, tells you how much to
trust it. Intervals are conformally calibrated against the errors the model has
actually been making on this stream, so they widen when the stream turns and
tighten when it settles.

```java
RCFCaster caster = RCFCaster.builder()
        .dimensions(baseDimensions * shingleSize)
        .shingleSize(shingleSize)
        .internalShinglingEnabled(true)
        .forecastHorizon(15)
        .transformMethod(TransformMethod.NORMALIZE)
        .calibration(Calibration.SIMPLE)
        .build();

for (double[] point : stream) {
    ForecastDescriptor result = caster.process(point, timestamp);
    RangeVector forecast = result.getTimedForecast().rangeVector;
    // forecast.values / .lower / .upper, laid out horizon-major:
    //   index i * baseDimensions + d  ==  step i ahead, dimension d
    float[] intervalAccuracy = result.getIntervalPrecision();  // how often past
                                                               // intervals held
}
```

### Shingle size, briefly

`dimensions` is the product of input dimensions and `shingleSize`, and
getting this wrong is the single most common mistake. A shingle is the
sliding window of recent observations that defines context: with `shingleSize=1`
a point is judged against the (here till now) global distribution; it is the conceptual 
definition of disconnectivity in time. A large shingle is the conceptual (and algorithmic)
guarantee of what is imminent is connected to immediate past, in the manner of a
higher-order Markov chain. Contextual anomalies — a value that is unremarkable in
isolation but wrong *there* — also need a shingle. Forecasting requires one.

Set `internalShinglingEnabled(true)` and let the forest build them; the models
come out smaller because the algorithm can use knowledge of the shingle to 
continually keep compressing the points.

Full parameter reference, `timeDecay` guidance, CLI runners, and benchmarks:
**[Java/README.md](Java/README.md)**.


## Memory, speed, concurrency and serialization

These knobs do not change what the model concludes. They change what it costs to
ask, and they are the answer to several complaints that would otherwise look
like limitations of the approach.

### Bounding box cache

Every tree memoises the bounding boxes of its internal nodes. The cache is a
memo of a deterministic function of the tree, so the scores are the same either
way — only the time and space to obtain them differ.

```java
RandomCutForest.builder()
        .boundingBoxCacheFraction(0.0)   // default is 1.0
        .build();
```

`1.0` caches everything and is fastest. `0.0` caches nothing and gives the
smallest model. Intermediate values cache approximately that fraction, so this
is a continuous dial rather than a switch, and it can be moved on a live model:

```java
forest.setBoundingBoxCacheFraction(0.0);   // shed memory under pressure
```

The knob exists because model size, not accuracy, is usually what stops a forest
being deployed — a few hundred forests on one host, or one forest on a device
with a fixed budget. Trading throughput for footprint at inference time is
usually the right trade, and it is reversible. Note that
`ThresholdedRandomCutForest.process` transiently raises a zero cache to a small
non-zero value for the duration of a single call and restores it afterwards,
because a completely cold tree makes the corrector's several traversals
needlessly expensive.

### Concurrent readers

A built forest can serve many threads at once:

```java
forest.setMultiRead(true);
// ... N threads calling getAnomalyScore / getNearNeighborsInSample / imputeMissingValues
forest.setMultiRead(false);
```

While `multiRead` is on, a traversal that reaches a cold cache slot computes the
box into caller-local scratch instead of filling the shared slot, so readers
never write to shared state and need no lock. Moreover, sequential traversal reuses visitors 
and the tree view — a deliberate choice for low GC pressure — and both are corrupted 
by simultaneous scoring. Updates are rejected outright
while it is set — `update` throws rather than corrupting the tree — which makes
the intended pattern explicit: build the model, freeze it, fan out inference (if desired, 
but multiple threads will create more resource use),
unfreeze to resume consuming the stream.

Independently of the above, one can set a parallelExecutionEnabled(true) flag -- as in the
prior versions to enable parallel scoring within a forest, which uses independent
visitors and therefore the visitors are not corrupted at cacheFraction = 0. But
update (which is a write operation) is not thread safe for concurrent accesses.
However, the performance of this option in 5.1.0 is subpar compared to sequential
executors. The parallelExecutionEnabled flag will be repurposed and improved in coming releases.
We recommend developing the discipline of using multiRead() for concurrent access. Finally,
note that one can always copy the exact state of forest (and have per thread copies).

```java
import org.streamingalgorithms.randomcutforest.RandomCutForest;
import org.streamingalgorithms.randomcutforest.state.RandomCutForestMapper;
import org.streamingalgorithms.randomcutforest.state.RandomCutForestState;

RandomCutForestMapper mapper = new RandomCutForestMapper(); 
mapper.setSaveTreeStateEnabled(true); 
mapper.setSaveExecutorContextEnabled(true);

RandomCutForestState state = mapper.toState(forest);
RandomCutForest newForest = mapper.toModel(state);
```

The above primitive is doubly useful in serialization. The mapper and state classes are in core.
This repo continues to provide support for legacy JSON models used in OpenSearch, unchanged from upstream. 
However, we recommend
using the POJO provided in RandomCutForestState directly using modern methods like Apache Fory
which is straightforward to use and has been benchmarked in benchmark/.../SerializationBenchmark.java

```java
import org.streamingalgorithms.randomcutforest.RandomCutForest;
// import appropriate FORY version
var fory = Fory.builder().withLanguage(Language.JAVA).requireClassRegistration(false).build();
byte[] wire = fory.serialize(state);
RandomCutForestState restored = (RandomCutForestState) fory.deserialize(wire);
RandomCutForest deserialized = mapper.toModel(restored);
```
---

## Reading that forecast

The animation at the top is one `RCFCaster` consuming a two-dimensional stream
point by point. Around the two-thirds mark the generating process changes — a
different seasonality, phase and amplitude — with no warning and no retraining.

| | |
| --- | --- |
| **Black** | the observed series (one plotted dimension) |
| **Blue line and band** | the forecast over the next 15 steps, with its calibrated interval |
| **Brown band** | the error distribution actually observed over the last 15 steps |
| **Magenta** | interval accuracy: the fraction of past intervals that contained the truth, against guides at 0.0, 0.8 and 1.0 |
| **Grey vertical** | now |

The magenta trace is the point of the whole exercise. A forecast that reports
80% intervals and hits 80% is honest; one that reports 80% and hits 40% is a
liability, and you would like to find that out from the model rather than from
production. Watch what it does at the regime change — the intervals widen, the
accuracy dips, and both recover.

Reproduce it:

```bash
cd Java
mvn package -DexcludedGroups=functional
java --add-modules jdk.incubator.vector \
     -jar examples/target/randomcutforest-examples-*-jar-with-dependencies.jar rcf_cast
```

---


## Reading that detection

<p align="center">
  <img src="docs/images/thresholded_multi_dim.gif" alt="ThresholdedRandomCutForest detecting injected anomalies in a two dimensional periodic stream" width="820">
</p>

<p align="center">
  <em>Figure 4. <code>ThresholdedRandomCutForest</code> on a two dimensional periodic stream, with autoAdjust=true<br>
  Produced by <a href="Java/examples/src/main/java/org/streamingalgorithms/randomcutforest/examples/ThresholdedRCFMovie.java">ThresholdedRCFMovie</a>.</em>
</p>

The two input dimensions are cosines of the same period differing in phase and
amplitude, so this is not a time series plot: it is the phase portrait, x
against y. The underlying process is a closed ellipse, noise thickens it into an
annulus, and the observation travels around it once per period. An anomaly is a
radial excursion off the ring, which is a great deal easier to see than the same
event buried in two stacked traces.

| | |
| --- | --- |
| **Grey cloud** | every observation so far — the annulus the model is learning |
| **Grey line** | the last revolution, so the direction of travel is visible |
| **Black ring** | ground truth: where noise was actually replaced by an anomaly |
| **Red** | `getAnomalyGrade() > 0`, drawn at the timestamp the model *blamed* |
| **Green** | `getExpectedValuesList()[0]` — where the model says the point belonged |
| **Dashed grey** | detection lag: from the timestamp that fired back to the one blamed |
| **Coloured dots** | suppressed timestamps, one colour per `CorrectionMode` |

Two things in that legend are worth dwelling on, because they are the parts of
the API that reward attention.

**The red marker is not where the alarm rang.** When `getRelativeIndex()` is
negative the model is telling you that the responsible observation was several
steps back, and that it has only now accumulated enough evidence to say so. By
the time the alarm arrives the stream has usually returned to the ring, so
plotting at the firing timestamp would put the marker on a perfectly ordinary
point. The dashed leader is that lag drawn explicitly. This is not a defect to
be tuned away; a streaming detector that never revised its attribution would
simply be worse. Note also that a detection can arrive with no expected value at
all, when the forest cannot produce a confident imputation — the example draws
those hollow rather than pretending they were explained.

**The green marker is a correction, not a score.** The forest returns the value
it believed should have been there, which is what makes root cause work
tractable: the arrow from red to green is the inferred change, and comparing it
to the injected change tells you whether the model recovered the right
dimension as well as the right timestamp.

### Flags worth turning on

The defaults are a compromise, and this example is a cheap way to find out
whether the compromise suits your stream. Each of the following changes the
picture visibly.

```java
ThresholdedRandomCutForest.builder()
        .dimensions(baseDimensions * shingleSize)
        .shingleSize(shingleSize)
        .scoringStrategy(ScoringStrategy.MULTI_MODE)   // or MULTI_MODE_RECALL
        .autoAdjust(true)
        .alertOnce(true)
        .build();
```

**`scoringStrategy`** — default `EXPECTED_INVERSE_DEPTH`, which is an attempted
goldilocks setting and will not suit every stream. `MULTI_MODE` consults a
distance based mode alongside the depth based one and trades recall for
precision; `MULTI_MODE_RECALL` trades the other way, taking the distance verdict
when the primary scorer has stayed quiet for longer than a shingle. `DISTANCE`
uses the density mode alone. Streams whose "normal" is several distinct
behaviours rather than one — see `multimodal_example` — are exactly where the
default struggles and the multi-mode strategies earn their keep.

**`autoAdjust`** — default false. With it on, the corrector learns level shifts
and concept drift and stops raising the same alarm repeatedly; with it off, a
level shift produces a sustained run of alerts, which is the correct behaviour
if detecting the shift *is* the goal. Turning it on in this example visibly
increases the number of `CONDITIONAL_FORECAST` suppressions, because the
corrector gains a mechanism it did not previously have. Run it both ways.

**`alertOnce`** — a blunter instrument for the same problem, suppressing repeats
during drift. Do not use it for genuinely extended anomalies, which it will
truncate.

The `CorrectionMode` on every descriptor tells you which of these fired and why
a candidate was suppressed, so the flags are inspectable rather than magic.

Reproduce it:

```bash
cd Java
mvn package -DexcludedGroups=functional
java --add-modules jdk.incubator.vector \
     -jar examples/target/randomcutforest-examples-*-jar-with-dependencies.jar Thresholded_RCF_movie
```
---

## Examples

The [`examples`](Java/examples/src/main/java/org/streamingalgorithms/randomcutforest/examples/)
module is the real documentation. Each one is a self-contained, runnable
scenario; several plot as they go.

```bash
java --add-modules jdk.incubator.vector \
     -jar examples/target/randomcutforest-examples-*-jar-with-dependencies.jar --help
```

| Command                                                                      | What it shows                                                                                                                                                          |
|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `rcf_cast`                                                                   | Calibrated forecasting through a regime change. **The one to read first.**                                                                                             |
| `gapped_rcf_cast`                                                            | Forecasting when the stream has holes in it                                                                                                                            |
| `Thresholded_Multi_Dim_example`                                              | Typical multiDimensional anomaly detection                                                                                                                             |
| `Thresholded_Predictive_example`                                             | Predictive forecast across multiple correlated series                                                                                                                  |
| `Conditional_Predictive_example`                                             | Imputation used as prediction                                                                                                                                          |
| `multimodal_example`                                                         | Streams whose "normal" is several distinct things at once                                                                                                              |
| `density`                                                                    | Directional, dynamic density estimation                                                                                                                                |
| `near_neighbor`                                                              | Dynamic nearest-neighbour queries against the sketch                                                                                                                   |
| `summarization`, `multi_summarize`, `string_summarize`, `centroid_summarize` | Clustering and multi-centroid summarisation, including over strings                                                                                                    |
| `Thresholded_RCF_movie`                                                      | Anomaly detection as a phase portrait: blame, expected values, detection lag, and every suppression token. **Try it with `autoAdjust` and the multi-mode strategies.** |

---

## Repository layout

```
Java/                 the reference implementation
  core/               RandomCutForest — trees, sampling, traversal, raw estimation, 
                      model persistence (state and mappers)
  parkservices/       ThresholdedRandomCutForest, RCFCaster — the layer you
                      probably want: grades, calibration, transforms
  serialization/      legacy JSON converters
  examples/           runnable scenarios and plots
  benchmark/          JMH microbenchmarks
  testutils/          internal test scaffolding
Rust/                 Rust implementation
python_rcf_wrapper/   Python bindings
```

The split between `core` and `parkservices` is deliberate and matters when you
choose a dependency. `core` gives you an estimate — an anomaly score, an
extrapolation. Turning estimations into decisions is hard, and anomaly 
detection deployments fail because the 'exciting' part does not 
align with 'non-exciting' part -- the notion of exciting is to the beholder.  The gestalt 
any algorithm corresponds to no single line of code being the weakest link. `parkservices` is where that work
lives. Its defaults differ from `core`'s on purpose: `internalShinglingEnabled`
is true there, for instance, because it is the natural choice in that context.

Consider `core` if you want a new scoring function -- and write that against
the traversal API using newly written visitors. The library will make it stream automatically. 
If that is not a goal, consider `parkservices` or your own code to build 
domain specific adaptations.

---

## Releases

Java artifacts are published to Maven Central under the `org.streamingalgorithms`
namespace. Jars are compiled, tested and signed by GitHub Actions runners — see
[Java/RELEASING.md](Java/RELEASING.md) for the process and
[`.github/workflows/`](.github/workflows/) for the workflows themselves.

GitHub tags carry a `-java` or `-rust` suffix (`5.1.0-java`) because the two
implementations release independently.

## Relationship to the upstream project

This is a fork of [random-cut-forest-by-aws](https://github.com/aws/random-cut-forest-by-aws),
originally developed at Amazon and released under Apache 2.0. This fork is maintained
independently under the `streamingalgorithms` organization and is **not
affiliated with or endorsed by Amazon or AWS**.

The Maven coordinates changed accordingly. If you are migrating from the AWS
artifacts, the package root moved from `com.amazon.randomcutforest` to
`org.streamingalgorithms.randomcutforest`. See the [release notes](Java/RELEASE-NOTES.md) for the
rest.

## Contributing

Issues and pull requests are welcome — including "the documentation didn't
explain this" issues, which are as useful as bug reports. See
[CONTRIBUTING.md](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).

Please do not report security issues in public issues; see [SECURITY.md](SECURITY.md).


## License

Apache License 2.0 — see [LICENSE](LICENSE).

Copyright 2019 Amazon.com, Inc. or its affiliates.
Copyright 2026 The streamingalgorithms authors.
