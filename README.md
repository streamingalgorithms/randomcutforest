# Random Cut Forest

This repository contains implementations of the Random Cut Forest (RCF)
probabilistic data structure. RCFs were originally developed for use in a
nonparametric anomaly detection algorithm for streaming data. Later algorithms
based on RCFs were developed for density estimation, imputation, and forecasting.

This project is a fork of
[random-cut-forest-by-aws](https://github.com/aws/random-cut-forest-by-aws),
originally developed at Amazon and released under the Apache 2.0 license. It is
maintained independently under the streamingalgorithms organization and is not
affiliated with or endorsed by Amazon or AWS.

The different directories correspond to implementations in different languages,
with language-specific bindings for greater flexibility of use.

The `randomcutforest-core` package provides raw estimation (such as an anomaly
score, or extrapolation over a forecast horizon). The `randomcutforest-parkservices`
package builds on that with higher-level capabilities — `ThresholdedRandomCutForest`
and `RCFCaster` — for turning raw scores into anomaly determinations or calibrated
forecasts. The `randomcutforest-examples` package showcases example scenarios and
parameter settings, many of which are built as tests.

## References

* Guha, S., Mishra, N., Roy, G., & Schrijvers, O. (2016, June). Robust random cut
  forest based anomaly detection on streams. In *International Conference on
  Machine Learning* (pp. 2712-2721).

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Please do not report security issues in public issues. See [SECURITY.md](SECURITY.md).

## License

Licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).

## Copyright

Copyright 2019 Amazon.com, Inc. or its affiliates.
Copyright 2026 The streamingalgorithms authors.
