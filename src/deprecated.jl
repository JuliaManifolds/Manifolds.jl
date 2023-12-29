
@deprecate default_estimation_method(M::AbstractManifold, f) default_approximation_method(
    M,
    f,
)
@deprecate ExtrinsicEstimation() ExtrinsicEstimation(EfficientEstimator())
