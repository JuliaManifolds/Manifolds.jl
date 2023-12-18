
@deprecate default_estimation_method(M::AbstractManifold, f) default_approximation_method(
    M,
    f,
)

@deprecate ExtrinsicEstimation() ExtrinsicEstimation(EfficientEstimator())

function Statistics.mean!(
    M::AbstractManifold,
    y,
    x,
    w,
    ::ExtrinsicEstimation;
    extrinsic_method=default_approximation_mthod(get_embedding(M), mean),
    kwargs...,
)
    Base.depwarn(
        "The Keyword Argument `extrinsic_method` is deprecated use `ExtrinsicEstimators field instead",
        mean!,
    )
    return Statistics.mean!(M, y, x, w, ExtrinsicEstimation(extrinsic_method); kwargs...)
end

function Statistics.median!(
    M::AbstractManifold,
    y,
    x::AbstractVector,
    w::AbstractVector,
    ::ExtrinsicEstimation;
    extrinsic_method::AbstractApproximationMethod=default_approximation_mthod(
        get_embedding(M),
        median,
    ),
    kwargs...,
)
    Base.depwarn(
        "The Keyword Argument `extrinsic_method` is deprecated use `ExtrinsicEstimators field instead",
        mean!,
    )
    return Statistics.median!(M, y, x, w, ExtrinsicEstimation(extrinsic_method); kwargs...)
end
