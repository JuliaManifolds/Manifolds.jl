"""
    AbstractEstimationMethod

Deprecated alias for `AbstractApproximationMethod`
"""
const AbstractEstimationMethod = AbstractApproximationMethod

_unit_weights(n::Int) = StatsBase.UnitWeights{Float64}(n)

function Base.show(io::IO, method::GeodesicInterpolationWithinRadius)
    return print(io, "GeodesicInterpolationWithinRadius($(method.radius))")
end

"""
    Statistics.cov(
        M::AbstractManifold,
        x::AbstractVector;
        basis::AbstractBasis=DefaultOrthonormalBasis(),
        tangent_space_covariance_estimator::CovarianceEstimator=SimpleCovariance(;
            corrected=true,
        ),
        mean_estimation_method::AbstractApproximationMethod=GradientDescentEstimation(),
        inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
            M, eltype(x),
        ),
    )

Estimate the covariance matrix of a set of points `x` on manifold `M`. Since the covariance matrix
on a manifold is a rank 2 tensor, the function returns its coefficients in basis induced by
the given tangent space basis. See Section 5 of [Pennec:2006](@cite) for details.

The mean is calculated using the specified `mean_estimation_method` using
[mean](@ref Statistics.mean(::AbstractManifold, ::AbstractVector, ::AbstractApproximationMethod),
and tangent vectors at this mean are calculated using the provided `inverse_retraction_method`.
Finally, the covariance matrix in the tangent plane is estimated using the Euclidean space
 estimator `tangent_space_covariance_estimator`. The type `CovarianceEstimator` is defined
 in [`StatsBase.jl`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
 and examples of covariance estimation methods can be found in
 [`CovarianceEstimation.jl`](https://github.com/mateuszbaran/CovarianceEstimation.jl/).
"""
function Statistics.cov(
    M::AbstractManifold,
    x::AbstractVector;
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    tangent_space_covariance_estimator::CovarianceEstimator=SimpleCovariance(;
        corrected=true,
    ),
    mean_estimation_method::AbstractApproximationMethod=default_approximation_method(
        M,
        cov,
    ),
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M,
        eltype(x),
    ),
)
    μ = mean(M, x, mean_estimation_method)
    Xs_at_μ = map(p -> inverse_retract(M, μ, p, inverse_retraction_method), x)
    Xcs = map(X -> get_coordinates(M, μ, X, basis), Xs_at_μ)
    return cov(
        tangent_space_covariance_estimator,
        reduce(hcat, Xcs);
        mean=zero(Xcs[1]),
        dims=2,
    )
end

function default_approximation_method(
    ::EmptyTrait,
    ::AbstractDecoratorManifold,
    ::typeof(cov),
)
    return GradientDescentEstimation()
end
function default_approximation_method(::AbstractManifold, ::typeof(cov))
    return GradientDescentEstimation()
end

@doc raw"""
    mean(M::AbstractManifold, x::AbstractVector[, w::AbstractWeights]; kwargs...)

Compute the (optionally weighted) Riemannian center of mass also known as
Karcher mean of the vector `x` of points on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`, defined
as the point that satisfies the minimizer
````math
\argmin_{y ∈ \mathcal M} \frac{1}{2 \sum_{i=1}^n w_i} \sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}^2(y,x_i),
````
where ``\mathrm{d}_{\mathcal M}`` denotes the Riemannian [`distance`](@ref).

In the general case, the [`GradientDescentEstimation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.GradientDescentEstimation) is used to compute the mean.
    mean(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::AbstractApproximationMethod=default_approximation_method(M, mean);
        kwargs...,
    )

Compute the mean using the specified `method`.

    mean(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::GradientDescentEstimation;
        p0=x[1],
        stop_iter=100,
        retraction::AbstractRetractionMethod = default_retraction_method(M),
        inverse_retraction::AbstractInverseRetractionMethod = default_retraction_method(M, eltype(x)),
        kwargs...,
    )

Compute the mean using the gradient descent scheme [`GradientDescentEstimation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.GradientDescentEstimation).

Optionally, provide `p0`, the starting point (by default set to the first data
point). `stop_iter` denotes the maximal number of iterations to perform and the
`kwargs...` are passed to [`isapprox`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#Base.isapprox-Tuple{AbstractManifold,%20Any,%20Any}) to stop, when the minimal change
between two iterates is small. For more stopping criteria check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction.

The Theory stems from [Karcher:1977](@cite) and is also described in [PennecArsigny:2012](@cite)
as the exponential barycenter.
The algorithm is further described in[AfsariTronVidal:2013](@cite).
"""
mean(::AbstractManifold, ::Any...)

#
# dispatch on method first to allow Euclidean defaults to hit
function Statistics.mean(M::AbstractManifold, x::AbstractVector, kwargs...)
    return mean(M, x, default_approximation_method(M, mean, eltype(x)); kwargs...)
end
function Statistics.mean(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractVector,
    kwargs...,
)
    return mean(M, x, w, default_approximation_method(M, mean, eltype(x)); kwargs...)
end

function Statistics.mean(
    M::AbstractManifold,
    x::AbstractVector,
    method::AbstractApproximationMethod;
    kwargs...,
)
    y = allocate_result(M, mean, x[1])
    return mean!(M, y, x, method; kwargs...)
end
function Statistics.mean(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractVector,
    method::AbstractApproximationMethod;
    kwargs...,
)
    y = allocate_result(M, mean, x[1])
    return mean!(M, y, x, w, method; kwargs...)
end

@doc raw"""
    mean!(M::AbstractManifold, y, x::AbstractVector[, w::AbstractWeights]; kwargs...)
    mean!(
        M::AbstractManifold,
        y,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::AbstractApproximationMethod;
        kwargs...,
    )

Compute the [`mean`](@ref mean(::AbstractManifold, args...)) in-place in `y`.
"""
mean!(::AbstractManifold, ::Any...)

function Statistics.mean!(
    M::AbstractManifold,
    y,
    x::AbstractVector,
    method::AbstractApproximationMethod=default_approximation_method(M, mean);
    kwargs...,
)
    w = _unit_weights(length(x))
    return mean!(M, y, x, w, method; kwargs...)
end
function Statistics.mean!(
    M::AbstractManifold,
    y,
    x::AbstractVector,
    w::AbstractVector,
    ::GradientDescentEstimation;
    p0=x[1],
    stop_iter=100,
    retraction::AbstractRetractionMethod=default_retraction_method(M, eltype(x)),
    inverse_retraction::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M,
        eltype(x),
    ),
    kwargs...,
)
    n = length(x)
    if length(w) != n
        throw(
            DimensionMismatch(
                "The number of weights ($(length(w))) does not match the number of points for the mean ($(n)).",
            ),
        )
    end
    copyto!(y, p0)
    yold = allocate_result(M, mean, y)
    v = zero_vector(M, y)
    vtmp = copy(v)
    α = w ./ cumsum(w)
    for _ in 1:stop_iter
        copyto!(yold, y)
        # Online weighted mean
        @inbounds inverse_retract!(M, v, yold, x[1], inverse_retraction)
        @inbounds for j in 2:n
            iszero(w[j]) && continue
            inverse_retract!(M, vtmp, yold, x[j], inverse_retraction)
            v .+= α[j] .* (vtmp .- v)
        end
        retract!(M, y, yold, v, 0.5, retraction)
        isapprox(M, y, yold; kwargs...) && break
    end
    return y
end

"""
    mean(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::GeodesicInterpolation;
        shuffle_rng=nothing,
        retraction::AbstractRetractionMethod = default_retraction_method(M, eltype(x)),
        inverse_retraction::AbstractInverseRetractionMethod = default_inverse_retraction_method(M, eltype(x)),
        kwargs...,
    )

Estimate the Riemannian center of mass of `x` in an online fashion using
repeated weighted geodesic interpolation. See
[`GeodesicInterpolation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.GeodesicInterpolation) for details.

If `shuffle_rng` is provided, it is used to shuffle the order in which the
points are considered for computing the mean.

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction.
"""
mean(::AbstractManifold, ::AbstractVector, ::AbstractVector, ::GeodesicInterpolation)

function Statistics.mean!(
    M::AbstractManifold,
    q,
    x::AbstractVector,
    w::AbstractVector,
    ::GeodesicInterpolation;
    shuffle_rng::Union{AbstractRNG,Nothing}=nothing,
    retraction::AbstractRetractionMethod=default_retraction_method(M, eltype(x)),
    inverse_retraction::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M,
        eltype(x),
    ),
    kwargs...,
)
    n = length(x)
    if length(w) != n
        throw(
            DimensionMismatch(
                "The number of weights ($(length(w))) does not match the number of points for the mean ($(n)).",
            ),
        )
    end
    order = shuffle_rng === nothing ? (1:n) : shuffle(shuffle_rng, 1:n)
    @inbounds begin
        j = order[1]
        s = w[j]
        copyto!(q, x[j])
    end
    v = zero_vector(M, q)
    ytmp = allocate_result(M, mean, q)
    @inbounds for i in 2:n
        iszero(w[i]) && continue
        j = order[i]
        s += w[j]
        t = w[j] / s
        inverse_retract!(M, v, q, x[j], inverse_retraction)
        retract!(M, ytmp, q, v, t, retraction)
        copyto!(q, ytmp)
    end
    return q
end

"""
    mean(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::GeodesicInterpolationWithinRadius;
        kwargs...,
    )

Estimate the Riemannian center of mass of `x` using
[`GeodesicInterpolationWithinRadius`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.GeodesicInterpolationWithinRadius).

See [`mean`](@ref mean(::AbstractManifold, ::AbstractVector, ::AbstractVector, ::GeodesicInterpolation))
for a description of `kwargs`.
"""
Statistics.mean(
    ::AbstractManifold,
    ::AbstractVector,
    ::AbstractVector,
    ::GeodesicInterpolationWithinRadius,
)

function Statistics.mean!(
    M::AbstractManifold,
    q,
    x::AbstractVector,
    w::AbstractVector,
    method::GeodesicInterpolationWithinRadius;
    shuffle_rng=nothing,
    kwargs...,
)
    mean!(M, q, x, w, GeodesicInterpolation(); shuffle_rng=shuffle_rng, kwargs...)
    radius = method.radius
    injectivity_radius(M, q) ≤ radius && return q
    for i in eachindex(x)
        @inbounds if distance(M, q, x[i]) ≥ radius
            return mean!(M, q, x, w, GradientDescentEstimation(); p0=q, kwargs...)
        end
    end
    return q
end
function Statistics.mean!(
    M::AbstractManifold,
    q,
    x::AbstractVector,
    w::AbstractVector,
    ::CyclicProximalPointEstimation;
    p0=x[1],
    stop_iter=1000000,
    retraction::AbstractRetractionMethod=default_retraction_method(M, eltype(x)),
    inverse_retraction::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M,
        eltype(x),
    ),
    kwargs...,
)
    n = length(x)
    if length(w) != n
        throw(
            DimensionMismatch(
                "The number of weights ($(length(w))) does not match the number of points for the mean ($(n)).",
            ),
        )
    end
    copyto!(q, p0)
    yold = allocate_result(M, mean, q)
    ytmp = copy(yold)
    X = zero_vector(M, q)
    wv = convert(AbstractVector, w) ./ sum(w)
    for i in 1:stop_iter
        λ = 0.5 / i
        copyto!(yold, q)
        for j in 1:n
            @inbounds t = (2 * λ * wv[j]) / (1 + 2 * λ * wv[j])
            @inbounds inverse_retract!(M, X, q, x[j], inverse_retraction)
            retract!(M, ytmp, q, X, t, retraction)
            copyto!(q, ytmp)
        end
        isapprox(M, q, yold; kwargs...) && break
    end
    return q
end

"""
    mean(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::ExtrinsicEstimation;
        kwargs...,
    )

Estimate the Riemannian center of mass of `x` using
[`ExtrinsicEstimation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.ExtrinsicEstimation), i.e. by computing the mean in the embedding and projecting
the result back.

See [`mean`](@ref mean(::AbstractManifold, ::AbstractVector, ::AbstractVector, ::GeodesicInterpolation))
for a description of the remaining `kwargs`.
"""
Statistics.mean(
    ::AbstractManifold,
    ::AbstractVector,
    ::AbstractVector,
    ::ExtrinsicEstimation,
)

function Statistics.mean!(
    M::AbstractManifold,
    y,
    x::AbstractVector,
    w::AbstractVector,
    e::ExtrinsicEstimation;
    extrinsic_method::Union{AbstractEstimationMethod,Nothing}=nothing,
    kwargs...,
)
    if !isnothing(extrinsic_method)
        Base.depwarn(
            "The Keyword Argument `extrinsic_method` is deprecated use `ExtrinsicEstimators` field instead",
            :mean!,
        )
        e = ExtrinsicEstimation(extrinsic_method)
    end
    embedded_x = map(p -> embed(M, p), x)
    embedded_y = mean(get_embedding(M), embedded_x, w, e.extrinsic_estimation; kwargs...)
    project!(M, y, embedded_y)
    return y
end

function default_approximation_method(::EmptyTrait, ::AbstractManifold, ::typeof(mean))
    return GradientDescentEstimation()
end;
function default_approximation_method(::AbstractManifold, ::typeof(mean))
    return GradientDescentEstimation()
end;

@doc raw"""
    median(M::AbstractManifold, x::AbstractVector[, w::AbstractWeights]; kwargs...)
    median(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::AbstractApproximationMethod;
        kwargs...,
    )

Compute the (optionally weighted) Riemannian median of the vector `x` of points on the
[`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`, defined as the point that satisfies the minimizer
````math
\argmin_{y ∈ \mathcal M} \frac{1}{\sum_{i=1}^n w_i} \sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}(y,x_i),
````
where ``\mathrm{d}_{\mathcal M}`` denotes the Riemannian [`distance`](@ref).
This function is nonsmooth (i.e nondifferentiable).

In the general case, the [`CyclicProximalPointEstimation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.CyclicProximalPointEstimation) is used to compute the
median. However, this default may be overloaded for specific manifolds.

Compute the median using the specified `method`.
"""
Statistics.median(::AbstractManifold, ::Any...)

function default_approximation_method(
    ::EmptyTrait,
    ::AbstractDecoratorManifold,
    ::typeof(median),
)
    return CyclicProximalPointEstimation()
end
function default_approximation_method(::AbstractManifold, ::typeof(median))
    return CyclicProximalPointEstimation()
end

"""
    median(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::CyclicProximalPointEstimation;
        p0=x[1],
        stop_iter=1000000,
        retraction::AbstractRetractionMethod = default_retraction_method(M, eltype(x),),
        inverse_retraction::AbstractInverseRetractionMethod = default_inverse_retraction_method(M, eltype(x),),
        kwargs...,
    )

Compute the median using [`CyclicProximalPointEstimation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.CyclicProximalPointEstimation).

Optionally, provide `p0`, the starting point (by default set to the first
data point). `stop_iter` denotes the maximal number of iterations to perform
and the `kwargs...` are passed to [`isapprox`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#Base.isapprox-Tuple{AbstractManifold,%20Any,%20Any}) to stop, when the minimal
change between two iterates is small. For more stopping criteria check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction.

The algorithm is further described in [Bacak:2014](@cite).


"""
Statistics.median(
    ::AbstractManifold,
    ::AbstractVector,
    ::AbstractVector,
    ::CyclicProximalPointEstimation,
)

"""
    median(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::ExtrinsicEstimation;
        kwargs...,
    )

Estimate the median of `x` using [`ExtrinsicEstimation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.ExtrinsicEstimation), i.e. by computing the median
in the embedding and projecting the result back.

See [`median`](@ref median(::AbstractManifold, ::AbstractVector, ::AbstractVector, ::CyclicProximalPointEstimation))
for a description of `kwargs`.
"""
Statistics.median(
    ::AbstractManifold,
    ::AbstractVector,
    ::AbstractVector,
    ::ExtrinsicEstimation,
)

@doc raw"""
    median(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::WeiszfeldEstimation;
        α = 1.0,
        p0=x[1],
        stop_iter=2000,
        retraction::AbstractRetractionMethod = default_retraction_method(M, eltype(x)),
        inverse_retraction::AbstractInverseRetractionMethod = default_inverse_retraction_method(M, eltype(x)),
        kwargs...,
    )

Compute the median using [`WeiszfeldEstimation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.WeiszfeldEstimation).

Optionally, provide `p0`, the starting point (by default set to the first
data point). `stop_iter` denotes the maximal number of iterations to perform
and the `kwargs...` are passed to [`isapprox`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#Base.isapprox-Tuple{AbstractManifold,%20Any,%20Any}) to stop, when the minimal
change between two iterates is small. For more stopping criteria check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

The parameter ``α\in (0,2]`` is a step size.

The algorithm is further described in [FletcherVenkatasubramanianJoshi:2008](@cite),
especially the update rule in Eq. (6), i.e. Let ``q_{k}`` denote the current
iterate, ``n`` the number of points ``x_1,\ldots,x_n``, and

```math
I_k = \bigl\{ i \in \{1,\ldots,n\} \big| x_i \neq q_k \bigr\}
```

all indices of points that are not equal to the current iterate. Then the update reads
``q_{k+1} = \exp_{q_k}(αX)``, where

```math
X = \frac{1}{s}\sum_{i\in I_k} \frac{w_i}{d_{\mathcal M}(q_k,x_i)}\log_{q_k}x_i
\quad
\text{ with }
\quad
s = \sum_{i\in I_k} \frac{w_i}{d_{\mathcal M}(q_k,x_i)},
```

and where ``\mathrm{d}_{\mathcal M}`` denotes the Riemannian [`distance`](@ref).

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction, which by default use the exponential and logarithmic map,
respectively.
"""
Statistics.median(
    ::AbstractManifold,
    ::AbstractVector,
    ::AbstractVector,
    ::WeiszfeldEstimation,
)

#
# dispatch on the method first before allocating to allow Euclidean defaults to hit
function Statistics.median(M::AbstractManifold, x::AbstractVector; kwargs...)
    return median(M, x, default_approximation_method(M, median, eltype(x)))
end
function Statistics.median(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return median(M, x, w, default_approximation_method(M, median, eltype(x)))
end

function Statistics.median(
    M::AbstractManifold,
    x::AbstractVector,
    method::AbstractApproximationMethod;
    kwargs...,
)
    y = allocate_result(M, median, x[1])
    return median!(M, y, x, method; kwargs...)
end
function Statistics.median(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractVector,
    method::AbstractApproximationMethod;
    kwargs...,
)
    y = allocate_result(M, median, x[1])
    return median!(M, y, x, w, method; kwargs...)
end

@doc raw"""
    median!(M::AbstractManifold, y, x::AbstractVector[, w::AbstractWeights]; kwargs...)
    median!(
        M::AbstractManifold,
        y,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::AbstractApproximationMethod;
        kwargs...,
    )

computes the [`median`](@ref) in-place in `y`.
"""
median!(::AbstractManifold, ::Any...)
function Statistics.median!(
    M::AbstractManifold,
    q,
    x::AbstractVector,
    method::AbstractApproximationMethod=default_approximation_method(M, median, eltype(x));
    kwargs...,
)
    w = _unit_weights(length(x))
    return median!(M, q, x, w, method; kwargs...)
end
function Statistics.median!(
    M::AbstractManifold,
    q,
    x::AbstractVector,
    w::AbstractVector,
    ::CyclicProximalPointEstimation;
    p0=x[1],
    stop_iter=1000000,
    retraction::AbstractRetractionMethod=default_retraction_method(M, eltype(x)),
    inverse_retraction::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M,
        eltype(x),
    ),
    kwargs...,
)
    n = length(x)
    if length(w) != n
        throw(
            DimensionMismatch(
                "The number of weights ($(length(w))) does not match the number of points for the median ($(n)).",
            ),
        )
    end
    copyto!(q, p0)
    yold = allocate_result(M, median, q)
    ytmp = copy(yold)
    v = zero_vector(M, q)
    wv = convert(AbstractVector, w) ./ sum(w)
    for i in 1:stop_iter
        λ = 0.5 / i
        copyto!(yold, q)
        for j in 1:n
            @inbounds t = min(λ * wv[j] / distance(M, q, x[j]), 1.0)
            @inbounds inverse_retract!(M, v, q, x[j], inverse_retraction)
            retract!(M, ytmp, q, v, t, retraction)
            copyto!(q, ytmp)
        end
        isapprox(M, q, yold; kwargs...) && break
    end
    return q
end

function Statistics.median!(
    M::AbstractManifold,
    y,
    x::AbstractVector,
    w::AbstractVector,
    e::ExtrinsicEstimation;
    extrinsic_method=nothing,
    kwargs...,
)
    if !isnothing(extrinsic_method)
        Base.depwarn(
            "The Keyword Argument `extrinsic_method` is deprecated use `ExtrinsicEstimators` field instead",
            :median!,
        )
        e = ExtrinsicEstimation(extrinsic_method)
    end
    embedded_x = map(p -> embed(M, p), x)
    embedded_y = median(get_embedding(M), embedded_x, w, e.extrinsic_estimation; kwargs...)
    project!(M, y, embedded_y)
    return y
end

function Statistics.median!(
    M::AbstractManifold,
    q,
    x::AbstractVector,
    w::AbstractVector,
    ::WeiszfeldEstimation;
    p0=x[1],
    stop_iter=2000,
    α=1.0,
    retraction::AbstractRetractionMethod=default_retraction_method(M, eltype(x)),
    inverse_retraction::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M,
        eltype(x),
    ),
    kwargs...,
)
    n = length(x)
    if length(w) != n
        throw(
            DimensionMismatch(
                "The number of weights ($(length(w))) does not match the number of points for the median ($(n)).",
            ),
        )
    end
    copyto!(q, p0)
    yold = allocate_result(M, median, q)
    ytmp = copy(yold)
    d = zeros(n)
    v = zero_vector(M, q)
    wv = convert(AbstractVector, w) ./ sum(w)
    for i in 1:stop_iter
        d .= [distance(M, q, xi) for xi in x] # compute distances
        # compute new weights / exclude points xi=q
        d .= [di > 0 ? wi / di : zero(typeof(wi / di)) for (di, wi) in zip(d, w)]
        copyto!(yold, q)
        zero_vector!(M, v, q)
        for j in 1:n
            @inbounds v .+= d[j] * inverse_retract(M, q, x[j], inverse_retraction)
        end
        retract!(M, ytmp, q, v, α / sum(d), retraction)
        copyto!(q, ytmp)
        isapprox(M, q, yold; kwargs...) && break
    end
    return q
end

@doc raw"""
    var(M, x, m=mean(M, x); corrected=true)
    var(M, x, w::AbstractWeights, m=mean(M, x, w); corrected=false)

compute the (optionally weighted) variance of a `Vector` `x` of `n` data points
on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`, i.e.

````math
\frac{1}{c} \sum_{i=1}^n w_i d_{\mathcal M}^2 (x_i,m),
````
where `c` is a correction term, see
[Statistics.var](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var).
The mean of `x` can be specified as `m`, and the corrected variance
can be activated by setting `corrected=true`. All further `kwargs...` are passed
to the computation of the mean (if that is not provided).
"""
var(::AbstractManifold, ::Any)
function Statistics.var(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractWeights,
    m;
    corrected::Bool=false,
)
    wv = convert(AbstractVector, w)
    s = sum(eachindex(x, w)) do i
        return @inbounds w[i] * distance(M, m, x[i])^2
    end
    c = StatsBase.varcorrection(w, corrected)
    return c * s
end
function Statistics.var(M::AbstractManifold, x::AbstractVector, m; corrected::Bool=true)
    n = length(x)
    w = _unit_weights(n)
    return var(M, x, w, m; corrected=corrected)
end
function Statistics.var(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractWeights;
    kwargs...,
)
    return mean_and_var(M, x, w; kwargs...)[2]
end
function Statistics.var(M::AbstractManifold, x::AbstractVector; kwargs...)
    return mean_and_var(M, x; kwargs...)[2]
end

@doc raw"""
    std(M, x, m=mean(M, x); corrected=true, kwargs...)
    std(M, x, w::AbstractWeights, m=mean(M, x, w); corrected=false, kwargs...)

compute the optionally weighted standard deviation of a `Vector` `x` of `n` data
points on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  `M`, i.e.

````math
\sqrt{\frac{1}{c} \sum_{i=1}^n w_i d_{\mathcal M}^2 (x_i,m)},
````
where `c` is a correction term, see
[Statistics.std](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std).
The mean of `x` can be specified as `m`, and the corrected variance
can be activated by setting `corrected=true`.
"""
Statistics.std(M::AbstractManifold, args...; kwargs...) = sqrt(var(M, args...; kwargs...))

@doc raw"""
    mean_and_var(M::AbstractManifold, x::AbstractVector[, w::AbstractWeights]; kwargs...) -> (mean, var)

Compute the [`mean`](@ref mean(::AbstractManifold, args...)) and the [`var`](@ref)iance
simultaneously. See those functions for a description of the arguments.

    mean_and_var(
        M::AbstractManifold,
        x::AbstractVector
        [w::AbstractWeights,]
        method::AbstractApproximationMethod;
        kwargs...,
    ) -> (mean, var)

Use the `method` for simultaneously computing the mean and variance. To use
a mean-specific method, call [`mean`](@ref mean(::AbstractManifold, args...)) and then
[`var`](@ref).
"""
mean_and_var(M::AbstractManifold, ::Any...)
function StatsBase.mean_and_var(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractWeights,
    method::AbstractApproximationMethod=default_approximation_method(M, mean, eltype(x));
    corrected=false,
    kwargs...,
)
    m = mean(M, x, w, method; kwargs...)
    v = var(M, x, w, m; corrected=corrected)
    return m, v
end
function StatsBase.mean_and_var(
    M::AbstractManifold,
    x::AbstractVector,
    method::AbstractApproximationMethod=default_approximation_method(
        M,
        mean_and_var,
        eltype(x),
    );
    corrected=true,
    kwargs...,
)
    n = length(x)
    w = _unit_weights(n)
    return mean_and_var(M, x, w, method; corrected=corrected, kwargs...)
end
function default_approximation_method(
    ::EmptyTrait,
    M::AbstractDecoratorManifold,
    ::typeof(mean_and_var),
)
    return default_approximation_method(M, mean)
end
function default_approximation_method(M::AbstractManifold, ::typeof(mean_and_var))
    return default_approximation_method(M, mean)
end

@doc raw"""
    mean_and_var(
        M::AbstractManifold,
        x::AbstractVector
        [w::AbstractWeights,]
        method::GeodesicInterpolation;
        shuffle_rng::Union{AbstractRNG,Nothing} = nothing,
        retraction::AbstractRetractionMethod = default_retraction_method(M, eltype(x)),
        inverse_retraction::AbstractInverseRetractionMethod = default_inverse_retraction_method(M, eltype(x)),
        kwargs...,
    ) -> (mean, var)

Use the repeated weighted geodesic interpolation to estimate the mean.
Simultaneously, use a Welford-like recursion to estimate the variance.

If `shuffle_rng` is provided, it is used to shuffle the order in which the
points are considered. Optionally, pass `retraction` and `inverse_retraction`
method types to specify the (inverse) retraction.

See [`GeodesicInterpolation`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.GeodesicInterpolation) for details on the geodesic
interpolation method.

!!! note
    The Welford algorithm for the variance is experimental and is not guaranteed
    to give accurate results except on [`Euclidean`](@ref).
"""
function StatsBase.mean_and_var(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractWeights,
    ::GeodesicInterpolation;
    shuffle_rng::Union{AbstractRNG,Nothing}=nothing,
    corrected=false,
    retraction::AbstractRetractionMethod=default_retraction_method(M, eltype(x)),
    inverse_retraction::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M,
        eltype(x),
    ),
    kwargs...,
)
    n = length(x)
    if length(w) != n
        throw(
            DimensionMismatch(
                "The number of weights ($(length(w))) does not match the number of points for the mean ($(n)).",
            ),
        )
    end
    order = shuffle_rng === nothing ? (1:n) : shuffle(shuffle_rng, 1:n)
    @inbounds begin
        j = order[1]
        s = w[j]
        y = copy(x[j])
    end
    v = zero_vector(M, y)
    M₂ = zero(number_eltype(v))
    ytmp = allocate_result(M, mean, y)
    @inbounds for i in 2:n
        j = order[i]
        snew = s + w[j]
        t = w[j] / snew
        inverse_retract!(M, v, y, x[j], inverse_retraction)
        retract!(M, ytmp, y, v, t, retraction)
        d = norm(M, y, v)
        copyto!(y, ytmp)
        M₂ += t * s * d^2
        s = snew
    end
    c = StatsBase.varcorrection(w, corrected)
    σ² = c * M₂
    return y, σ²
end

"""
    mean_and_var(
        M::AbstractManifold,
        x::AbstractVector
        [w::AbstractWeights,]
        method::GeodesicInterpolationWithinRadius;
        kwargs...,
    ) -> (mean, var)

Use repeated weighted geodesic interpolation to estimate the mean.
Simultaneously, use a Welford-like recursion to estimate the variance.

See [`GeodesicInterpolationWithinRadius`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.GeodesicInterpolationWithinRadius) and
[`mean_and_var`](@ref mean_and_var(::AbstractManifold, ::AbstractVector, ::AbstractWeights, ::GeodesicInterpolation))
for more information.
"""
function StatsBase.mean_and_var(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractWeights,
    method::GeodesicInterpolationWithinRadius;
    shuffle_rng=nothing,
    corrected=false,
    kwargs...,
)
    y, v = mean_and_var(
        M,
        x,
        w,
        GeodesicInterpolation();
        shuffle_rng=shuffle_rng,
        corrected=corrected,
        kwargs...,
    )
    radius = method.radius
    injectivity_radius(M, y) ≤ radius && return y, v
    for i in eachindex(x)
        @inbounds if distance(M, y, x[i]) ≥ radius
            mean!(M, y, x, w, GradientDescentEstimation(); p0=y, kwargs...)
            v = var(M, x, w, y; corrected=corrected)
            return y, v
        end
    end
    return y, v
end

@doc raw"""
    mean_and_std(M::AbstractManifold, x::AbstractVector[, w::AbstractWeights]; kwargs...) -> (mean, std)

Compute the [`mean`](@ref mean(::AbstractManifold, args...)) and the standard deviation
[`std`](@ref) simultaneously.

    mean_and_std(
        M::AbstractManifold,
        x::AbstractVector
        [w::AbstractWeights,]
        method::AbstractApproximationMethod;
        kwargs...,
    ) -> (mean, var)

Use the `method` for simultaneously computing the mean and standard deviation.
To use a mean-specific method, call [`mean`](@ref mean(::AbstractManifold, args...)) and
then [`std`](@ref).
"""
function StatsBase.mean_and_std(M::AbstractManifold, args...; kwargs...)
    m, v = mean_and_var(M, args...; kwargs...)
    return m, sqrt(v)
end
function default_approximation_method(M::AbstractManifold, ::typeof(mean_and_std))
    return default_approximation_method(M, mean)
end

"""
    moment(M::AbstractManifold, x::AbstractVector, k::Int[, w::AbstractWeights], m=mean(M, x[, w]))

Compute the `k`th central moment of points in `x` on manifold `M`. Optionally
provide weights `w` and/or a precomputed
[`mean`](@ref mean(::AbstractManifold, args...)).
"""
function StatsBase.moment(
    M::AbstractManifold,
    x::AbstractVector,
    k::Int,
    w::AbstractWeights,
    m=mean(M, x, w),
)
    s = sum(eachindex(x, w)) do i
        return @inbounds w[i] * distance(M, m, x[i])^k
    end
    return s / sum(w)
end
function StatsBase.moment(M::AbstractManifold, x::AbstractVector, k::Int, m=mean(M, x))
    w = _unit_weights(length(x))
    return moment(M, x, k, w, m)
end

"""
    skewness(M::AbstractManifold, x::AbstractVector, k::Int[, w::AbstractWeights], m=mean(M, x[, w]))

Compute the standardized skewness of points in `x` on manifold `M`. Optionally
provide weights `w` and/or a precomputed
[`mean`](@ref mean(::AbstractManifold, args...)) `m`.
"""
function StatsBase.skewness(M::AbstractManifold, x::AbstractVector, w::AbstractWeights)
    m, s = mean_and_std(M, x, w; corrected=false)
    return moment(M, x, 3, w, m) / s^3
end
function StatsBase.skewness(M::AbstractManifold, x::AbstractVector, w::AbstractWeights, m)
    return moment(M, x, 3, w, m) / std(M, x, w, m; corrected=false)^3
end
function StatsBase.skewness(M::AbstractManifold, x::AbstractVector, args...)
    w = _unit_weights(length(x))
    return skewness(M, x, w, args...)
end

"""
    kurtosis(M::AbstractManifold, x::AbstractVector, k::Int[, w::AbstractWeights], m=mean(M, x[, w]))

Compute the excess kurtosis of points in `x` on manifold `M`. Optionally
provide weights `w` and/or a precomputed
[`mean`](@ref mean(::AbstractManifold, args...)) `m`.
"""
function StatsBase.kurtosis(M::AbstractManifold, x::AbstractVector, w::AbstractWeights)
    m, v = mean_and_var(M, x, w; corrected=false)
    return moment(M, x, 4, w, m) / v^2 - 3
end
function StatsBase.kurtosis(M::AbstractManifold, x::AbstractVector, w::AbstractWeights, m)
    return moment(M, x, 4, w, m) / var(M, x, w, m; corrected=false)^2 - 3
end
function StatsBase.kurtosis(M::AbstractManifold, x::AbstractVector, args...)
    w = _unit_weights(length(x))
    return kurtosis(M, x, w, args...)
end

#
# decorate default method for a few functions
for mf in [mean, median, cov, var, mean_and_std, mean_and_var]
    @eval @trait_function default_approximation_method(
        M::AbstractDecoratorManifold,
        f::typeof($mf),
    ) (no_empty,)
    eval(
        quote
            function default_approximation_method(
                ::TraitList{IsEmbeddedSubmanifold},
                M::AbstractDecoratorManifold,
                f::typeof($mf),
            )
                return default_approximation_method(get_embedding(M), f)
            end
        end,
    )
end
@trait_function Statistics.mean(M::AbstractDecoratorManifold, x::AbstractVector)
