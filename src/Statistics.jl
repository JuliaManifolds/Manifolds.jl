using StatsBase: AbstractWeights, Weights, ProbabilityWeights, values, varcorrection, covm

_unit_weights(n::Int) = ProbabilityWeights(ones(n), n)

@doc doc"""
    GeodesicInterpolationMethod <: AbstractMethod

Repeated weighted geodesic interpolation method for estimating the Riemannian
center of mass.

The algorithm proceeds with the following simple online update:

```math
\begin{aligned}
\mu_1 &= x_1\\
t_k &= \frac{w_k}{\sum_{i=1}^k w_i}\\
\mu_{k} &= \gamma_{\mu_{k-1}}(x_k; t_k),
\end{aligned}
```

where $x_k$ are points, $w_k$ are weights, $\mu_k$ is the $k$th estimate of the
mean, and $\gamma_x(y; t)$ is the point at time $t$ along the
[`shortest_geodesic`](@ref) between points $x,y \in \mathcal M$. The algorithm
terminates when all $x_k$ have been considered. In the [`Euclidean`](@ref) case,
this exactly computes the weighted mean.

The algorithm has been shown to converge asymptotically with the sample size for
the following manifolds equipped with their default metrics when all sampled
points are in an open geodesic ball about the mean with corresponding radius:

* [`Euclidean`](@ref): $\infty$
* [`SymmetricPositiveDefinite`](@ref): $\infty$
* [`Sphere`](@ref): $\frac{\pi}{2}$
* `Grassmannian`: $\frac{\pi}{4}$
* `Stiefel`/[`Rotations`](@ref): $\frac{\pi}{2 \sqrt 2}$

For more information on the geodesic interpolation method, see the following
papers:

> 1. Ho J.; Cheng G.; Salehian H.; Vemuri B. C.; Recursive Karcher expectation
>    estimators and geometric law of large numbers.
>    Proceedings of the 16th International Conference on Artificial Intelligence
>    and Statistics (2013), pp. 325–332.
>    [pdf](http://proceedings.mlr.press/v31/ho13a.pdf).
> 2. Salehian H.; Chakraborty R.; Ofori E.; Vaillancourt D.; An efficient
>    recursive estimator of the Fréchet mean on a hypersphere with applications
>    to Medical Image Analysis.
>    Mathematical Foundations of Computational Anatomy (2015).
>    [pdf](https://www-sop.inria.fr/asclepios/events/MFCA15/Papers/MFCA15_4_2.pdf).
> 3. Chakraborty R.; Vemuri B. C.; Recursive Fréchet Mean Computation on the
>    Grassmannian and Its Applications to Computer Vision.
>    Proceedings of the IEEE International Conference on Computer Vision (ICCV) (2015),
>    pp. 4229-4237.
>    doi: [10.1109/ICCV.2015.481](https://doi.org/10.1109/ICCV.2015.481),
>    [link](http://openaccess.thecvf.com/content_iccv_2015/html/Chakraborty_Recursive_Frechet_Mean_ICCV_2015_paper.html).
> 4. Chakraborty R.; Vemuri B. C.; Statistics on the (compact) Stiefel manifold:
>    Theory and Applications.
>    The Annals of Statistics (2019), 47(1), pp. 415-438.
>    doi: [10.1214/18-AOS1692](https://doi.org/10.1214/18-AOS1692),
>    arxiv: [1708.00045](https://arxiv.org/abs/1708.00045).
"""
struct GeodesicInterpolationMethod <: AbstractMethod end

@doc doc"""
    mean(M::Manifold, x::AbstractVector [, w::AbstractWeights]; kwargs...)

Compute the (optionally weighted) Riemannian center of mass also known as
Karcher mean of the vector `x` of points on the [`Manifold`](@ref) `M`, defined
as the point that satisfies the minimizer
````math
\argmin_{y\in\mathcal M} \frac{1}{2 \sum_{i=1}^n w_i} \sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}^2(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemannian [`distance`](@ref).

In the general case, the [`GradientMethod`](@ref) is used to compute the mean.
However, this default may be overloaded for specific manifolds.

    mean(M::Manifold, x::AbstractVector, [w::AbstractWeights], method::AbstractMethod; kwargs...)

Compute the mean using the specified `method`.

    mean(
        M::Manifold,
        x::AbstractVector,
        w::AbstractWeights,
        method::GradientMethod;
        x0=x[1],
        stop_iter=100,
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
    )

Compute the mean using the gradient descent scheme [`GradientMethod`](@ref).

Optionally, provide `x0`, the starting point (by default set to the first data
point). `stop_iter` denotes the maximal number of iterations to perform and the
`kwargs...` are passed to [`isapprox`](@ref) to stop, when the minimal change
between two iterates is small. For more stopping criteria check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction.

The algorithm is further described in
> Afsari, B; Tron, R.; Vidal, R.: On the Convergence of Gradient
> Descent for Finding the Riemannian Center of Mass,
> SIAM Journal on Control and Optimization (2013), 51(3), pp. 2230–2260,
> doi: [10.1137/12086282X](https://doi.org/10.1137/12086282X),
> arxiv: [1201.0925](https://arxiv.org/abs/1201.0925)
"""
mean(::Manifold, args...)

@doc doc"""
    mean!(M::Manifold, y, x::AbstractVector [, w::AbstractWeights]; kwargs...)
    mean!(M::Manifold, y, x::AbstractVector, [w::AbstractWeights,] method::AbstractMethod; kwargs...)

Compute the [`mean`](@ref) in-place in `y`.
"""
mean!(::Manifold, args...)

function mean(M::Manifold, x::AbstractVector, method::AbstractMethod...; kwargs...)
    y = similar_result(M, mean, x[1])
    return mean!(M, y, x, method...; kwargs...)
end

function mean(M::Manifold, x::AbstractVector, w::AbstractWeights, method::AbstractMethod...; kwargs...)
    y = similar_result(M, mean, x[1])
    return mean!(M, y, x, w, method...; kwargs...)
end

function mean!(M::Manifold, y, x::AbstractVector, method::AbstractMethod...; kwargs...)
    w = _unit_weights(length(x))
    return mean!(M, y, x, w, method...; kwargs...)
end

function mean!(M::Manifold, y, x::AbstractVector, w::AbstractWeights; kwargs...)
    return mean!(M, y, x, w, GradientMethod(); kwargs...)
end

function mean!(
    M::Manifold,
    y,
    x::AbstractVector,
    w::AbstractWeights,
    ::GradientMethod;
    x0 = x[1],
    stop_iter=100,
    retraction::AbstractRetractionMethod = ExponentialRetraction(),
    inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
    kwargs...
)
    n = length(x)
    (length(w) != n) && throw(DimensionMismatch("The number of weights ($(length(w))) does not match the number of points for the mean ($(n))."))
    y === x0 || copyto!(y, x0)
    yold = similar_result(M, mean, y)
    copyto!(yold,y)
    v = zero_tangent_vector(M, y)
    vtmp = copy(v)
    α = w ./ cumsum(w)
    for i=1:stop_iter
        copyto!(yold,y)
        # Online weighted mean
        @inbounds inverse_retract!(M, v, yold, x[1], inverse_retraction)
        @inbounds for j in 2:n
            inverse_retract!(M, vtmp, yold, x[j], inverse_retraction)
            v .+= α[j] .* (vtmp .- v)
        end
        retract!(M, y, yold, v, 0.5, retraction)
        isapprox(M,y,yold; kwargs...) && break
    end
    return y
end

"""
    mean!(
        M::Manifold,
        y,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::GeodesicInterpolationMethod;
        shuffle_rng=nothing,
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
    )

Estimate the Riemannian center of mass of `x` in an online fashion using
repeated weighted geodesic interpolation. See
[`GeodesicInterpolationMethod`](@ref) for details.

If `shuffle_rng` is provided, it is used to shuffle the order in which the
points are considered for computing the mean.

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction.
"""
function mean!(
        M::Manifold,
        y,
        x::AbstractVector,
        w::AbstractWeights,
        ::GeodesicInterpolationMethod;
        shuffle_rng::Union{AbstractRNG,Nothing} = nothing,
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
)
    n = length(x)
    (length(w) != n) && throw(DimensionMismatch("The number of weights ($(length(w))) does not match the number of points for the mean ($(n))."))
    order = shuffle_rng === nothing ? (1:n) : shuffle(shuffle_rng, 1:n)
    @inbounds begin
        j = order[1]
        s = w[j]
        copyto!(y, x[j])
    end
    v = zero_tangent_vector(M, y)
    ytmp = similar_result(M, mean, y)
    @inbounds for i in 2:n
        j = order[i]
        s += w[j]
        t = w[j] / s
        inverse_retract!(M, v, y, x[j], inverse_retraction)
        retract!(M, ytmp, y, v, t, retraction)
        copyto!(y, ytmp)
    end
    return y
end

@doc doc"""
    median(M::Manifold, x::AbstractVector [, w::AbstractWeights]; kwargs...)

Compute the (optionally weighted) Riemannian median of the vector `x` of points on the
[`Manifold`](@ref) `M`, defined as the point that satisfies the minimizer
````math
\argmin_{y\in\mathcal M} \frac{1}{\sum_{i=1}^n w_i} \sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemannian [`distance`](@ref).
This function is nonsmooth (i.e nondifferentiable).

In the general case, the [`CyclicProximalPointMethod`](@ref) is used to compute the
median. However, this default may be overloaded for specific manifolds.

    median(M::Manifold, x::AbstractVector, [w::AbstractWeights,] method::AbstractMethod; kwargs...)

Compute the median using the specified `method`.

    median(
        M::Manifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::CyclicProximalPointMethod;
        x0=x[1],
        stop_iter=1000000,
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
    )

Compute the median using [`CyclicProximalPointMethod`](@ref).

Optionally, provide `x0`, the starting point (by default set to the first
data point). `stop_iter` denotes the maximal number of iterations to perform
and the `kwargs...` are passed to [`isapprox`](@ref) to stop, when the minimal
change between two iterates is small. For more stopping criteria check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction.

The algorithm is further described in Algorithm 4.3 and 4.4 in
> Bačák, M: Computing Medians and Means in Hadamard Spaces.
> SIAM Journal on Optimization (2014), 24(3), pp. 1542–1566,
> doi: [10.1137/140953393](https://doi.org/10.1137/140953393),
> arxiv: [1210.2145](https://arxiv.org/abs/1210.2145)
"""
median(::Manifold, args...)

@doc doc"""
    median!(M::Manifold, y, x::AbstractVector [, w::AbstractWeights]; kwargs...)
    median!(M::Manifold, y, x::AbstractVector, [w::AbstractWeights,] method::AbstractMethod; kwargs...)

computes the [`median`](@ref) in-place in `y`.
"""
median!(::Manifold, args...)

function median(M::Manifold, x::AbstractVector, method::AbstractMethod...; kwargs...)
    y = similar_result(M, median, x[1])
    return median!(M, y, x, method...; kwargs...)
end

function median(M::Manifold, x::AbstractVector, w::AbstractWeights, method::AbstractMethod...; kwargs...)
    y = similar_result(M, median, x[1])
    return median!(M, y, x, w, method...; kwargs...)
end

function median!(M::Manifold, y, x::AbstractVector, method::AbstractMethod...; kwargs...)
    w = _unit_weights(length(x))
    return median!(M, y, x, w, method...; kwargs...)
end

function median!(M::Manifold, y, x::AbstractVector, w::AbstractWeights; kwargs...)
    return median!(M, y, x, w, CyclicProximalPointMethod(); kwargs...)
end

function median!(
    M::Manifold,
    y,
    x::AbstractVector,
    w::AbstractWeights,
    ::CyclicProximalPointMethod;
    x0=x[1],
    stop_iter=1000000,
    retraction::AbstractRetractionMethod = ExponentialRetraction(),
    inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
    kwargs...
)
    n = length(x)
    (length(w) != n) && throw(DimensionMismatch("The number of weights ($(length(w))) does not match the number of points for the median ($(n))."))
    y === x0 || copyto!(y, x0)
    yold = similar_result(M,median,y)
    ytmp = copy(yold)
    v = zero_tangent_vector(M,y)
    wv = convert(Vector, w) ./ w.sum
    for i=1:stop_iter
        λ =  .5 / i
        copyto!(yold,y)
        for j in 1:n
            @inbounds t = min( λ * wv[j] / distance(M,y,x[j]), 1. )
            @inbounds inverse_retract!(M, v, y, x[j], inverse_retraction)
            retract!(M, ytmp, y, v, t, retraction)
            copyto!(y,ytmp)
        end
        isapprox(M, y, yold; kwargs...) && break
    end
    return y
end

@doc doc"""
    var(M, x, m=mean(M, x); corrected=true, kwargs...)
    var(M, x, w::AbstractWeights, m=mean(M, x, w); corrected=false, kwargs...)

compute the (optionally weighted) variance of a `Vector` `x` of `n` data points
on the [`Manifold`](@ref) `M`, i.e.

````math
\frac{1}{c} \sum_{i=1}^n w_i d_{\mathcal M}^2 (x_i,m),
````
where `c` is a correction term, see
[Statistics.var](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var).
The mean of `x` can be specified as `m`, and the corrected variance
can be activated by setting `corrected=true`. All further `kwargs...` are passed
to the computation of the mean (if that is not provided).
"""
var(M::Manifold, args...)

function var(
    M::Manifold,
    x::AbstractVector,
    w::AbstractWeights,
    m;
    corrected::Bool = false,
)
    wv = convert(Vector, w)
    s = sum(eachindex(x, w)) do i
        return @inbounds w[i] * distance(M, m, x[i])^2
    end
    c = varcorrection(w, corrected)
    return c * s
end

function var(
    M::Manifold,
    x::AbstractVector,
    m;
    corrected::Bool = true,
)
    n = length(x)
    w = _unit_weights(n)
    return var(M, x, w, m; corrected = corrected)
end

var(M::Manifold, x::AbstractVector, w::AbstractWeights; kwargs...) = mean_and_var(M, x, w; kwargs...)[2]
var(M::Manifold, x::AbstractVector; kwargs...) = mean_and_var(M, x; kwargs...)[2]

@doc doc"""
    std(M, x, m=mean(M, x); corrected=true, kwargs...)
    std(M, x, w::AbstractWeights, m=mean(M, x, w); corrected=false, kwargs...)

compute the optionally weighted standard deviation of a `Vector` `x` of `n` data
points on the [`Manifold`](@ref) `M`, i.e.

````math
\sqrt{\frac{1}{c} \sum_{i=1}^n w_i d_{\mathcal M}^2 (x_i,m)},
````
where `c` is a correction term, see
[Statistics.std](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.std).
The mean of `x` can be specified as `m`, and the corrected variance
can be activated by setting `corrected=true`.
"""
std(M::Manifold, args...; kwargs...) = sqrt(var(M, args...; kwargs...))

@doc doc"""
    cov(M, x, m=mean(M, x); corrected=true, kwargs...)
    cov(M, x, w::AbstractWeights, m=mean(M, x, w); corrected=false, kwargs...)

Compute the (optionally weighted) covariance of a `Vector` `x` of `n` data
points on the [`Manifold`](@ref) `M`, i.e.

````math
\frac{1}{c} \sum_{i=1}^n w_i (\log_{\mu} x_i) (\log_{\mu} x_i)^\mathrm{T},
````

where `c` is a correction term, see
[Statistics.cov](https://juliastats.org/StatsBase.jl/stable/cov/#Statistics.cov).
The mean of `x` can be specified as `m`, and the corrected covariance
can be activated by setting `corrected=true`. All further `kwargs...` are passed
to the computation of the mean (if that is not provided).

!!! note
    This function currently only supports manifolds with real vector
    representations.
"""
cov(M::Manifold, args...)

function cov(M::Manifold, x::AbstractVector, w::AbstractWeights, m; corrected::Bool=false)
    v0 = zero_tangent_vector(M, m)
    v = hcat(log.(Ref(M), Ref(m), x)...)
    return covm(v, v0, w, 2; corrected = corrected)
end

function cov(M::Manifold, x::AbstractVector, m; corrected::Bool = true)
    n = length(x)
    w = _unit_weights(n)
    return cov(M, x, w, m; corrected = corrected)
end

cov(M::Manifold, x::AbstractVector, w::AbstractWeights; kwargs...) = mean_and_cov(M, x, w; kwargs...)[2]
cov(M::Manifold, x::AbstractVector; kwargs...) = mean_and_cov(M, x; kwargs...)[2]

@doc doc"""
    mean_and_var(M::Manifold, x::AbstractVector [, w::AbstractWeights]; kwargs...) -> (mean, var)

Compute the [`mean`](@ref) and the [`var`](@ref)iance simultaneously. See those
functions for a description of the arguments.

    mean_and_var(
        M::Manifold,
        x::AbstractVector
        [w::AbstractWeights,]
        method::AbstractMethod;
        kwargs...,
    ) -> (mean, var)

Use the `method` for simultaneously computing the mean and variance. To use
a mean-specific method, call [`mean`](@ref) and then [`var`](@ref).
"""
mean_and_var(M::Manifold, args...)

function mean_and_var(M::Manifold, x::AbstractVector, w::AbstractWeights, method::AbstractMethod...; corrected=false, kwargs...)
    m = mean(M, x, w, method...; kwargs...)
    v = var(M, x, w, m; corrected = corrected)
    return m, v
end

function mean_and_var(M::Manifold, x::AbstractVector, method::AbstractMethod...; corrected=true, kwargs...)
    n = length(x)
    w = _unit_weights(n)
    return mean_and_var(M, x, w, method...; corrected = corrected, kwargs...)
end

@doc doc"""
    mean_and_var(
        M::Manifold,
        x::AbstractVector
        [w::AbstractWeights,]
        method::GeodesicInterpolationMethod;
        shuffle_rng::Union{AbstractRNG,Nothing} = nothing,
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
    ) -> (mean, var)

Use the repeated weighted geodesic interpolation to estimate the mean.
Simultaneously, use a Welford-like recursion to estimate the variance.

If `shuffle_rng` is provided, it is used to shuffle the order in which the
points are considered. Optionally, pass `retraction` and `inverse_retraction`
method types to specify the (inverse) retraction.

See [`GeodesicInterpolationMethod`](@ref) for details on the geodesic
interpolation method.

!!! note
    The Welford algorithm for the variance is experimental and is not guaranteed
    to give accurate results except on [`Euclidean`](@ref).
"""
function mean_and_var(
        M::Manifold,
        x::AbstractVector,
        w::AbstractWeights,
        ::GeodesicInterpolationMethod;
        shuffle_rng::Union{AbstractRNG,Nothing} = nothing,
        corrected = false,
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
)
    n = length(x)
    (length(w) != n) && throw(DimensionMismatch("The number of weights ($(length(w))) does not match the number of points for the mean ($(n))."))
    order = shuffle_rng === nothing ? (1:n) : shuffle(shuffle_rng, 1:n)
    @inbounds begin
        j = order[1]
        s = w[j]
        y = copy(x[j])
    end
    v = zero_tangent_vector(M, y)
    M₂ = zero(eltype(v))
    ytmp = similar_result(M, mean, y)
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
    c = varcorrection(w, corrected)
    σ² = c * M₂
    return y, σ²
end

@doc doc"""
    mean_and_std(M::Manifold, x::AbstractVector [, w::AbstractWeights]; kwargs...) -> (mean, std)

Compute the [`mean`](@ref) and the standard deviation [`std`](@ref)
simultaneously.

    mean_and_std(
        M::Manifold,
        x::AbstractVector
        [w::AbstractWeights,]
        method::AbstractMethod;
        kwargs...,
    ) -> (mean, var)

Use the `method` for simultaneously computing the mean and standard deviation.
To use a mean-specific method, call [`mean`](@ref) and then [`std`](@ref).
"""
function mean_and_std(M::Manifold, args...; kwargs...)
    m, v = mean_and_var(M, args...; kwargs...)
    return m, sqrt(v)
end

@doc doc"""
    mean_and_cov(M::Manifold, x::AbstractVector [, w::AbstractWeights]; kwargs...) -> (mean, cov)

Compute the [`mean`](@ref) and the [`cov`](@ref)ariance simultaneously. See
those functions for a description of the arguments.

!!! note
    This function currently only supports manifolds with real vector
    representations.

    mean_and_cov(
        M::Manifold,
        x::AbstractVector
        [w::AbstractWeights,]
        method::AbstractMethod;
        kwargs...,
    ) -> (mean, cov)

Use the `method` for simultaneously computing the mean and covariance. To use
a mean-specific method, call [`mean`](@ref) and then [`cov`](@ref).
"""
mean_and_cov(M::Manifold, args...)

function mean_and_cov(M::Manifold, x::AbstractVector, w::AbstractWeights, method::AbstractMethod...; corrected=false, kwargs...)
    m = mean(M, x, w, method...; kwargs...)
    Σ = cov(M, x, w, m; corrected = corrected)
    return m, Σ
end

function mean_and_cov(M::Manifold, x::AbstractVector, method::AbstractMethod...; corrected=true, kwargs...)
    n = length(x)
    w = _unit_weights(n)
    return mean_and_cov(M, x, w, method...; corrected = corrected, kwargs...)
end

"""
    moment(M::Manifold, x::AbstractVector, k::Int, [w::AbstractWeights,] m=mean(M, x [, w]))

Compute the `k`th central moment of points in `x` on manifold `M`. Optionally
provide weights `w` and/or a precomputed [`mean`](@ref).
"""
function moment(M::Manifold, x::AbstractVector, k::Int, w::AbstractWeights, m = mean(M, x, k, w))
    s = sum(eachindex(x, w)) do i
        return @inbounds w[i] * distance(M, m, x[i])^k
    end
    return s / w.sum
end

function moment(M::Manifold, x::AbstractVector, k::Int, m = mean(M, x))
    w = _unit_weights(length(x))
    return moment(M, x, k, w, m)
end

"""
    skewness(M::Manifold, x::AbstractVector, k::Int, [w::AbstractWeights,] m=mean(M, x [, w]))

Compute the standardized skewness of points in `x` on manifold `M`. Optionally
provide weights `w` and/or a precomputed [`mean`](@ref) `m`.
"""
function skewness(M::Manifold, x::AbstractVector, w::AbstractWeights)
    m, s = mean_and_std(M, x, w; corrected = false)
    return moment(M, x, 3, w, m) / s^3
end

function skewness(M::Manifold, x::AbstractVector, w::AbstractWeights, m)
    return moment(M, x, 3, w, m) / std(M, x, w, m; corrected = false)^3
end

function skewness(M::Manifold, x::AbstractVector, args...)
    w = _unit_weights(length(x))
    return skewness(M, x, w, args...)
end

"""
    kurtosis(M::Manifold, x::AbstractVector, k::Int, [w::AbstractWeights,] m=mean(M, x [, w]))

Compute the excess kurtosis of points in `x` on manifold `M`. Optionally
provide weights `w` and/or a precomputed [`mean`](@ref) `m`.
"""
function kurtosis(M::Manifold, x::AbstractVector, w::AbstractWeights)
    m, v = mean_and_var(M, x, w; corrected = false)
    return moment(M, x, 4, w, m) / v^2 - 3
end

function kurtosis(M::Manifold, x::AbstractVector, w::AbstractWeights, m)
    return moment(M, x, 4, w, m) / var(M, x, w, m; corrected = false)^2 - 3
end

function kurtosis(M::Manifold, x::AbstractVector, args...)
    w = _unit_weights(length(x))
    return kurtosis(M, x, w, args...)
end
