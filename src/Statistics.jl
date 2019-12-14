using StatsBase: AbstractWeights, Weights, ProbabilityWeights, values, varcorrection

_mean_weights(n::Int) = Weights(ones(n), n)
_var_weights(n::Int) = ProbabilityWeights(ones(n), n)

@doc doc"""
    GeodesicInterpolationMethod <: AbstractMethod

Repeated weighted geodesic interpolation method for estimating the Riemannian
center of mass.

The algorithm proceeds with the following simple online update:

```math
\begin{align}
\mu_1 &= x_1\\
t_k &= \frac{w_k}{\sum_{i=1}^k w_i}\\
\mu_{k} &= \gamma_{\mu_{k-1}}(x_k; t_k),
\end{align}
```

where $x_k$ are points, $w_k$ are weights, $\mu_k$ is the $k$th estimate of the
mean, and $\gamma_x(y; t)$ is the point at time $t$ along the
[`shortest_geodesic`](@ref) between points $x,y \in \mathcal M$. The algorithm
terminates when all $x_k$ have been considered. In the [`Euclidean`](@ref) case,
this exactly computes the weighted mean.

The algorithm has been shown to converge asymptotically with the sample size for
the following manifolds when all sampled points are in an open geodesic ball
about the mean with corresponding radius:

* [`Euclidean`](@ref): $\infty$
* [`SymmetricPositiveDefinite`](@ref) with the [`LinearAffineMetric`](@ref): $\infty$
* [`Sphere`](@ref): $\frac{\pi}{2}$
* `Grassmannian`: $\frac{\pi}{4}$
* `Stiefel`/['Rotations'](@ref): $\frac{\pi}{2 \sqrt 2}$

For more information on the geodesic interpolation method, see the following
papers:

1. Ho J.,; Cheng G.; Salehian H.; Vemuri B. C.; Recursive Karcher expectation
   estimators and geometric law of large numbers.  
   Proceedings of the 16th International Conference on Artificial Intelligence
   and Statistics (2013), pp. 325–332.  
   [pdf](http://proceedings.mlr.press/v31/ho13a.pdf)
2. Salehian H.; Chakraborty R.; Ofori E.; Vaillancourt D.; An efficient
   recursive estimator of the Fréchet mean on a hypersphere with applications
   to Medical Image Analysis.  
   Mathematical Foundations of Computational Anatomy (2015).  
   [pdf](https://www-sop.inria.fr/asclepios/events/MFCA15/Papers/MFCA15_4_2.pdf)
3. Chakraborty R.; Vemuri B. C.; Recursive Fréchet Mean Computation on the
   Grassmannian and Its Applications to Computer Vision.  
   Proceedings of the IEEE International Conference on Computer Vision (ICCV) (2015),
   pp. 4229-4237.  
   doi: [10.1109/ICCV.2015.481](https://doi.org/10.1109/ICCV.2015.481)  
   [pdf](http://openaccess.thecvf.com/content_iccv_2015/html/Chakraborty_Recursive_Frechet_Mean_ICCV_2015_paper.html)
4. Chakraborty R.; Vemuri B. C.; Statistics on the (compact) Stiefel manifold:
   Theory and Applications.  
   The Annals of Statistics (2019), 47(1), pp. 415-438.  
   doi: [10.1214/18-AOS1692](https://doi.org/10.1214/18-AOS1692)  
   arxiv: [1708.00045](https://arxiv.org/abs/1708.00045)
"""
struct GeodesicInterpolationMethod <: AbstractMethod end

@doc doc"""
    mean(M::Manifold, x::AbstractVector, [w::AbstractWeights]; kwargs...)

Compute the (optionally weighted) Riemannian center of mass also known as
Karcher mean of the vector `x` of points on the [`Manifold`](@ref) `M`, defined
as the point that satisfies the minimizer
````math
\argmin_{y\in\mathcal M} \frac{1}{2 \sum w}\sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}^2(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemannian [`distance`](@ref).

In the general case, the [`GradientMethod`](@ref) is used to compute the mean.
However, this default may overloaded for specific manifolds.

    mean(M::Manifold, x::AbstractVector, [w::AbstractWeights], method::AbstractMethod; kwargs...)

Compute the mean using the specified `method`.

    mean(
        M::Manifold,
        x::AbstractVector,
        w::AbstractWeights,
        method::GradientMethod;
        x0=x[1],
        stop_iter=100,
        kwargs...
    )

Compute the mean using the gradient descent scheme [`GradientMethod`](@ref).

Optionally, provide `x0`, the starting point (by default set to the first
data point). `stop_iter` denotes the maximal number of iterations to perform and
the `kwargs...` are passed to [`isapprox`](@ref) to stop, when the minimal change
between two iterates is small. For more stopping criteria
check the [`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

The algorithm is further described in
> Afsari, B; Tron, R.; Vidal, R.: On the Convergence of Gradient
> Descent for Finding the Riemannian Center of Mass,
> SIAM Journal on Control and Optimization (2013), 51(3), pp. 2230–2260,
> doi: [10.1137/12086282X](https://doi.org/10.1137/12086282X),
> arxiv: [1201.0925](https://arxiv.org/abs/1201.0925)
"""
mean

@doc doc"""
    mean!(M::Manifold, y, x::AbstractVector, [w::AbstractWeights]; kwargs...)
    mean!(M::Manifold, y, x::AbstractVector, [w::AbstractWeights], method::AbstractMethod; kwargs...)

Compute the [`mean`](@ref) in-place in `y`.
"""
mean!

function mean(M::Manifold, x::AbstractVector; kwargs...)
    y = similar_result(M, mean, x[1])
    return mean!(M, y, x; kwargs...)
end

function mean(M::Manifold, x::AbstractVector, w::AbstractWeights; kwargs...)
    y = similar_result(M, mean, x[1])
    return mean!(M, y, x, w; kwargs...)
end

function mean(M::Manifold, x::AbstractVector, w::AbstractWeights, method::AbstractMethod; kwargs...)
    y = similar_result(M, mean, x[1])
    return mean!(M, y, x, w, method; kwargs...)
end

function mean(M::Manifold, x::AbstractVector, method::AbstractMethod; kwargs...)
    y = similar_result(M, mean, x[1])
    return mean!(M, y, x, method; kwargs...)
end

function mean!(M::Manifold, y, x::AbstractVector; kwargs...)
    w = _mean_weights(length(x))
    return mean!(M, y, x, w; kwargs...)
end

function mean!(M::Manifold, y, x::AbstractVector, method::AbstractMethod; kwargs...)
    w = _mean_weights(length(x))
    return mean!(M, y, x, w, method; kwargs...)
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
    kwargs...
) where {T}
    x0 === nothing || copyto!(y, x0)
    yold = similar_result(M, mean, y)
    copyto!(yold,y)
    v0 = zero_tangent_vector(M, y)
    v = map(_ -> copy(v0), x)
    wreg = convert(Vector, w) ./ (2 * w.sum)
    for i=1:stop_iter
        copyto!(yold,y)
        log!.(Ref(M), v, Ref(yold), x)
        vreg = sum(wreg .* v)
        exp!(M, y, yold, vreg)
        isapprox(M,y,yold; kwargs...) && break
    end
    return y
end

"""
    mean!(
        M::Manifold,
        y,
        x::AbstractVector,
        w::AbstractWeights,
        method::GeodesicInterpolationMethod;
        shuffle_rng=nothing,
        kwargs...,
    )

Estimate the Riemannian center of mass of `x` in an online fashion using
repeated weighted geodesic interpolation. See
[`GeodesicInterpolationMethod`](@ref) for details.

If `shuffle_rng` is provided, it is used to shuffle the order in which the
points are considered for computing the mean.
"""
function mean!(
        M::Manifold,
        y,
        x::AbstractVector,
        w::AbstractWeights,
        ::GeodesicInterpolationMethod;
        shuffle_rng::Union{AbstractRNG,Nothing} = nothing,
        kwargs...,
)
    n = length(x)
    (length(w) != n) && throw(DimensionMismatch("The number of weights ($(length(w))) does not match the number of points for the median ($(n))."))
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
        log!(M, v, y, x[j])
        exp!(M, ytmp, y, v, t)
        copyto!(y, ytmp)
    end
    return y
end

@doc doc"""
    median(M::Manifold, x::AbstractVector, [w::AbstractWeights]; kwargs...)

Compute the (optionally weighted) Riemannian median of the vector `x` of points on the
[`Manifold`](@ref) `M`, defined as the point that satisfies the minimizer
````math
\argmin_{y\in\mathcal M} \frac{1}{\sum w} \sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemannian [`distance`](@ref).
This function is nonsmooth (i.e nondifferentiable).

In the general case, the [`CyclicProximalPointMethod`](@ref) is used to compute the
median. However, this default may overloaded for specific manifolds.

    median(M::Manifold, x::AbstractVector, [w::AbstractWeights], method::AbstractMethod; kwargs...)

Compute the median using the specified `method`.

    median(
        M::Manifold,
        x::AbstractVector,
        w::AbstractWeights,
        method::CyclicProximalPointMethod;
        x0=x[1],
        stop_iter=1000000,
        kwargs...
    )

Compute the median using [`CyclicProximalPointMethod`](@ref).

Optionally, provide `x0`, the starting point (by default set to the first
data point). Set `x0` to `nothing` to use the intitial value of `y` as the
starting point. `stop_iter` denotes the maximal number of iterations to perform
and the `kwargs...` are passed to [`isapprox`](@ref) to stop, when the minimal
change between two iterates is small. For more stopping criteria check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

The algorithm is further described in Algorithm 4.3 and 4.4 in
> Bačák, M: Computing Medians and Means in Hadamard Spaces.
> SIAM Journal on Optimization (2014), 24(3), pp. 1542–1566,
> doi: [10.1137/140953393](https://doi.org/10.1137/140953393),
> arxiv: [1210.2145](https://arxiv.org/abs/1210.2145)
"""
median

@doc doc"""
    median!(M::Manifold, y, x::AbstractVector, [w::AbstractWeights]; kwargs...)
    median!(M::Manifold, y, x::AbstractVector, [w::AbstractWeights], method::AbstractMethod; kwargs...)

computes the [`median`](@ref) in-place in `y`.
"""
median!

function median(M::Manifold, x::AbstractVector; kwargs...)
    y = similar_result(M, median, x[1])
    return median!(M, y, x; kwargs...)
end

function median(M::Manifold, x::AbstractVector, w::AbstractWeights; kwargs...)
    y = similar_result(M, median, x[1])
    return median!(M, y, x, w; kwargs...)
end

function median(M::Manifold, x::AbstractVector, w::AbstractWeights, method::AbstractMethod; kwargs...)
    y = similar_result(M, median, x[1])
    return median!(M, y, x, w, method; kwargs...)
end

function median(M::Manifold, x::AbstractVector, method::AbstractMethod; kwargs...)
    y = similar_result(M, median, x[1])
    return median!(M, y, x, median; kwargs...)
end

function median!(M::Manifold, y, x::AbstractVector; kwargs...)
    w = _mean_weights(length(x))
    return median!(M, y, x, w; kwargs...)
end

function median!(M::Manifold, y, x::AbstractVector, method::AbstractMethod; kwargs...)
    w = _mean_weights(length(x))
    return median!(M, y, x, w, method; kwargs...)
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
    kwargs...
) where {T}
    x0 === nothing || copyto!(y, x0)
    n = length(x)
    yold = similar_result(M,median,y)
    ytmp = copy(yold)
    (length(w) != n) && throw(DimensionMismatch("The number of weights ($(length(w))) does not match the number of points for the median ($(n))."))
    v = zero_tangent_vector(M,y)
    wv = convert(Vector, w) ./ w.sum
    for i=1:stop_iter
        λ =  .5 / i
        copyto!(yold,y)
        for j in 1:n
            @inbounds t = min( λ * wv[j] / distance(M,y,x[j]), 1. )
            @inbounds log!(M, v, y, x[j])
            exp!(M, ytmp, y, v, t)
            copyto!(y,ytmp)
        end
        isapprox(M, y, yold; kwargs...) && break
    end
    return y
end

@doc doc"""
    var(M, x, w::AbstractWeights, m=nothing; corrected=false, kwargs...)

compute the (weighted) variance of a `Vector` `x` of `n` data points on the
[`Manifold`](@ref) `M`, i.e.

````math
\frac{1}{c} \sum_{i=1}^n w_i d_{\mathcal M}^2 (x_i,m),
````
where `c` is a correction term, see
[Statistics.var](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var).
The (weighted) mean of `x` can be specified as `m`, and the corrected variance
can be activated by setting `corrected=true`. All further `kwargs...` are passed
to the computation of the mean (if that is not provided).
"""
function var(
    M::Manifold,
    x::AbstractVector,
    w::AbstractWeights,
    m = nothing;
    corrected::Bool = false,
    kwargs...
)
    if (m === nothing)
        m = mean(M, x, w; kwargs...)
    end
    wv = convert(Vector, w)
    sqdist = wv .* distance.(Ref(M), Ref(m), x) .^ 2
    s = sum(sqdist)
    c = varcorrection(w, corrected)
    return c * s
end

@doc doc"""
    var(M, x, m=nothing; corrected=true, kwargs...)

compute the variance of a `Vector` `x` of `n` data points on the
[`Manifold`](@ref) `M`, i.e.

````math
\frac{1}{c} \sum_{i=1}^n d_{\mathcal M}^2 (x_i,m),
````
where `c` is a correction term, see
[Statistics.var](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var).
and `m` is the provideed mean of `x`. The uncorrected variance
can be activated by setting `corrected=false`.
"""
function var(
    M::Manifold,
    x::AbstractVector,
    m = nothing;
    corrected::Bool = true,
    kwargs...
)
    if (m === nothing)
        m = mean(M, x; kwargs...)
    end
    n = length(x)
    w = _var_weights(n)
    return var(M, x, w, m; corrected = corrected, kwargs...)
end

@doc doc"""
    std(M, x, w::AbstractWeights, m=nothing; corrected=false, kwargs...)

compute the (weighted) standard deviation of a `Vector` `x` of `n` data points on the
[`Manifold`](@ref) `M`, i.e.

````math
\sqrt{\frac{1}{c} \sum_{i=1}^n w_i d_{\mathcal M}^2 (x_i,m)},
````
where `c` is a correction term, see
[Statistics.var](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var).
The (weighted) mean of `x` can be specified as `m`, and the corrected variance
can be activated by setting `corrected=true`. All further `kwargs...` are passed
to the computation of the mean (if that is not provided).
"""
function std(
    M::Manifold,
    x::AbstractVector,
    w::AbstractWeights,
    m = nothing;
    corrected::Bool = false,
    kwargs...
)
    return sqrt(var(M, x, w, m; corrected = corrected, kwargs...))
end

@doc doc"""
    std(M, x, m=nothing; corrected=true, kwargs...)

compute the variance of a `Vector` `x` of `n` data points on the
[`Manifold`](@ref) `M`, i.e.

````math
\sqrt{\frac{1}{c} \sum_{i=1}^n d_{\mathcal M}^2 (x_i,m)},
````
where `c` is a correction term, see
[Statistics.var](https://juliastats.org/StatsBase.jl/stable/scalarstats/#Statistics.var).
and `m` is the provideed mean of `x`.
"""
function std(
    M::Manifold,
    x::AbstractVector,
    m = nothing;
    corrected::Bool = true,
    kwargs...
)
    return sqrt(var(M, x, m; corrected = corrected, kwargs...))
end

@doc doc"""
    mean_and_var(M, x, w; corrected = false, kwargs...) -> (mean, var)
    mean_and_var(M, x; corrected = true, kwargs...) -> (mean, var)

compute the [`mean`](@ref) `m` and the [`var`](@ref)iance `v` simultaneously.
"""
function mean_and_var(
    M::Manifold,
    x::AbstractVector,
    w::AbstractWeights;
    corrected = false,
    kwargs...
)
    m = mean(M, x, w; kwargs...)
    v = var(M, x, w, m; corrected = corrected, kwargs...)
    return m, v
end

function mean_and_var(
    M::Manifold,
    x::AbstractVector;
    corrected = true,
    kwargs...
)
    n = length(x)
    w = _var_weights(n)
    m = mean(M, x, w; kwargs...)
    v = var(M, x, w, m; corrected = corrected, kwargs...)
    return m, v
end

@doc doc"""
    mean_and_std(M, x, w; corrected = false, kwargs...) -> (mean, std)
    mean_and_std(M, x; corrected = true, kwargs...) -> (mean, std)

compute the [`mean`](@ref) `m` and the standard deviation [`std`](@ref) `s`
simultaneously.
"""
function mean_and_std(
    M::Manifold,
    x::AbstractVector,
    w::AbstractWeights;
    corrected = false,
    kwargs...
)
    m = mean(M, x, w; kwargs...)
    s = std(M, x, w, m; corrected = corrected, kwargs...)
    return m, s
end

function mean_and_std(
    M::Manifold,
    x::AbstractVector;
    corrected = true,
    kwargs...
)
    m, v = mean_and_var(M, x; corrected = corrected, kwargs...)
    return m, √v
end
