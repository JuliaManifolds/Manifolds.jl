"""
    AbstractEstimationMethod

Abstract type for defining statistical estimation methods.
"""
abstract type AbstractEstimationMethod end

"""
    GradientDescentEstimation <: AbstractEstimationMethod

Method for estimation using gradient descent.
"""
struct GradientDescentEstimation <: AbstractEstimationMethod end

"""
    CyclicProximalPointEstimation <: AbstractEstimationMethod

Method for estimation using the cyclic proximal point technique.
"""
struct CyclicProximalPointEstimation <: AbstractEstimationMethod end

"""
    ExtrinsicEstimation <: AbstractEstimationMethod

Method for estimation in the ambient space and projecting to the manifold.

For [`mean`](@ref) estimation, [`GeodesicInterpolation`](@ref) is used for mean estimation
in the ambient space.
"""
struct ExtrinsicEstimation <: AbstractEstimationMethod end

"""
    WeiszfeldEstimation <: AbstractEstimationMethod

Method for estimation using the Weiszfeld algorithm for the [`median`](@ref)
"""
struct WeiszfeldEstimation <: AbstractEstimationMethod end

_unit_weights(n::Int) = StatsBase.UnitWeights{Float64}(n)

@doc raw"""
    GeodesicInterpolation <: AbstractEstimationMethod

Repeated weighted geodesic interpolation method for estimating the Riemannian
center of mass.

The algorithm proceeds with the following simple online update:

```math
\begin{aligned}
μ_1 &= x_1\\
t_k &= \frac{w_k}{\sum_{i=1}^k w_i}\\
μ_{k} &= γ_{μ_{k-1}}(x_k; t_k),
\end{aligned}
```

where $x_k$ are points, $w_k$ are weights, $μ_k$ is the $k$th estimate of the
mean, and $γ_x(y; t)$ is the point at time $t$ along the
[`shortest_geodesic`](@ref shortest_geodesic(::AbstractManifold, ::Any, ::Any, ::Real))
between points $x,y ∈ \mathcal M$. The algorithm
terminates when all $x_k$ have been considered. In the [`Euclidean`](@ref) case,
this exactly computes the weighted mean.

The algorithm has been shown to converge asymptotically with the sample size for
the following manifolds equipped with their default metrics when all sampled
points are in an open geodesic ball about the mean with corresponding radius
(see [`GeodesicInterpolationWithinRadius`](@ref)):

* All simply connected complete Riemannian manifolds with non-positive sectional
  curvature at radius $∞$ [^Cheng2016], in particular:
    + [`Euclidean`](@ref)
    + [`SymmetricPositiveDefinite`](@ref) [^Ho2013]
* Other manifolds:
    + [`Sphere`](@ref): $\frac{π}{2}$ [^Salehian2015]
    + [`Grassmann`](@ref): $\frac{π}{4}$ [^Chakraborty2015]
    + [`Stiefel`](@ref)/[`Rotations`](@ref): $\frac{π}{2 \sqrt 2}$ [^Chakraborty2019]

For online variance computation, the algorithm additionally uses an analogous
recursion to the weighted Welford algorithm [^West1979].

[^Ho2013]:
    > Ho J.; Cheng G.; Salehian H.; Vemuri B. C.; Recursive Karcher expectation
    > estimators and geometric law of large numbers.
    > Proceedings of the 16th International Conference on Artificial Intelligence
    > and Statistics (2013), pp. 325–332.
    > [pdf](http://proceedings.mlr.press/v31/ho13a.pdf).
[^Salehian2015]:
    > Salehian H.; Chakraborty R.; Ofori E.; Vaillancourt D.; An efficient
    > recursive estimator of the Fréchet mean on a hypersphere with applications
    > to Medical Image Analysis.
    > Mathematical Foundations of Computational Anatomy (2015).
    > [pdf](https://www-sop.inria.fr/asclepios/events/MFCA15/Papers/MFCA15_4_2.pdf).
[^Chakraborty2015]:
    > Chakraborty R.; Vemuri B. C.; Recursive Fréchet Mean Computation on the
    > Grassmannian and Its Applications to Computer Vision.
    > Proceedings of the IEEE International Conference on Computer Vision (ICCV) (2015),
    > pp. 4229-4237.
    > doi: [10.1109/ICCV.2015.481](https://doi.org/10.1109/ICCV.2015.481),
    > [link](http://openaccess.thecvf.com/content_iccv_2015/html/Chakraborty_Recursive_Frechet_Mean_ICCV_2015_paper.html).
[^Cheng2016]:
    > Cheng G.; Ho J.; Salehian H.; Vemuri B. C.; Recursive Computation of the
    > Fréchet Mean on Non-positively Curved Riemannian Manifolds with Applications.
    > Riemannian Computing in Computer Vision. Springer, Cham (2016), pp. 21-43.
    > doi: [10.1007/978-3-319-22957-7_2](https://doi.org/10.1007/978-3-319-22957-7_2),
    > [pdf](https://www.cise.ufl.edu/~vemuri/paperphp/article.php?y=2016&i=5).
[^Chakraborty2019]:
    > Chakraborty R.; Vemuri B. C.; Statistics on the (compact) Stiefel manifold:
    > Theory and Applications.
    > The Annals of Statistics (2019), 47(1), pp. 415-438.
    > doi: [10.1214/18-AOS1692](https://doi.org/10.1214/18-AOS1692),
    > arxiv: [1708.00045](https://arxiv.org/abs/1708.00045).
[^West1979]:
    > West D. H. D.; Updating Mean and Variance Estimates: An Improved Method.
    > Communications of the ACM (1979), 22(9), pp. 532–535.
    > doi: [10.1145/359146.359153](https://doi.org/10.1145/359146.359153).
"""
struct GeodesicInterpolation <: AbstractEstimationMethod end

"""
    GeodesicInterpolationWithinRadius{T} <: AbstractEstimationMethod

Estimation of Riemannian center of mass using [`GeodesicInterpolation`](@ref)
with fallback to [`GradientDescentEstimation`](@ref) if any points are outside of a
geodesic ball of specified `radius` around the mean.

# Constructor

    GeodesicInterpolationWithinRadius(radius)
"""
struct GeodesicInterpolationWithinRadius{T} <: AbstractEstimationMethod
    radius::T

    function GeodesicInterpolationWithinRadius(radius::T) where {T}
        radius > 0 && return new{T}(radius)
        return throw(
            DomainError("The radius must be strictly postive, received $(radius)."),
        )
    end
end

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
        mean_estimation_method::AbstractEstimationMethod=GradientDescentEstimation(),
        inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
            M,
        ),
    )

Estimate the covariance matrix of a set of points `x` on manifold `M`. Since the covariance matrix
on a manifold is a rank 2 tensor, the function returns its coefficients in basis induced by
the given tangent space basis. See Section 5 of [^Pennec2006] for details.

The mean is calculated using the specified `mean_estimation_method` using
[mean](@ref Statistics.mean(::AbstractManifold, ::AbstractVector, ::AbstractEstimationMethod),
and tangent vectors at this mean are calculated using the provided `inverse_retraction_method`.
Finally, the covariance matrix in the tangent plane is estimated using the Euclidean space
 estimator `tangent_space_covariance_estimator`. The type `CovarianceEstimator` is defined
 in [`StatsBase.jl`](https://juliastats.org/StatsBase.jl/stable/cov/#StatsBase.CovarianceEstimator)
 and examples of covariance estimation methods can be found in
 [`CovarianceEstimation.jl`](https://github.com/mateuszbaran/CovarianceEstimation.jl/).

[^Pennec2006]:
    > X. Pennec, “Intrinsic Statistics on Riemannian Manifolds: Basic Tools for Geometric
    > Measurements,” J Math Imaging Vis, vol. 25, no. 1, p. 127, Jul. 2006,
    > doi: 10.1007/s10851-006-6228-4.
"""
function Statistics.cov(
    M::AbstractManifold,
    x::AbstractVector;
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    tangent_space_covariance_estimator::CovarianceEstimator=SimpleCovariance(;
        corrected=true,
    ),
    mean_estimation_method::AbstractEstimationMethod=GradientDescentEstimation(),
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M,
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

@doc raw"""
    mean(M::AbstractManifold, x::AbstractVector[, w::AbstractWeights]; kwargs...)

Compute the (optionally weighted) Riemannian center of mass also known as
Karcher mean of the vector `x` of points on the [`AbstractManifold`](@ref) `M`, defined
as the point that satisfies the minimizer
````math
\argmin_{y ∈ \mathcal M} \frac{1}{2 \sum_{i=1}^n w_i} \sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}^2(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemannian [`distance`](@ref).

In the general case, the [`GradientDescentEstimation`](@ref) is used to compute the mean.
    mean(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::AbstractEstimationMethod;
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
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
    )

Compute the mean using the gradient descent scheme [`GradientDescentEstimation`](@ref).

Optionally, provide `p0`, the starting point (by default set to the first data
point). `stop_iter` denotes the maximal number of iterations to perform and the
`kwargs...` are passed to [`isapprox`](@ref) to stop, when the minimal change
between two iterates is small. For more stopping criteria check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction.

The Theory stems from[^Karcher1977] and is also described in[^PennecArsigny2013]
as the exponential barycenter.
The algorithm is further described in[^Afsari2013].

[^Afsari2013]:
    > Afsari, B; Tron, R.; Vidal, R.: On the Convergence of Gradient
    > Descent for Finding the Riemannian Center of Mass,
    > SIAM Journal on Control and Optimization (2013), 51(3), pp. 2230–2260,
    > doi: [10.1137/12086282X](https://doi.org/10.1137/12086282X),
    > arxiv: [1201.0925](https://arxiv.org/abs/1201.0925)
[^PennecArsigny2013]:
    > Pennec X., Arsigny V.: Exponential Barycenters of the Canonical Cartan Connection and
    > Invariant Means on Lie Groups.
    > In: Nielsen F., Bhatia R. (eds) Matrix Information Geometry, (2013), pp. 123-166.
    > doi: [10.1007/978-3-642-30232-9_7](https://doi.org/10.1007/978-3-642-30232-9_7),
    > hal: [https://hal.inria.fr/hal-00699361/document](https://hal.inria.fr/hal-00699361/document)
[^Karcher1977]:
    > Karcher, H.: Riemannian center of mass and mollifier smoothing.
    > Communications on Pure Applied Mathematics (1977), 30, pp. 509–541.
    > doi [10.1002/cpa.3160300502](https://doi.org/10.1002/cpa.3160300502)
"""
mean(::AbstractManifold, ::Any...)
function Statistics.mean(
    M::AbstractManifold,
    x::AbstractVector,
    method::AbstractEstimationMethod...;
    kwargs...,
)
    y = allocate_result(M, mean, x[1])
    return mean!(M, y, x, method...; kwargs...)
end
function Statistics.mean(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractVector,
    method::AbstractEstimationMethod...;
    kwargs...,
)
    y = allocate_result(M, mean, x[1])
    return mean!(M, y, x, w, method...; kwargs...)
end

@doc raw"""
    mean!(M::AbstractManifold, y, x::AbstractVector[, w::AbstractWeights]; kwargs...)
    mean!(
        M::AbstractManifold,
        y,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::AbstractEstimationMethod;
        kwargs...,
    )

Compute the [`mean`](@ref mean(::AbstractManifold, args...)) in-place in `y`.
"""
mean!(::AbstractManifold, ::Any...)
function Statistics.mean!(
    M::AbstractManifold,
    y,
    x::AbstractVector,
    method::AbstractEstimationMethod...;
    kwargs...,
)
    w = _unit_weights(length(x))
    return mean!(M, y, x, w, method...; kwargs...)
end
function Statistics.mean!(
    M::AbstractManifold,
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return mean!(M, y, x, w, GradientDescentEstimation(); kwargs...)
end
function Statistics.mean!(
    M::AbstractManifold,
    y,
    x::AbstractVector,
    w::AbstractVector,
    ::GradientDescentEstimation;
    p0=x[1],
    stop_iter=100,
    retraction::AbstractRetractionMethod=ExponentialRetraction(),
    inverse_retraction::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
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
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
    )

Estimate the Riemannian center of mass of `x` in an online fashion using
repeated weighted geodesic interpolation. See
[`GeodesicInterpolation`](@ref) for details.

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
    retraction::AbstractRetractionMethod=ExponentialRetraction(),
    inverse_retraction::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
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
[`GeodesicInterpolationWithinRadius`](@ref).

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
    retraction::AbstractRetractionMethod=ExponentialRetraction(),
    inverse_retraction::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
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

@trait_function Statistics.mean!(
    M::AbstractDecoratorManifold,
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)

function decorator_transparent_dispatch(
    ::typeof(mean!),
    ::AbstractEmbeddedManifold,
    args...;
    kwargs...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(mean!),
    ::AbstractEmbeddedManifold{𝔽,<:TransparentIsometricEmbedding},
    args...;
    kwargs...,
) where {𝔽}
    return Val(:transparent)
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
[`ExtrinsicEstimation`](@ref), i.e. by computing the mean in the embedding and projecting
the result back.
You can specify an `extrinsic_method` to specify which mean estimation method to use in the embedding,
which defaults to [`GeodesicInterpolation`](@ref).

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
    ::ExtrinsicEstimation;
    extrinsic_method::AbstractEstimationMethod=GeodesicInterpolation(),
    kwargs...,
)
    embedded_x = map(p -> embed(M, p), x)
    embedded_y = mean(get_embedding(M), embedded_x, w, extrinsic_method; kwargs...)
    project!(M, y, embedded_y)
    return y
end

@doc raw"""
    median(M::AbstractManifold, x::AbstractVector[, w::AbstractWeights]; kwargs...)
    median(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::AbstractEstimationMethod;
        kwargs...,
    )

Compute the (optionally weighted) Riemannian median of the vector `x` of points on the
[`AbstractManifold`](@ref) `M`, defined as the point that satisfies the minimizer
````math
\argmin_{y ∈ \mathcal M} \frac{1}{\sum_{i=1}^n w_i} \sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemannian [`distance`](@ref).
This function is nonsmooth (i.e nondifferentiable).

In the general case, the [`CyclicProximalPointEstimation`](@ref) is used to compute the
median. However, this default may be overloaded for specific manifolds.

Compute the median using the specified `method`.
"""
Statistics.median(::AbstractManifold, ::Any...)

"""
    median(
        M::AbstractManifold,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::CyclicProximalPointEstimation;
        p0=x[1],
        stop_iter=1000000,
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
    )

Compute the median using [`CyclicProximalPointEstimation`](@ref).

Optionally, provide `p0`, the starting point (by default set to the first
data point). `stop_iter` denotes the maximal number of iterations to perform
and the `kwargs...` are passed to [`isapprox`](@ref) to stop, when the minimal
change between two iterates is small. For more stopping criteria check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction.

The algorithm is further described in [^Bačák2014].

[^Bačák2014]:
    > Bačák, M: Computing Medians and Means in Hadamard Spaces.
    > SIAM Journal on Optimization (2014), 24(3), pp. 1542–1566,
    > doi: [10.1137/140953393](https://doi.org/10.1137/140953393),
    > arxiv: [1210.2145](https://arxiv.org/abs/1210.2145)
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
        extrinsic_method = CyclicProximalPointEstimation(),
        kwargs...,
    )

Estimate the median of `x` using [`ExtrinsicEstimation`](@ref), i.e. by computing the median
in the embedding and projecting the result back.
You can specify an `extrinsic_method` to specify which median estimation method to use in
the embedding, which defaults to [`CyclicProximalPointEstimation`](@ref).

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
        retraction::AbstractRetractionMethod = ExponentialRetraction(),
        inverse_retraction::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
        kwargs...,
    )

Compute the median using [`WeiszfeldEstimation`](@ref).

Optionally, provide `p0`, the starting point (by default set to the first
data point). `stop_iter` denotes the maximal number of iterations to perform
and the `kwargs...` are passed to [`isapprox`](@ref) to stop, when the minimal
change between two iterates is small. For more stopping criteria check the
[`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

The parameter ``α\in (0,2]`` is a step size.

The algorithm is further described in [^FletcherVenkatasubramanianJoshi2008],
especially the update rule in Eq. (6), i.e. Let ``q_{k}`` denote the current
iterate, $n$ the number of points ``x_1,\ldots,x_n``, and

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

and where $\mathrm{d}_{\mathcal M}$ denotes the Riemannian [`distance`](@ref).

Optionally, pass `retraction` and `inverse_retraction` method types to specify
the (inverse) retraction, which by default use the exponential and logarithmic map,
respectively.

[^FletcherVenkatasubramanianJoshi2008]:
    > Fletcher, T., Venkatasubramanian, S., Joshi, S:
    > Robust statistics on Riemannian manifolds via the geometric median
    > 2008 IEEE Conference on Computer Vision and Pattern Recognition,
    > doi: [10.1109/CVPR.2008.4587747](https://doi.org/10.1109/CVPR.2008.4587747),
"""
Statistics.median(
    ::AbstractManifold,
    ::AbstractVector,
    ::AbstractVector,
    ::WeiszfeldEstimation,
)

function Statistics.median(
    M::AbstractManifold,
    x::AbstractVector,
    method::AbstractEstimationMethod...;
    kwargs...,
)
    y = allocate_result(M, median, x[1])
    return median!(M, y, x, method...; kwargs...)
end
function Statistics.median(
    M::AbstractManifold,
    x::AbstractVector,
    w::AbstractVector,
    method::AbstractEstimationMethod...;
    kwargs...,
)
    y = allocate_result(M, median, x[1])
    return median!(M, y, x, w, method...; kwargs...)
end

@doc raw"""
    median!(M::AbstractManifold, y, x::AbstractVector[, w::AbstractWeights]; kwargs...)
    median!(
        M::AbstractManifold,
        y,
        x::AbstractVector,
        [w::AbstractWeights,]
        method::AbstractEstimationMethod;
        kwargs...,
    )

computes the [`median`](@ref) in-place in `y`.
"""
median!(::AbstractManifold, ::Any...)
function Statistics.median!(
    M::AbstractManifold,
    q,
    x::AbstractVector,
    method::AbstractEstimationMethod...;
    kwargs...,
)
    w = _unit_weights(length(x))
    return median!(M, q, x, w, method...; kwargs...)
end
function Statistics.median!(
    M::AbstractManifold,
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)
    return median!(M, y, x, w, CyclicProximalPointEstimation(); kwargs...)
end
function Statistics.median!(
    M::AbstractManifold,
    q,
    x::AbstractVector,
    w::AbstractVector,
    ::CyclicProximalPointEstimation;
    p0=x[1],
    stop_iter=1000000,
    retraction::AbstractRetractionMethod=ExponentialRetraction(),
    inverse_retraction::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
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
    ::ExtrinsicEstimation;
    extrinsic_method::AbstractEstimationMethod=CyclicProximalPointEstimation(),
    kwargs...,
)
    embedded_x = map(p -> embed(M, p), x)
    embedded_y = median(get_embedding(M), embedded_x, w, extrinsic_method; kwargs...)
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
    retraction::AbstractRetractionMethod=ExponentialRetraction(),
    inverse_retraction::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
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

@trait_function Statistics.median!(
    M::AbstractDecoratorManifold,
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
)

function decorator_transparent_dispatch(
    ::typeof(median!),
    ::AbstractEmbeddedManifold,
    args...;
    kwargs...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(median!),
    ::AbstractEmbeddedManifold{𝔽,<:TransparentIsometricEmbedding},
    args...;
    kwargs...,
) where {𝔽}
    return Val(:transparent)
end

@doc raw"""
    var(M, x, m=mean(M, x); corrected=true)
    var(M, x, w::AbstractWeights, m=mean(M, x, w); corrected=false)

compute the (optionally weighted) variance of a `Vector` `x` of `n` data points
on the [`AbstractManifold`](@ref) `M`, i.e.

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
points on the [`AbstractManifold`](@ref) `M`, i.e.

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
        method::AbstractEstimationMethod;
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
    method::AbstractEstimationMethod...;
    corrected=false,
    kwargs...,
)
    m = mean(M, x, w, method...; kwargs...)
    v = var(M, x, w, m; corrected=corrected)
    return m, v
end
function StatsBase.mean_and_var(
    M::AbstractManifold,
    x::AbstractVector,
    method::AbstractEstimationMethod...;
    corrected=true,
    kwargs...,
)
    n = length(x)
    w = _unit_weights(n)
    return mean_and_var(M, x, w, method...; corrected=corrected, kwargs...)
end

@doc raw"""
    mean_and_var(
        M::AbstractManifold,
        x::AbstractVector
        [w::AbstractWeights,]
        method::GeodesicInterpolation;
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

See [`GeodesicInterpolation`](@ref) for details on the geodesic
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
    retraction::AbstractRetractionMethod=ExponentialRetraction(),
    inverse_retraction::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
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

See [`GeodesicInterpolationWithinRadius`](@ref) and
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
        method::AbstractEstimationMethod;
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
