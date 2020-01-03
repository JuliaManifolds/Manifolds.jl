import LinearAlgebra: norm
using StatsBase: AbstractWeights

@doc doc"""
    Euclidean{T<:Tuple} <: Manifold

Euclidean vector space $\mathbb R^n$.

# Constructor

    Euclidean(n)

generates the $n$-dimensional vector space $\mathbb R^n$.

    Euclidean(n₁,n₂,...,nᵢ; field=ℝ)

generates the (2*)$n_1n_2\cdot\ldots n_i$-dimensional vector space $\mathbb F^{n_1, n_2, \ldots, n_i}$, whose
elements are interpreted as $n_1 \times,n_2\times\cdots\times n_i$ arrays, e.g. for
two parameters as matrices. The default `field=ℝ` can also be set to `field=ℂ`.
"""
struct Euclidean{N<:Tuple,T} <: Manifold where {N, T<: AbstractField} end
Euclidean(n::Vararg{Int,N};field::AbstractField=ℝ) where N = Euclidean{Tuple{n...},field}()

"""
    EuclideanMetric <: RiemannianMetric

a general type for any manifold that employs the Euclidean Metric, for example
the [`Euclidean`](@ref) manifold itself, or the [`Sphere`](@ref), where every
tangent space (as a plane in the embedding) uses this metric (in the embedding).

Since the metric is independent of the field type, this metric is also used for
the Hermitian metrics, i.e. metrics that are analogous to the `EuclideanMetric`
but where the field type of the manifold is `ℂ`.

This metric is the default metric for example for the [`Euclidean`](@ref) manifold.
"""
struct EuclideanMetric <: RiemannianMetric end

det_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = one(eltype(x))

"""
    distance(M::Euclidean, x, y)

compute the Euclidean distance between two points on the [`Euclidean`](@ref)
manifold `M`, i.e. for vectors it's just the norm of the difference, for matrices
and higher order arrays, the matrix and ternsor Frobenius norm, respectively.
"""
distance(::Euclidean, x, y) = norm(x .- y)

@doc doc"""
    exp(M::Euclidean, x, v)

compute the exponential map on the [`Euclidean`](@ref) manifold `M` from `x` in direction
`v`, which in this case is just
````math
\exp_x v = x + v.
````
"""
exp(::Euclidean, ::Any...)
exp!(M::Euclidean, y, x, v) = (y .= x .+ v)

"""
    flat(M::Euclidean, x, w)

since cotangent and tangent vectors can directly be identified in the [`Euclidean`](@ref)
case, this yields just the identity for a tangent vector `w` in the tangent space
of `x` on `M`. The result is returned also in place in `v`.
"""
flat(::Euclidean,::Any...)
function flat!(M::Euclidean, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

@doc doc"""
    injectivity_radius(M::Euclidean)

returns the injectivity radius on the [`Euclidean`](@ref) `M`, which is $\infty$.
"""
injectivity_radius(::Euclidean) = Inf

@doc doc"""
    inner(M::Euclidean, x, v, w)

compute the inner product on the [`Euclidean`](@ref) `M`, which is just
the inner product on the real-valued or complex valued vector space

````math
g_x(v,w) = w^{\mathrm{H}}v,
````

where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
inner(::Euclidean, ::Any...)
@inline inner(::Euclidean, x, v, w) = dot(v, w)
@inline inner(::MetricManifold{<:Manifold,EuclideanMetric}, x, v, w) = dot(v, w)

inverse_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = local_metric(M, x)

is_default_metric(::Euclidean,::EuclideanMetric) = Val(true)

local_metric(::MetricManifold{<:Manifold,EuclideanMetric}, x) = Diagonal(ones(SVector{size(x, 1),eltype(x)}))

@doc doc"""
    log(M::Euclidean, x, y)

computes the logarithmic map on the [`Euclidean`](@ref) `M` from `x` to `y`,
which in this case is just
````math
\log_x y = y - x.
````
"""
log!(M::Euclidean, v, x, y) = (v .= y .- x)

log_local_metric_density(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = zero(eltype(x))

"""
    manifold_dimension(M::Euclidean)

returns the manifold dimension of the [`Euclidean`](@ref) `M`, i.e.
the product of all array dimensions.
"""
@generated manifold_dimension(::Euclidean{N,ℝ}) where {N} = *(N.parameters...)
@generated manifold_dimension(::Euclidean{N,ℂ}) where {N} = 2*( *(N.parameters...) )

mean(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}; kwargs...) = mean(x)
mean(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}, w::AbstractWeights; kwargs...) = mean(x, w)
mean(::Euclidean, x::AbstractVector; kwargs...) = mean(x)
mean!(M::Euclidean, y, x::AbstractVector, w::AbstractVector; kwargs...) =
    mean!(M, y, x, w, GeodesicInterpolation(); kwargs...)

function mean_and_var(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}; kwargs...)
    m, v = mean_and_var(x; kwargs...)
    return m, sum(v)
end

function mean_and_var(
    ::Euclidean{Tuple{1}},
    x::AbstractVector{<:Number},
    w::AbstractWeights;
    corrected = false,
    kwargs...,
)
    m, v = mean_and_var(x, w; corrected = corrected, kwargs...)
    return m, sum(v)
end

mean_and_var(M::Euclidean, x::AbstractVector, w::AbstractWeights; kwargs...) =
    mean_and_var(M, x, w, GeodesicInterpolation(); kwargs...)

median(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}; kwargs...) = median(x)
median(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}, w::AbstractWeights; kwargs...) = median(x, w)
median!(::Euclidean{Tuple{1}}, y, x::AbstractVector; kwargs...) = copyto!(y, [median(vcat(x...))])
median!(::Euclidean{Tuple{1}}, y, x::AbstractVector, w::AbstractWeights; kwargs...) =
    copyto!(y, [median(vcat(x...), w)])

@doc doc"""
    norm(M::Euclidean, x, v)

compute the norm of a tangent vector `v` at `x` on the [`Euclidean`](@ref)
`M`, i.e. since every tangent space can be identified with `M` itself
in this case, just the (Frobenius) norm of `v`.
"""
norm(::Euclidean, x, v) = norm(v)
norm(::MetricManifold{<:Manifold,EuclideanMetric}, x, v) = norm(v)

"""
    normal_tvector_distribution(M::Euclidean, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(M::Euclidean{Tuple{N}}, x, σ) where N
    d = Distributions.MvNormal(zero(x), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), x, d, project_vector!, x)
end

@doc doc"""
    project_point(M::Euclidean, x)

project an arbitrary point `x` onto the [`Euclidean`](@ref) `M`, which
is of course just the identity map.
"""
project_point(::Euclidean, ::Any...)
project_point!(M::Euclidean, x) = x

"""
    project_tangent(M::Euclidean, x, v)

project an arbitrary vector `v` into the tangent space of a point `x` on the
[`Euclidean`](@ref) `M`, which is just the identity, since any tangent
space of `M` can be identified with all of `M`.
"""
project_tangent(::Euclidean, ::Any...)
function project_tangent!(M::Euclidean, w, x, v)
    w .= v
    return w
end

"""
    projected_distribution(M::Euclidean, d, [x])

Wraps standard distribution `d` into a manifold-valued distribution. Generated
points will be of similar type to `x`. By default, the type is not changed.
"""
function projected_distribution(M::Euclidean, d, x)
    return ProjectedPointDistribution(M, d, project_point!, x)
end
function projected_distribution(M::Euclidean, d)
    return ProjectedPointDistribution(M, d, project_point!, rand(d))
end

"""
    representation_size(M::Euclidean)

returns the array dimensions required to represent an element on the
[`Euclidean`](@ref) `M`, i.e. the vector of all array dimensions.
"""
@generated representation_size(::Euclidean{N}) where N = Tuple(N.parameters...)

"""
    sharp(M::Euclidean, x, w)

since cotangent and tangent vectors can directly be identified in the [`Euclidean`](@ref)
case, this yields just the identity for a cotangent vector `w` in the tangent space
of `x` on `M`.
"""
sharp(::Euclidean,::Any...)
function sharp!(M::Euclidean, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

"""
    vector_transport_to(M::Euclidean, x, v, y, ::ParallelTransport)

parallel transport the vector `v` from the tangent space at `x` to the tangent space at `y`
on the [`Euclidean`](@ref) `M`, which simplifies to the identity.
"""
vector_transport_to(::Euclidean, ::Any, ::Any, ::Any, ::ParallelTransport)
function vector_transport_to!(M::Euclidean, vto, x, v, y, ::ParallelTransport)
    vto .= v
    return vto
end

var(::Euclidean, x::AbstractVector; kwargs...) = sum(var(x; kwargs...))
var(::Euclidean, x::AbstractVector{T}, m::T; kwargs...) where {T} = sum(var(x; mean=m, kwargs...))

"""
    zero_tangent_vector(M::Euclidean, x)

compute a zero vector in the tangent space of `x` on the [`Euclidean`](@ref)
`M`, which here is just a zero filled array the same size as `x`.
"""
zero_tangent_vector(::Euclidean, ::Any...)

function zero_tangent_vector!(M::Euclidean, v, x)
    fill!(v, 0)
    return v
end
