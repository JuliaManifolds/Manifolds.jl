import LinearAlgebra: norm
using StatsBase: AbstractWeights

@doc doc"""
    Euclidean{T<:Tuple} <: Manifold

Euclidean vector space $\mathbb R^n$.

# Constructor

    Euclidean(n)

generates the $n$-dimensional vector space $\mathbb R^n$.

    Euclidean(n₁,n₂,...,nᵢ)

generates the $n_1n_2\cdot\ldots n_i$-dimensional vector space $\mathbb R^{n_1, n_2, \ldots, n_i}$, whose
elements are interpreted as $n_1 \times,n_2\times\cdots\times n_i$ arrays, e.g. for
two parameters as matrices.
"""
struct Euclidean{T<:Tuple} <: Manifold where {T} end
Euclidean(n::Vararg{Int,N}) where N = Euclidean{Tuple{n...}}()

"""
    representation_size(M::Euclidean{T})

returns the array dimensions required to represent an element on the
[`Euclidean`](@ref) manifold `M`, i.e. the vector of all array dimensions.
"""
@generated representation_size(::Euclidean{T}) where T = Tuple(T.parameters...)

"""
    manifold_dimension(M::Euclidean{T})

returns the manifold dimension of the [`Euclidean`](@ref) manifold `M`, i.e.
the product of all array dimensions.
"""
@generated manifold_dimension(::Euclidean{T}) where {T} = *(T.parameters...)

"""
    EuclideanMetric <: RiemannianMetric

a general type for any manifold that employs the Euclidean Metric, for example
the [`Euclidean`](@ref) manifold itself, or the [`Sphere`](@ref), where every
tangent space (as a plane in the embedding) uses this metric (in the embedding).
"""
struct EuclideanMetric <: RiemannianMetric end
is_default_metric(::Euclidean{T},::EuclideanMetric) where {T} = Val(true)


local_metric(::MetricManifold{<:Manifold,EuclideanMetric}, x) = Diagonal(ones(SVector{size(x, 1),eltype(x)}))

inverse_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = local_metric(M, x)

det_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = one(eltype(x))

log_local_metric_density(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = zero(eltype(x))

@inline inner(::Euclidean, x, v, w) = dot(v, w)
@inline inner(::MetricManifold{<:Manifold,EuclideanMetric}, x, v, w) = dot(v, w)

"""
    distance(M::Euclidean,x,y)

compute the Euclidean distance between two points on the [`Euclidean`](@ref)
manifold `M`, i.e. for vectors it's just the norm of the difference, for matrices
and higher order arrays, the matrix and ternsor Frobenius norm, respectively.
"""
distance(::Euclidean, x, y) = norm(x .- y)
"""
    norm(M::Euclidean,x,v)

compute the norm of a tangent vector `v` at `x` on the [`Euclidean`](@ref)
manifold `M`, i.e. since every tangent space can be identified with `M` itself
in this case, just the (Frobenius) norm of `v`.
"""
norm(::Euclidean, x, v) = norm(v)
norm(::MetricManifold{<:Manifold,EuclideanMetric}, x, v) = norm(v)

"""
    exp!(M::Euclidean,y, x, v)

compute the exponential map on the [`Euclidean`](@ref) manifold `M` from `x`
in direction `v` in place, i.e. in `y`, which in this case is just
````math
    y = x + v
````
"""
exp!(M::Euclidean, y, x, v) = (y .= x .+ v)
"""
    log!(M::Euclidean, v, x, y)

compute the logarithmic map on the [`Euclidean`](@ref) manifold `M` from `x`
tpo `y`, stored in the direction `v` in place, i.e. in `v`, which in this case is just
````math
    v = y - x
````
"""
log!(M::Euclidean, v, x, y) = (v .= y .- x)
"""
    zero_tangent_vector!(M::Euclidean, v, x)

compute a zero vector in the tangent space of `x` on the [`Euclidean`](@ref)
manifold `M`, whcih here is just a zero filled array the same size as the
in place parameter `v` which is assumed to have the same `size` as `x`.
"""
function zero_tangent_vector!(M::Euclidean, v, x)
    fill!(v, 0)
    return v
end

"""
    project_point!(M::Euclidean, x)

project an arbitrary point `x` onto the [`Euclidean`](@ref) manifold `M`, which
is of course just the identity map.
"""
project_point!(M::Euclidean, x) = x

"""
    project_tangent!(M::Euclidean, w, x, v)

project an arbitrary vector `v` into the tangent space of a point `x` on the
[`Euclidean`](@ref) manifold `M`, which is just the identity, since any tangent
space of `M` can be identified with all of `M`.
"""
function project_tangent!(M::Euclidean, w, x, v)
    w .= v
    return w
end
"""
    vector_transport_to!(M::Euclidean, vto, x, v, y, ::ParallelTransport)

parallel transport the vector `v` from the tangent space at `x` to the
tangent space at `y` on the [`Euclidean`](@ref) manifold `M`,
i.e. the in place `w` is just set to `v`.
"""
function vector_transport_to!(M::Euclidean, vto, x, v, y, ::ParallelTransport)
    vto .= v
    return vto
end
"""
    flat!(M::Euclidean, v, x, w)

since cotangent and tangent vectors can directly be identified in the [`Euclidean`](@ref)
case, this yields just the identity for a tangent vector `w` in the tangent space
of `x` on `M`. The result is returned also in place in `v`.
"""
function flat!(M::Euclidean, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
end
"""
    sharp!(M::Euclidean, v, x, w)

since cotangent and tangent vectors can directly be identified in the [`Euclidean`](@ref)
case, this yields just the identity for a cotangent vector `w` in the tangent space
of `x` on `M`. The result is returned also in place in `v`.
"""
function sharp!(M::Euclidean, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    copyto!(v.data, w.data)
    return v
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
    normal_tvector_distribution(S::Euclidean, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(M::Euclidean{Tuple{N}}, x, σ) where N
    d = Distributions.MvNormal(zero(x), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), x, d, project_vector!, x)
end

mean(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}; kwargs...) = mean(x)
mean(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}, w::AbstractWeights; kwargs...) = mean(x, w)
mean(::Euclidean, x::AbstractVector; kwargs...) = mean(x)
mean!(M::Euclidean, y, x::AbstractVector, w::AbstractVector; kwargs...) =
    mean!(M, y, x, w, GeodesicInterpolation(); kwargs...)

median(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}; kwargs...) = median(x)
median(::Euclidean{Tuple{1}}, x::AbstractVector{<:Number}, w::AbstractWeights; kwargs...) = median(x, w)
median!(::Euclidean{Tuple{1}}, y, x::AbstractVector; kwargs...) = copyto!(y, [median(vcat(x...))])
median!(::Euclidean{Tuple{1}}, y, x::AbstractVector, w::AbstractWeights; kwargs...) =
    copyto!(y, [median(vcat(x...), w)])

var(::Euclidean, x::AbstractVector; kwargs...) = sum(var(x; kwargs...))
var(::Euclidean, x::AbstractVector{T}, m::T; kwargs...) where {T} = sum(var(x; mean=m, kwargs...))

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
