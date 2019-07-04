@doc doc"""
    Euclidean{T<:Tuple} <: Manifold

Euclidean vector space $\mathbb R^n$.

# Constructor

    Euclidean(n)

generates the $n$-dimensional vector space $\mathbb R^n$.

   Euclidean(m, n)

generates the $mn$-dimensional vector space $\mathbb R^{m \times n}$, whose
elements are interpreted as $m \times n$ matrices.
"""
struct Euclidean{T<:Tuple} <: Manifold where {T} end

Euclidean(n::Int) = Euclidean{Tuple{n}}()
Euclidean(m::Int, n::Int) = Euclidean{Tuple{m,n}}()

function representation_size(::Euclidean{Tuple{n}}, ::Type{T}) where {n,T<:Union{MPoint, TVector, CoTVector}}
    return (n,)
end

function representation_size(::Euclidean{Tuple{m,n}}, ::Type{T}) where {m,n,T<:Union{MPoint, TVector, CoTVector}}
    return (m,n)
end

@generated manifold_dimension(::Euclidean{T}) where {T} = *(T.parameters...)

struct EuclideanMetric <: RiemannianMetric end

@traitimpl HasMetric{Euclidean,EuclideanMetric}

function local_metric(::MetricManifold{<:Manifold,EuclideanMetric}, x)
    return Diagonal(ones(SVector{size(x, 1),eltype(x)}))
end

function inverse_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, x)
    return local_metric(M, x)
end

det_local_metric(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = one(eltype(x))

log_local_metric_density(M::MetricManifold{<:Manifold,EuclideanMetric}, x) = zero(eltype(x))

@inline inner(::Euclidean, x, v, w) = dot(v, w)
@inline inner(::MetricManifold{<:Manifold,EuclideanMetric}, x, v, w) = dot(v, w)

distance(::Euclidean, x, y) = norm(x-y)
norm(::Euclidean, x, v) = norm(v)
norm(::MetricManifold{<:Manifold,EuclideanMetric}, x, v) = norm(v)

exp!(M::Euclidean, y, x, v) = (y .= x + v)

log!(M::Euclidean, v, x, y) = (v .= y - x)

function zero_tangent_vector!(M::Euclidean, v, x)
    fill!(v, 0)
    return v
end

proj!(M::Euclidean, x) = x

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
    return ProjectedPointDistribution(M, d, proj!, x)
end

function projected_distribution(M::Euclidean, d)
    return ProjectedPointDistribution(M, d, proj!, rand(d))
end

"""
    normal_tvector_distribution(S::Euclidean, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(M::Euclidean{Tuple{N}}, x, σ) where N
    d = Distributions.MvNormal(zero(x), σ)
    return ProjectedTVectorDistribution(M, x, d, project_tangent!, x)
end
