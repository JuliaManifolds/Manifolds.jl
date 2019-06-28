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

function representation_size(::Euclidean{S}, ::Type{T}) where {S,T<:Union{MPoint, TVector, CoTVector}}
    return Size(S.parameters[1]...)
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

inner(::Euclidean, x, v, w) = dot(v, w)
inner(::MetricManifold{<:Manifold,EuclideanMetric}, x, v, w) = dot(v, w)

norm(::Euclidean, x, v) = norm(v)
norm(::MetricManifold{<:Manifold,EuclideanMetric}, x, v) = norm(v)

exp!(M::Euclidean, y, x, v) = (y .= x + v)

log!(M::Euclidean, v, x, y) = (v .= y - x)
