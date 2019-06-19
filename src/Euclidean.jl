struct Euclidean{T<:Tuple} <: Manifold where {T} end

Euclidean(n::Int) = Euclidean{Tuple{n}}()
Euclidean(m::Int, n::Int) = Euclidean{Tuple{m,n}}()

@generated manifold_dimension(::Euclidean{T}) where {T} = *(T.parameters...)

struct EuclideanMetric <: RiemannianMetric end

end

end

local_metric(::MetricManifold{<:Manifold,EuclideanMetric}, x)= I

inverse_local_metric(::MetricManifold{<:Manifold,EuclideanMetric}, x) = I

inner(::Euclidean, x, v, w) = dot(v, w)
inner(M::MetricManifold{<:Manifold,EuclideanMetric}, args...) = inner(manifold(M), args...)

exp!(M::Euclidean, y, x, v) = y .= x + v
exp!(M::MetricManifold{<:Euclidean,EuclideanMetric}, args...) = exp!(manifold(M), args...)

log!(M::Euclidean, v, x, y) = v .= y - x
log!(M::MetricManifold{<:Euclidean,EuclideanMetric}, args...) = log!(manifold(M), args...)
