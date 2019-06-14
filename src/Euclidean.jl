struct Euclidean{T<:Tuple} <: Manifold where {T} end

Euclidean(n::Int) = Euclidean{Tuple{n}}()
Euclidean(m::Int, n::Int) = Euclidean{Tuple{m,n}}()

@generated dimension(::Euclidean{T}) where {T} = *(T.parameters...)

struct EuclideanMetric <: RiemannianMetric end

struct TransformedEuclideanMetric{G,IG} <: RiemannianMetric
   g::G
   g⁻¹::IG
end

function TransformedEuclideanMetric(g::Union{AbstractMatrix,UniformScaling})
   return TransformedEuclideanMetric(g, inv(g))
end

local_metric(::MetricManifold{<:Manifold,EuclideanMetric}, x)= I
local_metric(M::MetricManifold{<:Manifold,TransformedEuclideanMetric}, x) = metric(M).g

inverse_local_metric(::MetricManifold{<:Manifold,EuclideanMetric}, x) = I
function inverse_local_metric(M::MetricManifold{<:Manifold,TransformedEuclideanMetric}, x)
   return metric(M).g⁻¹
end

inner(::Euclidean, x, v, w) = dot(v, w)
inner(M::MetricManifold{<:Manifold,EuclideanMetric}, args...) = inner(manifold(M), args...)

exp!(M::Euclidean, y, x, v) = y .= x + v
exp!(M::MetricManifold{<:Euclidean,EuclideanMetric}, args...) = exp!(manifold(M), args...)

function exp!(M::MetricManifold{<:Euclidean,TransformedEuclideanMetric}, y, x, v)
   y .= x + inverse_local_metric(M) * v
end

log!(M::Euclidean, v, x, y) = v .= y - x
log!(M::MetricManifold{<:Euclidean,EuclideanMetric}, args...) = log!(manifold(M), args...)

function log!(M::MetricManifold{<:Euclidean,TransformedEuclideanMetric}, v, x, y)
   v .= local_metric(M) * (y - x)
end
