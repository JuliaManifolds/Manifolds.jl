struct Euclidean{T<:Tuple} <: Manifold where {T} end

Euclidean(n::Int) = Euclidean{Tuple{n}}()
Euclidean(m::Int, n::Int) = Euclidean{Tuple{m,n}}()

@generated dimension(::Euclidean{T}) where {T} = sum(T.parameters)

struct EuclideanMetric{M} <: RiemannianMetric{M}
   manifold::M
end

struct TransformedEuclideanMetric{M,G<:AbstractMatrix,IG<:AbstractMatrix} <: RiemannianMetric{M}
   manifold::M
   metric::G
   inv_metric::IG
end

function TransformedEuclideanMetric(manifold, metric)
   return TransformedEuclideanMetric(manifold, metric, inv(metric))
end

local_matrix(g::EuclideanMetric, x) = I
local_matrix(g::TransformedEuclideanMetric, x) = g.metric

inverse_local_matrix(g::EuclideanMetric, x) = I
inverse_local_matrix(g::TransformedEuclideanMetric, x) = g.inv_metric

dot(m::Euclidean, x, v, w) = dot(v, w)
dot(g::EuclideanMetric{<:Euclidean}, args...) = dot(manifold(g), args...)

exp!(m::Euclidean, y, x, v) = y .= x + v
exp!(g::EuclideanMetric{<:Euclidean}, args...) = exp!(manifold(g), args...)
exp!(g::TransformedEuclideanMetric{<:Euclidean}, y, x, v) = y .= x + g.inv_metric * v

log!(m::Euclidean, v, x, y) = v .= y - x
log!(g::EuclideanMetric{<:Euclidean}, args...) = exp!(manifold(g), args...)
log!(g::TransformedEuclideanMetric{<:Euclidean}, v, x, y) = v .= g.metric * (y - x)
