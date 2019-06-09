struct Euclidean{T<:Tuple} <: Manifold where {T} end

Euclidean(n::Int) = Euclidean{Tuple{n}}()
Euclidean(m::Int, n::Int) = Euclidean{Tuple{m,n}}()

@generated dimension(::Euclidean{T}) where {T} = sum(T.parameters)

struct EuclideanMetric{M} <: RiemannianMetric{M}
   manifold::M
end

struct TransformedEuclideanMetric{M,G<:AbstractMatrix} <: RiemannianMetric{M}
   g::G
   manifold::M
end


local_matrix(g::EuclideanMetric, x) = I
local_matrix(g::TransformedEuclideanMetric, x) = g.metric

function local_matrix!(g::EuclideanMetric, G, x)
   n = dimension(manifold(g))
   G .= Diagonal(ones(n))
end

local_matrix!(g::TransformedEuclideanMetric, G, x) = G .= g.g
