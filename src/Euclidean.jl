struct Euclidean{T<:Tuple} <: Manifold where {T} end

Euclidean(n::Int) = Euclidean{Tuple{n}}()
Euclidean(m::Int, n::Int) = Euclidean{Tuple{m,n}}()

@generated dimension(::Euclidean{T}) where {T} = sum(T.parameters)

struct EuclideanMetric{M} <: RiemannianMetric{M}
   m::M
end

struct TransformedEuclideanMetric{M,G<:AbstractMatrix} <: RiemannianMetric{M}
   m::M
   g::G
end

manifold(g::EuclideanMetric) = g.m
manifold(g::TransformedEuclideanMetric) = g.m

local_matrix(g::EuclideanMetric, x) = I

function local_matrix!(g::EuclideanMetric, G, x)
   n = dimension(manifold(g))
   G .= Diagonal(ones(n))
end

local_matrix!(g::TransformedEuclideanMetric, G, x) = G .= g.g
