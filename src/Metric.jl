abstract type Metric{M<:Manifold} end

convert(::Type{Manifold}, g::Metric) = manifold(g)
convert(::Type{G}, m::M) where {M<:Manifold,G<:Metric{M}} = G(m)
convert(::Type{Metric}, m::Manifold) = EuclideanMetric(m)

abstract type RiemannianMetric{M} <: Metric{M} end

abstract type LorentzianMetric{M} <: Metric{M} end

struct EuclideanMetric{M} <: RiemannianMetric{M}
   m::M
end

struct TransformedEuclideanMetric{M,G<:AbstractMatrix} <: RiemannianMetric{M}
   m::M
   g::G
end

manifold(g::Metric) = error("Not implemented")
manifold(g::EuclideanMetric) = g.m
manifold(g::TransformedEuclideanMetric) = g.m

function local_matrix(g::Metric, x)
   n = dimension(manifold(g))
   G = similar(x, n, n)
   local_matrix!(g::Metric, G, x)
end

local_matrix!(g::Metric, G, x) = error("Not implemented")

local_matrix(g::EuclideanMetric, x) = I

function local_matrix!(g::EuclideanMetric, G, x)
   n = dimension(manifold(g))
   G .= Diagonal(ones(n))
end

local_matrix!(g::TransformedEuclideanMetric, G, x) = G .= g.g
