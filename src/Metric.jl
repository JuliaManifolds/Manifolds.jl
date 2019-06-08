abstract type Metric{M<:Manifold} end

convert(::Type{Manifold}, g::Metric) = manifold(g)
convert(::Type{G}, m::M) where {M<:Manifold,G<:Metric{M}} = G(m)

abstract type RiemannianMetric{M} <: Metric{M} end

abstract type LorentzianMetric{M} <: Metric{M} end

manifold(g::Metric) = error("Not implemented")

function local_matrix(g::Metric, x)
   n = dimension(manifold(g))
   G = similar(x, n, n)
   local_matrix!(g::Metric, G, x)
end

local_matrix!(g::Metric, G, x) = error("Not implemented")
