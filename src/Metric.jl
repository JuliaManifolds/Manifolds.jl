abstract type Metric{M<:Manifold} end

convert(::Type{Manifold}, g::Metric) = manifold(g)
convert(::Type{G}, m::M) where {M<:Manifold,G<:Metric{M}} = G(m)

abstract type RiemannianMetric{M} <: Metric{M} end

abstract type LorentzianMetric{M} <: Metric{M} end

manifold(g::Metric) = g.manifold

local_matrix(g::Metric, x) = error("Not implemented")

inverse_local_matrix(g::Metric, x) = inv(local_matrix(g, x))

distance(g::Metric, x, y) = norm(g, x, log(g, x, y))

dot(g::Metric, x, v, w) = dot(v, local_matrix(g, x) * w)

norm(g::Metric, x, v) = sqrt(dot(g, x, v, v))

angle(g::Metric, x, v, w) = dot(g, x, v, w) / norm(g, x, v) / norm(g, x, w)

exp!(g::Metric, y, x, v) = error("Not implemented")
exp(g::Metric, x, v) = exp!(M, copy(x), x, v)

log!(g::Metric, v, x, y) = error("Not implemented")

geodesic(g::Metric, x, y, t) = exp(g, x, log(g, x, y), t)

function log(g::Metric, x, y)
    v = zero_tangent_vector(manifold(g), x)
    log!(g, v, x, y)
    return v
end

injectivity_radius(g::Metric, x) = Inf
injectivity_radius(g::Metric) = Inf
