abstract type Metric{M<:Manifold} end

abstract type RiemannianMetric{M} <: Metric{M} end

abstract type LorentzianMetric{M} <: Metric{M} end

convert(::Type{Manifold}, g::Metric) = manifold(g)
convert(::Type{G}, m::M) where {M<:Manifold,G<:Metric{M}} = G(m)

struct SemiRiemannianManifold{M<:Manifold,G<:Metric{M}} <: Manifold
    metric::G
end

SemiRiemannianManifold(manifold, metric) = SemiRiemannianManifold(metric)

manifold(g::Metric) = g.manifold
manifold(m::SemiRiemannianManifold) = manifold(m.metric)

metric(m::SemiRiemannianManifold) = m.metric

local_matrix(g::Metric, x) = error("Not implemented")

inverse_local_matrix(g::Metric, x) = inv(local_matrix(g, x))

distance(g::Metric, x, y) = norm(g, x, log(g, x, y))
distance(m::SemiRiemannianManifold, args...; kwargs...) = distance(metric(m), args...; kwargs...)

dot(g::Metric, x, v, w) = dot(v, local_matrix(g, x) * w)
dot(m::SemiRiemannianManifold, args...; kwargs...) = dot(metric(m), args...; kwargs...)

norm(g::Metric, x, v) = sqrt(dot(g, x, v, v))
norm(m::SemiRiemannianManifold, args...; kwargs...) = norm(metric(m), args...; kwargs...)

angle(g::Metric, x, v, w) = dot(g, x, v, w) / norm(g, x, v) / norm(g, x, w)
angle(m::SemiRiemannianManifold, args...; kwargs...) = angle(metric(m), args...; kwargs...)

exp!(g::Metric, y, x, v) = error("Not implemented")
exp!(m::SemiRiemannianManifold, args...; kwargs...) = exp!(metric(m), args...; kwargs...)

exp(g::Metric, x, v) = exp!(g, copy(x), x, v)
exp(m::SemiRiemannianManifold, args...; kwargs...) = exp(metric(m), args...; kwargs...)

geodesic(g::Metric, x, y, t) = exp(g, x, log(g, x, y), t)
geodesic(m::SemiRiemannianManifold, args...; kwargs...) = exp(metric(g), args...; kargs...)

log!(g::Metric, v, x, y) = error("Not implemented")
log!(m::SemiRiemannianManifold, args...; kwargs...) = log!(metric(m), args...; kwargs...)

function log(g::Metric, x, y)
    v = zero_tangent_vector(manifold(g), x)
    log!(g, v, x, y)
    return v
end

log(m::SemiRiemannianManifold, args...; kwargs...) = log(metric(m), args...; kwargs...)
