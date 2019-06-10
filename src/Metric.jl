abstract type Metric end

abstract type RiemannianMetric <: Metric end

abstract type LorentzianMetric <: Metric end

struct MetricManifold{M<:Manifold,G<:Metric} <: Manifold
    manifold::M
    metric::G
end

manifold(M::MetricManifold) = M.manifold

metric(M::MetricManifold) = M.metric

local_metric(M::MetricManifold, x) = error("Not implemented")

inverse_local_metric(M::MetricManifold, x) = inv(local_metric(M, x))

dot(M::MetricManifold, x, v, w) = dot(v, local_matrix(M, x) * w)

exp!(M::MetricManifold, y, x, v) = error("Not implemented")

log!(M::MetricManifold, v, x, y) = error("Not implemented")
