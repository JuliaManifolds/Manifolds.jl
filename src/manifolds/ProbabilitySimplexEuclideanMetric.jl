exp!(::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric}, q, p, X) = (q .= p .+ X)
function exp!(
    ::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric},
    q,
    p,
    X,
    t::Number,
)
    return (q .= p .+ t .* X)
end

@doc raw"""
    manifold_volume(::MetricManifold{ℝ,<:ProbabilitySimplex{n},<:EuclideanMetric})) where {n}

Return the volume of the [`ProbabilitySimplex`](@ref) with the Euclidean metric.
The formula reads ``\frac{\sqrt{n+1}}{n!}``
"""
function manifold_volume(M::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric})
    n = get_parameter(M.manifold.size)[1]
    return sqrt(n + 1) / factorial(n)
end

@doc raw"""
    volume_density(::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric}, p, X)

Compute the volume density at point `p` on [`ProbabilitySimplex`](@ref) `M` for tangent
vector `X`. It is equal to 1.
"""
function volume_density(::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric}, p, X)
    return one(eltype(X))
end
