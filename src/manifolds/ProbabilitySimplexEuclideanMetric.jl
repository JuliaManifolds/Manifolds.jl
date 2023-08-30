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
function manifold_volume(
    ::MetricManifold{ℝ,<:ProbabilitySimplex{n},<:EuclideanMetric},
) where {n}
    return sqrt(n + 1) / factorial(n)
end

@doc raw"""
    rand(::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric}; vector_at=nothing, σ::Real=1.0)


When `vector_at` is `nothing`, return a random (uniform) point `x` on the [`ProbabilitySimplex`](@ref) with the Euclidean metric
manifold `M` by normalizing independent exponential draws to unit sum, see [Devroye:1986](@cite), Theorems 2.1 and 2.2 on p. 207 and 208, respectively.

When `vector_at` is not `nothing`, return a (Gaussian) random vector from the tangent space
``T_{p}\mathrm{\Delta}^n``by shifting a multivariate Gaussian with standard deviation `σ`
to have a zero component sum.

"""
rand(::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric}; σ::Real=1.0)

function Random.rand!(
    rng::AbstractRNG,
    M::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric},
    pX;
    vector_at=nothing,
    σ=one(eltype(pX)),
)
    if isnothing(vector_at)
        Random.randexp!(rng, pX)
        LinearAlgebra.normalize!(pX, 1)
    else
        Random.randn!(rng, pX)
        pX .= (pX .- mean(pX)) .* σ
    end
    return pX
end

@doc raw"""
    volume_density(::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric}, p, X)

Compute the volume density at point `p` on [`ProbabilitySimplex`](@ref) `M` for tangent
vector `X`. It is equal to 1.
"""
function volume_density(::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric}, p, X)
    return one(eltype(X))
end
