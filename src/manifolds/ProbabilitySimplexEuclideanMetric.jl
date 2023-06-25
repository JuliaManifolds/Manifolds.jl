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
    rand(::MetricManifold{ℝ,<:ProbabilitySimplex,<:EuclideanMetric}; vector_at=nothing, σ::Real=1.0)


When `vector_at` is `nothing`, return a random (uniform) point `x` on the [`ProbabilitySimplex`](@ref) with the Euclidean metric
manifold `M` by normalizing independent exponential draws to unit sum, see [^Devroye1986], Theorems 2.1 and 2.2 on p. 207 and 208, respectively.

When `vector_at` is not `nothing`, return a (Gaussian) random vector from the tangent space
``T_{p}\mathrm{\Delta}^n``by shifting a multivariate Gaussian with standard deviation `σ` 
to have a zero component sum.

[^Devroye1986]:
    > Devroye, L.:
    > _Non-Uniform Random Variate Generation_.
    > Springer New York, NY, 1986.
    > doi: [10.1007/978-1-4613-8643-8](https://doi.org/10.1007/978-1-4613-8643-8)
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
