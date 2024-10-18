module ManifoldsDistributionsRecursiveArrayToolsExt

if isdefined(Base, :get_extension)
    using Manifolds
    using Distributions
    using Random
    using LinearAlgebra

    import Manifolds: uniform_distribution

    using RecursiveArrayTools: ArrayPartition
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manifolds
    using ..Distributions
    using ..Random

    import ..Manifolds: uniform_distribution

    using ..RecursiveArrayTools: ArrayPartition
end

## product manifold

"""
    ProductPointDistribution(M::ProductManifold, distributions)

Product distribution on manifold `M`, combined from `distributions`.
"""
struct ProductPointDistribution{
    TM<:ProductManifold,
    TD<:(NTuple{N,Distribution} where {N}),
} <: MPointDistribution{TM}
    manifold::TM
    distributions::TD
end

function ProductPointDistribution(M::ProductManifold, distributions::MPointDistribution...)
    return ProductPointDistribution{typeof(M),typeof(distributions)}(M, distributions)
end
function ProductPointDistribution(distributions::MPointDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    return ProductPointDistribution(M, distributions...)
end

"""
    ProductFVectorDistribution([type::VectorSpaceFiber], [x], distrs...)

Generates a random vector at point `x` from vector space (a fiber of a tangent
bundle) of type `type` using the product distribution of given distributions.

Vector space type and `x` can be automatically inferred from distributions `distrs`.
"""
struct ProductFVectorDistribution{
    TSpace<:VectorSpaceFiber{<:Any,<:ProductManifold},
    TD<:(NTuple{N,Distribution} where {N}),
} <: FVectorDistribution{TSpace}
    type::TSpace
    distributions::TD
end

function ProductFVectorDistribution(distributions::FVectorDistribution...)
    M = ProductManifold(map(d -> support(d).space.manifold, distributions)...)
    fiber_type = support(distributions[1]).space.fiber_type
    if !all(d -> support(d).space.fiber_type == fiber_type, distributions)
        error(
            "Not all distributions have support in vector spaces of the same type, which is currently not supported",
        )
    end
    # Probably worth considering sum spaces in the future?
    p = ArrayPartition(map(d -> support(d).space.point, distributions)...)
    return ProductFVectorDistribution(Fiber(M, p, fiber_type), distributions)
end

function Random.rand(rng::AbstractRNG, d::ProductFVectorDistribution)
    return ArrayPartition(map(d -> rand(rng, d), d.distributions)...)
end
function Random.rand(rng::AbstractRNG, d::ProductPointDistribution)
    return ArrayPartition(map(d -> rand(rng, d), d.distributions)...)
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductFVectorDistribution,
    X::ArrayPartition,
)
    map(
        (t1, t2) -> Distributions._rand!(rng, t1, t2),
        d.distributions,
        submanifold_components(d.type.manifold, X),
    )
    return X
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductPointDistribution,
    p::ArrayPartition,
)
    map(
        (t1, t2) -> Distributions._rand!(rng, t1, t2),
        d.distributions,
        submanifold_components(d.manifold, p),
    )
    return p
end

Distributions.support(d::ProductPointDistribution) = MPointSupport(d.manifold)
function Distributions.support(tvd::ProductFVectorDistribution)
    return FVectorSupport(tvd.type)
end

function uniform_distribution(M::ProductManifold)
    return ProductPointDistribution(M, map(uniform_distribution, M.manifolds))
end
function uniform_distribution(M::ProductManifold, p)
    return ProductPointDistribution(
        M,
        map(uniform_distribution, M.manifolds, submanifold_components(M, p)),
    )
end


end
