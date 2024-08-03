module ManifoldsDistributionsRecursiveArrayToolsExt

if isdefined(Base, :get_extension)
    using Manifolds
    using Distributions
    using RecursiveArrayTools
    using Random
else
    # imports need to be relative for Requires.jl-based workflows:
    # https://github.com/JuliaArrays/ArrayInterface.jl/pull/387
    using ..Manifolds
    using ..Distributions
    using ..RecursiveArrayTools
    using ..Random
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
    v::AbstractArray{<:Number},
)
    return copyto!(v, rand(rng, d))
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductFVectorDistribution,
    X::ArrayPartition,
)
    map(
        t -> Distributions._rand!(rng, t[1], t[2]),
        d.distributions,
        submanifold_components(d.space.manifold, X),
    )
    return X
end
function Distributions._rand!(
    rng::AbstractRNG,
    d::ProductPointDistribution,
    x::AbstractArray{<:Number},
)
    return copyto!(x, rand(rng, d))
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

end
