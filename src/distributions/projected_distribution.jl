@decorator_transparent_signature projected_distribution(M::AbstractDecoratorManifold, d, p)
@decorator_transparent_signature projected_distribution(M::AbstractDecoratorManifold, d)

"""
    ProjectedPoint(M::Manifold, d, proj!, p)

Generates a random point in ambient space of `M` and projects it to `M`
using function `proj!`. Generated arrays are of type `TResult`, which can be
specified by providing the `p` argument.
"""
struct ProjectedPoint{TResult,TM<:Manifold,TD<:Distribution,TProj} <: MPointDistribution{TM}
    manifold::TM
    distribution::TD
    proj!::TProj
end

function ProjectedPoint(M::Manifold, d::Distribution, proj!, ::TResult) where {TResult}
    return ProjectedPoint{TResult,typeof(M),typeof(d),typeof(proj!)}(M, d, proj!)
end

function Random.rand(rng::AbstractRNG, d::ProjectedPoint{TResult}) where {TResult}
    p = convert(TResult, Random.rand(rng, d.distribution))
    return d.proj!(d.manifold, p, p)
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::ProjectedPoint,
    p::AbstractArray{<:Number},
)
    Distributions._rand!(rng, d.distribution, p)
    return d.proj!(d.manifold, p, p)
end

Distributions.support(d::ProjectedPoint) = MPointSupport(d.manifold)

"""
    ProjectedFVector(type::VectorBundleFibers, p, d, project!)

Generates a random vector from ambient space of manifold `type.manifold`
at point `p` and projects it to vector space of type `type` using function
`project!`, see [`project`](@ref) for documentation.
Generated arrays are of type `TResult`.
"""
struct ProjectedFVector{
    TResult,
    TSpace<:VectorBundleFibers,
    ManifoldPoint,
    TD<:Distribution,
    TProj,
} <: FVectorDistribution{TSpace,ManifoldPoint}
    type::TSpace
    point::ManifoldPoint
    distribution::TD
    project!::TProj
end

function ProjectedFVector(
    type::VectorBundleFibers,
    p,
    d::Distribution,
    project!,
    ::TResult,
) where {TResult}
    return ProjectedFVector{
        TResult,
        typeof(type),
        typeof(p),
        typeof(d),
        typeof(project!),
    }(
        type,
        p,
        d,
        project!,
    )
end

function Random.rand(rng::AbstractRNG, d::ProjectedFVector{TResult}) where {TResult}
    X = convert(TResult, reshape(Random.rand(rng, d.distribution), size(d.point)))
    return d.project!(d.type, X, d.point, X)
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::ProjectedFVector,
    X::AbstractArray{<:Number},
)
    # calling _rand!(rng, d.d, v) doesn't work for all arrays types
    return copyto!(X, Random.rand(rng, d))
end

function Distributions.support(tvd::ProjectedFVector)
    return FVectorSupport(tvd.type, tvd.point)
end
