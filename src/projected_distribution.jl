"""
    ProjectedPointDistribution(M::Manifold, d, proj!, p)

Generates a random point in ambient space of `M` and projects it to `M`
using function `proj!`. Generated arrays are of type `TResult`, which can be
specified by providing the `p` argument.
"""
struct ProjectedPointDistribution{TResult,TM<:Manifold,TD<:Distribution,TProj} <:
       MPointDistribution{TM}
    manifold::TM
    distribution::TD
    proj!::TProj
end

function ProjectedPointDistribution(
    M::Manifold,
    d::Distribution,
    proj!,
    ::TResult,
) where {TResult}
    return ProjectedPointDistribution{TResult,typeof(M),typeof(d),typeof(proj!)}(
        M,
        d,
        proj!,
    )
end

function Random.rand(rng::AbstractRNG, d::ProjectedPointDistribution{TResult}) where {TResult}
    p = convert(TResult, rand(rng, d.distribution))
    return d.proj!(d.manifold, p, p)
end

function Distributions._rand!(rng::AbstractRNG, d::ProjectedPointDistribution, p::AbstractArray{<:Number})
    Distributions._rand!(rng, d.distribution, p)
    return d.proj!(d.manifold, p, p)
end

Distributions.support(d::ProjectedPointDistribution) = MPointSupport(d.manifold)

"""
    ProjectedFVectorDistribution(type::VectorBundleFibers, p, d, project!)

Generates a random vector from ambient space of manifold `type.manifold`
at point `p` and projects it to vector space of type `type` using function
`project!`, see [`project`](@ref) for documentation.
Generated arrays are of type `TResult`.
"""
struct ProjectedFVectorDistribution{
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

function ProjectedFVectorDistribution(
    type::VectorBundleFibers,
    p,
    d::Distribution,
    project!,
    ::TResult,
) where {TResult}
    return ProjectedFVectorDistribution{
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

function Random.rand(rng::AbstractRNG, d::ProjectedFVectorDistribution{TResult}) where {TResult}
    X = convert(TResult, reshape(rand(rng, d.distribution), size(d.point)))
    return d.project!(d.type, X, d.point, X)
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::ProjectedFVectorDistribution,
    X::AbstractArray{<:Number},
)
    # calling _rand!(rng, d.d, v) doesn't work for all arrays types
    return copyto!(X, rand(rng, d))
end

Distributions.support(tvd::ProjectedFVectorDistribution) = FVectorSupport(tvd.type, tvd.point)
