"""
    ProjectedPointDistribution(M::AbstractManifold, d, proj!, p)

Generates a random point in ambient space of `M` and projects it to `M`
using function `proj!`. Generated arrays are of type `TResult`, which can be
specified by providing the `p` argument.
"""
struct ProjectedPointDistribution{TResult,TM<:AbstractManifold,TD<:Distribution,TProj} <:
       MPointDistribution{TM}
    manifold::TM
    distribution::TD
    proj!::TProj
end

function ProjectedPointDistribution(
    M::AbstractManifold,
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

"""
    projected_distribution(M::AbstractManifold, d, [p=rand(d)])

Wrap the standard distribution `d` into a manifold-valued distribution. Generated
points will be of similar type to `p`. By default, the type is not changed.
"""
function projected_distribution(M::AbstractManifold, d, p=rand(d))
    return ProjectedPointDistribution(M, d, project!, p)
end

function Random.rand(
    rng::AbstractRNG,
    d::ProjectedPointDistribution{TResult},
) where {TResult}
    p = convert(TResult, rand(rng, d.distribution))
    return d.proj!(d.manifold, p, p)
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::ProjectedPointDistribution,
    p::AbstractArray{<:Number},
)
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

function Random.rand(
    rng::AbstractRNG,
    d::ProjectedFVectorDistribution{TResult},
) where {TResult}
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

"""
    normal_tvector_distribution(M::Euclidean, p, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `p`.
"""
function normal_tvector_distribution(M::AbstractManifold, p, σ)
    d = Distributions.MvNormal(zero(vec(p)), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), p, d, project!, p)
end

function Distributions.support(tvd::ProjectedFVectorDistribution)
    return FVectorSupport(tvd.type, tvd.point)
end
