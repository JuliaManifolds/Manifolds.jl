
"""
    ProjectedPointDistribution(m::Manifold, d, proj!, x)

Generates a random point in ambient space of `m` and projects it to `m`
using function `proj!`. Generated arrays are of type `TResult`, which can be
specified by providing the `x` argument.
"""
struct ProjectedPointDistribution{TResult,TM<:Manifold,TD<:Distribution,TProj} <:
       MPointDistribution{TM}
    manifold::TM
    d::TD
    proj!::TProj
end

function ProjectedPointDistribution(
    M::Manifold,
    d::Distribution,
    proj!,
    x::TResult,
) where {TResult}
    return ProjectedPointDistribution{TResult,typeof(M),typeof(d),typeof(proj!)}(
        M,
        d,
        proj!,
    )
end

function rand(rng::AbstractRNG, d::ProjectedPointDistribution{TResult}) where {TResult}
    x = convert(TResult, rand(rng, d.d))
    d.proj!(d.manifold, x)
    return x
end
function _rand!(rng::AbstractRNG, d::ProjectedPointDistribution, x::AbstractArray{<:Number})
    _rand!(rng, d.d, x)
    d.proj!(d.manifold, x)
    return x
end

function support(d::ProjectedPointDistribution)
    return MPointSupport(d.manifold)
end

"""
    ProjectedFVectorDistribution(type::VectorBundleFibers, x, d, project_vector!)

Generates a random vector from ambient space of manifold `type.manifold`
at point `x` and projects it to vector space of type `type` using function
`project_vector!`, see [`project_vector`](@ref) for documentation.
Generated arrays are of type `TResult`.
"""
struct ProjectedFVectorDistribution{
    TResult,
    TSpace<:VectorBundleFibers,
    TX,
    TD<:Distribution,
    TProj,
} <: FVectorDistribution{TSpace,TX}
    type::TSpace
    x::TX
    d::TD
    project_vector!::TProj
end

function ProjectedFVectorDistribution(
    type::VectorBundleFibers,
    x,
    d::Distribution,
    project_vector!,
    xt::TResult,
) where {TResult}
    return ProjectedFVectorDistribution{
        TResult,
        typeof(type),
        typeof(x),
        typeof(d),
        typeof(project_vector!),
    }(
        type,
        x,
        d,
        project_vector!,
    )
end

function rand(rng::AbstractRNG, d::ProjectedFVectorDistribution{TResult}) where {TResult}
    v = convert(TResult, reshape(rand(rng, d.d), size(d.x)))
    d.project_vector!(d.type, v, d.x, v)
    return v
end
function _rand!(
    rng::AbstractRNG,
    d::ProjectedFVectorDistribution,
    v::AbstractArray{<:Number},
)
    # calling _rand!(rng, d.d, v) doesn't work for all arrays types
    copyto!(v, rand(rng, d))
    return v
end

function support(tvd::ProjectedFVectorDistribution)
    return FVectorSupport(tvd.type, tvd.x)
end
