
"""
    ProjectedPointDistribution(m::Manifold, d, proj!, x)

Generates a random point in ambient space of `m` and projects it to `m`
using function `proj!`. Generated arrays are of type `TResult`, which can be
specified by providing the `x` argument.
"""
struct ProjectedPointDistribution{TResult, TM<:Manifold, TD<:Distribution, TProj} <: MPointDistribution{TM}
    manifold::TM
    d::TD
    proj!::TProj
end

function ProjectedPointDistribution(M::Manifold, d::Distribution, proj!, x::TResult) where TResult
    return ProjectedPointDistribution{TResult, typeof(M), typeof(d), typeof(proj!)}(M, d, proj!)
end

function rand(rng::AbstractRNG, d::ProjectedPointDistribution{TResult}) where TResult
    x = convert(TResult, rand(rng, d.d))
    d.proj!(d.manifold, x)
    return x
end

function _rand!(rng::AbstractRNG, d::ProjectedPointDistribution, x::AbstractArray{<:Number})
    rand!(rng, d.d, x)
    d.proj!(d.manifold, x)
    return x
end


"""
    ProjectedTVectorDistribution(m::Manifold, d, proj!)

Generates a random tangent vector in ambient space of `m` and projects it
to `m` using function `project_tangent!`.
Generated arrays are of type `TResult`.
"""
struct ProjectedTVectorDistribution{TResult, TM<:Manifold, TX, TD<:Distribution, TProj} <: MPointDistribution{TM}
    manifold::TM
    x::TX
    d::TD
    project_tangent!::TProj
end

function ProjectedTVectorDistribution(M::Manifold, x, d::Distribution, project_tangent!, xt::TResult) where TResult
    return ProjectedTVectorDistribution{TResult, typeof(M), typeof(x), typeof(d), typeof(project_tangent!)}(M, x, d, project_tangent!)
end

function get_support(tvd::ProjectedTVectorDistribution)
    return TVectorSupport(tvd.manifold, tvd.x)
end

function rand(rng::AbstractRNG, d::ProjectedTVectorDistribution{TResult}) where TResult
    v = convert(TResult, reshape(rand(rng, d.d), size(d.x)))
    d.project_tangent!(d.manifold, v, d.x, v)
    return v
end

function _rand!(rng::AbstractRNG, d::ProjectedTVectorDistribution, v::AbstractArray{<:Number})
    rand!(rng, d.d, v)
    d.project_tangent!(d.manifold, v, d.x, v)
    return v
end
