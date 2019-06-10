
"""
	ProjectedDistribution(m::Manifold, d, proj!)

Generates a random point in ambient space of `m` and projects it to `m`
using function `proj!`. Generated arrays are of type `TResult`.
"""
struct ProjectedDistribution{TResult, TM<:Manifold, TD<:Distribution, TProj} <: MPointDistribution{TM}
	manifold::TM
	d::TD
	proj!::TProj
end

function ProjectedDistribution(M::Manifold, d::Distribution, proj!, x::TResult) where TResult
	return ProjectedDistribution{TResult, typeof(M), typeof(d), typeof(proj!)}(M, d, proj!)
end

function rand(rng::AbstractRNG, d::ProjectedDistribution{TResult}) where TResult
	x = convert(TResult, rand(rng, d.d))
	d.proj!(d.manifold, x)
    return x
end

function _rand!(rng::AbstractRNG, d::ProjectedDistribution, x::AbstractArray{<:Number})
	rand!(rng, d.d, x)
	d.proj!(d.manifold, x)
    return x
end
