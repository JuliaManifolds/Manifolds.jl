

"""
	RetractedDistribution(m::Manifold, d)

Generates a random point in ambient space of `m` and retracts it to `m`.
"""
struct RetractedDistribution{TM<:Manifold, TD<:Distribution} <: MPointDistribution{TM}
	manifold::TM
	d::TD
end

function rand(rng::AbstractRNG, d::RetractedDistribution)
	x = rand(rng, d.d)
	retr!(d.manifold, x)
    return x
end

function _rand!(rng::AbstractRNG, d::RetractedDistribution, x::AbstractArray{<:Number})
	rand!(rng, d.d, x)
	retr!(d.manifold, x)
    return x
end

# example:
# using Distributions
# d = ManifoldMuseum.RetractedDistribution(ManifoldMuseum.Sphere((3,)), Distributions.MvNormal([0.0, 0.0, 0.0], 1.0))
# rand(d)
