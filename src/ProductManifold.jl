@doc doc"""
    ProductManifold{TM<:Tuple, TRanges<:Tuple, TSizes<:Tuple} <: Manifold

Product manifold $M_1 \times M_2 \times \dots \times M_n$ with product geometry.
`TRanges` and `TSizes` statically define the relationship between representation
of the product manifold and representations of point, tangent vectors
and cotangent vectors of respective manifolds.

# Constructor

    ProductManifold(M_1, M_2, ..., M_n)

generates the product manifold $M_1 \times M_2 \times \dots \times M_n$.
Alternatively, the same manifold can be contructed using the `×` operator:
`M_1 × M_2 × M_3`.
"""
struct ProductManifold{TM<:Tuple} <: Manifold
    manifolds::TM
end

ProductManifold(manifolds::Manifold...) = ProductManifold{typeof(manifolds)}(manifolds)

function cross(M1::Manifold, M2::Manifold)
    return ProductManifold(M1, M2)
end

function cross(M1::ProductManifold, M2::Manifold)
    return ProductManifold(M1.manifolds..., M2)
end

function cross(M1::Manifold, M2::ProductManifold)
    return ProductManifold(M1, M2.manifolds...)
end

function cross(M1::ProductManifold, M2::ProductManifold)
    return ProductManifold(M1.manifolds..., M2.manifolds...)
end

"""
    submanifold(M::ProductManifold, i::Integer)

Extract the `i`th factor of the product manifold `M`.
"""
function submanifold(M::ProductManifold, i::Integer)
    return M.manifolds[i]
end

"""
    submanifold(M::ProductManifold, i::Val)

Extract the factor of the product manifold `M` indicated by indices in `i`.
For example, for `i` equal to `Val((1, 3))` the product manifold constructed
from the first and the third factor is returned.
"""
function submanifold(M::ProductManifold, i::Val)
    return ProductManifold(select_from_tuple(M.manifolds, i))
end

"""
    submanifold(M::ProductManifold, i::AbstractVector)

Extract the factor of the product manifold `M` indicated by indices in `i`.
For example, for `i` equal to `[1, 3]` the product manifold constructed
from the first and the third factor is returned.

This function is not type-stable, for better preformance use
`submanifold(M::ProductManifold, i::Val)`.
"""
submanifold(M::ProductManifold, i::AbstractVector) = submanifold(M, Val(tuple(i...)))

function isapprox(M::ProductManifold, x, y; kwargs...)
    return all(t -> isapprox(t...; kwargs...), ziptuples(M.manifolds, x.parts, y.parts))
end

function isapprox(M::ProductManifold, x, v, w; kwargs...)
    return all(t -> isapprox(t...; kwargs...), ziptuples(M.manifolds, x.parts, v.parts, w.parts))
end

function representation_size(M::ProductManifold)
    return (mapreduce(m -> prod(representation_size(m)), +, M.manifolds),)
end

manifold_dimension(M::ProductManifold) = mapreduce(manifold_dimension, +, M.manifolds)

struct ProductMetric <: Metric end

function det_local_metric(M::MetricManifold{ProductManifold, ProductMetric}, x::ProductArray)
    dets = map(det_local_metric, M.manifolds, x.parts)
    return prod(dets)
end

function inner(M::ProductManifold, x, v, w)
    subproducts = map(inner, M.manifolds, x.parts, v.parts, w.parts)
    return sum(subproducts)
end

function norm(M::ProductManifold, x, v)
    norms_squared = map(norm, M.manifolds, x.parts, v.parts).^2
    return sqrt(sum(norms_squared))
end

function distance(M::ProductManifold, x, y)
    return sqrt(sum(map(distance, M.manifolds, x.parts, y.parts).^2))
end

function exp!(M::ProductManifold, y, x, v)
    map(exp!, M.manifolds, y.parts, x.parts, v.parts)
    return y
end

function exp(M::ProductManifold, x::ProductRepr, v::ProductRepr)
    return ProductRepr(map(exp, M.manifolds, x.parts, v.parts)...)
end

function log!(M::ProductManifold, v, x, y)
    map(log!, M.manifolds, v.parts, x.parts, y.parts)
    return v
end

function log(M::ProductManifold, x::ProductRepr, y::ProductRepr)
    return ProductRepr(map(log, M.manifolds, x.parts, y.parts)...)
end

function injectivity_radius(M::ProductManifold, x)
    return min(map(injectivity_radius, M.manifolds, x.parts)...)
end

function injectivity_radius(M::ProductManifold)
    return min(map(injectivity_radius, M.manifolds)...)
end

"""
    ProductRetraction(retractions::AbstractRetractionMethod...)

Product retraction of `retractions`. Works on [`ProductManifold`](@ref).
"""
struct ProductRetraction{TR<:Tuple} <: AbstractRetractionMethod
    retractions::TR
end

ProductRetraction(retractions::AbstractRetractionMethod...) = ProductRetraction{typeof(retractions)}(retractions)

function retract!(M::ProductManifold, y, x, v, method::ProductRetraction)
    map(retract!, M.manifolds, y.parts, x.parts, v.parts, method.retractions)
    return y
end

struct InverseProductRetraction{TR<:Tuple} <: AbstractInverseRetractionMethod
    inverse_retractions::TR
end

"""
    InverseProductRetraction(inverse_retractions::AbstractInverseRetractionMethod...)

Product inverse retraction of `inverse_retractions`.
Works on [`ProductManifold`](@ref).
"""
InverseProductRetraction(inverse_retractions::AbstractInverseRetractionMethod...) = InverseProductRetraction{typeof(inverse_retractions)}(inverse_retractions)

function inverse_retract!(M::ProductManifold, v, x, y, method::InverseProductRetraction)
    map(inverse_retract!, M.manifolds, v.parts, x.parts, y.parts, method.inverse_retractions)
    return v
end

"""
    is_manifold_point(M::ProductManifold, x; kwargs...)

Check whether `x` is a valid point on the [`ProductManifold`](@ref) `M`.

The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_manifold_point(M::ProductManifold, x::ProductRepr; kwargs...)
    return all(t -> is_manifold_point(t...; kwargs...), ziptuples(M.manifolds, x.parts))
end

function is_manifold_point(M::ProductManifold, x::ProductArray; kwargs...)
    return all(t -> is_manifold_point(t...; kwargs...), ziptuples(M.manifolds, x.parts))
end

"""
    is_tangent_vector(M::ProductManifold, x, v; kwargs... )

Check whether `v` is a tangent vector to `x` on the [`ProductManifold`](@ref)
`M`, i.e. atfer [`is_manifold_point`](@ref)`(M, x)`, and all projections to
base manifolds must be respective tangent vectors.

The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_tangent_vector(M::ProductManifold, x::ProductRepr, v::ProductRepr; kwargs...)
    is_manifold_point(M, x)
    return all(t -> is_tangent_vector(t...; kwargs...), ziptuples(M.manifolds, x.parts, v.parts))
end

function is_tangent_vector(M::ProductManifold, x::ProductArray, v::ProductArray; kwargs...)
    is_manifold_point(M, x)
    return all(t -> is_tangent_vector(t...; kwargs...), ziptuples(M.manifolds, x.parts, v.parts))
end

"""
    ProductPointDistribution(M::ProductManifold, distributions)

Product distribution on manifold `M`, combined from `distributions`.
"""
struct ProductPointDistribution{TM<:ProductManifold, TD<:(NTuple{N,Distribution} where N)} <: MPointDistribution{TM}
    manifold::TM
    distributions::TD
end

function ProductPointDistribution(M::ProductManifold, distributions::MPointDistribution...)
    return ProductPointDistribution{typeof(M), typeof(distributions)}(M, distributions)
end

function ProductPointDistribution(distributions::MPointDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    return ProductPointDistribution(M, distributions...)
end

function support(d::ProductPointDistribution)
    return MPointSupport(d.manifold)
end

function rand(rng::AbstractRNG, d::ProductPointDistribution)
    return ProductRepr(map(d -> rand(rng, d), d.distributions)...)
end

function _rand!(rng::AbstractRNG, d::ProductPointDistribution, x::AbstractArray{<:Number})
    x .= rand(rng, d)
    return x
end

function _rand!(rng::AbstractRNG, d::ProductPointDistribution, x::ProductRepr)
    map(t -> _rand!(rng, t[1], t[2]), d.distributions, x.parts)
    return x
end

"""
    ProductTVectorDistribution([m::ProductManifold], [x], distrs...)

Generates a random tangent vector at point `x` from manifold `m` using the
product distribution of given distributions.

Manifold and `x` can be automatically inferred from distributions `distrs`.
"""
struct ProductTVectorDistribution{TM<:ProductManifold, TD<:(NTuple{N,Distribution} where N), TX} <: TVectorDistribution{TM, TX}
    manifold::TM
    x::TX
    distributions::TD
end

function ProductTVectorDistribution(M::ProductManifold, x::Union{AbstractArray, MPoint, ProductRepr}, distributions::TVectorDistribution...)
    return ProductTVectorDistribution{typeof(M), typeof(distributions), typeof(x)}(M, x, distributions)
end

function ProductTVectorDistribution(M::ProductManifold, distributions::TVectorDistribution...)
    x = ProductRepr(map(d -> support(d).x, distributions))
    return ProductTVectorDistribution(M, x, distributions...)
end

function ProductTVectorDistribution(distributions::TVectorDistribution...)
    M = ProductManifold(map(d -> support(d).manifold, distributions)...)
    x = ProductRepr(map(d -> support(d).x, distributions)...)
    return ProductTVectorDistribution(M, x, distributions...)
end

function support(tvd::ProductTVectorDistribution)
    return TVectorSupport(tvd.manifold, ProductRepr(map(d -> support(d).x, tvd.distributions)...))
end

function rand(rng::AbstractRNG, d::ProductTVectorDistribution)
    return ProductRepr(map(d -> rand(rng, d), d.distributions)...)
end

function _rand!(rng::AbstractRNG, d::ProductTVectorDistribution, v::AbstractArray{<:Number})
    v .= rand(rng, d)
    return v
end

function _rand!(rng::AbstractRNG, d::ProductTVectorDistribution, v::ProductRepr)
    map(t -> _rand!(rng, t[1], t[2]), d.distributions, v.parts)
    return v
end
