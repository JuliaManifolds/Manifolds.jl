
"""
    ArrayPowerRepresentation

Representation of points and tangent vectors on a power manifold using multidimensional
arrays where first dimensions are equal to [`representation_size`](@ref) of the
wrapped manifold and the following ones are equal to the number of elements in each
direction.

[`Torus`](@ref) uses this representation.
"""
struct ArrayPowerRepresentation <: AbstractPowerRepresentation end

@doc raw"""
    PowerMetric <: Metric

Represent the [`Metric`](@ref) on an [`AbstractPowerManifold`](@ref), i.e. the inner
product on the tangent space is the sum of the inner product of each elements
tangent space of the power manifold.
"""
struct PowerMetric <: Metric end

function PowerManifold(M::Manifold{𝔽}, size::Integer...) where {𝔽}
    return PowerManifold{𝔽,typeof(M),Tuple{size...},ArrayPowerRepresentation}(M)
end

"""
    PowerPointDistribution(M::AbstractPowerManifold, distribution)

Power distribution on manifold `M`, based on `distribution`.
"""
struct PowerPointDistribution{TM<:AbstractPowerManifold,TD<:MPointDistribution,TX} <:
       MPointDistribution{TM}
    manifold::TM
    distribution::TD
    point::TX
end

"""
    PowerFVectorDistribution([type::VectorBundleFibers], [x], distr)

Generates a random vector at a `point` from vector space (a fiber of a tangent
bundle) of type `type` using the power distribution of `distr`.

Vector space type and `point` can be automatically inferred from distribution `distr`.
"""
struct PowerFVectorDistribution{
    TSpace<:VectorBundleFibers{<:VectorSpaceType,<:AbstractPowerManifold},
    TD<:FVectorDistribution,
    TX,
} <: FVectorDistribution{TSpace,TX}
    type::TSpace
    point::TX
    distribution::TD
end

const PowerManifoldMultidimensional =
    AbstractPowerManifold{𝔽,<:Manifold{𝔽},ArrayPowerRepresentation} where {𝔽}

Base.:^(M::Manifold, n) = PowerManifold(M, n...)

function allocate_result(M::PowerManifoldNested, f::typeof(flat), w::TFVector, x)
    alloc = [allocate(_access_nested(w.data, i)) for i in get_iterator(M)]
    return FVector(CotangentSpace, alloc)
end
function allocate_result(M::PowerManifoldNested, f::typeof(sharp), w::CoTFVector, x)
    alloc = [allocate(_access_nested(w.data, i)) for i in get_iterator(M)]
    return FVector(TangentSpace, alloc)
end

default_metric_dispatch(::AbstractPowerManifold, ::PowerMetric) = Val(true)

function det_local_metric(
    M::MetricManifold{PowerMetric,𝔽,<:AbstractPowerManifold{𝔽}},
    p::AbstractArray,
) where {𝔽}
    result = one(number_eltype(p))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        result *= det_local_metric(M.manifold, _read(M, rep_size, p, i))
    end
    return result
end

@doc raw"""
    flat(M::AbstractPowerManifold, p, X::FVector{TangentSpaceType})

use the musical isomorphism to transform the tangent vector `X` from the tangent space at
`p` on an [`AbstractPowerManifold`](@ref) `M` to a cotangent vector.
This can be done elementwise for each entry of `X` (and `p`).
"""
flat(::AbstractPowerManifold, ::Any...)

function flat!(M::AbstractPowerManifold, ξ::CoTFVector, p, X::TFVector)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        flat!(
            M.manifold,
            FVector(CotangentSpace, _write(M, rep_size, ξ.data, i)),
            _read(M, rep_size, p, i),
            FVector(TangentSpace, _read(M, rep_size, X.data, i)),
        )
    end
    return ξ
end

function Random.rand(rng::AbstractRNG, d::PowerFVectorDistribution)
    fv = zero_vector(d.type, d.point)
    Distributions._rand!(rng, d, fv)
    return fv
end
function Random.rand(rng::AbstractRNG, d::PowerPointDistribution)
    x = allocate_result(d.manifold, rand, d.point)
    Distributions._rand!(rng, d, x)
    return x
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::PowerFVectorDistribution,
    v::AbstractArray,
)
    PM = d.type.manifold
    rep_size = representation_size(PM.manifold)
    for i in get_iterator(d.type.manifold)
        copyto!(d.distribution.point, _read(PM, rep_size, d.point, i))
        Distributions._rand!(rng, d.distribution, _read(PM, rep_size, v, i))
    end
    return v
end
function Distributions._rand!(rng::AbstractRNG, d::PowerPointDistribution, x::AbstractArray)
    M = d.manifold
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        Distributions._rand!(rng, d.distribution, _write(M, rep_size, x, i))
    end
    return x
end

Base.@propagate_inbounds @inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::AbstractArray,
    i::Tuple,
)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
Base.@propagate_inbounds @inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::HybridArray,
    i::Tuple,
)
    return x[rep_size_to_colons(rep_size)..., i...]
end

function representation_size(M::PowerManifold{𝔽,<:Manifold,TSize}) where {𝔽,TSize}
    return (representation_size(M.manifold)..., size_to_tuple(TSize)...)
end

@doc raw"""
    sharp(M::AbstractPowerManifold, p, ξ::FVector{CotangentSpaceType})

Use the musical isomorphism to transform the cotangent vector `ξ` from the tangent space at
`p` on an [`AbstractPowerManifold`](@ref) `M` to a tangent vector.
This can be done elementwise for every entry of `ξ` (and `p`).
"""
sharp(::AbstractPowerManifold, ::Any...)

function sharp!(M::AbstractPowerManifold, X::TFVector, p, ξ::CoTFVector)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        sharp!(
            M.manifold,
            FVector(TangentSpace, _write(M, rep_size, X.data, i)),
            _read(M, rep_size, p, i),
            FVector(CotangentSpace, _read(M, rep_size, ξ.data, i)),
        )
    end
    return X
end

function Base.show(
    io::IO,
    M::PowerManifold{𝔽,TM,TSize,ArrayPowerRepresentation},
) where {𝔽,TM,TSize}
    return print(io, "PowerManifold($(M.manifold), $(join(TSize.parameters, ", ")))")
end

Distributions.support(tvd::PowerFVectorDistribution) = FVectorSupport(tvd.type, tvd.point)
Distributions.support(d::PowerPointDistribution) = MPointSupport(d.manifold)

function vector_bundle_transport(fiber::VectorSpaceType, M::PowerManifold)
    return PowerVectorTransport(ParallelTransport())
end

@inline function _write(
    M::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::AbstractArray,
    i::Tuple,
)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
