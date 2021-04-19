
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

function PowerManifold(M::AbstractManifold{𝔽}, size::Integer...) where {𝔽}
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
    AbstractPowerManifold{𝔽,<:AbstractManifold{𝔽},ArrayPowerRepresentation} where {𝔽}

Base.:^(M::AbstractManifold, n) = PowerManifold(M, n...)

function allocate_result(M::PowerManifoldNested, ::typeof(flat), w::TFVector, x)
    alloc = [allocate(_access_nested(w.data, i)) for i in get_iterator(M)]
    return FVector(CotangentSpace, alloc)
end
function allocate_result(M::PowerManifoldNested, f::typeof(get_point), x)
    return [allocate_result(M.manifold, f, _access_nested(x, i)) for i in get_iterator(M)]
end
function allocate_result(M::PowerManifoldNested, f::typeof(get_point_coordinates), p)
    return invoke(
        allocate_result,
        Tuple{AbstractManifold,typeof(get_point_coordinates),Any},
        M,
        f,
        p,
    )
end
function allocate_result(M::PowerManifoldNested, ::typeof(sharp), w::CoTFVector, x)
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
    flat(M::AbstractPowerManifold, p, X)

use the musical isomorphism to transform the tangent vector `X` from the tangent space at
`p` on an [`AbstractPowerManifold`](@ref) `M` to a cotangent vector.
This can be done elementwise for each entry of `X` (and `p`).
"""
flat(::AbstractPowerManifold, ::Any...)

function flat!(M::AbstractPowerManifold, ξ::RieszRepresenterCotangentVector, p, X)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        p_i = _read(M, rep_size, p, i)
        flat!(
            M.manifold,
            RieszRepresenterCotangentVector(M.manifold, p_i, _write(M, rep_size, ξ.X, i)),
            p_i,
            _read(M, rep_size, X, i),
        )
    end
    return ξ
end

Base.@propagate_inbounds function Base.getindex(
    p::AbstractArray,
    M::PowerManifoldMultidimensional,
    I::Integer...,
)
    return collect(get_component(M, p, I...))
end
Base.@propagate_inbounds function Base.getindex(
    p::AbstractArray{T,N},
    M::PowerManifoldMultidimensional,
    I::Vararg{Integer,N},
) where {T,N}
    return get_component(M, p, I...)
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
    rep_size::Tuple{},
    x::AbstractArray,
    i::NTuple{N,Int},
) where {N}
    return x[i...]
end
Base.@propagate_inbounds @inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::HybridArray,
    i::Tuple,
)
    return x[rep_size_to_colons(rep_size)..., i...]
end
Base.@propagate_inbounds @inline function _read(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple{},
    x::HybridArray,
    i::NTuple{N,Int},
) where {N}
    # disambiguation
    return x[i...]
end

function representation_size(M::PowerManifold{𝔽,<:AbstractManifold,TSize}) where {𝔽,TSize}
    return (representation_size(M.manifold)..., size_to_tuple(TSize)...)
end

@doc raw"""
    sharp(M::AbstractPowerManifold, p, ξ::RieszRepresenterCotangentVector)

Use the musical isomorphism to transform the cotangent vector `ξ` from the tangent space at
`p` on an [`AbstractPowerManifold`](@ref) `M` to a tangent vector.
This can be done elementwise for every entry of `ξ` (and `p`).
"""
sharp(::AbstractPowerManifold, ::Any...)

function sharp!(M::AbstractPowerManifold, X, p, ξ::RieszRepresenterCotangentVector)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        p_i = _read(M, rep_size, p, i)
        sharp!(
            M.manifold,
            _write(M, rep_size, X, i),
            p_i,
            RieszRepresenterCotangentVector(M.manifold, p_i, _read(M, rep_size, ξ.X, i)),
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
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::AbstractArray,
    i::Tuple,
)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
