
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
    PowerMetric <: AbstractMetric

Represent the [`AbstractMetric`](@ref) on an `AbstractPowerManifold`, i.e. the inner
product on the tangent space is the sum of the inner product of each elements
tangent space of the power manifold.
"""
struct PowerMetric <: AbstractMetric end

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

for PowerRepr in [PowerManifoldNested, PowerManifoldNestedReplacing]
    @eval begin
        function allocate_result(::$PowerRepr, ::typeof(get_point), a)
            return error("Operation unsupported")
        end
        function allocate_result(M::$PowerRepr, f::typeof(get_parameters), p)
            return invoke(
                allocate_result,
                Tuple{AbstractManifold,typeof(get_parameters),Any},
                M,
                f,
                p,
            )
        end
    end
end

"""
    change_representer(M::AbstractPowerManifold, ::AbstractMetric, p, X)

Since the metric on a power manifold decouples, the change of a representer can be done elementwise
"""
change_representer(::AbstractPowerManifold, ::AbstractMetric, ::Any, ::Any)

function change_representer!(M::AbstractPowerManifold, Y, G::AbstractMetric, p, X)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        change_representer!(
            M.manifold,
            _write(M, rep_size, Y, i),
            G,
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
        )
    end
    return Y
end

"""
    change_metric(M::AbstractPowerManifold, ::AbstractMetric, p, X)

Since the metric on a power manifold decouples, the change of metric can be done elementwise.
"""
change_metric(M::AbstractPowerManifold, ::AbstractMetric, ::Any, ::Any)

function change_metric!(M::AbstractPowerManifold, Y, G::AbstractMetric, p, X)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        change_metric!(
            M.manifold,
            _write(M, rep_size, Y, i),
            G,
            _read(M, rep_size, p, i),
            _read(M, rep_size, X, i),
        )
    end
    return Y
end

@doc raw"""
    flat(M::AbstractPowerManifold, p, X)

use the musical isomorphism to transform the tangent vector `X` from the tangent space at
`p` on an [`AbstractPowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractPowerManifold)  `M` to a cotangent vector.
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

function Random.rand!(M::AbstractPowerManifold, pX; vector_at=nothing, kwargs...)
    rep_size = representation_size(M.manifold)
    if vector_at === nothing
        for i in get_iterator(M)
            rand!(M.manifold, _write(M, rep_size, pX, i))
        end
    else
        for i in get_iterator(M)
            rand!(
                M.manifold,
                _write(M, rep_size, pX, i);
                vector_at=_read(M, rep_size, vector_at, i),
            )
        end
    end
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::AbstractPowerManifold,
    pX;
    vector_at=nothing,
    kwargs...,
)
    rep_size = representation_size(M.manifold)
    if vector_at === nothing
        for i in get_iterator(M)
            rand!(rng, M.manifold, _write(M, rep_size, pX, i))
        end
    else
        for i in get_iterator(M)
            rand!(
                rng,
                M.manifold,
                _write(M, rep_size, pX, i);
                vector_at=_read(M, rep_size, vector_at, i),
            )
        end
    end
    return pX
end
function Random.rand!(M::PowerManifoldNestedReplacing, pX; vector_at=nothing, kwargs...)
    if vector_at === nothing
        for i in get_iterator(M)
            pX[i...] = rand(M.manifold; kwargs...)
        end
    else
        for i in get_iterator(M)
            pX[i...] = rand(M.manifold; vector_at=vector_at[i...], kwargs...)
        end
    end
    return pX
end
function Random.rand!(
    rng::AbstractRNG,
    M::PowerManifoldNestedReplacing,
    pX;
    vector_at=nothing,
    kwargs...,
)
    if vector_at === nothing
        for i in get_iterator(M)
            pX[i...] = rand(rng, M.manifold; kwargs...)
        end
    else
        for i in get_iterator(M)
            pX[i...] = rand(rng, M.manifold; vector_at=vector_at[i...], kwargs...)
        end
    end
    return pX
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

function Base.view(
    p::AbstractArray,
    M::PowerManifoldMultidimensional,
    I::Union{Integer,Colon,AbstractVector}...,
)
    rep_size = representation_size(M.manifold)
    return _write(M, rep_size, p, I...)
end

function representation_size(M::PowerManifold{𝔽,<:AbstractManifold,TSize}) where {𝔽,TSize}
    return (representation_size(M.manifold)..., size_to_tuple(TSize)...)
end

@doc raw"""
    sharp(M::AbstractPowerManifold, p, ξ::RieszRepresenterCotangentVector)

Use the musical isomorphism to transform the cotangent vector `ξ` from the tangent space at
`p` on an [`AbstractPowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractPowerManifold)  `M` to a tangent vector.
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
    return ParallelTransport()
end

@inline function _write(
    ::PowerManifoldMultidimensional,
    rep_size::Tuple,
    x::AbstractArray,
    i::Tuple,
)
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
