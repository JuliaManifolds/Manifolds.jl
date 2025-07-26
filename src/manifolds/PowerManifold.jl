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

Represent the [`AbstractMetric`](@extref `ManifoldsBase.AbstractMetric`)
on an `AbstractPowerManifold`, i.e. the inner
product on the tangent space is the sum of the inner product of each elements
tangent space of the power manifold.
"""
struct PowerMetric <: AbstractMetric end

function PowerManifold(
        M::AbstractManifold{𝔽},
        size::Integer...;
        parameter::Symbol = :field,
    ) where {𝔽}
    size_w = wrap_type_parameter(parameter, size)
    return PowerManifold{𝔽, typeof(M), typeof(size_w), ArrayPowerRepresentation}(M, size_w)
end

const PowerManifoldMultidimensional =
    AbstractPowerManifold{𝔽, <:AbstractManifold{𝔽}, ArrayPowerRepresentation} where {𝔽}

Base.:^(M::AbstractManifold, n) = PowerManifold(M, n...)

function allocate(::PowerManifoldNestedReplacing, x::AbstractArray{<:SArray})
    return similar(x)
end

for PowerRepr in [PowerManifoldNested, PowerManifoldNestedReplacing]
    @eval begin
        function allocate_result(::$PowerRepr, ::typeof(get_point), a)
            return error("Operation unsupported")
        end
        function allocate_result(M::$PowerRepr, f::typeof(get_parameters), p)
            return invoke(
                allocate_result,
                Tuple{AbstractManifold, typeof(get_parameters), Any},
                M,
                f,
                p,
            )
        end
    end
end

function allocate_result(M::PowerManifoldNestedReplacing, f, ::Identity, x...)
    return allocate_result(M, f, x...)
end

@doc raw"""
    flat(M::AbstractPowerManifold, p, X)

use the musical isomorphism to transform the tangent vector `X` from the tangent space at
`p` on an [`AbstractPowerManifold`](@extref `ManifoldsBase.AbstractPowerManifold`)  `M` to a cotangent vector.
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
        p::AbstractArray{T, N},
        M::PowerManifoldMultidimensional,
        I::Vararg{Integer, N},
    ) where {T, N}
    return get_component(M, p, I...)
end

@doc raw"""
    manifold_volume(M::PowerManifold)

Return the manifold volume of an [`PowerManifold`](@extref `ManifoldsBase.PowerManifold`) `M`.
"""
function manifold_volume(M::PowerManifold)
    size = get_parameter(M.size)
    return manifold_volume(M.manifold)^prod(size)
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
        i::NTuple{N, Int},
    ) where {N}
    return x[i...]
end

function Base.view(
        p::AbstractArray,
        M::PowerManifoldMultidimensional,
        I::Union{Integer, Colon, AbstractVector}...,
    )
    rep_size = representation_size(M.manifold)
    return _write(M, rep_size, p, I...)
end

function representation_size(M::PowerManifold)
    return (representation_size(M.manifold)..., get_parameter(M.size)...)
end

@doc raw"""
    Y = riemannian_Hessian(M::AbstractPowerManifold, p, G, H, X)
    riemannian_Hessian!(M::AbstractPowerManifold, Y, p, G, H, X)

Compute the Riemannian Hessian ``\operatorname{Hess} f(p)[X]`` given the
Euclidean gradient ``∇ f(\tilde p)`` in `G` and the Euclidean Hessian ``∇^2 f(\tilde p)[\tilde X]`` in `H`,
where ``\tilde p, \tilde X`` are the representations of ``p,X`` in the embedding,.

On an abstract power manifold, this decouples and can be computed elementwise.
"""
riemannian_Hessian(M::AbstractPowerManifold, p, G, H, X)

function riemannian_Hessian!(M::AbstractPowerManifold, Y, p, G, H, X)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        riemannian_Hessian!(
            M.manifold,
            _write(M, rep_size, Y, i),
            _read(M, rep_size, p, i),
            _read(M, rep_size, G, i),
            _read(M, rep_size, H, i),
            _read(M, rep_size, X, i),
        )
    end
    return Y
end

@doc raw"""
    sharp(M::AbstractPowerManifold, p, ξ::RieszRepresenterCotangentVector)

Use the musical isomorphism to transform the cotangent vector `ξ` from the tangent space at
`p` on an [`AbstractPowerManifold`](@extref `ManifoldsBase.AbstractPowerManifold`)  `M` to a tangent vector.
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
        M::PowerManifold{𝔽, TM, TypeParameter{TSize}, ArrayPowerRepresentation},
    ) where {𝔽, TM <: AbstractManifold{𝔽}, TSize}
    return print(
        io,
        "PowerManifold($(M.manifold), $(join(TSize.parameters, ", ")), parameter=:type)",
    )
end
function Base.show(
        io::IO,
        M::PowerManifold{𝔽, TM, <:Tuple, ArrayPowerRepresentation},
    ) where {𝔽, TM <: AbstractManifold{𝔽}}
    size = get_parameter(M.size)
    return print(io, "PowerManifold($(M.manifold), $(join(size, ", ")))")
end

@doc raw"""
    volume_density(M::PowerManifold, p, X)

Return volume density on the [`PowerManifold`](@extref `ManifoldsBase.PowerManifold`) `M`, i.e. product of constituent
volume densities.
"""
function volume_density(M::PowerManifold, p, X)
    density = one(float(eltype(X)))
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        p_i = _read(M, rep_size, p, i)
        X_i = _read(M, rep_size, X, i)
        density *= volume_density(M.manifold, p_i, X_i)
    end
    return density
end

@inline function _write(
        ::PowerManifoldMultidimensional,
        rep_size::Tuple,
        x::AbstractArray,
        i::Tuple,
    )
    return view(x, rep_size_to_colons(rep_size)..., i...)
end
