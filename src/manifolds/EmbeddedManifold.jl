"""
    AbstractEmbeddingType

A type used to specify properties of an [`AbstractEmbeddedManifold`](@ref).
"""
abstract type AbstractEmbeddingType end

"""
    AbstractEmbeddedManifold{T<:AbstractEmbeddingType} <: AbstractDecoratorManifold

An abstract type for embedded manifolds, which acts as an [`AbstractDecoratorManifold`](@ref).
The functions of the manifold that is embedded can hence be just passed on to the embedding.
The embedding is further specified by an [`AbstractEmbeddingType`](@ref).

This means, that technically an embedded manifold is a decorator for the embedding, i.e.
functions of this type get, in the semi-transparent way of the
[`AbstractDecoratorManifold`](@ref), passed on to the embedding.
"""
abstract type AbstractEmbeddedManifold{T<:AbstractEmbeddingType} <:
              AbstractDecoratorManifold end

"""
DefaultEmbeddingType <: AbstractEmbeddingType

A type of default embedding that does not have any special properties.
"""
struct DefaultEmbeddingType <: AbstractEmbeddingType end

"""
    AbstractIsometricEmbeddingType <: AbstractEmbeddingType

Characterizes an embedding as isometric. For this case the [`inner`](@ref) product
is passed from the embedded manifold to the embedding.
"""
abstract type AbstractIsometricEmbeddingType <: AbstractEmbeddingType end


"""
    DefaultIsometricEmbeddingType <: AbstractIsometricEmbeddingType

An isometric embedding type that acts as a default, i.e. it has no specifig properties
beyond its isometric property.
"""
struct DefaultIsometricEmbeddingType <: AbstractIsometricEmbeddingType end

"""
    TransparentIsometricEmbedding <: AbstractIsometricEmbeddingType

Specify that an embedding is the default isometric embedding. This even inherits
logarithmic and exponential map as well as retraction and inverse retractions from the
embedding.

For an example, see [`SymmetricMatrices`](@ref) which are isometrically embedded in
the Euclidean space of matrices but also inherit exponential and logarithmic maps.
"""
struct TransparentIsometricEmbedding <: AbstractIsometricEmbeddingType end

"""
    EmbeddedManifold{MT <: Manifold, NT <: Manifold, ET} <: AbstractEmbeddedManifold{ET}

A type to represent that a [`Manifold`](@ref) `M` of type `MT` is indeed an emebedded
manifold and embedded into the manifold `N` of type `NT`.
Based on the [`AbstractEmbeddingType`](@ref) `ET`, this introduces methods for `M` by
passing them through to embedding `N`.

# Fields

* `manifold` the manifold that is an embedded manifold
* `embedding` a second manifold, the first one is embedded into

# Constructor

    EmbeddedManifold(M, N, e=TransparentIsometricEmbedding())

Generate the `EmbeddedManifold` of the [`Manifold`](@ref) `M` into the
[`Manifold`](@ref) `N` with [`AbstractEmbeddingType`](@ref) `e` that by default is the most
transparent [`TransparentIsometricEmbedding`](@ref)
"""
struct EmbeddedManifold{MT<:Manifold,NT<:Manifold,ET} <: AbstractEmbeddedManifold{ET}
    manifold::MT
    embedding::NT
end
function EmbeddedManifold(
    M::MT,
    N::NT,
    e::ET = TransparentIsometricEmbedding(),
) where {MT<:Manifold,NT<:Manifold,ET<:AbstractEmbeddingType}
    return EmbeddedManifold{MT,NT,ET}(M, N)
end

"""
    base_manifold(M::AbstractEmbeddedManifold, d::Val{N} = Val(-1))

Return the base manifold of `M`. While functions like `inner` might be overwritten to
use the (decorated) manifold representing the embedding, the base_manifold is the manifold
itself in the sense that detemining e.g. the [`is_default_metric`](@ref) does not check with
the embedding but with the manifold itself.
"""
base_manifold(M::AbstractEmbeddedManifold, d::Val{N}=Val(-1)) where {N} = M
base_manifold(M::EmbeddedManifold, d::Val{N}=Val(-1)) where {N} = M.M


"""
    check_manifold_point(M::AbstractEmbeddedManifold, p; kwargs)

check whether a point `p` is a valid point on the [`AbstractEmbeddedManifold`](@ref),
i.e. that `embed(M, p)` is a valid point on the embedded manifold.
"""
function check_manifold_point(M::AbstractEmbeddedManifold, p; kwargs...)
    q = embed(M,p)
    return invoke(
        check_manifold_point,
        Tuple{typeof(get_embedding(M)), typeof(q)},
        get_embedding(M),
        q;
        kwargs...
    )
end

"""
    check_tangent_vector(M::AbstractEmbeddedManifold, p, X; check_base_point = true, kwargs...)

check that `embed(M,p,X)` is a valid tangent to `embed(p,X)`, where `check_base_point`
determines whether the validity of `p` is checked, too.
"""
function check_tangent_vector(M::AbstractEmbeddedManifold, p, X; check_base_point = true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    q = embed(M,p)
    Y = embed(M,p,X)
    return invoke(
        check_tangent_vector,
        Tuple{typeof(get_embedding(M)), typeof(q), typeof(Y)},
        get_embedding(M),
        q,
        Y;
        check_base_point = check_base_point,
        kwargs...
    )
end
"""
    embed(M::AbstractEmbeddedManifold, p)

return the embedded representation of a point `p` on the [`AbstractEmbeddedManifold`](@ref)
`M`.

    embed(M::AbstractEmbeddedManifold, p, X)

return the embedded representation of a tangent vector `X` at point `p` on the
[`AbstractEmbeddedManifold`](@ref) `M`.
"""
embed(::AbstractEmbeddedManifold, ::Any...)

@decorator_transparent_function function embed(M::AbstractEmbeddedManifold, p)
    q = allocate(p)
    embed!(M, q, p)
    return q
end
@decorator_transparent_function function embed!(M::AbstractEmbeddedManifold, q, p)
    error("Embedding a point $(typeof(p)) on $(typeof(M)) not yet implemented.")
end

@decorator_transparent_function function embed(M::AbstractEmbeddedManifold, p, X)
    Y = allocate(X)
    embed!(M, Y, p, X)
    return Y
end
@decorator_transparent_function function embed!(M::AbstractEmbeddedManifold, Y, p, X)
    error("Embedding a tangent $(typeof(X)) at point $(typeof(p)) on $(typeof(M)) not yet implemented.")
end

decorated_manifold(M::AbstractEmbeddedManifold) = M.embedding

"""
    get_embedding(M::AbstractEmbeddedManifold)

Return the [`Manifold`](@ref) `N` an [`AbstractEmbeddedManifold`](@ref) is embedded into.
"""
get_embedding(::AbstractEmbeddedManifold)

@decorator_transparent_function function get_embedding(M::AbstractEmbeddedManifold)
    return decorated_manifold(M)
end

function show(
    io::IO,
    M::EmbeddedManifold{MT,NT,ET},
) where {MT<:Manifold,NT<:Manifold,ET<:AbstractEmbeddingType}
    print(io, "EmbeddedManifold($(M.manifold), $(M.embedding), $(ET()))")
end

function default_decorator_dispatch(M::EmbeddedManifold)
    return default_embedding_dispatch(M)
end

@doc doc"""
    default_embedding_dispatch(M::AbstractEmbeddedManifold)

This method indicates that an [`AbstractEmbeddedManifold`](@ref) is the default
and hence acts completely transparently and passes all functions transparently onwards.
This is used by the [`AbstractDecoratorManifold`](@ref) within
[`default_decorator_dispatch`](@ref).
By default this is set to `Val(false)`.
"""
default_embedding_dispatch(M::AbstractEmbeddedManifold) = Val(false)

function decorator_transparent_dispatch(
    ::typeof(check_manifold_point),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(check_tangent_vector),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(exp),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(exp),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(exp!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(exp!),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(get_basis),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(get_coordinates),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(get_vector),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(inner),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(inner),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract!),
    ::AbstractEmbeddedManifold,
    ::Any,
    ::Any,
    ::Any,
    ::LogarithmicInverseRetraction,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract!),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end

function decorator_transparent_dispatch(
    ::typeof(log),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(log),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(log!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(log!),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(::typeof(norm), ::AbstractEmbeddedManifold, args...)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(norm),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(manifold_dimension),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_point),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(project_point!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_point!),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_tangent),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(project_tangent),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_tangent!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_tangent!),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(retract),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(retract),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(retract!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(retract!),
    ::AbstractEmbeddedManifold,
    ::Any,
    ::Any,
    ::Any,
    ::ExponentialRetraction,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(retract!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(retract!),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along!),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction!),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to!),
    ::AbstractEmbeddedManifold{<:TransparentIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
