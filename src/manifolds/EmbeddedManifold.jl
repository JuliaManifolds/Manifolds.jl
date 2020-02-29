"""
    AbstractEmbeddingType

A type to provide properties of the embedding an [`AbstractEmbeddedManifold`](@ref) has.
"""
abstract type AbstractEmbeddingType end

"""
    AbstractEmbeddedManifold{T<:AbstractEmbeddingType} <: AbstractDecoratorManifold

An abstract type for an EmbeddedManifold, which acts as an [`AbstractDecoratorManifold`](@ref).
The functions of the manifold that is embedded can hence be just passed on to the embedding.
The embedding is further specified by an [`AbstractEmbeddingType`](@ref).

This means, that formally an embedded manifold is a decorator for the manifold it is embedded
into.
"""
abstract type AbstractEmbeddedManifold{T<:AbstractEmbeddingType} <: AbstractDecoratorManifold end

"""
    AbstractIsometricEmbeddingType <: AbstractEmbeddingType

Characterizes an that an embedding is isometric. For this case the [`inner`](@ref) product
is passed from the embedded manifold to the embedding.
"""
abstract type AbstractIsometricEmbeddingType <: AbstractEmbeddingType end

"""
    DefaultIsometricEmbedding <: AbstractIsometricEmbeddingType

Specify that an embedding is the default isometric embedding. This even inherits
logarithmic and exponential map as well as retraction and inverse retractions from the
embedding.

For an example, see [`SymmetricMatrices`](@ref) which are isometrically embedded in
the Euclidean space of matrices but also inherit exponential and logarithmic maps.
"""
struct DefaultIsometricEmbedding <: AbstractIsometricEmbeddingType end

"""
    EmbeddedManifold{MT <: Manifold, NT <: Manifold, ET} <: AbstractEmbeddedManifold{ET}

A type to represent that a [`Manifold`](@ref) `M` of type `MT` is indeed an emebedded
manifold and embedded into the manifold `N` of type `NT`.
Based on the [`AbstractEmbeddingType`](@ref) `ET`, this introduces functions for `M` by
passing through to embedding `N`.

# Fields

* `manifold` the manifold that is an embedded manifold
* `embedding` a second manifold, the first one is embedded into

# Constructor

    EmbeddedManifold(M,N,e=DefaultIsometricEmbedding)

Generate the `EmbeddedManifold` of the [`Manifold`](@ref) `M` into the
[`Manifold`](@ref) `N` with [`AbstractEmbeddingType`](@ref) `e` that by default is the most
transparent [`DefaultIsometricEmbedding`](@ref)
"""
struct EmbeddedManifold{MT <: Manifold, NT <: Manifold, ET} <: AbstractEmbeddedManifold{ET}
    manifold::MT
    embedding::NT
end
function EmbeddedManifold(
    M::MT,
    N::NT,
    e::ET=DefaultIsometricEmbedding()
) where {MT <: Manifold, NT <: Manifold, ET <: AbstractEmbeddingType}
    return EmbeddedManifold{MT,NT,ET}(M,N)
end
"""
    EmbeddedRetraction{R <: AbstractRetractionMethod} <: AbstractRetractionMethod

A retraction on an [`EmbeddedManifold`](@ref) by using the retraction `R` in the
embedding and projecting the result back to the manifold.

# Constructor

    EmbeddedInverseRetraction(m::AbstractRetractionMethod=ExponentialRetraction)

Generate the retraction that uses the [`AbstractRetractionMethod`](@ref) `m` in the
embedding, which defaults to the [`ExponentialRetraction`](@ref)
"""
struct EmbeddedRetraction{R <: AbstractRetractionMethod} <: AbstractRetractionMethod
    retraction_method::R
end
EmbeddedRetraction() = EmbeddedRetraction(ExponentialRetraction())
"""
    EmbeddedInverseRetraction{R <: AbstractInverseRetractionMethod} <: AbstractInverseRetractionMethod

An inverse retraction on an [`EmbeddedManifold`](@ref) by using the inverse retraction `R`
in the embedding and projecting the result back to the corresponding tangent space.

# Constructor

    EmbeddedInverseRetraction(m::AbstractInverseRetractionMethod=LogarithmicInverseRetraction)

Generate the inverse retraction that uses the [`AbstractInverseRetractionMethod`](@ref) `m`
in the embedding, which defaults to the [`LogarithmicInverseRetraction`](@ref).
"""
struct EmbeddedInverseRetraction{IR <: AbstractInverseRetractionMethod} <: AbstractInverseRetractionMethod
    inverse_retraction_method::IR
end
EmbeddedInverseRetraction() = EmbeddedInverseRetraction(LogarithmicInverseRetraction())

"""
    embed(M::AbstractEmbeddedManifold,p)

return the embedded representation of a point `p` on the [`AbstractEmbeddedManifold`](@ref)
`M`.

    embed(M::AbstractEmbeddedManifold,p,X)

return the embedded representation of a tangent vector `X` at point `p` on the
[`AbstractEmbeddedManifold`](@ref) `M`.
"""
embed(::AbstractEmbeddedManifold, ::Any...)

@decorator_transparent_function function embed(M::AbstractEmbeddedManifold,p)
    q = allocate(p)
    return embed!(M, q, p)
end
@decorator_transparent_function function embed!(M::AbstractEmbeddedManifold, q, p)
    error("Embedding a point $(typeof(p)) on $(typeof(M)) not yet implemented.")
end

@decorator_transparent_function function embed(M::AbstractEmbeddedManifold,p, X)
    q = allocate(p)
    return embed!(M, q, p)
end
@decorator_transparent_function function embed!(M::AbstractEmbeddedManifold, Y, p, X)
    error("Embedding a tangent $(X) at point $(typeof(p)) on $(typeof(M)) not yet implemented.")
end

"""
    inverse_retract(M::EmbeddedManifold, p, q, m::EmbeddedInverseRetraction)

Compute the inverse retraction on the [`AbstractEmbeddingType`](@ref) `M` by
using an [`AbstractInverseRetractionMethod`](@ref) in the embedding and projecting the
result back into the tangent space at `p`.
"""
inverse_retract(::AbstractEmbeddedManifold, ::Any, ::Any, ::EmbeddedInverseRetraction)

function inverse_retract!(M::AbstractEmbeddedManifold, X, p, q, m::EmbeddedInverseRetraction)
    x = allocate(q)
    y = allocate(p)
    embed!(M, x, p)
    embed!(M, y, q)
    retract!(M.embedding, X, x, y, m.inverse_retraction_method)
    project_tangent!(M, X, p, X)
    return X
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

"""
    inverse_retract(M::EmbeddedManifold, p, q, m::EmbeddedInverseRetraction)

Compute the retraction on the [`AbstractEmbeddingType`](@ref) `M` by
using an [`AbstractRetractionMethod`](@ref) in the embedding and projecting the
result back onto the manifold `M`.
"""
retract(::AbstractEmbeddedManifold, ::Any, ::Any, ::EmbeddedRetraction)

function retract!(M::AbstractEmbeddedManifold, q, p, X, m::EmbeddedRetraction)
    x = allocate(q)
    Z = allocate(X)
    embed!(M, x, p)
    embed!(M, Z, p, X)
    retract!(M.embedding, q, x, Z, m.retraction_method)
    project_point!(M, q, q)
    return q
end

function show(
    io::IO,
    M::EmbeddedManifold{MT,NT,ET}
) where {MT <: Manifold, NT <: Manifold, ET<:AbstractEmbeddingType}
    print(io, "EmbeddedManifold($(M.manifold), $(M.embedding), $(ET()))")
end

function default_decorator_dispatch(M::EmbeddedManifold)
    return default_embedding_dispatch(M)
end

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
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(exp),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(exp!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(exp!),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
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
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(inverse_retract),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
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
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end

function decorator_transparent_dispatch(
    ::typeof(log),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(log),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(log!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(log!),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(norm),
    ::AbstractEmbeddedManifold,
    args...,
)
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
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(project_point),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(project_point!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_point!),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_tangent),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(project_tangent),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_tangent!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_tangent!),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(retract),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(retract),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
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
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along!),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_direction!),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:parent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_to!),
    ::AbstractEmbeddedManifold{<:DefaultIsometricEmbedding},
    args...,
)
    return Val(:transparent)
end
