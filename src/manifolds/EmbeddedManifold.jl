abstract type AbstractEmbeddingType end
abstract type AbstractIsometricEmbeddingType <: AbstractEmbeddingType end

"""
    AbstractEmbeddedManifold <: AbstractDecoratorManifold

An abstract type for an EmbeddedManifold,
"""
abstract type AbstractEmbeddedManifold{T<:AbstractEmbeddingType} <: AbstractDecoratorManifold end

"""
    DefaultIsometricEmbedding <: AbstractEmbeddingType

"""
struct DefaultIsometricEmbedding <: AbstractIsometricEmbeddingType end

struct EmbeddedManifold{MT <: Manifold, NT <: Manifold, ET} <: AbstractEmbeddedManifold{ET}
    submanifold::MT
    manifold::NT
end

"""
    EmbeddedRetraction{R} <: AbstractRetractionMethod

Introduce a retraction on an [`EmbeddedManifold`](@ref) by using a retraction `R` in the
embedding and projecting the result back to the manifold.
"""
struct EmbeddedRetraction{R <: AbstractRetractionMethod} <: AbstractRetractionMethod
    retraction_method::R
end

"""
    EmbeddedInverseRetraction{R} <: AbstractInverseRetractionMethod

Introduce an inverse retraction on an [`EmbeddedManifold`](@ref) by using an inverse
retraction `R` in the embedding and projecting the result back to the tangent space
"""
struct EmbeddedInverseRetraction{IR <: AbstractInverseRetractionMethod} <: AbstractInverseRetractionMethod
    inverse_retraction_method::IR
end

@decorator_transparent_function function embed(M::AbstractEmbeddedManifold,p)
    q = allocate(p)
    return embed!(M, q, p)
end
@decorator_transparent_function function embed!(M::AbstractEmbeddedManifold, q, p)
    error("Embedding a point $(typeof(p)) on $(typeof(M)) not yet implemented.")
end

function inverse_retract!(M::MT, X, p, q, m::EmbeddedRetraction) where {MT <: EmbeddedManifold}
    x = allocate(q)
    y = allocate(p)
    embed!(M, x, p)
    embed!(M, y, q)
    retract!(M.embedding, X, x, y)
    project_tangent!(M, X, p, X)
    return q
end

@decorator_transparent_function function get_embedding(M::AbstractEmbeddedManifold)
    return M.manifold
end

function decorator_transparent_dispatch(
    ::typeof(project_point!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(project_tangent!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end


function retract!(M::MT, q, p, X, m::EmbeddedRetraction) where {MT <: AbstractEmbeddedManifold}
    x = allocate(q)
    Z = allocate(X)
    embed!(M, x, p)
    embed!(M, Z, p, X)
    retract!(get_embedding(M), q, x, Z)
    project_point!(M, q, q)
    return q
end

function show(io::IO, ::EmbeddedManifold{M,N}) where {N <: Manifold, M <: Manifold}
    print(io, "EmbeddedManifold($(M),$(N))")
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
    ::typeof(exp!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(exp!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
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
    ::typeof(log!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(log!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
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
    ::typeof(vector_transport_along!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(:intransparent)
end
function decorator_transparent_dispatch(
    ::typeof(vector_transport_along!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
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
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
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
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(:transparent)
end
