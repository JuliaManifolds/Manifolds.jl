"""
    AbstractEmbeddedManifold

"""
abstract type AbstractEmbeddedManifold <: AbstractDecoratorManifold end

abstract type AbstractEmbeddingType end
abstract type AbstractIsometricEmbedding <: AbstractEmbeddingType end

"""
    IsometricEmbedding <: AbstractEmbeddingType

"""
struct DefaultIsometricEmbedding <: AbstractIsometricEmbedding end

struct EmbeddedManifold{MT <: Manifold, NT <: Manifold, ET <: AbstractEmbeddingType} <: AbstractEmbeddedManifold
    submanifold::MT
    manifold::NT
    type::ET
end

"""
    EmbeddedRetraction{R} <: AbstractRetractionMethod

Introduce a retraction on an [`EmbeddedManifold`](@ref) by using a retraction `R` in the
embedding and projecting the result back to the manifold.
"""
struct EmbeddedRetraction{R <: AbstractRetractionMethod} <: AbstractRetractionMethod
    retractionMethod::R
end

"""
    EmbeddedInverseRetraction{R} <: AbstractInverseRetractionMethod

Introduce an inverse retraction on an [`EmbeddedManifold`](@ref) by using an inverse
retraction `R` in the embedding and projecting the result back to the tangent space
"""
struct EmbeddedInverseRetraction{IR <: AbstractInverseRetractionMethod} <: AbstractInverseRetractionMethod
    inverseretracttionMethod::IR
end

val_is_decorator_transparent(::EM, ::typeof(project_point!)) where {EM <: AbstractEmbeddedManifold} = Val(false)
val_is_decorator_transparent(::EM, ::typeof(project_tangent!)) where {EM <: AbstractEmbeddedManifold} = Val(false)


function embed(M::MT,p) where {MT <: AbstractEmbeddedManifold}
    q = allocate(p)
    return embed!(M, q, p)
end
function embed!(M, q, p)
    error("Embedding a point $(typeof(p)) on $(typeof(M)) not yet implemented.")
end

val_is_decorator_transparent(::EM, ::typeof(inner)) where {EM <: AbstractEmbeddedManifold} = Val(false)
val_is_decorator_transparent(::EmbeddedManifold{M,N,T}, ::typeof(inner)) where {M,N, T<:AbstractIsometricEmbedding} = Val(true)

function inverse_retract!(M::MT, X, p, q, m::EmbeddedRetraction) where {MT <: EmbeddedManifold}
    x = allocate(q)
    y = allocate(p)
    embed!(M, x, p)
    embed!(M, y, q)
    retract!(M.embedding, X, x, y)
    project_tangent!(M, X, p, X)
    return q
end

val_is_decorator_transparent(::EM, ::typeof(manifold_dimension)) where {EM <: AbstractEmbeddedManifold} = Val(false)

val_is_default_decorator(M::EM) where {EM <: EmbeddedManifold} = is_default_embedding(M)

get_embedding(M::AbstractEmbeddedManifold) = M.manifold

val_is_decorator_transparent(::EM, ::typeof(norm)) where {EM <: AbstractEmbeddedManifold} = Val(false)
val_is_decorator_transparent(::EmbeddedManifold{M,N,T}, ::typeof(norm)) where {M,N,T<:AbstractIsometricEmbedding} = Val(true)

function retract!(M::MT, q, p, X, m::EmbeddedRetraction) where {MT <: AbstractEmbeddedManifold}
    x = allocate(q)
    Z = allocate(X)
    embedd!(M, x, p)
    embedd!(M, Z, p, X)
    retract!(get_embedding(M), q, x, Z)
    project_point!(M, q)
    return q
end

function show(io::IO, ::EmbeddedManifold{M,N}) where {N <: Manifold, M <: Manifold}
    print(io, "EmbeddedManifold($(M),$(N))")
end
