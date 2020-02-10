"""
    AbstractEmbeddedManifold

"""
abstract type AbstractEmbeddedManifold <: AbstractDecoratorManifold end

abstract type AbstractEmbeddingType end

"""
    IsometricEmbedding <: AbstractEmbeddingType

"""
struct IsometricEmbedding <: AbstractEmbeddingType end

struct EmbeddedManifold{MT <: Manifold, NT <: Manifold, ET <: AbstractEmbeddingType} <: AbstractEmbeddedManifold
    manifold::MT
    embedding::NT
    type::ET
end

struct EmbeddedRetraction{R <: AbstractRetractionMethod} <: AbstractRetractionMethod
    retractionMethod::R
end
struct EmbeddedInverseRetraction{IR <: AbstractInverseRetractionMethod} <: AbstractInverseRetractionMethod
    inverseretracttionMethod::IR
end

function embed(M::MT,p) where {MT <: AbstractEmbeddedManifold}
    q = allocate(p)
    return embed!(M, q, p)
end
function embed!(M, q, p)
    error("Embedding a point $(typeof(p)) on $(typeof(M)) not yet implemented.")
end

is_decorator_transparent(::EM, ::typeof(inner)) where {EM <: AbstractEmbeddedManifold} = false
is_decorator_transparent(::EmbeddedManifold{M,N,IsometricEmbedding}, ::typeof(inner)) where {M,N} = true

function inverse_retract!(M::MT, X, p, q, m::EmbeddedRetraction) where {MT <: EmbeddedManifold}
    x = allocate(q)
    y = allocate(p)
    embedd!(M, x, p)
    embedd!(M, y, q)
    retract!(M.embedding, X, x, y)
    project_tangent!(M, X, p, X)
    return q
end

is_default_decorator(M::EM) where {EM <: EmbeddedManifold} = is_default_embedding(M)

get_embedding(M::AbstractEmbeddedManifold) = M.embedding


is_decorator_transparent(::EM, ::typeof(norm)) where {EM <: AbstractEmbeddedManifold} = false
is_decorator_transparent(::EmbeddedManifold{M,N,IsometricEmbedding}, ::typeof(norm)) where {M,N} = true

is_decorator_transparent(::EM, ::typeof(project_point!)) where {EM <: AbstractEmbeddedManifold} = false
is_decorator_transparent(::EmbeddedManifold{M,N,IsometricEmbedding}, ::typeof(project_point!)) where {M,N} = true

is_decorator_transparent(::EM, ::typeof(project_tangent!)) where {EM <: AbstractEmbeddedManifold} = false
is_decorator_transparent(::EmbeddedManifold{M,N,IsometricEmbedding}, ::typeof(project_tangent!)) where {M,N} = true

function retract!(M::MT, q, p, X, m::EmbeddedRetraction) where {MT <: AbstractEmbeddedManifold}
    x = allocate(q)
    Z = allocate(X)
    embedd!(M, x, p)
    embedd!(M, Z, p, X)
    retract!(M.embedding, q, x, Z)
    project_point!(M, q)
    return q
end

function show(io::IO, ::EmbeddedManifold{M,N}) where {N <: Manifold, M <: Manifold}
    print(io, "EmbeddedManifold($(M),$(N))")
end
