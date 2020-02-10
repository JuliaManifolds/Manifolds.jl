"""
    AbstractEmbeddedManifold

"""
struct AbstractEmbeddedManifold end

"""
    AbstrectIsometricallyEmbeddedManifold

"""
struct AbstractIsometricallyEmbeddedManifold end

struct EmbeddedManifold{MT <: Manifold, NT <: Manifold} <: AbstractDecoratorManifold
    manifold::MT
    embedding::NT
end

struct EmbeddedRetraction{R <: AbstractRetractionMethod} <: AbstractRetractionMethod
    retractionMethod::R
end
struct EmbeddedInverseRetraction{IR <: AbstractInverseRetractionMethod} <: AbstractInverseRetractionMethod
    inverseretracttionMethod::IR
end

function embedd(M::MT,p) where {MT <: AbstractEmbeddedManifold}
    q = allocate(p)
    return embedd!(M, q, p)
end
function embedd!(M, q, p)
    error("Embedding a point $(typeof(p)) on $(typeof(M)) not yet implemented.")
end

function inner(M::MT, p, X, Y) where {MT <: AbstractIsometricallyEmbeddedManifold}
    return inner(M.embedding,p,X,X)
end

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

getEmbedding(M::AbstractEmbeddedManifold) = M.embedding

function norm(M::MT, p, X, Y) where {MT <: AbstractIsometricallyEmbeddedManifold}
    return norm(M.embedding,p,X,X)
end

function project_point!(M::MT, q, p) where {
        MT <: AbstractIsometricallyEmbeddedManifold
    }
    return project_point!(M.manifold, q, p)
end

function project_tangent!(M::MT, X, p, A) where {
        MT <: AbstractIsometricallyEmbeddedManifold
    }
    return project_tangent!(M.manifold, X, p, A)
end

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
