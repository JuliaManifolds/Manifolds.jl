abstract type AbstractEmbeddingType end
abstract type AbstractIsometricEmbeddingType <: AbstractEmbeddingType end

"""
    DefaultIsometricEmbedding <: AbstractEmbeddingType

"""
struct DefaultIsometricEmbedding <: AbstractIsometricEmbeddingType end

"""
    AbstractEmbeddedManifold

"""
abstract type AbstractEmbeddedManifold{T<:AbstractEmbeddingType} <: AbstractDecoratorManifold end

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

function val_is_decorator_transparent(
    ::typeof(project_point!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end
function val_is_decorator_transparent(
    ::typeof(project_tangent!),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end

function embed(M::MT,p) where {MT <: AbstractEmbeddedManifold}
    q = allocate(p)
    return embed!(M, q, p)
end
function embed!(M, q, p)
    error("Embedding a point $(typeof(p)) on $(typeof(M)) not yet implemented.")
end

function get_basis(M::AbstractEmbeddedManifold, p, B::ArbitraryOrthonormalBasis)
    return invoke(get_basis, Tuple{Manifold,Any,ArbitraryOrthonormalBasis}, M, p, B)
end
function get_basis(M::AbstractEmbeddedManifold, p, B::AbstractPrecomputedOrthonormalBasis)
    return invoke(get_basis, Tuple{Manifold,Any,AbstractPrecomputedOrthonormalBasis}, M, p, B)
end
function get_basis(M::AbstractEmbeddedManifold, p, B::ProjectedOrthonormalBasis{:svd,ℝ})
    return invoke(get_basis, Tuple{Manifold,Any,ProjectedOrthonormalBasis{:svd,ℝ}}, M, p, B)
end

function get_coordinates(M::AbstractEmbeddedManifold, p, X, B::AbstractPrecomputedOrthonormalBasis{ℝ})
    return invoke(get_coordinates, Tuple{Manifold,Any,Any,AbstractPrecomputedOrthonormalBasis{ℝ}}, M, p, X, B)
end
function get_coordinates(M::AbstractEmbeddedManifold, p, X, B::AbstractPrecomputedOrthonormalBasis)
    return invoke(get_coordinates, Tuple{Manifold,Any,Any,AbstractPrecomputedOrthonormalBasis}, M, p, X, B)
end

function get_vector(M::AbstractEmbeddedManifold, p, X, B::AbstractPrecomputedOrthonormalBasis)
    return invoke(get_vector, Tuple{Manifold,Any,Any,AbstractPrecomputedOrthonormalBasis}, M, p, X, B)
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

get_embedding(M::AbstractEmbeddedManifold) = M.manifold

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

val_is_default_decorator(M::EM) where {EM <: EmbeddedManifold} = is_default_embedding(M)
function val_is_decorator_transparent(
    ::typeof(check_manifold_point),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end
function val_is_decorator_transparent(
    ::typeof(check_tangent_vector),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end
function val_is_decorator_transparent(
    ::typeof(inner),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end
function val_is_decorator_transparent(
    ::typeof(norm),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end
function val_is_decorator_transparent(
    ::typeof(manifold_dimension),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end
function val_is_decorator_transparent(
    ::typeof(exp!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(true)
end
function val_is_decorator_transparent(
    ::typeof(get_basis),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end
function val_is_decorator_transparent(
    ::typeof(get_coordinates),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end
function val_is_decorator_transparent(
    ::typeof(get_vector),
    ::AbstractEmbeddedManifold,
    args...,
)
    return Val(false)
end
function val_is_decorator_transparent(
    ::typeof(inner),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(true)
end
function val_is_decorator_transparent(
    ::typeof(log!),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(true)
end
function val_is_decorator_transparent(
    ::typeof(norm),
    ::AbstractEmbeddedManifold{<:AbstractIsometricEmbeddingType},
    args...,
)
    return Val(true)
end
