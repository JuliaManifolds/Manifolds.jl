"""
    ExplicitEmbeddedBackend{TF<:NamedTuple} <: AbstractDiffBackend

A backend to use with the [`RiemannianProjectionBackend`](@ref) or the [`TangentDiffBackend`](@ref),
when you have explicit formulae for the gradient in the embedding available.


# Constructor
    ExplicitEmbeddedBackend(M::AbstractManifold; kwargs)

Construct an [`ExplicitEmbeddedBackend`](@ref) in the embedding `M`,
where currently the following keywords may be used

* `gradient` for a(n allocating) gradient function `gradient(M, p)` defined in the embedding
* `gradient!` for a mutating gradient function `gradient!(M, X, p)`.

Note that the gradient functions are defined on the embedding manifold `M` passed to the Backend as well
"""
struct ExplicitEmbeddedBackend{TM<:AbstractManifold,TF<:NamedTuple} <: AbstractDiffBackend
    manifold::TM
    functions::TF
end
function ExplicitEmbeddedBackend(M::TM; kwargs...) where {TM<:AbstractManifold}
    return ExplicitEmbeddedBackend{TM,typeof(values(kwargs))}(M, values(kwargs))
end

function _gradient(f, p, e::ExplicitEmbeddedBackend)
    g = get(e.functions, :gradient, Missing())
    g === missing &&
        throw(MissingException("The provided Embedded backend does not provide a gradient"))
    return g(e.manifold, p)
end

function _gradient!(f, X, p, e::ExplicitEmbeddedBackend)
    g! = get(e.functions, :gradient!, Missing())
    g! === missing && throw(
        MissingException(
            "The provided Embedded backend does not provide a mutating gradient",
        ),
    )
    g!(e.manifold, X, p)
    return X
end
