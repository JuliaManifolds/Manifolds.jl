
"""
    AdditionOperation <: AbstractGroupOperation

Group operation that consists of simple addition.
"""
struct AdditionOperation <: AbstractGroupOperation end

Base.:+(e::Identity{AdditionOperation}) = e
Base.:+(e::Identity{AdditionOperation}, ::Identity{AdditionOperation}) = e
Base.:+(::Identity{AdditionOperation}, p) = p
Base.:+(p, ::Identity{AdditionOperation}) = p

Base.:-(e::Identity{AdditionOperation}) = e
Base.:-(e::Identity{AdditionOperation}, ::Identity{AdditionOperation}) = e
Base.:-(::Identity{AdditionOperation}, p) = -p
Base.:-(p, ::Identity{AdditionOperation}) = p

Base.:*(e::Identity{AdditionOperation}, p) = e
Base.:*(p, e::Identity{AdditionOperation}) = e
Base.:*(e::Identity{AdditionOperation}, ::Identity{AdditionOperation}) = e

const AdditionGroupTrait = TraitList{<:IsGroupManifold{AdditionOperation}}

adjoint_action(::AdditionGroupTrait, G::AbstractDecoratorManifold, p, X) = X

function adjoint_action!(::AdditionGroupTrait, G::AbstractDecoratorManifold, Y, p, X)
    return copyto!(G, Y, p, X)
end

identity_element(::AdditionGroupTrait, G::AbstractDecoratorManifold, p::Number) = zero(p)

function identity_element!(::AdditionGroupTrait, G::AbstractDecoratorManifold, p) where {𝔽}
    return fill!(p, zero(eltype(p)))
end

Base.inv(::AdditionGroupTrait, G::AbstractDecoratorManifold, p) = -p
Base.inv(::AdditionGroupTrait, G::AbstractDecoratorManifold, e::Identity) = e

inv!(::AdditionGroupTrait, G::AbstractDecoratorManifold, q, p) = copyto!(G, q, -p)
function inv!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    q,
    ::Identity{AdditionOperation},
)
    return identity_element!(G, q)
end
function inv!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    q::Identity{AdditionOperation},
    e::Identity{AdditionOperation},
)
    return q
end

function is_identity(::AdditionGroupTrait, G::AbstractDecoratorManifold, q; kwargs...)
    return isapprox(G, q, zero(q); kwargs...)
end
# resolve ambiguities
function is_identity(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    ::Identity{AdditionOperation};
    kwargs...,
)
    return true
end
function is_identity(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    ::Identity;
    kwargs...,
)
    return false
end

compose(::AdditionGroupTrait, G::AbstractDecoratorManifold, p, q) = p + q

function compose!(::AdditionGroupTrait, G::AbstractDecoratorManifold, x, p, q)
    x .= p .+ q
    return x
end

function translate_diff(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    ::ActionDirection,
)
    return X
end

function translate_diff!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    ::ActionDirection,
)
    return copyto!(G, Y, p, X)
end

function inverse_translate_diff(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    ::ActionDirection,
)
    return X
end

function inverse_translate_diff!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    Y,
    p,
    q,
    X,
    ::ActionDirection,
)
    return copyto!(G, Y, p, X)
end

exp_lie(::AdditionGroupTrait, G::AbstractDecoratorManifold, X) = X

exp_lie!(::AdditionGroupTrait, G::AbstractDecoratorManifold, q, X) = copyto!(G, q, X)

log_lie(::AdditionGroupTrait, G::AbstractDecoratorManifold, q) = q
function log_lie(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    ::Identity{AdditionOperation},
)
    return zero_vector(G, identity_element(G))
end
log_lie!(::AdditionGroupTrait, G::AbstractDecoratorManifold, X, q) = copyto!(G, X, q)
function log_lie!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    X,
    ::Identity{AdditionOperation},
)
    return zero_vector!(G, X, identity_element(G))
end

lie_bracket(::AdditionGroupTrait, G::AbstractDecoratorManifold, X, Y) = zero(X)

lie_bracket!(::AdditionGroupTrait, G::AbstractDecoratorManifold, Z, X, Y) = fill!(Z, 0)
