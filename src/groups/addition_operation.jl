
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

function adjoint_action(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    X,
    ::LeftAction,
)
    _lie_groups_depwarn_move(adjoint_action, :adjoint)
    return X
end
function adjoint_action!(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    Y,
    p,
    X,
    ::LeftAction,
)
    _lie_groups_depwarn_move(adjoint_action, :adjoint)
    return copyto!(G, Y, p, X)
end

"""
    adjoint_inv_diff(::AdditionGroupTrait, G::AbstractDecoratorManifold, p, X)

Compute the value of pullback of additive matrix inversion ``p ↦ -p`` at ``X``, i.e. ``-X``.
"""
function adjoint_inv_diff(::AdditionGroupTrait, G::AbstractDecoratorManifold, p, X)
    return -X
end
function adjoint_inv_diff!(::AdditionGroupTrait, G::AbstractDecoratorManifold, Y, p, X)
    Y .= X
    Y .*= -1
    return Y
end

identity_element(::AdditionGroupTrait, G::AbstractDecoratorManifold, p::Number) = zero(p)

function identity_element!(::AdditionGroupTrait, G::AbstractDecoratorManifold, p)
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

"""
    inv_diff(::AdditionGroupTrait, G::AbstractDecoratorManifold, p, X)

Compute the value of differential of additive matrix inversion ``p ↦ -p`` at ``X``, i.e. ``-X``.
"""
function inv_diff(::AdditionGroupTrait, G::AbstractDecoratorManifold, p, X)
    return -X
end
function inv_diff!(::AdditionGroupTrait, G::AbstractDecoratorManifold, Y, p, X)
    Y .= X
    Y .*= -1
    return Y
end

function is_identity(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    q::T;
    atol::Real=sqrt(prod(representation_size(G))) * eps(real(float(number_eltype(T)))),
    kwargs...,
) where {T}
    return isapprox(G, q, zero(q); atol=atol, kwargs...)
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

exp_lie(::AdditionGroupTrait, G::AbstractDecoratorManifold, X) = X

exp_lie!(::AdditionGroupTrait, G::AbstractDecoratorManifold, q, X) = copyto!(G, q, X)

function inverse_translate_diff(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    ::ActionDirectionAndSide,
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
    ::ActionDirectionAndSide,
)
    return copyto!(G, Y, p, X)
end

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

function translate_diff(
    ::AdditionGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    X,
    ::ActionDirectionAndSide,
)
    return X
end
