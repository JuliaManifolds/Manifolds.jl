
"""
    MultiplicationOperation <: AbstractGroupOperation

Group operation that consists of multiplication.
"""
struct MultiplicationOperation <: AbstractGroupOperation end

const MultiplicationGroupTrait = TraitList{<:IsGroupManifold{<:MultiplicationOperation}}

Base.:*(e::Identity{MultiplicationOperation}) = e
Base.:*(::Identity{MultiplicationOperation}, p) = p
Base.:*(p, ::Identity{MultiplicationOperation}) = p
Base.:*(e::Identity{MultiplicationOperation}, ::Identity{MultiplicationOperation}) = e
Base.:*(::Identity{MultiplicationOperation}, e::Identity{AdditionOperation}) = e
Base.:*(e::Identity{AdditionOperation}, ::Identity{MultiplicationOperation}) = e

Base.:/(p, ::Identity{MultiplicationOperation}) = p
Base.:/(::Identity{MultiplicationOperation}, p) = inv(p)
Base.:/(e::Identity{MultiplicationOperation}, ::Identity{MultiplicationOperation}) = e

Base.:\(p, ::Identity{MultiplicationOperation}) = inv(p)
Base.:\(::Identity{MultiplicationOperation}, p) = p
Base.:\(e::Identity{MultiplicationOperation}, ::Identity{MultiplicationOperation}) = e

LinearAlgebra.det(::Identity{MultiplicationOperation}) = true
LinearAlgebra.adjoint(e::Identity{MultiplicationOperation}) = e

function identity_element!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p::AbstractMatrix,
)
    return copyto!(p, I)
end

function identity_element!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p::AbstractArray,
)
    if length(p) == 1
        fill!(p, one(eltype(p)))
    else
        throw(DimensionMismatch("Array $p cannot be set to identity element of group $G"))
    end
    return p
end

function is_identity(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q::Number;
    kwargs...,
)
    return isapprox(G, q, one(q); kwargs...)
end
function is_identity(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q::AbstractArray{<:Any,0};
    kwargs...,
)
    return isapprox(G, q[], one(q[]); kwargs...)
end
function is_identity(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q::AbstractMatrix;
    kwargs...,
)
    return isapprox(G, q, I; kwargs...)
end

LinearAlgebra.mul!(q, ::Identity{MultiplicationOperation}, p) = copyto!(q, p)
LinearAlgebra.mul!(q, p, ::Identity{MultiplicationOperation}) = copyto!(q, p)
function LinearAlgebra.mul!(
    q::AbstractMatrix,
    ::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
)
    return copyto!(q, I)
end
function LinearAlgebra.mul!(
    q,
    ::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
)
    return copyto!(q, one(q))
end
function LinearAlgebra.mul!(
    q::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
    ::Identity{MultiplicationOperation},
)
    return q
end
Base.one(e::Identity{MultiplicationOperation}) = e

Base.inv(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, p) = inv(p)
function Base.inv(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    e::Identity{MultiplicationOperation},
)
    return e
end

inv!(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, q, p) = copyto!(q, inv(G, p))
function inv!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q,
    ::Identity{MultiplicationOperation},
)
    return identity_element!(G, q)
end
function inv!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q::Identity{MultiplicationOperation},
    e::Identity{MultiplicationOperation},
)
    return q
end

compose(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, p, q) = p * q

function compose!(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, x, p, q)
    return mul!_safe(x, p, q)
end

function inverse_translate(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    ::LeftAction,
)
    return p \ q
end
function inverse_translate(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    ::RightAction,
)
    return q / p
end

function inverse_translate!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    x,
    p,
    q,
    conv::ActionDirection,
)
    return copyto!(x, inverse_translate(G, p, q, conv))
end

function exp_lie!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    q,
    X::Union{Number,AbstractMatrix},
)
    copyto!(q, exp(X))
    return q
end

function log_lie!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    X::AbstractMatrix,
    q::AbstractMatrix,
)
    return log_safe!(X, q)
end
function log_lie!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    X,
    ::Identity{MultiplicationOperation},
)
    return zero_vector!(G, X, identity_element(G))
end

function lie_bracket(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, X, Y)
    return mul!(X * Y, Y, X, -1, true)
end

function lie_bracket!(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, Z, X, Y)
    mul!(Z, X, Y)
    mul!(Z, Y, X, -1, true)
    return Z
end
