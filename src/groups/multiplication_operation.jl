
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

Base.inv(e::Identity{MultiplicationOperation}) = e

LinearAlgebra.det(::Identity{MultiplicationOperation}) = true
LinearAlgebra.adjoint(e::Identity{MultiplicationOperation}) = e

@doc raw"""
    adjoint_inv_diff(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, p, X)

Compute the value of differential of matrix inversion ``p ↦ p^{-1}`` at ``X``.
When tangent vectors are represented in Lie algebra in a left-invariant way, the formula
reads ``-p^\mathrm{T}X(p^{-1})^\mathrm{T}``. For matrix groups with ambient space tangent
vectors, the formula would read ``-(p^{-1})^\mathrm{T}X(p^{-1})^\mathrm{T}``. See the
section about matrix inverse in [Giles:2008](@cite).
"""
function adjoint_inv_diff(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, p, X)
    return -(p' * X * inv(G, p)')
end
function adjoint_inv_diff(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p::AbstractArray{<:Number,0},
    X::AbstractArray{<:Number,0},
)
    p_inv = inv(p[])
    return -(p[] * X * p_inv)
end

function adjoint_inv_diff!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    Y,
    p,
    X,
)
    p_inv = inv(p)
    Z = X * p_inv'
    mul!(Y, p', Z)
    Y .*= -1
    return Y
end

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

"""
    inv_diff(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, p, X)

Compute the value of differential of matrix inversion ``p ↦ p^{-1}`` at ``X``.
When tangent vectors are represented in Lie algebra in a left-invariant way, the formula
reads ``-pXp^{-1}``. For matrix groups with ambient space tangent vectors, the formula would
read ``-p^{-1}Xp^{-1}``. See the section about matrix inverse in [Giles:2008](@cite).
"""
function inv_diff(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, p, X)
    return -(p * X * inv(G, p))
end
function inv_diff(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p::AbstractArray{<:Number,0},
    X::AbstractArray{<:Number,0},
)
    p_inv = inv(p[])
    return -(p[] * X * p_inv)
end

function inv_diff!(::MultiplicationGroupTrait, G::AbstractDecoratorManifold, Y, p, X)
    p_inv = inv(p)
    Z = X * p_inv
    mul!(Y, p, Z)
    Y .*= -1
    return Y
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
    ::LeftForwardAction,
)
    return p \ q
end
function inverse_translate(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    p,
    q,
    ::RightBackwardAction,
)
    return q / p
end

function inverse_translate!(
    ::MultiplicationGroupTrait,
    G::AbstractDecoratorManifold,
    x,
    p,
    q,
    conv::ActionDirectionAndSide,
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
