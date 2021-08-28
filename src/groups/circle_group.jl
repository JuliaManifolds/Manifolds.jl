@doc raw"""
    CircleGroup <: GroupManifold{Circle{ℂ},MultiplicationOperation}

The circle group is the complex circle ([`Circle(ℂ)`](@ref)) equipped with
the group operation of complex multiplication ([`MultiplicationOperation`](@ref)).
"""
const CircleGroup = GroupManifold{ℂ,Circle{ℂ},MultiplicationOperation}

CircleGroup() = GroupManifold(Circle{ℂ}(), MultiplicationOperation())

Base.show(io::IO, ::CircleGroup) = print(io, "CircleGroup()")

invariant_metric_dispatch(::CircleGroup, ::ActionDirection) = Val(true)

default_metric_dispatch(::MetricManifold{ℂ,CircleGroup,EuclideanMetric}) = Val(true)

adjoint_action(::CircleGroup, p, X) = X

adjoint_action!(::CircleGroup, Y, p, X) = copyto!(Y, X)

function _compose(G::CircleGroup, p::AbstractVector, q::AbstractVector)
    return map(compose, repeated(G), p, q)
end

_compose!(G::CircleGroup, x, p, q) = copyto!(x, compose(G, p, q))

identity_element(G::CircleGroup) = 1.0
identity_element(::CircleGroup, p::Number) = one(p)
identity_element(::CircleGroup, p::AbstractArray) = map(i -> one(eltype(p)), p)

Base.inv(G::CircleGroup, p::AbstractVector) = map(inv, repeated(G), p)

function inverse_translate(
    ::CircleGroup,
    p::AbstractVector,
    q::AbstractVector,
    ::LeftAction,
)
    return map(/, q, p)
end
function inverse_translate(
    ::CircleGroup,
    p::AbstractVector,
    q::AbstractVector,
    ::RightAction,
)
    return map(/, q, p)
end

lie_bracket(::CircleGroup, X, Y) = zero(X)

lie_bracket!(::CircleGroup, Z, X, Y) = fill!(Z, 0)

translate_diff(::GT, p, q, X, ::ActionDirection) where {GT<:CircleGroup} = map(*, p, X)
function translate_diff(
    ::CircleGroup,
    ::Identity{MultiplicationOperation},
    q,
    X,
    ::ActionDirection,
)
    return X
end

function translate_diff!(G::CircleGroup, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end

function exp_lie(::CircleGroup, X)
    return map(X) do imθ
        θ = imag(imθ)
        sinθ, cosθ = sincos(θ)
        return Complex(cosθ, sinθ)
    end
end

exp_lie!(G::CircleGroup, q, X) = (q .= exp_lie(G, X))

function log_lie(::CircleGroup, q)
    return map(q) do z
        cosθ, sinθ = reim(z)
        θ = atan(sinθ, cosθ)
        return θ * im
    end
end
log_lie(::CircleGroup, e::Identity{MultiplicationOperation}) = 0.0 * im

_log_lie!(G::CircleGroup, X, q) = (X .= log_lie(G, q))

function number_of_coordinates(G::CircleGroup, B::AbstractBasis)
    return number_of_coordinates(base_manifold(G), B)
end
