@doc raw"""
    CircleGroup <: GroupManifold{Circle{ℂ},MultiplicationOperation}

The circle group is the complex circle ([`Circle(ℂ)`](@ref)) equipped with
the group operation of complex multiplication ([`MultiplicationOperation`](@ref)).
"""
const CircleGroup = GroupManifold{Circle{ℂ},MultiplicationOperation}

CircleGroup() = GroupManifold(Circle{ℂ}(), MultiplicationOperation())

show(io::IO, ::CircleGroup) = print(io, "CircleGroup()")

has_invariant_metric(::CircleGroup, ::ActionDirection) = Val(true)

default_metric_dispatch(::MetricManifold{CircleGroup,EuclideanMetric}) = Val(true)

function compose(G::CircleGroup, p::AbstractVector, q::AbstractVector)
    return map(compose, repeated(G), p, q)
end

compose!(G::CircleGroup, x, p, q) = copyto!(x, compose(G, p, q))

identity(::CircleGroup, p::AbstractVector) = map(one, p)
identity(G::GT, e::Identity{GT}) where {GT<:CircleGroup} = e

identity!(::CircleGroup, q::AbstractVector, p) = copyto!(q, 1)
identity!(::GT, q::AbstractVector, ::Identity{GT}) where {GT<:CircleGroup} = copyto!(q, 1)

inv(G::CircleGroup, p::AbstractVector) = map(inv, repeated(G), p)
inv(G::GT, e::Identity{GT}) where {GT<:CircleGroup} = e

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

translate_diff(::GT, p, q, X, ::ActionDirection) where {GT<:CircleGroup} = map(*, p, X)
function translate_diff(
    ::GT,
    ::Identity{GT},
    q,
    X,
    ::ActionDirection,
) where {GT<:CircleGroup}
    return X
end

function translate_diff!(G::CircleGroup, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end

function group_exp(G::CircleGroup, X)
    return map(X) do imθ
        θ = imag(imθ)
        sinθ, cosθ = sincos(θ)
        return Complex(cosθ, sinθ)
    end
end

group_exp!(G::CircleGroup, q, X) = (q .= group_exp(G, X))

function group_log(G::CircleGroup, q)
    return map(q) do z
        cosθ, sinθ = reim(z)
        θ = atan(sinθ, cosθ)
        return θ * im
    end
end

group_log!(G::CircleGroup, X, q) = (X .= group_log(G, q))
