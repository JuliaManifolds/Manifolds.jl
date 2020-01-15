@doc doc"""
    CircleGroup <: GroupManifold{Circle{ℂ},MultiplicationOperation}

The circle group is the complex circle ([`Circle(ℂ)`](@ref)) equipped with
the group operation of complex multiplication ([`MultiplicationOperation`](@ref)).
"""
const CircleGroup = GroupManifold{Circle{ℂ},MultiplicationOperation}

CircleGroup() = GroupManifold(Circle{ℂ}(), MultiplicationOperation())

show(io::IO, ::CircleGroup) = print(io, "CircleGroup()")

function compose(G::CircleGroup, x::AbstractVector, y::AbstractVector)
    return map(compose, repeated(G), x, y)
end

compose!(G::CircleGroup, z, x, y) = copyto!(z, compose(G, x, y))

identity(::CircleGroup, x::AbstractVector) = map(one, x)
identity(G::GT, e::Identity{GT}) where {GT<:CircleGroup} = e

identity!(::CircleGroup, y::AbstractVector, x) = copyto!(y, 1)
identity!(::GT, y::AbstractVector, ::Identity{GT}) where {GT<:CircleGroup} = copyto!(y, 1)

inv(G::CircleGroup, x::AbstractVector) = map(inv, repeated(G), x)
inv(G::GT, e::Identity{GT}) where {GT<:CircleGroup} = e

function inverse_translate(
    ::CircleGroup,
    x::AbstractVector,
    y::AbstractVector,
    ::LeftAction,
)
    return map(/, y, x)
end
function inverse_translate(
    ::CircleGroup,
    x::AbstractVector,
    y::AbstractVector,
    ::RightAction,
)
    return map(/, y, x)
end

translate_diff(::GT, x, y, v, ::ActionDirection) where {GT<:CircleGroup} = map(*, x, v)
function translate_diff(
    ::GT,
    ::Identity{GT},
    y,
    v,
    ::ActionDirection,
) where {GT<:CircleGroup}
    return v
end

function translate_diff!(G::CircleGroup, vout, x, y, v, conv::ActionDirection)
    return copyto!(vout, translate_diff(G, x, y, v, conv))
end
