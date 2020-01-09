@doc doc"""
    SpecialOrthogonal{n} <: GroupManifold{Rotations{n},MultiplicationOperation}

Special orthogonal group $\mathrm{SO}(n)$ represented by rotation matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = GroupManifold{Rotations{n},MultiplicationOperation}

function SpecialOrthogonal(n)
    return SpecialOrthogonal{n}(Rotations(n), MultiplicationOperation())
end

inv!(G::SpecialOrthogonal, y, x) = copyto!(y, inv(G, x))
inv!(G::AG, y, e::Identity{AG}) where {AG<:SpecialOrthogonal} = identity!(G, y, e)

inv(G::SpecialOrthogonal, x) = transpose(x)
inv(::AG, e::Identity{AG}) where {AG<:SpecialOrthogonal} = e

show(io::IO, ::SpecialOrthogonal{n}) where {n} = print(io, "SpecialOrthogonal($(n))")

translate_diff(::SpecialOrthogonal, x, y, v, ::LeftAction) = v
translate_diff(::SpecialOrthogonal, x, y, v, ::RightAction) = transpose(x) * v * x

function translate_diff!(
    G::SpecialOrthogonal,
    vout,
    x,
    y,
    v,
    conv::ActionDirection = LeftAction(),
)
    copyto!(vout, translate_diff(G, x, y, v, conv))
    return vout
end

inverse_translate_diff(::SpecialOrthogonal, x, y, v, ::LeftAction) = v
inverse_translate_diff(::SpecialOrthogonal, x, y, v, ::RightAction) = x * v * transpose(x)

function inverse_translate_diff!(
    G::SpecialOrthogonal,
    vout,
    x,
    y,
    v,
    conv::ActionDirection = LeftAction(),
)
    copyto!(vout, inverse_translate_diff(G, x, y, v, conv))
    return vout
end
