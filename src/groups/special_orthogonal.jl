@doc raw"""
    SpecialOrthogonal{n} <: GroupManifold{Rotations{n},MultiplicationOperation}

Special orthogonal group $\mathrm{SO}(n)$ represented by rotation matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = GroupManifold{Rotations{n},MultiplicationOperation}

SpecialOrthogonal(n) = SpecialOrthogonal{n}(Rotations(n), MultiplicationOperation())

show(io::IO, ::SpecialOrthogonal{n}) where {n} = print(io, "SpecialOrthogonal($(n))")

inv(::SpecialOrthogonal, x) = transpose(x)

inverse_translate(G::SpecialOrthogonal, x, y, conv::LeftAction) = inv(G, x) * y
inverse_translate(G::SpecialOrthogonal, x, y, conv::RightAction) = y * inv(G, x)

translate_diff(::SpecialOrthogonal, x, y, v, ::LeftAction) = v
translate_diff(G::SpecialOrthogonal, x, y, v, ::RightAction) = inv(G, x) * v * x

function translate_diff!(G::SpecialOrthogonal, vout, x, y, v, conv::ActionDirection)
    return copyto!(vout, translate_diff(G, x, y, v, conv))
end

function inverse_translate_diff(G::SpecialOrthogonal, x, y, v, conv::ActionDirection)
    return translate_diff(G, inv(G, x), y, v, conv)
end

function inverse_translate_diff!(G::SpecialOrthogonal, vout, x, y, v, conv::ActionDirection)
    return copyto!(vout, inverse_translate_diff(G, x, y, v, conv))
end
