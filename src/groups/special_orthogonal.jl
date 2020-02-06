@doc raw"""
    SpecialOrthogonal{n} <: GroupManifold{Rotations{n},MultiplicationOperation}

Special orthogonal group $\mathrm{SO}(n)$ represented by rotation matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = GroupManifold{Rotations{n},MultiplicationOperation}

SpecialOrthogonal(n) = SpecialOrthogonal{n}(Rotations(n), MultiplicationOperation())

show(io::IO, ::SpecialOrthogonal{n}) where {n} = print(io, "SpecialOrthogonal($(n))")

inv(::SpecialOrthogonal, p) = transpose(p)

inverse_translate(G::SpecialOrthogonal, p, q, conv::LeftAction) = inv(G, p) * q
inverse_translate(G::SpecialOrthogonal, p, q, conv::RightAction) = q * inv(G, p)

translate_diff(::SpecialOrthogonal, p, q, X, ::LeftAction) = X
translate_diff(G::SpecialOrthogonal, p, q, X, ::RightAction) = inv(G, p) * X * p

function translate_diff!(G::SpecialOrthogonal, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end

function inverse_translate_diff(G::SpecialOrthogonal, p, q, X, conv::ActionDirection)
    return translate_diff(G, inv(G, p), q, X, conv)
end

function inverse_translate_diff!(G::SpecialOrthogonal, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end
