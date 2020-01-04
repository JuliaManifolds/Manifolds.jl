@doc doc"""
    SpecialOrthogonal{N} <: GroupManifold{Rotations{N},MultiplicationOperation}

Special orthogonal group `\mathrm{SO}(N)` represented by rotation matrices.

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{N} = GroupManifold{Rotations{N},MultiplicationOperation}

function SpecialOrthogonal(n)
    return SpecialOrthogonal{n}(Rotations(n), MultiplicationOperation())
end

# optimized inverseion for the special orthogonal group
function inv!(G::SpecialOrthogonal, y, x)
    copyto!(y, inv(G, x))
    return y
end
function inv!(G::AG, y, e::Identity{AG}) where {AG<:SpecialOrthogonal}
    identity!(G, y, e)
    return y
end

inv(G::SpecialOrthogonal, x) = transpose(x)
inv(::AG, e::Identity{AG}) where {AG<:SpecialOrthogonal} = e

function show(io::IO, ::SpecialOrthogonal{N}) where {N}
    print(io, "SpecialOrthogonal($(N))")
end

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
