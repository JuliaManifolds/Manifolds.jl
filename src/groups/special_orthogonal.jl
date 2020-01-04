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
    copyto!(y, transpose(x))
    return y
end

function inv!(G::AG, y, x::Identity{AG}) where AG<:SpecialOrthogonal
    error("inv! not implemented on $(typeof(G)) for elements $(typeof(y)) and $(typeof(x))")
end

function inv(G::SpecialOrthogonal, x)
    return transpose(x)
end

inv(::AG, e::Identity{AG}) where AG<:SpecialOrthogonal = e

function show(io::IO, ::SpecialOrthogonal{N}) where {N}
    print(io, "SpecialOrthogonal($(N))")
end
