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

function show(io::IO, ::SpecialOrthogonal{N}) where {N}
    print(io, "SpecialOrthogonal($(N))")
end
