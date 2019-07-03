"""
    SpecialOrthogonal{N} <: GroupManifold{Rotations{N},MultiplicationOperation}

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

@traitimpl IsMatrixGroup{SpecialOrthogonal}
