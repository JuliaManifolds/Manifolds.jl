"""
    SpecialOrthogonal{N} <: GroupManifold{Rotations{N},MultiplicationOperation}

# Constructor

    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{N} = GroupManifold{Rotations{N},MultiplicationOperation}

function SpecialOrthogonal(n)
    return SpecialOrthogonal{n}(Rotations(n), MultiplicationOperation())
end

@traitimpl IsMatrixGroup{SpecialOrthogonal}

function show(io::IO, ::SpecialOrthogonal{N}) where {N}
    print(io, "SpecialOrthogonal($(N))")
end

function inner(::SpecialOrthogonal, e::Identity, ve, we)
    return inner(base_manifold(G), I, ve, we)
end
