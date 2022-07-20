@doc raw"""
    SpecialOrthogonal{n} <: GroupManifold{ℝ,Rotations{n},MultiplicationOperation}

Special orthogonal group ``\mathrm{SO}(n)`` represented by rotation matrices, see [`Rotations`](@ref).

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = GeneralUnitaryMultiplicationGroup{n,ℝ,DeterminantOneMatrices}

SpecialOrthogonal(n) = SpecialOrthogonal{n}(Rotations(n))

Base.inv(::SpecialOrthogonal, p) = transpose(p)
Base.inv(::SpecialOrthogonal, e::Identity{MultiplicationOperation}) = e

Base.show(io::IO, ::SpecialOrthogonal{n}) where {n} = print(io, "SpecialOrthogonal($(n))")
