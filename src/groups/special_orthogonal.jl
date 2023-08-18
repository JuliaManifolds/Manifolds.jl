@doc raw"""
    SpecialOrthogonal{n} <: GroupManifold{ℝ,Rotations{n},MultiplicationOperation}

Special orthogonal group ``\mathrm{SO}(n)`` represented by rotation matrices, see [`Rotations`](@ref).

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{n} = GeneralUnitaryMultiplicationGroup{n,ℝ,DeterminantOneMatrices}

function SpecialOrthogonal(n; parameter::Symbol=:field)
    return GeneralUnitaryMultiplicationGroup(Rotations(n; parameter=parameter))
end

Base.inv(::SpecialOrthogonal, p) = transpose(p)
Base.inv(::SpecialOrthogonal, e::Identity{MultiplicationOperation}) = e

function Base.show(io::IO, ::SpecialOrthogonal{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "SpecialOrthogonal($(n); parameter=:type)")
end
function Base.show(io::IO, M::SpecialOrthogonal{Tuple{Int}})
    n = get_n(M)
    return print(io, "SpecialOrthogonal($(n))")
end
