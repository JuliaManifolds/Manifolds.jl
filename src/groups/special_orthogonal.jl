@doc raw"""
    SpecialOrthogonal{T} = GeneralUnitaryMultiplicationGroup{T,ℝ,DeterminantOneMatrixType}

Special orthogonal group ``\mathrm{SO}(n)`` represented by rotation matrices, see [`Rotations`](@ref).

# Constructor
    SpecialOrthogonal(n)
"""
const SpecialOrthogonal{T} = GeneralUnitaryMultiplicationGroup{T,ℝ,DeterminantOneMatrixType}

function SpecialOrthogonal(n; parameter::Symbol=:type)
    _lie_groups_depwarn_move(SpecialOrthogonal, :SpecialOrthogonalGroup)
    return GeneralUnitaryMultiplicationGroup(Rotations(n; parameter=parameter))
end

"""
    adjoint_matrix(::SpecialOrthogonal{TypeParameter{Tuple{2}}}, p)

Compte the adjoint matrix for [`SpecialOrthogonal`](@ref)`(2)` at point `p`, which is equal
to `1`. See [SolaDerayAtchuthan:2021](@cite), Appendix A.
"""
adjoint_matrix(::SpecialOrthogonal{TypeParameter{Tuple{2}}}, p) = @SMatrix [1]
"""
    adjoint_matrix(::SpecialOrthogonal{TypeParameter{Tuple{3}}}, p)

Compte the adjoint matrix for [`SpecialOrthogonal`](@ref)`(3)` at point `p`, which is equal
to `p`. See [Chirikjian:2012](@cite), Section 10.6.6.
"""
adjoint_matrix(::SpecialOrthogonal{TypeParameter{Tuple{3}}}, p) = p

adjoint_matrix!(::SpecialOrthogonal{TypeParameter{Tuple{2}}}, J, p) = fill!(J, 1)
adjoint_matrix!(::SpecialOrthogonal{TypeParameter{Tuple{3}}}, J, p) = copyto!(J, p)

Base.inv(::SpecialOrthogonal, p) = transpose(p)
Base.inv(::SpecialOrthogonal, e::Identity{MultiplicationOperation}) = e

function Base.show(io::IO, ::SpecialOrthogonal{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "SpecialOrthogonal($(n))")
end
function Base.show(io::IO, M::SpecialOrthogonal{Tuple{Int}})
    n = get_parameter(M.manifold.size)[1]
    return print(io, "SpecialOrthogonal($(n); parameter=:field)")
end
