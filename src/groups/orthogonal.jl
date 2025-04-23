@doc raw"""
    Orthogonal{T} = GeneralUnitaryMultiplicationGroup{T,ℝ,AbsoluteDeterminantOneMatrixType}

Orthogonal group $\mathrm{O}(n)$ represented by [`OrthogonalMatrices`](@ref).

# Constructor

    Orthogonal(n::Int; parameter::Symbol=:type)
"""
const Orthogonal{T} =
    GeneralUnitaryMultiplicationGroup{T,ℝ,AbsoluteDeterminantOneMatrixType}

function Orthogonal(n::Int; parameter::Symbol=:type)
    _lie_groups_depwarn_move(Orthogonal, :OrthogonalGroup)
    return GeneralUnitaryMultiplicationGroup(OrthogonalMatrices(n; parameter=parameter))
end

function Base.show(io::IO, ::Orthogonal{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "Orthogonal($(n))")
end
function Base.show(io::IO, M::Orthogonal{Tuple{Int}})
    n = get_parameter(M.manifold.size)[1]
    return print(io, "Orthogonal($(n); parameter=:field)")
end
