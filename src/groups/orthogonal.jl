@doc raw"""
    Orthogonal{n} = GeneralUnitaryMultiplicationGroup{n,ℝ,AbsoluteDeterminantOneMatrices}

Orthogonal group $\mathrm{O}(n)$ represented by [`OrthogonalMatrices`](@ref).

# Constructor

    Orthogonal(n)
"""
const Orthogonal{n} = GeneralUnitaryMultiplicationGroup{n,ℝ,AbsoluteDeterminantOneMatrices}

Orthogonal(n) = Orthogonal{n}(OrthogonalMatrices(n))

show(io::IO, ::Orthogonal{n}) where {n} = print(io, "Orthogonal($(n))")
