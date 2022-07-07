@doc raw"""
     OrthogonalMatrices{n} =  GeneralUnitaryMatrices{n,ℝ,AbsoluteDeterminantOneMatrices}

The manifold of (real) orthogonal matrices ``\mathrm{O}(n)``.

    OrthogonalMatrices(n)
"""
const OrthogonalMatrices{n} = GeneralUnitaryMatrices{n,ℝ,AbsoluteDeterminantOneMatrices}

OrthogonalMatrices(n) = OrthogonalMatrices{n}()

function Base.show(io::IO, ::OrthogonalMatrices{n}) where {n}
    return print(io, "OrthogonalMatrices($(n))")
end
