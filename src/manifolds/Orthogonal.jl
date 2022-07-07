@doc raw"""
     OrthogonalMatrices{n}

The manifold of (real) orthogonal matrices ``\mathrm{O}(n)``.
This is the special case of {`UnitaryMatrices`}(@ref) over the reals.

    OrthogonalMatrices(n)

The constructor is equivalent to calling [`Unitary(n,ℝ)`](@ref).
"""
const OrthogonalMatrices{n} = AbstractUnitaryMatrices{n,ℝ,AbsoluteDeterminantOneMatrices}

OrthogonalMatrices(n) = OrthogonalMatrices{n}()
