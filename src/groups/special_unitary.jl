#@doc
raw"""
    SpecialUnitary{n} <: GroupManifold{ℂ,???,MultiplicationOperation}

Special orthogonal group $\mathrm{SU}(n)$ represented by unitary mazrices of determinant 1.
This is stil modelled upon [`UnitaryMatrices`](@ref) since compared to this manifold,
only the checks have to be modified

# Constructor

    SpecialUnitary(n)

Generate the Lie group of ``n×n`` unitary matrices.
"""
const SpecialUnitary{n} = GeneralUnitaryMultiplicationGroup{
    n,
    ℝ,
    GeneralUnitaryMatrices{n,ℂ,DeterminantOneMatrices},
}

SpecialUnitary(n) = SpecialOrthogonal{n}()
