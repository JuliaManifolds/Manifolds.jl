@doc raw"""
    const SpecialUnitaryMatrices{T} = GeneralUnitaryMatrices{ℂ, T, DeterminantOneMatrixType}

The manifold ``SU(n)`` of ``n×n`` complex matrices such that

```math
    p^{\mathrm{H}}p = \mathrm{I}_n \text{ and } \det(p) = 1,
```

where ``p^{\mathrm{H}}`` is the conjugate transpose of ``p`` and ``\mathrm{I}_n`` is the ``n×n`` identity matrix.

The tangent spaces are given by

```math
    T_pU(n) \coloneqq \bigl\{
    X \big| pY \text{ where } Y \text{ is skew symmetric and traceless, i. e. } Y = -Y^{\mathrm{H}} \text{ and } \operatorname{tr}(Y) = 0
    \bigr\}
```

But note that tangent vectors are represented in the Lie algebra, i.e. just using ``Y`` in
the representation above.

# Constructor

    SpecialUnitaryMatrices(n; parameter::Symbol = :type)

see also [`Rotations`](@ref) for the real valued case.
"""
const SpecialUnitaryMatrices{T} = GeneralUnitaryMatrices{ℂ, T, DeterminantOneMatrixType}

function SpecialUnitaryMatrices(n::Int; parameter::Symbol = :type)
    size = wrap_type_parameter(parameter, (n,))
    return SpecialUnitaryMatrices{typeof(size)}(size)
end
