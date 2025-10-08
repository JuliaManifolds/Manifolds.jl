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

@doc raw"""
    manifold_dimension(M::SpecialUnitaryMatrices)

Return the dimension of the manifold of special unitary matrices.
```math
\dim_{\mathrm{SU}(n)} = n^2-1.
```
"""
function manifold_dimension(M::SpecialUnitaryMatrices)
    n = get_parameter(M.size)[1]
    return n^2 - 1
end

@doc raw"""
    manifold_volume(::SpecialUnitaryMatrices)

Volume of the manifold of complex general unitary matrices of determinant one. The formula
reads [BoyaSudarshanTilma:2003](@cite)

```math
\sqrt{n 2^{n-1}} π^{(n-1)(n+2)/2} \prod_{k=1}^{n-1}\frac{1}{k!}.
```
"""
function manifold_volume(M::SpecialUnitaryMatrices)
    n = get_parameter(M.size)[1]
    vol = sqrt(n * 2^(n - 1)) * π^(((n - 1) * (n + 2)) // 2)
    kf = 1
    for k in 1:(n - 1)
        kf *= k
        vol /= kf
    end
    return vol
end

@doc raw"""
    injectivity_radius(G::SpecialUnitaryMatrices)

Return the injectivity radius for general complex unitary matrix manifolds, where the determinant is ``+1``,
which is[^1]

```math
    \operatorname{inj}_{\mathrm{SU}(n)} = π \sqrt{2}.
```
[^1]
    > For a derivation of the injectivity radius, see [sethaxen.com/blog/2023/02/the-injectivity-radii-of-the-unitary-groups/](https://sethaxen.com/blog/2023/02/the-injectivity-radii-of-the-unitary-groups/).
"""
function injectivity_radius(::SpecialUnitaryMatrices)
    return π * sqrt(2.0)
end
