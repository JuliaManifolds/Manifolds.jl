# [Hermitian positive definite matrices](@id HermitianPositiveDefiniteSection)

```@docs
HermitianPositiveDefinite
```

For the real case see the [symmetric positive definite matrices](@ref SymmetricPositiveDefiniteSection).

## Functions

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/HermitianPositiveDefinite.jl"]
Order = [:function]
Public=true
Private=false
Filter = t -> t !== mean
```

## The affine invariant metric

[`HermitianPositiveDefinite`](@ref) is equipped with the [`AffineInvariantMetric`](@ref), so any call of the following functions with `H=HermitianPositiveDefinite(3)` yields the same as with `MetricManifold(H,AffineInvariantMetric())`.
The curvature tensor, the basis diagonalizing it, the minimal sectional curvature and the volume density are only available for the real case, see the [symmetric positive definite matrices](@ref SymmetricPositiveDefiniteSection).

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/HermitianPositiveDefiniteAffineInvariant.jl"]
Order = [:function]
```

## Statistics

```@autodocs
Modules = [Manifolds]
Pages   = ["HermitianPositiveDefinite.jl"]
Order = [:function]
Filter = t -> t === mean
```

## Efficient representation

When a point `p` is used in several occasions, it might be beneficial to store the eigenvalues
and vectors of `p` and optionally its square root and the inverse of the square root.
The [`MatrixSqrtManifoldPoint`](@ref) can be used for exactly that.

```@docs
MatrixSqrtManifoldPoint
```

and there are three internal functions to be able to use [`MatrixSqrtManifoldPoint`](@ref) interchangeably
with the default representation as a matrix.

```@docs
Manifolds.hpd_sqrt
Manifolds.hpd_sqrt_inv
Manifolds.hpd_sqrt_and_sqrt_inv
```

## Literature

```@bibliography
Pages = ["hermitianpositivedefinite.md"]
Canonical=false
```
