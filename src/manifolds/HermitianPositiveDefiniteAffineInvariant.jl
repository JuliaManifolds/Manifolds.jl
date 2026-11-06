@doc raw"""
    AffineInvariantMetric <: AbstractMetric

The linear affine metric is the metric for Hermitian, and in particular symmetric, positive definite matrices,
that employs matrix logarithms and exponentials, which yields a linear and affine metric.
"""
struct AffineInvariantMetric <: RiemannianMetric end

@doc raw"""
    change_representer(M::HermitianPositiveDefinite, E::EuclideanMetric, p, X)

Compute the representer of the linear function given by ``X ∈ T_p\mathcal M`` with respect to
the (default) [`AffineInvariantMetric`](@ref) on the [`HermitianPositiveDefinite`](@ref) `M`.
Here `X` represents the linear function on the tangent space at `p` with respect to the
[`EuclideanMetric`](@extref `ManifoldsBase.EuclideanMetric`) `g_E`.

To be precise we are looking for ``Z∈T_p\mathcal H(n)`` such that for all ``Y∈T_p\mathcal H(n)``
it holds

```math
⟨X,Y⟩ = \operatorname{Re}\operatorname{tr}(X^{\mathrm{H}}Y) = \operatorname{tr}(XY) = \operatorname{tr}(p^{-1}Zp^{-1}Y) = g_p(Z,Y)
```

and hence ``Z = pXp``.
"""
change_representer(::HermitianPositiveDefinite, ::EuclideanMetric, ::Any, ::Any)

function change_representer!(::HermitianPositiveDefinite, Y, ::EuclideanMetric, p, X)
    Y .= p * X * p
    return Y
end
