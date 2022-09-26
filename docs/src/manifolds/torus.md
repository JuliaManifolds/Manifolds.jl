# Torus

The torus $𝕋^d ≅ [-π,π)^d$ is modeled as an [`AbstractPowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractPowerManifold)  of the (real-valued) [`Circle`](@ref) and uses [`ArrayPowerRepresentation`](@ref).
Points on the torus are hence row vectors, $x ∈ ℝ^{d}$.

## Example

The following code can be used to make a three-dimensional torus $𝕋^3$ and compute a tangent vector:

```@example
using Manifolds
M = Torus(3)
p = [0.5, 0.0, 0.0]
q = [0.0, 0.5, 1.0]
X = log(M, p, q)
```

## Types and functions

Most functions are directly implemented for an [`AbstractPowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractPowerManifold)  with [`ArrayPowerRepresentation`](@ref) except the following special cases:

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Torus.jl"]
Order = [:type, :function]
```

## Curved Torus

Two-dimensional torus embedded in $ℝ^3$.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/CurvedTorus.jl"]
Order = [:type, :function]
```
