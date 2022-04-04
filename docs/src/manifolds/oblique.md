# Oblique manifold

The oblique manifold $\mathcal{OB}(n,m)$ is modeled as an [`AbstractPowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractPowerManifold)  of the (real-valued) [`Sphere`](@ref) and uses [`ArrayPowerRepresentation`](@ref).
Points on the torus are hence matrices, $x ∈ ℝ^{n,m}$.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Oblique.jl"]
Order = [:type]
```

## Functions

Most functions are directly implemented for an [`AbstractPowerManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.AbstractPowerManifold)  with [`ArrayPowerRepresentation`](@ref) except the following special cases:

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Oblique.jl"]
Order = [:function]
```
