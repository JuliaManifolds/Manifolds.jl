# Torus
The torus $\mathbb T^d \equiv [-π,π)^d$ is modeled as an [`AbstractPowerManifold`](@ref) of
the (real-valued) [`Circle`](@ref) and uses [`MultidimentionalArrayPowerRepresentation`](@ref).
Points on the torus are hence row vectors, $x\in\mathbb R^{d}$.

## Example

The following code can be used to make a three-dimensional torus $\mathbb{T}^3$ and compute a tangent vector.
```@example
using Manifolds
M = Torus(3)
x = [0.5, 0.0, 0.0]
y = [0.0, 0.5, 1.0]
v = log(M, x, y)
```

## Types and functions

Most functions are directly implemented for an [`AbstractPowerManifold`](@ref) with [`MultidimentionalArrayPowerRepresentation`](@ref) except the following special cases.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Torus.jl"]
Order = [:type, :function]
```
