# Torus
The torus $\mathbb T^d \equiv [-π,π)^d$ is modeled as a [`PowerManifold`] of
the (real-valued) [`Circle`](@ref). Points on the torus are hence column vectors, $x\in\mathbb R^{1\times d}$.

Most functions are directly implememnted for an [`AbstractPowerManifold`](@ref) despite the following special cases

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Torus.jl"]
Order = [:type, :function]
```
