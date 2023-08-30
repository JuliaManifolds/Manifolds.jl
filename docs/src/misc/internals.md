# Internal documentation

This page documents the internal types and methods of `Manifolds.jl`'s that might be of use for writing your own manifold.

## Functions

```@docs
Manifolds.eigen_safe
Manifolds.isnormal
Manifolds.log_safe
Manifolds.log_safe!
Manifolds.mul!_safe
Manifolds.nzsign
Manifolds.realify
Manifolds.realify!
Manifolds.select_from_tuple
Manifolds.symmetrize
Manifolds.symmetrize!
Manifolds.unrealify!
Manifolds.usinc
Manifolds.usinc_from_cos
Manifolds.vec2skew!
Manifolds.ziptuples
```

## Types in Extensions

```@autodocs
Modules = [Manifolds]
Pages = ["../ext/ManifoldsOrdinaryDiffEqDiffEqCallbacksExt.jl"]
Order = [:type, :function]
```