# Internal documentation

This page documents the internal types and methods of `Manifolds.jl`'s that might be of use for writing your own manifold.

## Functions

```@docs
Manifolds.eigen_safe
Manifolds.estimated_sectional_curvature
Manifolds.estimated_sectional_curvature_matrix
Manifolds.get_parameter_type
Manifolds.isnormal
Manifolds.log_safe
Manifolds.log_safe!
Manifolds.mul!_safe
Manifolds.normal_tvector_distribution
Manifolds.nzsign
Manifolds.projected_distribution
Manifolds.realify
Manifolds.realify!
Manifolds.symmetrize
Manifolds.symmetrize!
Manifolds.unrealify!
Manifolds.usinc
Manifolds.usinc_from_cos
Manifolds.vec2skew!
```

## Types in Extensions

```@autodocs
Modules = [Base.get_extension(Manifolds, :ManifoldsOrdinaryDiffEqDiffEqCallbacksExt)]
Order = [:type, :function]
```
