# [Hyperrectangle](@id HyperrectangleSection)

Hyperrectangle is a manifold with corners [Joyce:2010](@cite), and also a subset of the real [Euclidean](@ref Main.Manifolds.Euclidean) manifold.
It is useful for box-constrained optimization, for example it is implicitly used in the classic L-BFGS-B algorithm.

!!! note
    This is a manifold with corners. Some parts of its interface specific to this property are experimental and may change without a breaking release.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Hyperrectangle.jl"]
Order = [:type,:function]
```

## Literature

```@bibliography
Pages = ["hyperrectangle.md"]
Canonical=false
```
