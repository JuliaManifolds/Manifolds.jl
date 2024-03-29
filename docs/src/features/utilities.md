# Utilities

## Ease of notation

The following terms introduce a nicer notation for some operations, for example using the ∈ operator, ``p ∈ \mathcal M`` to determine whether ``p`` is a point on the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)  ``\mathcal M``.

````@docs
in
````

## Public documentation

```@docs
sectional_curvature_matrix
```

### Specific exception types

For some manifolds it is useful to keep an extra index, at which point on the manifold, the error occurred as well as to collect all errors that occurred on a manifold. This page contains the manifold-specific error messages this package introduces.

```@autodocs
Modules = [Manifolds]
Pages = ["errors.jl"]
Order = [:type, :function]
```
