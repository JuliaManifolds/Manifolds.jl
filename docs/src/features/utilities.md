# Ease of notation

The following terms introduce a nicer notation for some operations, for example using the ∈ operator, $p ∈ \mathcal M$, to determine whether $p$ is a point on the [`AbstractManifold`](@ref) $\mathcal M$.

````@docs
in
TangentSpace
````

# Public documentation

The following functions are of interest for extending and using the [`ProductManifold`](@ref).

```@docs
Manifolds.ShapeSpecification
submanifold_component
submanifold_components
Manifolds.ProductArray
ProductRepr
Manifolds.prod_point
Manifolds.StaticReshaper
Manifolds.ArrayReshaper
Manifolds.make_reshape
```

## Specific exception types

For some manifolds it is useful to keep an extra index, at which point on the manifold, the error occurred as well as to collect all errors that occurred on a manifold. This page contains the manifold-specific error messages this package introduces.

```@autodocs
Modules = [Manifolds]
Pages = ["errors.jl"]
Order = [:type, :function]
```
