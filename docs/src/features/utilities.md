# Ease of notation

The following terms introduce a nicer notation for some operations, for example using the ∈ operator, $p ∈ \mathcal M$, to determine whether $p$ is a point on the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  $\mathcal M$.

````@docs
in
TangentSpace
````

# Fallback for the exponential map: Solving the corresponding ODE

When additionally loading [`NLSolve.jl`](https://github.com/JuliaNLSolvers/NLsolve.jl) the following fallback for the exponential map is available.

```@autodocs
Modules = [Manifolds]
Pages = ["nlsolve.jl"]
Order = [:type, :function]
```

# Public documentation

The following functions are of interest for extending and using the [`ProductManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/metamanifolds/#ManifoldsBase.ProductManifold).

```@docs
submanifold_components
```

## Specific exception types

For some manifolds it is useful to keep an extra index, at which point on the manifold, the error occurred as well as to collect all errors that occurred on a manifold. This page contains the manifold-specific error messages this package introduces.

```@autodocs
Modules = [Manifolds]
Pages = ["errors.jl"]
Order = [:type, :function]
```
