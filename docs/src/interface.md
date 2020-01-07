# ManifoldsBase.jl – An Interface for Manifolds

The interface for a manifold is provided in the lightweight package
[`ManifoldsBase.jl](https://github.com/JuliaNLSolvers/ManifoldsBase.jl) separate
from the collection of manifolds in here. You can easily implement your algorithms
and even first own manifolds just using the interface.

The following functions are currently available from the interface.
If a manifold that you implement for your own package fits this interface,
we happily look forward to a
[Pull Request](https://github.com/JuliaNLSolvers/Manifolds.jl/compare) to add it here.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["ManifoldsBase.jl"]
Order = [:type, :function]
```

`DefaultManifold` is a simplified version of [`Euclidean`](@ref) and demonstrates a basic interface implementation.
```@docs
ManifoldsBase.DefaultManifold
```
