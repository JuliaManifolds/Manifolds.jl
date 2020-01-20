# Graph manifold

For a given graph $G(V,E)$ implemented using [`LightGraphs.jl`](https://juliagraphs.github.io/LightGraphs.jl/latest/), the [`GraphManifold`](@ref) models a [`PowerManifold`](@ref) either on the nodes or edges of the graph, depending on the [`GraphManifoldType`](@ref).
i.e., it's either a $\mathcal M^{\lvert V \rvert}$ for the case of a vertex manifold or a $\mathcal M^{\lvert E \rvert}$ for the case of a edge manifold.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/GraphManifold.jl"]
Order = [:type, :function]
```
