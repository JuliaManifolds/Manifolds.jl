# Product manifold

Product manifold $\mathcal M = \mathcal{M}_1 × \mathcal{M}_2 × … × \mathcal{M}_n$ of manifolds $\mathcal{M}_1, \mathcal{M}_2, …, \mathcal{M}_n$.
Points on the product manifold can be constructed using [`ProductRepr`](@ref) with canonical projections $Π_i : \mathcal{M} → \mathcal{M}_i$ for $i ∈ 1, 2, …, n$ provided by [`submanifold_component`](@ref).

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/ProductManifold.jl"]
Order = [:type, :function]
```
