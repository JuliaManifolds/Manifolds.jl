# [Product manifold](@id ProductManifoldSection)

Product manifold ``\mathcal M = \mathcal{M}_1 × \mathcal{M}_2 × … × \mathcal{M}_n`` of manifolds ``\mathcal{M}_1, \mathcal{M}_2, …, \mathcal{M}_n``.
Points on the product manifold can be constructed using `ArrayPartition` (from `RecursiveArrayTools.jl`) with canonical projections ``Π_i : \mathcal{M} → \mathcal{M}_i`` for ``i ∈ 1, 2, …, n`` provided by [`submanifold_component`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/metamanifolds/#ManifoldsBase.submanifold_component-Tuple).

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/ProductManifold.jl"]
Order = [:type, :function]
```
