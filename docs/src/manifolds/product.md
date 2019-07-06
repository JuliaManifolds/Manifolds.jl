# Product Manifold

Product manifold $M = M_1 \times M_2 \times \dots M_n$ of manifolds $M_1, M_2, \dots, M_n$. Points on the product manifold can be constructed using [`ProductMPoint`](@ref) with canonical projections $\Pi_i \colon M \to M_i$ for $i \in 1, 2, \dots, n$ provided by [`submanifold_component`](@ref).

```@autodocs
Modules = [Manifolds]
Pages = ["ProductManifold.jl"]
Order = [:type, :function]
```
