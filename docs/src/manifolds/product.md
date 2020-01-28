# Product manifold

Product manifold $M = M_1 ⨉ M_2 ⨉ … M_n$ of manifolds $M_1, M_2, …, M_n$.
Points on the product manifold can be constructed using [`ProductRepr`](@ref) with canonical projections $Π_i : M → M_i$ for $i ∈ 1, 2, …, n$ provided by [`submanifold_component`](@ref).

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/ProductManifold.jl"]
Order = [:type, :function]
```
