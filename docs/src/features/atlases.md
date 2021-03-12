# [Atlases and charts](@id atlases_and_charts)

Operations using atlases and charts are available through the following types and functions.

```@autodocs
Modules = [Manifolds,ManifoldsBase]
Pages = ["atlases.jl"]
Order = [:type, :function]
```

## Cotangent space

Related to atlases, there is also support for the cotangent space and coefficients of
cotangent vectors in bases of the cotangent space.

```@autodocs
Modules = [Manifolds,ManifoldsBase]
Pages = ["cotangent_space.jl"]
Order = [:type, :function]
```

## Musical isomorphisms

Functions [`sharp`](@ref) and [`flat`](@ref) implement musical isomorphisms for arbitrary vector bundles.

```@docs
sharp
flat
```
