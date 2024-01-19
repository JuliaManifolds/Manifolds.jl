# [Fiber bundles](@id FiberBundleSection)

Fiber bundle $E$ is a manifold that is built on top of another manifold $\mathcal M$ (base space).
It is characterized by a continuous function $Π : E → \mathcal M$. For each point $p ∈ \mathcal M$ the preimage of $p$ by $Π$, $Π^{-1}(\{p\})$ is called a fiber $F$.
Bundle projection can be performed using function [`bundle_projection`](@ref).

`Manifolds.jl` primarily deals with the case of trivial bundles, where $E$ can be topologically identified with a product $M×F$.

[Vector bundles](@ref VectorBundleSection) is a special case of a fiber bundle. Other examples include unit tangent bundle. Note that in general fiber bundles don't have a canonical Riemannian structure but can at least be equipped with an [Ehresmann connection](https://en.wikipedia.org/wiki/Ehresmann_connection), providing notions of parallel transport and curvature.

## Documentation

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/Fiber.jl", "manifolds/FiberBundle.jl"]
Order = [:constant, :type, :function]
```
