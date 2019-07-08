# Vector bundles

Vector bundle $E$ is a manifold that is built on top of another manifold $M$ (base space). It is characterized by a continuous function $\Pi \colon E \to M$, such that for each point $x \in M$ the preimage of $x$ by $\Pi$, $\Pi^{-1}(\{x\})$, has a structure of a vector space. These vector spaces are called fibers. Bundle projection can be performed using function [`bundle_projection`](@ref).

Tangent bundle is a simple example of a vector bundle, where each fiber is the tangent space at the specified point $x$.

```@autodocs
Modules = [Manifolds]
Pages = ["VectorBundle.jl"]
Order = [:type, :function]
```
