# Vector bundles

Vector bundle $E$ is a manifold that is built on top of another manifold $ℳ$ (base space).
It is characterized by a continuous function $Π \colon E → ℳ$, such that for each point $x \in ℳ$ the preimage of $x$ by $Π$, $Π^{-1}(\{x\})$, has a structure of a vector space.
These vector spaces are called fibers.
Bundle projection can be performed using function [`bundle_projection`](@ref).

Tangent bundle is a simple example of a vector bundle, where each fiber is the tangent space at the specified point $x$.
An object representing a tangent bundle can be obtained using the constructor called `TangentBundle`.

Fibers of a vector bundle are represented by the type `VectorBundleFibers`.
The important difference between functions operating on `VectorBundle` and `VectorBundleFibers` is that in the first case both a point on the underlying manifold and the vector are represented together (by a single argument) while in the second case only the vector part is present, while the point is supplied in a different argument where needed.

`VectorBundleFibers` refers to the whole set of fibers of a vector bundle.
There is also another type, [`VectorSpaceAtPoint`](@ref), that represents a specific fiber at a given point.
This distinction is made to reduce the need to repeatedly construct objects of type [`VectorSpaceAtPoint`](@ref) in certain usage scenarios.

## FVector

For cases where confusion between different types of vectors is possible, the type [`FVector`](@ref) can be used to express which type of vector space the vector belongs to.
It is used for example in musical isomorphisms (the [`flat`](@ref) and [`sharp`](@ref) functions) that are used to go from a tangent space to cotangent space and vice versa.

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/VectorBundle.jl"]
Order = [:type, :function]
```
