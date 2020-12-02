# Vector bundles

Vector bundle $E$ is a manifold that is built on top of another manifold $\mathcal M$ (base space).
It is characterized by a continuous function $Π : E → \mathcal M$, such that for each point $p ∈ \mathcal M$ the preimage of $p$ by $Π$, $Π^{-1}(\{p\})$, has a structure of a vector space.
These vector spaces are called fibers.
Bundle projection can be performed using function [`bundle_projection`](@ref).

Tangent bundle is a simple example of a vector bundle, where each fiber is the tangent space at the specified point $x$.
An object representing a tangent bundle can be obtained using the constructor called `TangentBundle`.

Fibers of a vector bundle are represented by the type `VectorBundleFibers`.
The important difference between functions operating on `VectorBundle` and `VectorBundleFibers` is that in the first case both a point on the underlying manifold and the vector are represented together (by a single argument) while in the second case only the vector part is present, while the point is supplied in a different argument where needed.

`VectorBundleFibers` refers to the whole set of fibers of a vector bundle.
There is also another type, [`VectorSpaceAtPoint`](@ref), that represents a specific fiber at a given point.
This distinction is made to reduce the need to repeatedly construct objects of type [`VectorSpaceAtPoint`](@ref) in certain usage scenarios.
This is also considered a manifold.

## FVector

For cases where confusion between different types of vectors is possible, the type [`FVector`](@ref) can be used to express which type of vector space the vector belongs to.
It is used for example in musical isomorphisms (the [`flat`](@ref) and [`sharp`](@ref) functions) that are used to go from a tangent space to cotangent space and vice versa.

## Example

The following code defines two points on a tangent bundle of the sphere $S^2$ and calculates distance between them, distance between their base points and norm of one of these tangent vectors.

```@example
using Manifolds
M = Sphere(2)
TB = TangentBundle(M)
p = ProductRepr([1.0, 0.0, 0.0], [0.0, 1.0, 3.0])
q = ProductRepr([0.0, 1.0, 0.0], [2.0, 0.0, -1.0])
println("Distance between p and q: ", distance(TB, p, q))
println("Distance between base points of p and q: ", distance(M, p[TB, :point], q[TB, :point]))
println("Norm of p: ", norm(M, p[TB, :point], p[TB, :vector]))
```

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/VectorBundle.jl"]
Order = [:type, :function]
```
