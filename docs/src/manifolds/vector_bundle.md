# [Vector bundles](@id VectorBundleSection)

Vector bundle $E$ is a special case of a [fiber bundle](@ref FiberBundleSection) where each fiber is a vector space.

Tangent bundle is a simple example of a vector bundle, where each fiber is the tangent space at the specified point $p$.
An object representing a tangent bundle can be obtained using the constructor called `TangentBundle`.

There is also another type, [`VectorSpaceFiber`](@ref), that represents a specific fiber at a given point.
This distinction is made to reduce the need to repeatedly construct objects of type [`VectorSpaceFiber`](@ref) in certain usage scenarios.
This is also considered a manifold.

## FVector

For cases where confusion between different types of vectors is possible, the type [`FVector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.FVector) can be used to express which type of vector space the vector belongs to.
It is used for example in musical isomorphisms (the [`flat`](@ref) and [`sharp`](@ref) functions) that are used to go from a tangent space to cotangent space and vice versa.

## Documentation

```@autodocs
Modules = [Manifolds, ManifoldsBase]
Pages = ["manifolds/VectorFiber.jl", "manifolds/VectorBundle.jl"]
Order = [:constant, :type, :function]
```

## Example

The following code defines a point on the tangent bundle of the sphere $S^2$ and a tangent vector to that point.

```@example tangent-bundle
using Manifolds
M = Sphere(2)
TB = TangentBundle(M)
p = ArrayPartition([1.0, 0.0, 0.0], [0.0, 1.0, 3.0])
X = ArrayPartition([0.0, 1.0, 0.0], [0.0, 0.0, -2.0])
```

An approximation of the exponential in the Sasaki metric using 1000 steps can be calculated as follows.

```@example tangent-bundle
q = retract(TB, p, X, SasakiRetraction(1000))
println("Approximation of the exponential map: ", q)
```
