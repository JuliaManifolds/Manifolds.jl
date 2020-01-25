# Graph manifold

For a given graph $G(V,E)$ implemented using [`LightGraphs.jl`](https://juliagraphs.github.io/LightGraphs.jl/latest/), the [`GraphManifold`](@ref) models a [`PowerManifold`](@ref) either on the nodes or edges of the graph, depending on the [`GraphManifoldType`](@ref).
i.e., it's either a $\mathcal M^{\lvert V \rvert}$ for the case of a vertex manifold or a $\mathcal M^{\lvert E \rvert}$ for the case of a edge manifold.

## Example:

To make a graph manifold over $\mathbb{R}^2$ with three vertices and two edges, one can use
```@example
using Manifolds
using LightGraphs
M = Euclidean(2)
x = [[1., 4.], [2., 5.], [3., 6.]]
y = [[4., 5.], [6., 7.], [8., 9.]]
z = [[6., 5.], [4., 3.], [2., 8.]]
G = SimpleGraph(3)
add_edge!(G, 1, 2)
add_edge!(G, 2, 3)
N = GraphManifold(G, M, VertexManifold())
```
It supports all [`AbstractPowerManifold`](@ref) operations (it is based on [`NestedPowerRepresentation`](@ref)) and furthermore it is possible to compute a graph logarithm:

```@setup graph-1
using Manifolds
using LightGraphs
M = Euclidean(2)
x = [[1., 4.], [2., 5.], [3., 6.]]
y = [[4., 5.], [6., 7.], [8., 9.]]
z = [[6., 5.], [4., 3.], [2., 8.]]
G = SimpleGraph(3)
add_edge!(G, 1, 2)
add_edge!(G, 2, 3)
N = GraphManifold(G, M, VertexManifold())
```
```@example graph-1
incident_log(N, x)
```

## Types and functions

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/GraphManifold.jl"]
Order = [:type, :function]
```
