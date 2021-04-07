# [Atlases and charts](@id atlases_and_charts)

Atlases on an $n$-dimensional manifold $\mathcal M$ are collections of charts $\{(U_i, \varphi_i) \colon i \in I\}$ such that $U_i \subseteq \mathcal M$ and each chart $\varphi_i$ is a function from $U_i$ to $\mathbb{R}^n$.
They provide a basic connection between manifolds and Euclidean spaces.

Most operations on manifolds in `Manifolds.jl` avoid operating in a chart through appropriate embeddings and formulas derived for particular manifolds, though atlases provide the most general way of working with manifolds.
They are extensively used in metric-related functions on [`MetricManifold`](@ref Main.Manifolds.MetricManifold)s.

Atlases are represented by objects of subtypes of [`AbstractAtlas`](@ref Main.Manifolds.AbstractAtlas).
There are no type restrictions for indices of charts in atlases.

Operations using atlases and charts are available through the following functions:

* [`get_chart_index`](@ref Main.Manifolds.get_chart_index) can be used to select an appropriate chart for the neighborhood of a given point.
* [`get_point_coordinates`](@ref Main.Manifolds.get_point_coordinates) converts a point to its coordinates in a chart.
* [`get_point`](@ref Main.Manifolds.get_point) converts coordinates in a chart to the point that corresponds to them.
* [`induced_basis`](@ref Main.Manifolds.induced_basis) returns a basis of a given vector space at a point induced by a chart.
* [`transition_map`](@ref Main.Manifolds.transition_map) converts coordinates of a point between two charts.

```@autodocs
Modules = [Manifolds,ManifoldsBase]
Pages = ["atlases.jl"]
Order = [:type, :function]
```

## Cotangent space and musical isomorphisms

Related to atlases, there is also support for the cotangent space and coefficients of cotangent vectors in bases of the cotangent space.

Functions [`sharp`](@ref) and [`flat`](@ref) implement musical isomorphisms for arbitrary vector bundles.

```@autodocs
Modules = [Manifolds,ManifoldsBase]
Pages = ["cotangent_space.jl"]
Order = [:type, :function]
```
