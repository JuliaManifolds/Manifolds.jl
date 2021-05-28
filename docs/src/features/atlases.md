# [Atlases and charts](@id atlases_and_charts)

Atlases on an ``n``-dimensional manifold $\mathcal M$ are collections of charts ``\mathcal A = \{(U_i, \varphi_i) \colon i \in I\}``, where ``I`` is a (finite or infinte) index family, such that ``U_i \subseteq \mathcal M`` is an open set and each chart ``\varphi_i: U_i \to \mathbb{R}^n`` is a homeomorphism. This means, that ``\varphi_i`` is bijecive – sometimes also called one-to-one and onto - and continuous, and its inverse ``\varphi_i^{-1}`` is continuous as well.
The inverse ``\varphi_i^{-1}`` is called (local) parametrization.
For an atlas ``\mathcal A`` we further require that

```math
\displaystyle\bigcup_{i\in I} U_i = \mathcal M.
```

We say that ``\varphi_i`` is a chart about ``p``, if ``p\in U_i``.
An atlas provides a connection between a manifold and the Euclidean space ``\mathbb{R}^n``, since
locally, a chart about ``p`` can be used to identify its neighborhood (as long as you stay in ``U_i``) with a subset of a Euclidean space.
Most manifolds we consider are smooth, i.e. any change of charts ``\varphi_i\circ\varphi_j^{-1}: \mathbb{R}^n\to\mathbb{R}^n``, where ``i,j\in I``, is a smooth function. These changes of charts are also called transition maps.

Most operations on manifolds in `Manifolds.jl` avoid operating in a chart through appropriate embeddings and formulas derived for particular manifolds, though atlases provide the most general way of working with manifolds.
Compared to these approaches, using an atlas is often more technical and time-consuming.
They are extensively used in metric-related functions on [`MetricManifold`](@ref Main.Manifolds.MetricManifold)s.

Atlases are represented by objects of subtypes of [`AbstractAtlas`](@ref Main.Manifolds.AbstractAtlas).
There are no type restrictions for indices of charts in atlases.

Operations using atlases and charts are available through the following functions:

* [`get_chart_index`](@ref Main.Manifolds.get_chart_index) can be used to select an appropriate chart for the neighborhood of a given point ``p``. This function should work deterministically, i.e. for a fixed ``p`` always return the same chart.
* [`get_parameters`](@ref Main.Manifolds.get_parameters) converts a point to its local coordinates, also called parameters with respect to the chart in a chart.
* [`get_point`](@ref Main.Manifolds.get_point) converts parameters (local coordinates) in a chart to the point that corresponds to them.
* [`induced_basis`](@ref Main.Manifolds.induced_basis) returns a basis of a given vector space at a point induced by a chart ``\varphi``, by taking the derivative of the coordinate functions ``\varphi^k``, ``k=1,\ldots,n``.
* [`transition_map`](@ref Main.Manifolds.transition_map) converts coordinates of a point between two charts, e.g. computes ``\varphi_i\circ\varphi_j^{-1}: \mathbb{R}^n\to\mathbb{R}^n``, ``i,j\in I``.

While an atlas could store charts as explicit functions, it is favourable, that the [`get_parameters`] actually implements a chart ``\varphi``, [`get_point`](@ref) its inverse, the prametrization ``\varphi^{-1}``.

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
