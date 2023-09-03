# Stiefel

## Common and metric independent functions

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Stiefel.jl"]
Order = [:type, :function]
```

## Default metric: the Euclidean metric

The [`EuclideanMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.EuclideanMetric) is obtained from the embedding of the Stiefel manifold in ``ℝ^{n,k}``.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/StiefelEuclideanMetric.jl"]
Order = [:function]
```

## The canonical metric
Any ``X∈T_p\mathcal M``, ``p∈\mathcal M``, can be written as

```math
X = pA + (I_n-pp^{\mathrm{T}})B,
\quad
A ∈ ℝ^{p×p} \text{ skew-symmetric},
\quad
B ∈ ℝ^{n×p} \text{ arbitrary.}
```

In the [`EuclideanMetric`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/manifolds.html#ManifoldsBase.EuclideanMetric), the elements from ``A`` are counted twice (i.e. weighted with a factor of 2).
The canonical metric avoids this.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/StiefelCanonicalMetric.jl"]
Order = [:type]
```

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/StiefelCanonicalMetric.jl"]
Order = [:function]
```

## The submersion or normal metric

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/StiefelSubmersionMetric.jl"]
Order = [:type, :function]
Public = true
Private = false
```

## Internal types and functions

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/StiefelSubmersionMetric.jl"]
Order = [:type, :function]
Public = false
Private = true
```

## Literature

```@bibliography
Pages = ["manifolds/stiefel.md"]
Canonical=false
```