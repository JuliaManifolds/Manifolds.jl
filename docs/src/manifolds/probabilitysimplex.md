# The probability simplex

```@autodocs
Modules = [Manifolds, Base]
Pages = ["manifolds/ProbabilitySimplex.jl"]
Order = [:type, :function]
Private=false
Public=true
```

## Euclidean metric

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/ProbabilitySimplexEuclideanMetric.jl"]
Order = [:type, :function]
Private=false
Public=true
```


## Real probability amplitudes

An isometric embedding of interior of [`ProbabilitySimplex`](@ref) in positive orthant of the
[`Sphere`](@ref) is established through functions [`simplex_to_amplitude`](@ref Manifolds.simplex_to_amplitude) and [`amplitude_to_simplex`](@ref Manifolds.amplitude_to_simplex). Some properties extend to the boundary but not all.

This embedding isometrically maps the Fisher-Rao metric on the open probability simplex to
the sphere of radius 1 with Euclidean metric. More details can be found in Section 2.2
of [AyJostLeSchwachhoefer:2017](@cite).

The name derives from the notion of probability amplitudes in quantum mechanics.
They are complex-valued and their squared norm corresponds to probability. This construction
restricted to real valued amplitudes results in this embedding.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/ProbabilitySimplex.jl"]
Order = [:type, :function]
Private=true
Public=false
```

## Literature

```@bibliography
Pages = ["manifolds/probabilitysimplex.md"]
Canonical=false
```