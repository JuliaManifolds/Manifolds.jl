# The probability simplex

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/ProbabilitySimplex.jl"]
Order = [:type, :function]
```

## Real probability amplitudes

An isometric embedding of interior of [`ProbabilitySimplex`] in positive orthant of the
[`Sphere`] is established through functions `simplex_to_amplitude` and `amplitude_to_simplex`. Some properties extend to the boundary but not all.

This embedding isometrically maps the Fisher-Rao metric on the open probability simplex to
the sphere of radius 1 with Euclidean metric. More details can be found in Section 2.2
of [^AyJostLeSchwachhöfer2017].

The name derives from the notion of probability amplitudes in quantum mechanics.
They are complex-valued and their squared norm corresponds to probability. This construction
restricted to real valued amplitudes results in this embedding.

```@docs
Manifolds.amplitude_to_simplex
Manifolds.amplitude_to_simplex_diff
Manifolds.simplex_to_amplitude
Manifolds.simplex_to_amplitude_diff
```

## Literature
