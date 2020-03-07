# Lorentzian Manifold

The [Lorentz manifold](https://en.wikipedia.org/wiki/Pseudo-Riemannian_manifold#Lorentzian_manifold) is a [pseudo-Riemannian manifold](https://en.wikipedia.org/wiki/Pseudo-Riemannian_manifold).
It is named after the Dutch physicist [Hendrik Lorentz](https://en.wikipedia.org/wiki/Hendrik_Lorentz) (1853–1928).
The default [`LorentzMetric`](@ref) is the [`MinkowskiMetric`](@ref) named after the German mathematician [Hermann Minkowski](https://en.wikipedia.org/wiki/Hermann_Minkowski) (1864–1909).

Within `Manifolds.jl` it is used as the embedding of the [`Hyperbolic`](@ref) space.

```@autodocs
Modules = [Manifolds]
Pages = ["Lorentz.jl"]
Order = [:type, :function]
```
