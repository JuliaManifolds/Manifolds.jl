# Euclidean space

The Euclidean space $ℝ^n$ is a simple model space, since it has curvature constantly zero everywhere; hence, nearly all operations simplify.
The easiest way to generate an Euclidean space is to use a field, i.e. [`AbstractNumbers`](@ref), e.g. to create the $ℝ^n$ or $ℝ^{n\times n}$ you can simply type `M = ℝ^n` or `ℝ^(n,n)`, respectively.

```@autodocs
Modules = [Manifolds]
Pages = ["manifolds/Euclidean.jl"]
Order = [:type,:function]
```
