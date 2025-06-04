@dpc raw"""
    HermitianPositiveDefinite{𝔽,T} <: AbstractDecoratorManifold{𝔽}

The manifold of hermitian positive definite matrices, i.e.

````math
\mathcal H(n) :=
\bigl\{
p ∈ 𝔽^{n×n}\ \big|\ a^\mathrm{T}pa > 0 \text{ for all } a ∈ 𝔽^{n}\backslash\{0\}
\bigr\},
````
where usually ``𝔽=ℂ``. For the case ``𝔽=ℝ`` this manifold simplified to the [`SymmetricPositiveDefinite`](@ref)

The tangent space at ``p∈\mathcal H(n)`` reads

```math
    T_p\mathcal H(n) =
    \bigl\{
        X \in 𝔽^{n×n} \big|\ X=X^\mathrm{H}
    \bigr\},
```
i.e. the set of hermitian matrices,
# Constructor

    HermitianPositiveDefinite(n, 𝔽=ℂ; parameter::Symbol=:type)

generates the manifold ``\mathcal P(n) \subset ℝ^{n×n}``
"""
struct HermitianPositiveDefinite{𝔽,T} <: AbstractDecoratorManifold{𝔽}
    size::T
end
function HermitianPositiveDefinite(n,𝔽::AbstractNumbers=ℂ; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, n)
    return HermitianPositiveDefinite{field,typeof(size)}(size)
end