@doc raw"""
    SpecialLinear{n,𝔽} <: AbstractDecoratorManifold

The special linear group ``\mathrm{SL}(n,𝔽)`` that is, the group of all invertible matrices
with unit determinant in ``𝔽^{n×n}``.

The Lie algebra ``𝔰𝔩(n, 𝔽) = T_e \mathrm{SL}(n,𝔽)`` is the set of all matrices in
``𝔽^{n×n}`` with trace of zero. By default, tangent vectors ``X_p ∈ T_p \mathrm{SL}(n,𝔽)``
for ``p ∈ \mathrm{SL}(n,𝔽)`` are represented with their corresponding Lie algebra vector
``X_e = p^{-1}X_p ∈ 𝔰𝔩(n, 𝔽)``.

The default metric is the same left-``\mathrm{GL}(n)``-right-``\mathrm{O}(n)``-invariant
metric used for [`GeneralLinear(n, 𝔽)`](@ref). The resulting geodesic on
``\mathrm{GL}(n,𝔽)`` emanating from an element of ``\mathrm{SL}(n,𝔽)`` in the direction of
an element of ``𝔰𝔩(n, 𝔽)`` is a closed subgroup of ``\mathrm{SL}(n,𝔽)``. As a result, most
metric functions forward to [`GeneralLinear`](@ref).
"""
struct SpecialLinear{n,𝔽} <: AbstractDecoratorManifold{𝔽} end

SpecialLinear(n, 𝔽::AbstractNumbers=ℝ) = SpecialLinear{n,𝔽}()

@inline function active_traits(f, ::SpecialLinear, args...)
    return merge_traits(
        IsGroupManifold(MultiplicationOperation()),
        IsEmbeddedSubmanifold(),
        HasLeftInvariantMetric(),
        IsDefaultMetric(EuclideanMetric()),
    )
end

function allocation_promotion_function(::SpecialLinear{n,ℂ}, f, args::Tuple) where {n}
    return complex
end

function check_point(G::SpecialLinear{n,𝔽}, p; kwargs...) where {n,𝔽}
    detp = det(p)
    if !isapprox(detp, 1; kwargs...)
        return DomainError(
            detp,
            "The matrix $(p) does not lie on $(G), since it does not have a unit " *
            "determinant.",
        )
    end
    return nothing
end

function check_vector(G::SpecialLinear, p, X; kwargs...)
    trX = tr(inverse_translate_diff(G, p, p, X, LeftAction()))
    if !isapprox(trX, 0; kwargs...)
        return DomainError(
            trX,
            "The matrix $(X) does not lie in the tangent space of $(G) at $(p), since " *
            "its Lie algebra representation is not traceless.",
        )
    end
    return nothing
end

embed(::SpecialLinear, p) = p
embed(::SpecialLinear, p, X) = X

get_embedding(::SpecialLinear{n,𝔽}) where {n,𝔽} = GeneralLinear(n, 𝔽)

inverse_translate_diff(::SpecialLinear, p, q, X, ::LeftAction) = X
inverse_translate_diff(::SpecialLinear, p, q, X, ::RightAction) = p * X / p

function inverse_translate_diff!(G::SpecialLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end

function manifold_dimension(G::SpecialLinear)
    return manifold_dimension(get_embedding(G)) - real_dimension(number_system(G))
end

@doc raw"""
    project(G::SpecialLinear, p)

Project ``p ∈ \mathrm{GL}(n, 𝔽)`` to the [`SpecialLinear`](@ref) group
``G=\mathrm{SL}(n, 𝔽)``.

Given the singular value decomposition of ``p``, written ``p = U S V^\mathrm{H}``, the
formula for the projection is

````math
\operatorname{proj}_{\mathrm{SL}(n, 𝔽)}(p) = U S D V^\mathrm{H},
````
where

````math
D_{ij} = δ_{ij} \begin{cases}
    1            & \text{ if } i ≠ n \\
    \det(p)^{-1} & \text{ if } i = n
\end{cases}.
````
"""
project(::SpecialLinear, p)

function project!(::SpecialLinear{n}, q, p) where {n}
    detp = det(p)
    isapprox(detp, 1) && return copyto!(q, p)
    F = svd(p)
    q .= F.U .* F.S'
    q[:, n] ./= detp
    mul!_safe(q, q, F.Vt)
    return q
end

@doc raw"""
    project(G::SpecialLinear, p, X)

Orthogonally project ``X ∈ 𝔽^{n × n}`` onto the tangent space of ``p`` to the
[`SpecialLinear`](@ref) ``G = \mathrm{SL}(n, 𝔽)``. The formula reads
````math
\operatorname{proj}_{p}
    = (\mathrm{d}L_p)_e ∘ \operatorname{proj}_{𝔰𝔩(n, 𝔽)} ∘ (\mathrm{d}L_p^{-1})_p
    \colon X ↦ X - \frac{\operatorname{tr}(X)}{n} I,
````
where the last expression uses the tangent space representation as the Lie algebra.
"""
project(::SpecialLinear, p, X)

function project!(G::SpecialLinear{n}, Y, p, X) where {n}
    inverse_translate_diff!(G, Y, p, p, X, LeftAction())
    Y[diagind(n, n)] .-= tr(Y) / n
    translate_diff!(G, Y, p, p, Y, LeftAction())
    return Y
end

Base.show(io::IO, ::SpecialLinear{n,𝔽}) where {n,𝔽} = print(io, "SpecialLinear($n, $𝔽)")

translate_diff(::SpecialLinear, p, q, X, ::LeftAction) = X
translate_diff(::SpecialLinear, p, q, X, ::RightAction) = p \ X * p

function translate_diff!(G::SpecialLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end
