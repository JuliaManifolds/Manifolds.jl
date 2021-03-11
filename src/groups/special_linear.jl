struct SpecialLinear{n,𝔽} <:
       AbstractGroupManifold{𝔽,MultiplicationOperation,TransparentIsometricEmbedding} end

SpecialLinear(n, 𝔽::AbstractNumbers=ℝ) = SpecialLinear{n,𝔽}()

function allocation_promotion_function(::SpecialLinear{n,ℂ}, f, args::Tuple) where {n}
    return complex
end

function check_manifold_point(G::SpecialLinear{n,𝔽}, p; kwargs...) where {n,𝔽}
    mpv = check_manifold_point(Euclidean(n, n; field=𝔽), p; kwargs...)
    mpv === nothing || return mpv
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
check_manifold_point(::GT, ::Identity{GT}; kwargs...) where {GT<:SpecialLinear} = nothing
function check_manifold_point(G::SpecialLinear, e::Identity; kwargs...)
    return DomainError(e, "The identity element $(e) does not belong to $(G).")
end

function check_tangent_vector(G::SpecialLinear, p, X; check_base_point=true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(G, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv =
        check_tangent_vector(decorated_manifold(G), p, X; check_base_point=false, kwargs...)
    mpv === nothing || return mpv
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

decorated_manifold(::SpecialLinear{n,𝔽}) where {n,𝔽} = GeneralLinear(n, 𝔽)

default_metric_dispatch(::SpecialLinear, ::EuclideanMetric) = Val(true)
default_metric_dispatch(::SpecialLinear, ::LeftInvariantMetric{EuclideanMetric}) = Val(true)

inverse_translate_diff(::SpecialLinear, p, q, X, ::LeftAction) = X
inverse_translate_diff(::SpecialLinear, p, q, X, ::RightAction) = p * X / p

function inverse_translate_diff!(G::SpecialLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end

function manifold_dimension(G::SpecialLinear)
    return manifold_dimension(decorated_manifold(G)) - real_dimension(number_system(G))
end

@doc raw"""
    project(G::SpecialLinear, p)

Project $p ∈ \mathrm{GL}(n, 𝔽)$ to the [`SpecialLinear`](@ref) group $G=\mathrm{SL}(n, 𝔽)$.

Given the singular value decomposition of $p$, written $p = U S V^\mathrm{H}$, the
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

Orthogonally project $X ∈ 𝔽^{n × n}$ onto the tangent space of $p$ to the
[`SpecialLinear`](@ref) $G = \mathrm{SL}(n, 𝔽)$. The formula reads
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

function decorator_transparent_dispatch(::typeof(project), ::SpecialLinear, args...)
    return Val(:parent)
end

Base.show(io::IO, ::SpecialLinear{n,𝔽}) where {n,𝔽} = print(io, "SpecialLinear($n, $𝔽)")

translate_diff(::SpecialLinear, p, q, X, ::LeftAction) = X
translate_diff(::SpecialLinear, p, q, X, ::RightAction) = p \ X * p

function translate_diff!(G::SpecialLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end
