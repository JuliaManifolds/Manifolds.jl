struct SpecialLinear{n,𝔽} <:
       AbstractGroupManifold{𝔽,MultiplicationOperation,TransparentIsometricEmbedding} end

SpecialLinear(n, 𝔽::AbstractNumbers = ℝ) = SpecialLinear{n,𝔽}()

function allocation_promotion_function(::SpecialLinear{n,ℂ}, f, args::Tuple) where {n}
    return complex
end

function check_manifold_point(G::SpecialLinear{n,𝔽}, p; kwargs...) where {n,𝔽}
    mpv = check_manifold_point(Euclidean(n, n; field = 𝔽), p; kwargs...)
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

function check_tangent_vector(G::SpecialLinear, p, X; check_base_point = true, kwargs...)
    if check_base_point
        mpe = check_manifold_point(G, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = check_tangent_vector(
        decorated_manifold(G),
        p,
        X;
        check_base_point = false,
        kwargs...,
    )
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

Base.show(io::IO, ::SpecialLinear{n,𝔽}) where {n,𝔽} = print(io, "SpecialLinear($n, $𝔽)")

translate_diff(::SpecialLinear, p, q, X, ::LeftAction) = X
translate_diff(::SpecialLinear, p, q, X, ::RightAction) = p \ X * p

function translate_diff!(G::SpecialLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end
