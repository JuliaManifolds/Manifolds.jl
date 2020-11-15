@doc raw"""
    GeneralLinear{n,𝔽} <: AbstractGroupManifold{𝔽,MultiplicationOperation}

The general linear group, that is, the group of all invertible matrices in $𝔽^{n×n}$.

The default metric is the left-$\mathrm{GL}(n)$-right-$\mathrm{O}(n)$-invariant metric whose
inner product is written
$$⟨X_p,Y_p⟩_p = ⟨p^{-1}X_p,p^{-1}Y_p⟩_\mathrm{F} = ⟨X_e, Y_e⟩_\mathrm{F},$$
where $X_e = p^{-1}X_p ∈ 𝔤l(n) = T_e \mathrm{GL}(n, 𝔽) = 𝔽^{n×n}$ is the corresponding
vector in the Lie algebra. In the default implementations, all tangent vectors $X_p$ are
instead represented with their corresponding Lie algebra vectors.

[^MartinNeff2016]:
    > Martin, R. J. and Neff, P.:
    > “Minimal geodesics on GL(n) for left-invariant, right-O(n)-invariant Riemannian metrics”,
    > Journal of Geometric Mechanics 8(3), pp. 323-357, 2016.
    > doi: [10.3934/jgm.2016010](https://doi.org/10.3934/jgm.2016010),
    > arXiv: [1409.7849v2](https://arxiv.org/abs/1409.7849v2).
[^AndruchowLarotondaRechtVarela2014]:
    > Andruchow E., Larotonda G., Recht L., and Varela A.:
    > “The left invariant metric in the general linear group”,
    > Journal of Geometry and Physics 86, pp. 241-257, 2014.
    > doi: [10.1016/j.geomphys.2014.08.009](https://doi.org/10.1016/j.geomphys.2014.08.009),
    > arXiv: [1109.0520v1](https://arxiv.org/abs/1109.0520v1).
"""
struct GeneralLinear{n,𝔽} <: AbstractGroupManifold{𝔽,MultiplicationOperation} end

GeneralLinear(n, 𝔽::AbstractNumbers = ℝ) = GeneralLinear{n,𝔽}()

function allocation_promotion_function(::GeneralLinear{n,ℂ}, f, ::Tuple) where {n}
    return complex
end

function check_manifold_point(G::GeneralLinear{n,𝔽}, p; kwargs...) where {n,𝔽}
    mpv = check_manifold_point(Euclidean(n, n; field = 𝔽); kwargs...)
    mpv === nothing || return mpv
    detp = det(p)
    if iszero(detp)
        return DomainError(
            detp,
            "The matrix $(p) does not lie on $(G), since it is not invertible.",
        )
    end
    return nothing
end

function check_tangent_vector(
    G::GeneralLinear{n,𝔽},
    p,
    X;
    check_base_point = true,
    kwargs...,
) where {n,𝔽}
    if check_base_point
        mpe = check_manifold_point(G, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = check_tangent_vector(Euclidean(n, n; field = 𝔽), p, X; kwargs...)
    mpv === nothing || return mpv
    return nothing
end

decorator_transparent_dispatch(::typeof(exp), ::GeneralLinear, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(retract!), ::GeneralLinear, args...) = Val(:parent)
decorator_transparent_dispatch(::typeof(log), ::GeneralLinear, args...) = Val(:parent)

function default_metric_dispatch(
    ::GeneralLinear{n,ℝ},
    ::LeftInvariantMetric{EuclideanMetric},
) where {n}
    return Val(true)
end

function exp!(G::GeneralLinear, q, p, X)
    expX = exp(X)
    compose!(G, q, expX', exp(X - X'))
    compose!(G, q, p, q)
    return q
end
function exp!(::GeneralLinear{1}, q, p, X)
    p1 = p isa Identity ? p : p[1]
    q[1] = p1 * exp(X[1])
    return q
end
function exp!(G::GeneralLinear{2}, q, p, X)
    A = SizedMatrix{2,2}(X')
    B = SizedMatrix{2,2}(X) - A
    compose!(G, q, exp(A), exp(B))
    compose!(G, q, p, q)
    return q
end

flat!(::GeneralLinear, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

get_coordinates(::GeneralLinear{n,ℝ}, p, X, ::DefaultOrthonormalBasis) where {n} = vec(X)

function get_coordinates!(
    ::GeneralLinear{n,ℝ},
    Xⁱ,
    p,
    X,
    ::DefaultOrthonormalBasis,
) where {n}
    return copyto!(Xⁱ, X)
end

function get_vector(::GeneralLinear{n,ℝ}, p, Xⁱ, ::DefaultOrthonormalBasis) where {n}
    return reshape(Xⁱ, n, n)
end

function get_vector!(::GeneralLinear{n,ℝ}, X, p, Xⁱ, ::DefaultOrthonormalBasis) where {n}
    return copyto!(X, Xⁱ)
end

function group_exp!(::GeneralLinear{1}, q, X)
    q[1] = exp(X[1])
    return q
end
group_exp!(::GeneralLinear{2}, q, X) = copyto!(q, exp(SizedMatrix{2,2}(X)))

function group_log!(::GeneralLinear{1}, X, p)
    X[1] = log(p[1])
    return X
end

inner(::GeneralLinear, p, X, Y) = dot(X, Y)

invariant_metric_dispatch(::GeneralLinear, ::LeftAction) = Val(true)

inverse_translate(::GeneralLinear, p, q, ::LeftAction) = p \ q
inverse_translate(::GeneralLinear, p, q, ::RightAction) = q / p

inverse_translate_diff(::GeneralLinear, p, q, X, ::LeftAction) = X
inverse_translate_diff(::GeneralLinear, p, q, X, ::RightAction) = p * X / p

function inverse_translate_diff!(G::GeneralLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, inverse_translate_diff(G, p, q, X, conv))
end

function _log_project_SOn_S⁺!(X, q, n = size(q, 1))
    F = svd(q)
    d = allocate(q, n)
    s = mean(F.S)
    fill!(d, s)
    d[n] = det(F.U) * det(F.Vt) * s
    expX = F.U * Diagonal(d) * F.Vt
    return copyto!(X, eltype(X) <: Real ? real(log_safe(expX)) : log_safe(expX))
end

function log!(G::GeneralLinear{n}, X, p, q) where {n}
    pinvq = inverse_translate(G, p, q, LeftAction())
    if isnormal(pinvq; atol = sqrt(eps(real(eltype(pinvq)))))
        copyto!(X, log_safe(pinvq))
    else
        𝔽 = number_system(G)
        if 𝔽 === ℝ
            e = Identity(G, pinvq)
            _log_project_SOn_S⁺!(X, pinvq, n)
            inverse_retraction = ApproximateInverseRetraction(ExponentialRetraction())
            inverse_retract!(G, X, e, pinvq, inverse_retraction; X0 = X)
        else
            # compute the equivalent logarithm on GL(dim(𝔽) * n, ℝ)
            Gᵣ = GeneralLinear(real_dimension(𝔽) * n, ℝ)
            pinvqᵣ = realify(pinvq, 𝔽)
            Xᵣ = realify(X, 𝔽)
            eᵣ = Identity(Gᵣ, pinvqᵣ)
            log!(Gᵣ, Xᵣ, eᵣ, pinvqᵣ)
            unrealify!(X, Xᵣ, 𝔽, n)
        end
    end
    return X
end
function log!(::GeneralLinear{1}, X, p, q)
    p1 = p isa Identity ? p : p[1]
    X[1] = p1 * log(q[1])
    return X
end

function manifold_dimension(::GeneralLinear{n,𝔽}) where {n,𝔽}
    return manifold_dimension(Euclidean(n, n; field = 𝔽))
end

LinearAlgebra.norm(::GeneralLinear, p, X) = norm(X)

project!(::GeneralLinear, Y, p, X) = copyto!(Y, X)

@generated representation_size(::GeneralLinear{n}) where {n} = (n, n)

sharp!(::GeneralLinear{n}, X::TFVector, p, ξ::CoTFVector) where {n} = copyto!(X, ξ)

Base.show(io::IO, ::GeneralLinear{n,𝔽}) where {n,𝔽} = print(io, "GeneralLinear($n, $𝔽)")

translate_diff(::GeneralLinear, p, q, X, ::LeftAction) = X
translate_diff(::GeneralLinear, p, q, X, ::RightAction) = p \ X * p

function translate_diff!(G::GeneralLinear, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end

zero_tangent_vector(::GeneralLinear, p) = zero(p)

zero_tangent_vector!(::GeneralLinear, X, p) = fill!(X, 0)
