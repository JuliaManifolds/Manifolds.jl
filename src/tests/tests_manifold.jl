@doc raw"""
    (p,X) = generate_tangent_vectors(M, p, m=default_inverse_retraction_method(M))

Generate a set `X` of tangent vectors on the manifold `M`
using the point tuple `p` of `n` points and the `AbstractInverseRetractionMethod` `m`.

The generated vectors are

```math
    \begin{align*}
    X_i &= \operatorname{retr}_{p_i}^{-1}p_{i+1},\qquad i=1,...n-1\\
    X_n &= 0_{p_n},
    \end{align*}
```
where ``0_p ∈ T_p\mathcal M`` denotes the zeros vector.

## keywords

* `minimal` (`false`) – reduce the ``X_i`` and ``p_i`` to just a tuple of length 1.

## Returns

* (`p`, `X`) – a tuple of points and corresponding tangent vectors

"""
function generate_tangent_vectors(
    M::AbstractManifold,
    p::NTuple{N,P},
    m::AbstractInverseRetractionMethod;
    minimal=false,
) where {N,P}
    if N == 1
        X = Tuple(zero_vector(M, p[1]))
    else
        if minimal # only compute first
            X = (inverse_retract(M, p[1], p[2], m),)
            return ((p[1],), (X,))
        end
        p1 = p[1:(end - 1)]
        p2 = p[2:end]
        X = Tuple([
            [inverse_retract(M, pi, pj, m) for (pi, pj) in zip(p1, p2)]...,
            zero_vector(M, p[end]),
        ])
    end
    return (p, X)
end

"""
    test_explog(M, p; kwargs...)
    test_explog(M, p, X; kwargs...)

Test suite for exponential and logarithmic map, where

* `M` is an `AbstractManifold`
* `p` is a point or a `Tuple` of points
* `X` is either provided or computed (using `log`)

## Keyword Arguments
* `minimal` (`false`) – whether to perform a minimal test (only the first elements of `p` and `X`) or not
* `in_place`(`true`) — whether to test the in-place function as well
* `atol_multiplier = 0`: change absolute tolerance in comparisons
    (0 use default, i.e. deactivate atol and use rtol).
* `rtol_multiplier = 1`: change the relative tolerance of exp/log tests
    (1 use default). This is deactivated if the `exp_log_atol_multiplier` is nonzero.

All further keyword arguments are passed down to both `is_point` and `is_vector` calls.
"""
function test_explog(M, p::NTuple{N,P}; minimal=false, kwargs...) where {N,P}
    (pL, X) = generate_tangent_vectors(M, p; minimal=minimal)
    return test_explog(M, pL, X; minimal=minimal, kwargs...)
end
function test_explog(
    M,
    p::NTuple{N,P},
    X::NTuple{N,T};
    minimal=false,
    in_place=true,
    atol_multiplier=0,
    rtol_multiplier=1,
    kwargs...,
) where {N,T,P}
    epsFp = find_eps(first(p))
    atp = atol_multiplier * epsFp
    rtp = atol_multiplier == 0.0 ? sqrt(epsFp) * rtol_multiplier : 0
    pL = minimal ? (p[1],) : p
    XL = minimal ? (X[1],) : X
    kw = (
        minimal=minimal,
        in_place=in_place,
        atol_multiplier=atol_multiplier,
        rtol_multiplier=rtol_multiplier,
        kwargs...,
    )
    test_exp(M, pL, XL; kw...)
    test_log(M, pL, XL; kw...)
    if in_place
        ri = allocate(M, first(pL))
        Zi = zero_vector(M, first(pL))
    end
    Test.@testset "Testing exp/log on $M" begin
        for (pi, Xi) in zip(pL, XL)
            qi = exp(M, pi, Xi)
            Yi = log(M, pi, qi)
            Test.@test isapprox(M, pi, Xi, Yi; atol=atp, rtol=rtp)
            if in_place
                exp!(M, ri, pi, Xi)
                Test.@test isapprox(M, ri, qi; atol=atp, rtol=rtp)
                log!(M, Zi, pi, qi)
                Test.@test isapprox(M, pi, Zi, Yi; atol=atp, rtol=rtp)
            end
        end
    end
end
"""

"""
function test_exp(M, p::NTuple{N,P}; minimal=false, kwargs...) where {N,P}
    (pL, X) = generate_tangent_vectors(M, p; minimal=minimal)
    return test_exp(M, pL, X; minimal=minimal, kwargs...)
end
function test_exp(
    M,
    p::NTuple{N,P},
    X::NTuple{N,T};
    minimal=false,
    in_place=true,
    atol_multiplier=0,
    rtol_multiplier=1,
    kwargs...,
) where {N,T,P}
    epsFp = find_eps(first(p))
    atp = atol_multiplier * epsFp
    rtp = atol_multiplier == 0.0 ? sqrt(epsFp) * rtol_multiplier : 0
    if minimal # Minimal: just check the very fist one
        pL = (p[1],)
        XL = (X[1],)
    else
        pL = p
        XL = X
    end
    in_place && (ri = allocate(M, first(pL)))
    Test.@testset "Testing exp on $M" begin
        for (pi, Xi) in zip(pL, XL)
            qi = exp(M, pi, Xi)
            Test.@test is_point(M, qi, true; atol=atp, rtol=rtp, kwargs...)
            Test.@test isapprox(M, pi, exp(M, pi, Xi, 0); atol=atp, rtol=rtp)
            Test.@test isapprox(M, qi, exp(M, pi, Xi, 1); atol=atp, rtol=rtp)
            if in_place
                exp!(M, ri, pi, Xi)
                Test.@test isapprox(M, ri, qi; atol=atp, rtol=rtp)
            end
        end
    end
end
"""

"""
function test_log(M, p::NTuple{N,P}; minimal=false, kwargs...) where {N,P}
    (pL, X) = generate_tangent_vectors(M, p; minimal=minimal)
    return test_log(M, pL, X; minimal=minimal, kwargs...)
end
function test_log(
    M,
    p::NTuple{N,P},
    X::NTuple{N,T};
    minimal=false,
    in_place=true,
    atol_multiplier=0,
    rtol_multiplier=1,
    kwargs...,
) where {N,T,P}
    epsFp = find_eps(first(p))
    atp = atol_multiplier * epsFp
    rtp = atol_multiplier == 0.0 ? sqrt(epsFp) * rtol_multiplier : 0
    if minimal # Minimal: just check the very fist one
        pL = (p[1],)
        qL = (X[1],)
    else
        pL = length(p) > 1 ? p[1:(end - 1)] : p
        qL = length(p) > 1 ? p[2:end] : p
    end
    in_place && (Yi = zero_vector(M, first(pL)))
    Test.@testset "Testing log on $M" begin
        for (pi, qi) in zip(pL, qL)
            Xi = log(M, pi, qi)
            Test.@test is_vector(M, pi, Xi, true; atol=atp, rtol=rtp, kwargs...)
            Test.@test isapprox(
                M,
                pi,
                log(M, pi, pi),
                zero_vector(M, pi);
                atol=atp,
                rtol=rtp,
            )
            if in_place
                log!(M, Yi, pi, qi)
                Test.@test isapprox(M, pi, Xi, Yi; atol=atp, rtol=rtp)
            end
        end
    end
end
"""
"""
function test_retr_and_inv(
    M,
    p,
    m::Tuple{R,I};
    kwargs...,
) where {R<:AbstractRetractionMethod,I<:AbstractInverseRetractionMethod}
    X = generate_tangent_vectors(M, p, m[2])
    return test_retr_and_inv(M, p, X, m; kwargs...)
end
function test_retr_and_inv(
    M,
    p,
    X,
    m::Tuple{R,I};
    minimal=false,
    in_place=true,
    atol_multiplier=0,
    rtol_multiplier=1,
    kwargs...,
) where {R<:AbstractRetractionMethod,I<:AbstractInverseRetractionMethod}
    epsFp = find_eps(first(p))
    atp = atol_multiplier * epsFp
    rtp = atol_multiplier == 0.0 ? sqrt(epsFp) * rtol_multiplier : 0
    pL = minimal ? (p[1],) : p
    XL = minimal ? (X[1],) : X
    kw = (
        minimal=minimal,
        in_place=in_place,
        atol_multiplier=atol_multiplier,
        rtol_multiplier=rtol_multiplier,
        kwargs...,
    )
    test_retr(M, p, X, m[1]; kw...)
    test_inv_retr(M, p, X, m[2]; kw...)
    if in_place
        ri = allocate(M, first(pL))
        Zi = zero_vector(M, first(pL))
    end
    Test.@testset "Testing $(M[1])/$(m[2]) on $M" begin
        for (pi, Xi) in zip(pL, XL)
            qi = retract(M, pi, Xi, m[1])
            Yi = inverse_retract(M, pi, qi, m[2])
            Test.@test isapprox(M, pi, Xi, Yi; atol=atp, rtol=rtp)
            if in_place
                retract!(M, ri, pi, Xi, m[1])
                Test.@test isapprox(M, ri, qi; atol=atp, rtol=rtp)
                inverse_retract!(M, Zi, pi, qi, m[2])
                Test.@test isapprox(M, pi, Zi, Yi; atol=atp, rtol=rtp)
            end
        end
    end
end
"""

"""
function test_inv_retr(M, p, m::AbstractInverseRetractionMethod; kwargs...)
    X = generate_tangent_vectors(M, p, m)
    return test_retr(M, p, X, m; kwargs...)
end
function test_inv_retr(
    M,
    p,
    X,
    m::AbstractInverseRetractionMethod;
    minimal=false,
    in_place=true,
    atol_multiplier=0,
    rtol_multiplier=1,
    kwargs...,
)
    epsFp = find_eps(first(p))
    atp = atol_multiplier * epsFp
    rtp = atol_multiplier == 0.0 ? sqrt(epsFp) * rtol_multiplier : 0
    if minimal # Minimal: just check the very fist one
        pL = (p[1],)
        qL = (X[1],)
    else
        pL = length(p) > 1 ? p[1:(end - 1)] : p
        qL = length(p) > 1 ? p[2:end] : p
    end
    in_place && (Yi = zero_vector(M, first(pL)))
    Test.@testset "Testing $m on $M" begin
        for (pi, qi) in zip(pL, qL)
            Xi = inverse_retract(M, pi, qi, m)
            Test.@test is_vector(M, pi, Xi, true; atol=atp, rtol=rtp, kwargs...)
            Test.@test isapprox(
                M,
                pi,
                inverse_retract(M, pi, pi, m),
                zero_vector(M, pi);
                atol=atp,
                rtol=rtp,
            )
            if in_place
                inverse_retract!(M, Yi, pi, qi, m)
                Test.@test isapprox(M, pi, Xi, Yi; atol=atp, rtol=rtp)
            end
        end
    end
end
"""

"""
function test_retr(
    M,
    p,
    m::AbstractRetractionMethod;
    inverse_retraction_method=default_inverse_retraction_method(M),
    kwargs...,
)
    X = generate_tangent_vectors(M, p, inverse_retraction_method)
    return test_retr(M, p, X, m; kwargs...)
end
function test_retr(
    M,
    p,
    X,
    m::AbstractRetractionMethod;
    minimal=false,
    in_place=true,
    atol_multiplier=0,
    rtol_multiplier=1,
    kwargs...,
)
    epsFp = find_eps(first(p))
    atp = atol_multiplier * epsFp
    rtp = atol_multiplier == 0.0 ? sqrt(epsFp) * rtol_multiplier : 0
    if minimal # Minimal: just check the very fist one
        pL = (p[1],)
        XL = (X[1],)
    else
        pL = p
        XL = X
    end
    ri = allocate(M, first(pL))
    Test.@testset "Testing $m on $M" begin
        for (pi, Xi) in zip(pL, XL)
            qi = retract(M, pi, Xi, m)
            Test.@test is_point(M, qi, true; atol=atp, rtol=rtp, kwargs...)
            Test.@test isapprox(M, pi, retract(M, pi, Xi, 0, m); atol=atp, rtol=rtp)
            Test.@test isapprox(M, qi, retract(M, pi, Xi, 1, m); atol=atp, rtol=rtp)
            if in_place
                retract!(M, ri, pi, Xi, m)
                Test.@test isapprox(M, qi, ri; atol=atp, rtol=rtp)
            end
        end
    end
end
