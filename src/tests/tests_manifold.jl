"""
    test_exp_log(M, p; kwargs...)
    test_exp_log(M, p, X; kwargs...)

Test suite for exponential and logarithmic map, where

* `M` is an `AbstractManifold`
* `p` is a point or a `Tuple` of points
* `X` is either provided or computed (using `log`)

## Keyword Arguments
* `minimal` (`false`) – whether to perform a minimal test (only the first elements of `p` and `X`) or not
* `in_place`(`true`) — whether to test the in-place function as well
- `atol_multiplier = 0`: change absolute tolerance in comparisons
    (0 use default, i.e. deactivate atol and use rtol).
- `rtol_multiplier = 1`: change the relative tolerance of exp/log tests
    (1 use default). This is deactivated if the `exp_log_atol_multiplier` is nonzero.
*
*
*
"""
function test_explog(M, p::NTuple{N,P}; kwargs...) where {N,P}
    if N == 1
        X = Tuple(zero_vector(M, p[1]))
    else
        p1 = p[1:(end - 1)]
        p2 = p[2:end]
        X = Tuple([[log(M, pi, pj) for (pi, pj) in zip(p1, p2)]..., zero_vector(M, p[end])])
    end
    return test_explog(M, p, X; kwargs...)
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
    if minimal # Minimal: just check the very fist one
        p = (p[1],)
        X = (X[1],)
    end
    Test.@testset "Testing exp/log on $M" begin
        for (pi, Xi) in zip(p, X)
            qi = exp(M, pi, Xi)
            Yi = log(M, pi, qi)
            Test.@test is_point(M, qi; kwargs...)
            Test.@test isapprox(M, pi, Xi, Yi; atol=atp, rtol=rtp)

            Test.@test isapprox(M, pi, exp(M, pi, Xi, 0); atol=atp, rtol=rtp)
            Test.@test isapprox(M, qi, exp(M, pi, Xi, 1); atol=atp, rtol=rtp)
            Test.@test isapprox(
                M,
                pi,
                log(M, pi, pi),
                zero_vector(M, pi);
                atol=atp,
                rtol=rtp,
            )
            if in_place
                ri = allocate(M, pi)
                exp!(M, ri, pi, Xi)
                Test.@test isapprox(M, ri, qi; atol=atp, rtol=rtp)
                Zi = zero_vector(M, pi)
                log!(M, Zi, pi, qi)
                Test.@test isapprox(M, pi, Xi, Yi; atol=atp, rtol=rtp)
            end
        end
    end
end
