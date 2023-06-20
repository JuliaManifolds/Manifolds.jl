#
# Test suite for exponential map
#

"""
    test_exp(M, p; kwargs...)

Call [`generate_tangent_vectors`](@ref) and call the test with tangent vectors.
all kwargs are passt on to the `test_exp(M,p,X)`.
"""
function test_exp(M, p::NTuple{N,P}; minimal=false, kwargs...) where {N,P}
    (pL, X) = generate_tangent_vectors(M, p; minimal=minimal)
    return test_exp(M, pL, X; minimal=minimal, kwargs...)
end
"""
    test_exp(M, p, X)

Perform generic tests for the exponential map for each pair of point and tangent vector
from zip(p,X).

# Keyword arguments

* `minimal` _ (`false`) – perform a minimal test only on `p[1]` and `X[1]`.
* `in_place` – (`true`) test `exp!`
* `in_place_self` – (`true`) test that `exp!(M, p, p, X)` works (i.e. has no side effects)
* `atol` _ (`0`) `atol` of `isapprox`checks in this test
* `rtol` - (`1` has no effect, if `atol_multiplier>0`) modify the default `rtol` of `isapprox by a factor.
"""
function test_exp(
    M,
    p::NTuple{N,P},
    X::NTuple{N,T};
    minimal=false,
    in_place=true,
    in_place_self=false,
    atol=0,
    rtol=atol > 0 ? 0 : sqrt(eps),
    kwargs...,
) where {N,T,P}
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
            Test.@test is_point(M, qi, true; atol=atol, rtol=rtol, kwargs...)
            Test.@test isapprox(M, pi, exp(M, pi, Xi, 0); atol=atol, rtol=rtol)
            Test.@test isapprox(M, qi, exp(M, pi, Xi, 1); atol=atol, rtol=rtol)
            if in_place
                exp!(M, ri, pi, Xi)
                Test.@test isapprox(M, ri, qi; atol=atol, rtol=rtol)
                #test self in_place has no side effects
                if in_place_self
                    ri = copy(M, pi)
                    exp!(M, ri, ri, Xi)
                    Test.@test isapprox(M, ri, qi; atol=atol, rtol=rtol)
                end
            end
        end
    end
end
"""
    test_exp(M, p; kwargs...)

Call [`generate_tangent_vectors`](@ref) and call the test with tangent vectors.
all kwargs are passt on to the `test_log(M,p,X)`.
"""
function test_log(M, p::NTuple{N,P}; minimal=false, kwargs...) where {N,P}
    (pL, X) = generate_tangent_vectors(M, p; minimal=minimal)
    return test_log(M, pL, X; minimal=minimal, kwargs...)
end

#
# Test suite for exp _and_ log, that they are inverses of each other
#

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
    in_place_self=false,
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
    test_exp(M, pL, XL; in_place_self=in_place_self, kw...)
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

#
# Test suite for is_point and is_vector
#
"""
    test_is_point(M, points, non_points; kwargs...)

Test suite for checking points, where

* `M` is an `AbstractManifold`
* `points` is a `Tuple` of points
* `non_points` is a `Tuple` of “points” that issue each of the errors a point can have.
  these are expected to throw `DomainErrors`.

## Keyword Arguments
* `minimal` (`false`) – whether to perform a minimal test (only the first elements of `p` and `X`) or not
* `atol_multiplier = 0`: change absolute tolerance in comparisons
    (0 use default, i.e. deactivate atol and use rtol).
* `rtol_multiplier = 1`: change the relative tolerance of exp/log tests
    (1 use default). This is deactivated if the `exp_log_atol_multiplier` is nonzero.

All further keyword arguments are passed down to both `is_point` and `is_vector` calls.
"""
function test_is_point(
    M,
    points,
    non_points;
    error=DomainError,
    errors=fill(error, length(non_points)),
    kwargs...,
)
    Test.@testset "Testing is_point on $M" begin
        for p in points # test that they are points, these also error if this is not the case
            Test.@test is_point(M, p, true; kwargs...)
        end
        for (p, e) in zip(non_points, errors)
            Test.@test_throws "$e" is_point(M, p, true; kwargs...)
        end
    end
end

"""
    test_is_vector(M, points, vectors, non_vectors, non_points; kwargs...)

Test suite for checking points, where

* `M` is an `AbstractManifold`
* `points` is a `Tuple` of points
* `vectors` is a `Tuple` of tangent vectors, same length (or less) as `points` and such that
  the `i`th vector is a tangent vector at the `i`th point
* `non_points` is a `Tuple` of “points” that issue each of the errors a point can have.
  and are used with the correspondin vectors avobe to thech `check_point=true`.
  these have to be less or equally many as `vectors`
* `non_vectors` is a `Tuple` of “vectors” that issue each of the errors a tangent vector can have.
  The check for the `i`th vector is done in the `i`th point from above

## Keyword Arguments
* `minimal` (`false`) – whether to perform a minimal test (only the first elements of `p` and `X`) or not
* `atol_multiplier = 0`: change absolute tolerance in comparisons
    (0 use default, i.e. deactivate atol and use rtol).
* `rtol_multiplier = 1`: change the relative tolerance of exp/log tests
    (1 use default). This is deactivated if the `exp_log_atol_multiplier` is nonzero.

All further keyword arguments are passed down to both `is_point` and `is_vector` calls.
"""
function test_is_vector(
    M,
    points,
    vectors,
    non_vectors,
    non_points;
    error=DomainError,
    errors=fill(error, length(non_vectors)),
    kwargs...,
)
    Test.@testset "Testing is_vector on $M" begin
        for (p, X) in zip(points, vectors)  # test that they are points, these also error if this is not the case
            Test.@test is_vector(M, p, X, true; kwargs...)
        end
        for (np, X) in zip(non_points, vectors) # check that the point errors yield  false
            Test.@test !is_vector(M, np, X, false, true; kwargs...)
        end
        for (p, nX, e) in zip(points, non_vectors, errors) #check that the point check errors
            Test.@test_throws e is_vector(M, p, nX, true; kwargs...)
        end
    end
end

#
# Test Suite for the logarithmic map
#

"""
    test_log(M, p, X)

Perform generic tests for the logarithmic map for each pair of point and tangent vector
from zip(p,X).

# Keyword arguments

* `minimal` _ (`false`) – perform a minimal test only on `p[1]` and `X[1]`.
* `in_place` – (`true`) test `exp!`
* `atol_multiplier` _ (`0`) modify the default `atol` of `isapprox by a factor.
* `rtol_multiplier` - (`1` has no effect, if `atol_multiplier>0`) modify the default `rtol` of `isapprox by a factor.
"""
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
    rtp = atol_multiplier == 0.0 ? sqrt(epsFp) * rtol_multiplier : 0.0
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
    test_retr_and_inv(M, p; kwargs...)

Call [`generate_tangent_vectors`](@ref) and call the test with tangent vectors.
all kwargs are passt on to the `test_retr_and_inv(M,p,X)`.
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
    test_inv_retr(M, p; kwargs...)

Call [`generate_tangent_vectors`](@ref) and call the test with tangent vectors.
all kwargs are passt on to the `test_int_retr(M,p,X)`.
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
    test_retr(M, p; kwargs...)

Call [`generate_tangent_vectors`](@ref) and call the test with tangent vectors.
all kwargs are passt on to the `test_retr(M,p,X)`.
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
