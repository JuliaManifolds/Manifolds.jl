
"""
    abstract type AbstractRetractionDiffArgumentMethod end

Abstract type for methods of computing differentials of retractions with respect to
the argument, for a fixed point.
"""
abstract type AbstractRetractionDiffArgumentMethod end

"""
    abstract type AbstractInverseRetractionDiffArgumentMethod end

Abstract type for methods of computing differentials of inverse retractions with respect to
the argument, for a fixed point.
"""
abstract type AbstractInverseRetractionDiffArgumentMethod end

"""
    FlatExpDiffArgumentMethod <: AbstractRetractionDiffArgumentMethod

Method of computing differential of the exponential map with respect to the argument
for flat manifolds.
"""
struct FlatExpDiffArgumentMethod <: AbstractInverseRetractionDiffArgumentMethod end

"""
    FlatLogDiffArgumentMethod <: AbstractRetractionDiffArgumentMethod

Method of computing differential of the exponential map with respect to the argument
for flat manifolds.
"""
struct FlatLogDiffArgumentMethod <: AbstractInverseRetractionDiffArgumentMethod end

"""
    FiniteDifferenceLogDiffArgumentMethod <: AbstractRetractionDiffArgumentMethod

Method of approximating differential of the logarithmic map with respect to the argument
using a finite difference-like scheme.
"""
struct FiniteDifferenceLogDiffArgumentMethod{
    TRetr<:AbstractRetractionMethod,
    TInvRetr<:AbstractInverseRetractionMethod,
    TH<:Real,
} <: AbstractInverseRetractionDiffArgumentMethod
    retr::TRetr
    invretr::TInvRetr
    h::TH
end

@doc raw"""
    inverse_retract_diff_argument(M::AbstractManifold, p, q, X, m::FiniteDifferenceLogDiffArgumentMethod)

Approximate the differential of the inverse retraction `m.invretr` using a finite difference
formula (see Eq. (16) in [^Zimmermann2019]):
```math
\frac{\operatorname{retr}^{-1}_q(\operatorname{retr}_p(hX)) - \operatorname{retr}^{-1}_q(\operatorname{retr}_p(-hX))}{2h}
```
where ``h`` is the finite difference step `m.h`, ``\operatorname{retr}^{-1}`` is the inverse
retraction `m.invretr` and ``\operatorname{retr}`` is the retraction `m.retr`.

[^Zimmermann2019]:
    > R. Zimmermann, “Hermite interpolation and data processing errors on Riemannian matrix
    > manifolds,” arXiv:1908.05875 [cs, math], Sep. 2019,
    > Available: http://arxiv.org/abs/1908.05875
"""
inverse_retract_diff_argument(
    M::AbstractManifold,
    p,
    q,
    X,
    ::FiniteDifferenceLogDiffArgumentMethod,
)

function inverse_retract_diff_argument!(
    M::AbstractManifold,
    Y,
    p,
    q,
    X,
    m::FiniteDifferenceLogDiffArgumentMethod,
)
    p_tmp = retract(M, p, m.h * X, m.retr)
    inverse_retract!(M, Y, q, p_tmp, m.invretr)
    retract!(M, p_tmp, p, -m.h * X, m.retr)
    X_tmp = inverse_retract(M, q, p_tmp, m.invretr)
    Y .= (Y .- X_tmp) ./ (2 .* h)
    return Y
end

"""
    retract_diff_argument(::AbstractManifold, p, X, Y, ::FlatExpDiffArgumentMethod)

Calculate differential of retraction with respect to argument on a flat manifold.
"""
retract_diff_argument(::AbstractManifold, p, X, Y, ::FlatExpDiffArgumentMethod) = Y

function retract_diff_argument!(
    M::AbstractManifold,
    Z,
    p,
    X,
    Y,
    ::FlatExpDiffArgumentMethod,
)
    return copyto!(M, Z, p, Y)
end

"""
    inverse_retract_diff_argument(::AbstractManifold, p, q, X, ::FlatLogDiffArgumentMethod)

Calculate differential of inverse retraction with respect to argument on a flat manifold.
"""
inverse_retract_diff_argument(::AbstractManifold, p, q, X, ::FlatLogDiffArgumentMethod) = X

function inverse_retract_diff_argument!(
    M::AbstractManifold,
    Y,
    p,
    q,
    X,
    ::FlatLogDiffArgumentMethod,
)
    return copyto!(M, Y, p, X)
end

@doc raw"""
    LieGroupExpDiffArgumentApprox(n::Int)

Approximate differential of exponential map based on Lie group exponential. See Theorem 1.7
of [^Helgason1978].

[^Helgason1978]:
    > S. Helgason, Differential Geometry, Lie Groups, and Symmetric Spaces, First Edition.
    > Academic Press, 1978.
"""
struct LieGroupExpDiffArgumentApprox <: AbstractRetractionDiffArgumentMethod
    n::Int
end

@doc raw"""
    retract_diff_argument(M::AbstractManifold, p, X, Y, ::LieGroupExpDiffArgumentApprox)

Approximate differential of exponential map based on Lie group exponential. The formula
reads (see Theorem 1.7 of [^Helgason1978])
```math
D_X \exp_{p}(X)[Y] = (\mathrm{d}L_{\exp_e(X)})_e\left(\sum_{k=0}^{n}\frac{(-1)^k}{(k+1)!}(\operatorname{ad}_X)^k(Y)\right)
```
where ``(\operatorname{ad}_X)^k(Y)`` is defined recursively as ``(\operatorname{ad}_X)^0(Y) = Y``,
``\operatorname{ad}_X^{k+1}(Y) = [X, \operatorname{ad}_X^k(Y)]``.

[^Helgason1978]:
    > S. Helgason, Differential Geometry, Lie Groups, and Symmetric Spaces, First Edition.
    > Academic Press, 1978.
"""
retract_diff_argument(M::AbstractManifold, p, X, Y, ::LieGroupExpDiffArgumentApprox)

function retract_diff_argument!(
    M::AbstractManifold,
    Z,
    p,
    X,
    Y,
    m::LieGroupExpDiffArgumentApprox,
)
    tmp = copy(M, p, Y)
    a = -1.0
    zero_vector!(M, Z, p)
    for k in 0:(m.n)
        a *= -1 // (k + 1)
        Z .+= a .* tmp
        if k < m.n
            copyto!(tmp, lie_bracket(M, X, tmp))
        end
    end
    q = exp(M, p, X)
    translate_diff!(M, Z, q, Identity(M), Z)
    return Z
end

"""
    default_retract_diff_argument_method(M::AbstractManifold, retraction::AbstractRetractionMethod)

The [`AbstractRetractionDiffArgumentMethod`](@ref) that is used when calling
[`retract_diff_argument`](@ref) without specifying the method.
"""
default_retract_diff_argument_method(
    M::AbstractManifold,
    retraction::AbstractRetractionMethod,
)

"""
    default_inverse_retract_diff_argument_method(M::AbstractManifold, inverse_retraction::AbstractInverseRetractionMethod)

The [`AbstractRetractionDiffArgumentMethod`](@ref) that is used when calling
[`inverse_retract_diff_argument`](@ref) without specifying the method.
"""
default_retract_diff_argument_method(
    M::AbstractManifold,
    inverse_retraction::AbstractInverseRetractionMethod,
)

@doc raw"""
    exp_diff_argument(M::AbstractManifold, p, X, Y)

Compute differential of the exponential map with respect to the argument for a fixed
base point `p`. The differential of function ``\exp_p: (T_p\mathcal M) → \mathcal M``
is a function ``D \exp_p: T_X(T_p\mathcal M) → T_{\exp_p X}\mathcal M``.

Note that through the isomorphism ``Y ∈ T_X(T_p\mathcal M) = T_p\mathcal M`` the argument
`Y` is still a tangent vector.
"""
function exp_diff_argument(M::AbstractManifold, p, X, Y)
    return retract_diff_argument(
        M,
        p,
        X,
        Y,
        default_retract_diff_argument_method(M, ExponentialRetraction()),
    )
end

function exp_diff_argument!(M::AbstractManifold, Z, p, X, Y)
    return retract_diff_argument!(
        M,
        Z,
        p,
        X,
        Y,
        default_retract_diff_argument_method(M, ExponentialRetraction()),
    )
end

@doc raw"""
    retract_diff_argument(M::AbstractManifold, p, X, Y, m::AbstractRetractionMethod)

Compute differential of the retraction `m` with respect to the argument for a fixed
base point `p`. The differential of function ``\operatorname{retr}_p: (T_p\mathcal M) → \mathcal M``
is a function ``D \operatorname{retr}_p: T_X(T_p\mathcal M) → T_{\operatorname{retr}_p X}\mathcal M``.

Note that through the isomorphism ``Y ∈ T_X(T_p\mathcal M) = T_p\mathcal M`` the argument
`Y` is still a tangent vector.
"""
function retract_diff_argument(M::AbstractManifold, p, X, Y)
    return retract_diff_argument(M, p, X, Y, default_retraction_method(M))
end
function retract_diff_argument(M::AbstractManifold, p, X, Y, m::AbstractRetractionMethod)
    return retract_diff_argument(M, p, X, Y, default_retract_diff_argument_method(M, m))
end
function retract_diff_argument(
    M::AbstractManifold,
    p,
    X,
    Y,
    m::AbstractRetractionDiffArgumentMethod,
)
    Z = allocate_result(M, retract_diff_argument, Y, X)
    return retract_diff_argument!(M, Z, p, X, Y, m)
end

function retract_diff_argument!(M::AbstractManifold, Z, p, X, Y)
    return retract_diff_argument!(M, Z, p, X, Y, default_retraction_method(M))
end
function retract_diff_argument!(
    M::AbstractManifold,
    Z,
    p,
    X,
    Y,
    m::AbstractRetractionMethod,
)
    return retract_diff_argument!(M, Z, p, X, Y, default_retract_diff_argument_method(M, m))
end

@doc raw"""
    inverse_retract_diff_argument(M::AbstractManifold, p, q, X, m::AbstractInverseRetractionMethod)

Compute differential of the inverse retraction `m` with respect to the argument for a fixed
base point `p`. The differential of function ``\operatorname{retr}^{-1}_p: \mathcal M → T_p\mathcal M``
is a function ``D \operatorname{retr}^{-1}_p: T_p M → T_{\operatorname{retr}^{-1}_p q}T_p\mathcal M``.

Note that through the isomorphism ``X ∈ T_{\operatorname{retr}^{-1}_p q}(T_p\mathcal M) = T_p \mathcal M``
the argument `X` is still a tangent vector.
"""
function inverse_retract_diff_argument(
    M::AbstractManifold,
    p,
    q,
    X,
    m::AbstractInverseRetractionMethod,
)
    return inverse_retract_diff_argument(
        M,
        p,
        q,
        X,
        default_inverse_retract_diff_argument_method(M, m),
    )
end

function inverse_retract_diff_argument!(
    M::AbstractManifold,
    Y,
    p,
    q,
    X,
    m::AbstractInverseRetractionMethod,
)
    return inverse_retract_diff_argument!(
        M,
        Y,
        p,
        q,
        X,
        default_inverse_retract_diff_argument_method(M, m),
    )
end
