@doc raw"""
    SymmetricPositiveSemidefiniteFixedRank{T,𝔽} <: AbstractDecoratorManifold{𝔽}

The [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)  `` \operatorname{SPS}_k(n)`` consisting of the real- or complex-valued
symmetric positive semidefinite matrices of size ``n×n`` and rank ``k``, i.e. the set

````math
\operatorname{SPS}_k(n) = \bigl\{
p  ∈ 𝔽^{n×n}\ \big|\ p^{\mathrm{H}} = p,
apa^{\mathrm{H}} \geq 0 \text{ for all } a ∈ 𝔽
\text{ and } \operatorname{rank}(p) = k\bigr\},
````
where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transpose,
and the field ``𝔽 ∈ \{ ℝ, ℂ\}``.
We sometimes ``\operatorname{SPS}_{k,𝔽}(n)``, when distinguishing the real- and complex-valued
manifold is important.

An element is represented by ``q ∈ 𝔽^{n×k}`` from the factorization ``p = qq^{\mathrm{H}}``.
Note that since for any unitary (orthogonal) ``A ∈ 𝔽^{n×n}`` we have
``(Aq)(Aq)^{\mathrm{H}} = qq^{\mathrm{H}} = p``, the representation is not unique, or in
other words, the manifold is a quotient manifold of ``𝔽^{n×k}``.

The tangent space at ``p``, ``T_p\operatorname{SPS}_k(n)``, is also represented
by matrices ``Y ∈ 𝔽^{n×k}`` and reads as

````math
T_p\operatorname{SPS}_k(n) = \bigl\{
X ∈ 𝔽^{n×n}\,|\,X = qY^{\mathrm{H}} + Yq^{\mathrm{H}}
\text{ i.e. } X = X^{\mathrm{H}}
\bigr\}.
````

Note that the metric used yields a non-complete manifold.
The metric was used in [JourneeBachAbsilSepulchre:2010](@cite)[MassartAbsil:2020](@cite).

# Constructor

    SymmetricPositiveSemidefiniteFixedRank(n::Int, k::Int, field::AbstractNumbers=ℝ; parameter::Symbol=:type)

Generate the manifold of ``n×n`` symmetric positive semidefinite matrices of rank ``k``
over the `field` of real numbers `ℝ` or complex numbers `ℂ`.
"""
struct SymmetricPositiveSemidefiniteFixedRank{T,𝔽} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function SymmetricPositiveSemidefiniteFixedRank(
    n::Int,
    k::Int,
    field::AbstractNumbers=ℝ;
    parameter::Symbol=:type,
)
    size = wrap_type_parameter(parameter, (n, k))
    return SymmetricPositiveSemidefiniteFixedRank{typeof(size),field}(size)
end

function active_traits(f, ::SymmetricPositiveSemidefiniteFixedRank, args...)
    return merge_traits(IsIsometricEmbeddedManifold())
end

@doc raw"""
    check_point(M::SymmetricPositiveSemidefiniteFixedRank, q; kwargs...)

Check whether `q` is a valid manifold point on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) `M`, i.e.
whether `p=q*q'` is a symmetric matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@extref ManifoldsBase number-system) `𝔽`.
The symmetry of `p` is not explicitly checked since by using `q` p is symmetric by construction.
The tolerance for the symmetry of `p` can and the rank of `q*q'` be set using `kwargs...`.
"""
function check_point(M::SymmetricPositiveSemidefiniteFixedRank, q; kwargs...)
    n, k = get_parameter(M.size)
    p = q * q'
    r = rank(p * p'; kwargs...)
    if r < k
        return DomainError(
            r,
            "The point $(p) does not lie on $M, since its rank is less than $(k).",
        )
    end
    return nothing
end

"""
    check_vector(M::SymmetricPositiveSemidefiniteFixedRank, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SymmetricPositiveSemidefiniteFixedRank`](@ref) `M`, i.e. `X` has to be a symmetric matrix of size `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](@extref ManifoldsBase number-system).

Due to the reduced representation this is fulfilled as soon as the matrix is of correct size.
"""
check_vector(M::SymmetricPositiveSemidefiniteFixedRank, q, Y; kwargs...)

function get_embedding(
    ::SymmetricPositiveSemidefiniteFixedRank{TypeParameter{Tuple{n,k}},𝔽},
) where {n,k,𝔽}
    return Euclidean(n, k; field=𝔽)
end
function get_embedding(
    M::SymmetricPositiveSemidefiniteFixedRank{Tuple{Int,Int},𝔽},
) where {𝔽}
    n, k = get_parameter(M.size)
    return Euclidean(n, k; field=𝔽, parameter=:field)
end

@doc raw"""
    distance(M::SymmetricPositiveSemidefiniteFixedRank, p, q)

Compute the distance between two points `p`, `q` on the
[`SymmetricPositiveSemidefiniteFixedRank`](@ref), which is the Frobenius norm of ``Y`` which
minimizes ``\lVert p - qY\rVert`` with respect to ``Y``.
"""
distance(M::SymmetricPositiveSemidefiniteFixedRank, p, q) = norm(M, p, log(M, p, q))

@doc raw"""
    exp(M::SymmetricPositiveSemidefiniteFixedRank, q, Y)

Compute the exponential map on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref),
which just reads

````math
    \exp_q Y = q+Y.
````
!!! note

    Since the manifold is represented in the embedding and is a quotient manifold,
    the exponential and logarithmic map are a bijection only with respect to the equivalence
    classes. Computing
    ````math
        q_2 = \exp_p(\log_pq)
    ````
    might yield a matrix ``q_2\neq q``, but they represent the same point on the quotient
    manifold, i.e. ``d_{\operatorname{SPS}_k(n)}(q_2,q) = 0``.
"""
exp(::SymmetricPositiveSemidefiniteFixedRank, ::Any, ::Any)

function exp!(::SymmetricPositiveSemidefiniteFixedRank, q2, q, Y)
    q2 .= q .+ Y
    return q2
end

@doc raw"""
    isapprox(M::SymmetricPositiveSemidefiniteFixedRank, p, q; kwargs...)

test, whether two points `p`, `q` are (approximately) nearly the same.
Since this is a quotient manifold in the embedding, the test is performed by checking
their distance, if they are not the same, i.e. that ``d_{\mathcal M}(p,q) \approx 0``, where
the comparison is performed with the classical `isapprox`.
The `kwargs...` are passed on to this accordingly.
"""
function _isapprox(
    M::SymmetricPositiveSemidefiniteFixedRank,
    p::T,
    q;
    atol::Real=sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
    kwargs...,
) where {T}
    return isapprox(norm(p - q), 0; atol=atol, kwargs...) ||
           isapprox(distance(M, p, q), 0; atol=atol, kwargs...)
end

"""
    is_flat(::SymmetricPositiveSemidefiniteFixedRank)

Return false. [`SymmetricPositiveSemidefiniteFixedRank`](@ref) is not a flat manifold. See Theorem A.18 in [MassartAbsil:2020](@cite).
"""
is_flat(M::SymmetricPositiveSemidefiniteFixedRank) = false

@doc raw"""
    log(M::SymmetricPositiveSemidefiniteFixedRank, q, p)

Compute the logarithmic map on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) manifold
by minimizing ``\lVert p - qY\rVert`` with respect to ``Y``.

!!! note

    Since the manifold is represented in the embedding and is a quotient manifold,
    the exponential and logarithmic map are a bijection only with respect to the equivalence
    classes. Computing
    ````math
        q_2 = \exp_p(\log_pq)
    ````
    might yield a matrix ``q_2≠q``, but they represent the same point on the quotient
    manifold, i.e. ``d_{\operatorname{SPS}_k(n)}(q_2,q) = 0``.
"""
log(::SymmetricPositiveSemidefiniteFixedRank, q, p)

function log!(::SymmetricPositiveSemidefiniteFixedRank, Z, q, p)
    s = svd(q' * p)
    return Z .= p * (s.V * s.U') - q
end

@doc raw"""
    manifold_dimension(M::SymmetricPositiveSemidefiniteFixedRank)

Return the dimension of the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) matrix `M` over the number system
`𝔽`, i.e.

````math
\begin{aligned}
\dim \operatorname{SPS}_{k,ℝ}(n) &= kn - \frac{k(k-1)}{2},\\
\dim \operatorname{SPS}_{k,ℂ}(n) &= 2kn - k^2,
\end{aligned}
````

where the last ``k^2`` is due to the zero imaginary part for Hermitian matrices diagonal
"""
manifold_dimension(::SymmetricPositiveSemidefiniteFixedRank)

function manifold_dimension(M::SymmetricPositiveSemidefiniteFixedRank{<:Any,ℝ})
    n, k = get_parameter(M.size)
    return k * n - div(k * (k - 1), 2)
end
function manifold_dimension(M::SymmetricPositiveSemidefiniteFixedRank{<:Any,ℂ})
    n, k = get_parameter(M.size)
    return 2 * k * n - k * k
end

function project!(::SymmetricPositiveSemidefiniteFixedRank, Z, q, Y)
    Z .= Y - q * lyap(q' * q, q' * Y - Y' * q)
    return Z
end

function Base.show(
    io::IO,
    ::SymmetricPositiveSemidefiniteFixedRank{TypeParameter{Tuple{n,k}},𝔽},
) where {n,k,𝔽}
    return print(io, "SymmetricPositiveSemidefiniteFixedRank($(n), $(k), $(𝔽))")
end
function Base.show(
    io::IO,
    M::SymmetricPositiveSemidefiniteFixedRank{Tuple{Int,Int},𝔽},
) where {𝔽}
    n, k = get_parameter(M.size)
    return print(
        io,
        "SymmetricPositiveSemidefiniteFixedRank($(n), $(k), $(𝔽); parameter=:field)",
    )
end

"""
     vector_transport_to(M::SymmetricPositiveSemidefiniteFixedRank, p, X, q)

 transport the tangent vector `X` at `p` to `q` by projecting it onto the tangent space
 at `q`.
 """
vector_transport_to(
    ::SymmetricPositiveSemidefiniteFixedRank,
    ::Any,
    ::Any,
    ::Any,
    ::ProjectionTransport,
)

function vector_transport_to_project!(M::SymmetricPositiveSemidefiniteFixedRank, Y, p, X, q)
    project!(M, Y, q, X)
    return Y
end

@doc raw"""
     zero_vector(M::SymmetricPositiveSemidefiniteFixedRank, p)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `p` on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) manifold `M`.
"""
zero_vector(::SymmetricPositiveSemidefiniteFixedRank, ::Any...)

function zero_vector!(::SymmetricPositiveSemidefiniteFixedRank, v, ::Any)
    return fill!(v, 0)
end
