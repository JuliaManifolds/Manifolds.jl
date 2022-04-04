@doc raw"""
    SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽} <: AbstractDecoratorManifold{𝔽}

The [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#ManifoldsBase.AbstractManifold)  $ \operatorname{SPS}_k(n)$ consisting of the real- or complex-valued
symmetric positive semidefinite matrices of size $n × n$ and rank $k$, i.e. the set

````math
\operatorname{SPS}_k(n) = \bigl\{
p  ∈ 𝔽^{n × n}\ \big|\ p^{\mathrm{H}} = p,
apa^{\mathrm{H}} \geq 0 \text{ for all } a ∈ 𝔽
\text{ and } \operatorname{rank}(p) = k\bigr\},
````
where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transpose,
and the field $𝔽 ∈ \{ ℝ, ℂ\}$.
We sometimes $\operatorname{SPS}_{k,𝔽}(n)$, when distinguishing the real- and complex-valued
manifold is important.

An element is represented by $q ∈ 𝔽^{n × k}$ from the factorization $p = qq^{\mathrm{H}}$.
Note that since for any unitary (orthogonal) $A ∈ 𝔽^{n × n}$ we have
$(Aq)(Aq)^{\mathrm{H}} = qq^{\mathrm{H}} = p$, the representation is not unique, or in
other words, the manifold is a quotient manifold of $𝔽^{n × k}$.

The tangent space at $p$, $T_p\operatorname{SPS}_k(n)$, is also represented
by matrices $Y ∈ 𝔽^{n × k}$ and reads as

````math
T_p\operatorname{SPS}_k(n) = \bigl\{
X ∈ 𝔽^{n × n}\,|\,X = qY^{\mathrm{H}} + Yq^{\mathrm{H}}
\text{ i.e. } X = X^{\mathrm{H}}
\bigr\}.
````

Note that the metric used yields a non-complete manifold.
The metric was used in[^JourneeBachAbsilSepulchre2010][^MassartAbsil2020].

# Constructor

    SymmetricPositiveSemidefiniteFixedRank(n::Int, k::Int, field::AbstractNumbers=ℝ)

Generate the manifold of $n × n$ symmetric positive semidefinite matrices of rank $k$
over the `field` of real numbers `ℝ` or complex numbers `ℂ`.

[^JourneeBachAbsilSepulchre2010]:
    > Journée, M., Bach, F., Absil, P.-A., and Sepulchre, R.:
    > “Low-Rank Optimization on the Cone of Positive Semidefinite Matrices”,
    > SIAM Journal on Optimization (20)5, pp. 2327–2351, 2010.
    > doi: [10.1137/080731359](https://doi.org/10.1137/080731359),
    > arXiv: [0807.4423](http://arxiv.org/abs/0807.4423).
[^MassartAbsil2020]:
    > Massart, E., Absil, P.-A.:
    > "Quotient Geometry with Simple Geodesics for the AbstractManifold of Fixed-Rank Positive-Semidefinite Matrices",
    > SIAM Journal on Matrix Analysis and Applications (41)1, pp. 171–198, 2020.
    > doi: [10.1137/18m1231389](https://doi.org/10.1137/18m1231389),
    > preprint: [sites.uclouvain.be/absil/2018.06](https://sites.uclouvain.be/absil/2018.06).
"""
struct SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽} <: AbstractDecoratorManifold{𝔽} end

function SymmetricPositiveSemidefiniteFixedRank(n::Int, k::Int, field::AbstractNumbers=ℝ)
    return SymmetricPositiveSemidefiniteFixedRank{n,k,field}()
end

function active_traits(f, ::SymmetricPositiveSemidefiniteFixedRank, args...)
    return merge_traits(IsIsometricEmbeddedManifold())
end

@doc raw"""
    check_point(M::SymmetricPositiveSemidefiniteFixedRank{n,𝔽}, q; kwargs...)

Check whether `q` is a valid manifold point on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) `M`, i.e.
whether `p=q*q'` is a symmetric matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#number-system) `𝔽`.
The symmetry of `p` is not explicitly checked since by using `q` p is symmetric by construction.
The tolerance for the symmetry of `p` can and the rank of `q*q'` be set using `kwargs...`.
"""
function check_point(
    M::SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽},
    q;
    kwargs...,
) where {n,k,𝔽}
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
    check_vector(M::SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SymmetricPositiveSemidefiniteFixedRank`](@ref) `M`, i.e. `X` has to be a symmetric matrix of size `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#number-system).

Due to the reduced representation this is fulfilled as soon as the matrix is of correct size.
"""
check_vector(M::SymmetricPositiveSemidefiniteFixedRank, q, Y; kwargs...)

function get_embedding(::SymmetricPositiveSemidefiniteFixedRank{N,K,𝔽}) where {N,K,𝔽}
    return Euclidean(N, K; field=𝔽)
end

@doc raw"""
    distance(M::SymmetricPositiveSemidefiniteFixedRank, p, q)

Compute the distance between two points `p`, `q` on the
[`SymmetricPositiveSemidefiniteFixedRank`](@ref), which is the Frobenius norm of $Y$ which
minimizes $\lVert p - qY\rVert$ with respect to $Y$.
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
    might yield a matrix $q_2\neq q$, but they represent the same point on the quotient
    manifold, i.e. $d_{\operatorname{SPS}_k(n)}(q_2,q) = 0$.
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
their distance, if they are not the same, i.e. that $d_{\mathcal M}(p,q) \approx 0$, where
the comparison is performed with the classical `isapprox`.
The `kwargs...` are passed on to this accordingly.
"""
function isapprox(M::SymmetricPositiveSemidefiniteFixedRank, p, q; kwargs...)
    return isapprox(norm(p - q), 0.0; kwargs...) ||
           isapprox(distance(M, p, q), 0.0; kwargs...)
end

@doc raw"""
    log(M::SymmetricPositiveSemidefiniteFixedRank, q, p)

Compute the logarithmic map on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) manifold
by minimizing $\lVert p - qY\rVert$ with respect to $Y$.

!!! note

    Since the manifold is represented in the embedding and is a quotient manifold,
    the exponential and logarithmic map are a bijection only with respect to the equivalence
    classes. Computing
    ````math
        q_2 = \exp_p(\log_pq)
    ````
    might yield a matrix $q_2\neq q$, but they represent the same point on the quotient
    manifold, i.e. $d_{\operatorname{SPS}_k(n)}(q_2,q) = 0$.
"""
log(::SymmetricPositiveSemidefiniteFixedRank, q, p)

function log!(::SymmetricPositiveSemidefiniteFixedRank, Z, q, p)
    s = svd(q' * p)
    return Z .= p * (s.V * s.U') - q
end

@doc raw"""
    manifold_dimension(M::SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽})

Return the dimension of the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) matrix `M` over the number system
`𝔽`, i.e.

````math
\begin{aligned}
\dim \operatorname{SPS}_{k,ℝ}(n) &= kn - \frac{k(k-1)}{2},\\
\dim \operatorname{SPS}_{k,ℂ}(n) &= 2kn - k^2,
\end{aligned}
````

where the last $k^2$ is due to the zero imaginary part for Hermitian matrices diagonal
"""
manifold_dimension(::SymmetricPositiveSemidefiniteFixedRank)

@generated function manifold_dimension(
    ::SymmetricPositiveSemidefiniteFixedRank{N,K,ℝ},
) where {N,K}
    return K * N - div(K * (K - 1), 2)
end
@generated function manifold_dimension(
    ::SymmetricPositiveSemidefiniteFixedRank{N,K,ℂ},
) where {N,K}
    return 2 * K * N - K * K
end

function project!(::SymmetricPositiveSemidefiniteFixedRank, Z, q, Y)
    Z .= Y - q * lyap(q' * q, q' * Y - Y' * q)
    return Z
end

function Base.show(io::IO, ::SymmetricPositiveSemidefiniteFixedRank{n,k,F}) where {n,k,F}
    return print(io, "SymmetricPositiveSemidefiniteFixedRank($(n), $(k), $(F))")
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

function vector_transport_to!(
    M::SymmetricPositiveSemidefiniteFixedRank,
    Y,
    p,
    X,
    q,
    ::ProjectionTransport,
)
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
