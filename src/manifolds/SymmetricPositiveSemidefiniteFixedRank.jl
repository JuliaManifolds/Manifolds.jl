@doc raw"""
    SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType}

The [`Manifold`](@ref) $ \operatorname{SPS}_k(n)$ consisting of the real- or complex-valued
symmetric positive semidefinite matrices of size $n × n$ and rank $k$, i.e. the set

````math
\operatorname{SPS}_k(n) = \bigl\{
p  ∈ 𝔽^{n × n}\ \big|\ p^{\mathrm{H}} = p,
apa^mathrm{H} \geq 0 \text{ for all } a ∈ 𝔽
\text{ and } \operatorname{rank}(p) = k\bigr\},
````
where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transpose,
and the field $𝔽 ∈ \{ ℝ, ℂ\}$.

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
\bigr\}
````

Note that the metric used yields a non-complete manifold.
The metric was used in[^JourneeBachAbsilSepulchre2010][^MassartAbsil2020].

# Constructor

    SymmetricPositiveSemidefiniteFixedRank(n::Int, k::Int, field::AbstractNumbers=ℝ)

Generate the manifold of $n × n$ symmetric positive semidefinite matrices of rank $k$.

[^JourneeBachAbsilSepulchre2010]:
    > Journée, M., Bach, F., Absil, P.-A., and Sepulchre, R.:
    > “Low-Rank Optimization on the Cone of Positive Semidefinite Matrices”,
    > SIAM Journal on Optimization (20)5, pp. 2327–2351, 2010.
    > doi: [10.1137/080731359](https://doi.org/10.1137/080731359),
    > arXiv: [0807.4423](http://arxiv.org/abs/0807.4423).
[^MassartAbsil2020]:
    > Massart, E., Absil, P.-A.:
    > "Quotient Geometry with Simple Geodesics for the Manifold of Fixed-Rank Positive-Semidefinite Matrices",
    > SIAM Journal on Matrix Analysis and Applications (41)1, pp. 171–198, 2020.
    > doi: [10.1137/18m1231389](https://doi.org/10.1137/18m1231389),
"""
struct SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

function SymmetricPositiveSemidefiniteFixedRank(n::Int, k::Int, field::AbstractNumbers = ℝ)
    return SymmetricPositiveSemidefiniteFixedRank{n,k,field}()
end

@doc raw"""
    check_manifold_point(M::SymmetricPositiveSemidefiniteFixedRank{n,𝔽}, q; kwargs...)

Check whether `q` is a valid manifold point on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) `M`, i.e.
whether `p=q*q'` is a symmetric matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@ref) `𝔽`.
The symmetry of `p` is not explicitly checked since by using `q` p is symmetric by construction.
The tolerance for the symmetry of `p` can and the rank of `q*q'` be set using `kwargs...`.
"""
function check_manifold_point(M::SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽}, q; kwargs...) where {n,k,𝔽}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(q)}, M, q; kwargs...)
    mpv === nothing || return mpv
    p = q*q'
    if rank(p*p';kwargs...) < k
        return DomainError(
            rank(p*p';kwargs...),
            "The point $(p) does not lie on $M, since its rank is less than $(k).",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽}, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SymmetricPositiveSemidefiniteFixedRank`](@ref) `M`, i.e. `X` has to be a symmetric matrix of size `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](@ref).
The optional parameter `check_base_point` indicates, whether to call
 [`check_manifold_point`](@ref)  for `p`.
The tolerance for the symmetry of `p` and `X` can be set using `kwargs...`.
"""
function check_tangent_vector(
    M::SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽},
    q,
    Y;
    check_base_point = true,
    kwargs...,
) where {n,k,𝔽}
    if check_base_point
        mpe = check_manifold_point(M, q; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)),typeof(q),typeof(Y)},
        M,
        q,
        Y;
        check_base_point = false, # already checked above
        kwargs...,
    )
    return mpv
end

decorated_manifold(::SymmetricPositiveSemidefiniteFixedRank{N,K,𝔽}) where {N,K,𝔽} = Euclidean(N, K; field = 𝔽)

embed!(::SymmetricPositiveSemidefiniteFixedRank, q, p) = copyto!(q, p)
embed!(::SymmetricPositiveSemidefiniteFixedRank, Y, ::Any, X) = copyto!(Y, X)

@doc raw"""
    exp(M::SymmetricPositiveSemidefiniteFixedRank, q, Y)

Compute the exponential map on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref),
which just reads

````math
    \exp_q Y = q+Y.
````
"""
exp(::SymmetricPositiveSemidefiniteFixedRank, ::Any, ::Any)

function exp!(::SymmetricPositiveSemidefiniteFixedRank, q2, q, Y)
    q2 .= q .+ Y
    return q2
end

@doc raw"""
    log(M::SymmetricPositiveSemidefiniteFixedRank, q, p)

Compute the logarithmic map on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) manifold
by minimizing $\lVert p - qY\rVert$ with respect to $Y$.
"""
log(::SymmetricPositiveSemidefiniteFixedRank, q, p)

function log!(::SymmetricPositiveSemidefiniteFixedRank, Z, q, p)
    s = svd(q'*p)
    Z .= p*(s.V*s.U') - q
end

@doc raw"""
    manifold_dimension(M::SymmetricPositiveSemidefiniteFixedRank{n,k,𝔽})

Return the dimension of the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) matrix `M` over the number system
`𝔽`, i.e.

````math
\begin{aligned}
\dim \mathrm{Sym}(n,ℝ) &= kn - \frac{k(k-1)}{2},\\
\dim \mathrm{Sym}(n,ℂ) &= 2kn - k^2,
\end{aligned}
````

where the last $k^2$ is due to the zero imaginary part for Hermitian matrices diagonal
"""
function manifold_dimension(::SymmetricPositiveSemidefiniteFixedRank) end
function manifold_dimension(::SymmetricPositiveSemidefiniteFixedRank{N,K,ℝ}) where {N,K}
    return K*N - div(K*(K-1),2)
end
function manifold_dimension(::SymmetricPositiveSemidefiniteFixedRank{N,K,ℂ}) where {N,K}
    return 2*K*N - K*K
end

function project!(::SymmetricPositiveSemidefiniteFixedRank, Z, q, Y)
    Z .= Y - q * lyap(q' * q, q' * Y - Y' * q)
    return Z
end

function Base.show(io::IO, ::SymmetricPositiveSemidefiniteFixedRank{n,k,F}) where {n,k,F}
    return print(io, "SymmetricPositiveSemidefiniteFixedRank($(n), $(k), $(F))")
end

@doc raw"""
     zero_tangent_vector(M::SymmetricPositiveSemidefiniteFixedRank, p)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `p` on the [`SymmetricPositiveSemidefiniteFixedRank`](@ref) manifold `M`.
"""
zero_tangent_vector(::SymmetricPositiveSemidefiniteFixedRank, ::Any...)

function zero_tangent_vector!(::SymmetricPositiveSemidefiniteFixedRank{N,K}, v, ::Any) where {N,K}
    fill!(v, 0)
    return v
end