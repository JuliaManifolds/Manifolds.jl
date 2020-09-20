@doc raw"""
    Grassmann{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType}

The Grassmann manifold $\operatorname{Gr}(n,k)$ consists of all subspaces spanned by $k$ linear independent
vectors $𝔽^n$, where $𝔽  ∈ \{ℝ, ℂ\}$ is either the real- (or complex-) valued vectors.
This yields all $k$-dimensional subspaces of $ℝ^n$ for the real-valued case and all $2k$-dimensional subspaces
of $ℂ^n$ for the second.

The manifold can be represented as

````math
\operatorname{Gr}(n,k) := \bigl\{ \operatorname{span}(p) : p ∈ 𝔽^{n × k}, p^\mathrm{H}p = I_k\},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian and
$I_k$ is the $k × k$ identity matrix. This means, that the columns of $p$
form an unitary basis of the subspace, that is a point on
$\operatorname{Gr}(n,k)$, and hence the subspace can actually be represented by
a whole equivalence class of representers.
Another interpretation is, that

````math
\operatorname{Gr}(n,k) = \operatorname{St}(n,k) / \operatorname{O}(k),
````

i.e the Grassmann manifold is the quotient of the [`Stiefel`](@ref) manifold and
the orthogonal group $\operatorname{O}(k)$ of orthogonal $k × k$ matrices.

The tangent space at a point (subspace) $x$ is given by

````math
T_x\mathrm{Gr}(n,k) = \bigl\{
X ∈ 𝔽^{n × k} :
X^{\mathrm{H}}p + p^{\mathrm{H}}X = 0_{k} \bigr\},
````

where $0_k$ is the $k × k$ zero matrix and $\overline{\cdot}$ the complex conjugate.

Note that a point $p ∈ \operatorname{Gr}(n,k)$ might be represented by
different matrices (i.e. matrices with unitary column vectors that span
the same subspace). Different representations of $p$ also lead to different
representation matrices for the tangent space $T_p\mathrm{Gr}(n,k)$

The manifold is named after
[Hermann G. Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).

# Constructor

    Grassmann(n,k,field=ℝ)

Generate the Grassmann manifold $\operatorname{Gr}(n,k)$, where the real-valued
case `field = ℝ` is the default.
"""
struct Grassmann{n,k,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultIsometricEmbeddingType} end

Grassmann(n::Int, k::Int, field::AbstractNumbers = ℝ) = Grassmann{n,k,field}()

function allocation_promotion_function(M::Grassmann{n,k,ℂ}, f, args::Tuple) where {n,k}
    return complex
end

@doc raw"""
    check_manifold_point(M::Grassmann{n,k,𝔽}, p)

Check whether `p` is representing a point on the [`Grassmann`](@ref) `M`, i.e. its
a `n`-by-`k` matrix of unitary column vectors and of correct `eltype` with respect to `𝔽`.
"""
function check_manifold_point(M::Grassmann{n,k,𝔽}, p; kwargs...) where {n,k,𝔽}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    c = p' * p
    if !isapprox(c, one(c); kwargs...)
        return DomainError(
            norm(c - one(c)),
            "The point $(p) does not lie on $(M), because x'x is not the unit matrix.",
        )
    end
    return nothing
end

@doc raw"""
    check_tangent_vector(M::Grassmann{n,k,𝔽}, p, X; check_base_point = true, kwargs...)

Check whether `X` is a tangent vector in the tangent space of `p` on
the [`Grassmann`](@ref) `M`, i.e. that `X` is of size and type as well as that

````math
    p^{\mathrm{H}}X + \overline{X^{\mathrm{H}}p} = 0_k,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian,
$\overline{\cdot}$ the (elementwise) complex conjugate, and $0_k$ the $k × k$ zero natrix.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
"""
function check_tangent_vector(
    M::Grassmann{n,k,𝔽},
    p,
    X;
    check_base_point = true,
    kwargs...,
) where {n,k,𝔽}
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)),typeof(p),typeof(X)},
        M,
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(p' * X + conj(X' * p), zeros(k, k); kwargs...)
        return DomainError(
            norm(p' * X + conj(X' * p)),
            "The matrix $(X) does not lie in the tangent space of $(p) on $(M), since p'X + X'p is not the zero matrix.",
        )
    end
    return nothing
end

decorated_manifold(M::Grassmann{N,K,𝔽}) where {N,K,𝔽} = Euclidean(N, K; field = 𝔽)

@doc raw"""
    distance(M::Grassmann, p, q)

Compute the Riemannian distance on [`Grassmann`](@ref) manifold `M`$= \mathrm{Gr}(n,k)$.

Let $USV = p^\mathrm{H}q$ denote the SVD decomposition of
$p^\mathrm{H}q$, where $\cdot^{\mathrm{H}}$ denotes the complex
conjugate transposed or Hermitian. Then the distance is given by
````math
d_{\mathrm{Gr}(n,k)}(p,q) = \operatorname{norm}(\operatorname{Re}(b)).
````
where

````math
b_{i}=\begin{cases}
0 & \text{if} \; S_i ≥ 1\\
\arccos(S_i) & \, \text{if} \; S_i<1.
\end{cases}
````
"""
function distance(M::Grassmann, p, q)
    p ≈ q && return zero(real(eltype(p)))
    a = svd(p' * q).S
    a[a .> 1] .= 1
    return sqrt(sum((real(acos.(a))) .^ 2))
end

embed!(::Grassmann, q, p) = (q .= p)

embed!(::Grassmann, Y, p, X) = (Y .= X)

@doc raw"""
    exp(M::Grassmann, p, X)

Compute the exponential map on the [`Grassmann`](@ref) `M`$= \mathrm{Gr}(n,k)$ starting in
`p` with tangent vector (direction) `X`. Let $X = USV$ denote the SVD decomposition of $X$.
Then the exponential map is written using

````math
z = p V\cos(S)V^\mathrm{H} + U\sin(S)V^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian and the
cosine and sine are applied element wise to the diagonal entries of $S$. A final QR
decomposition $z=QR$ is performed for numerical stability reasons, yielding the result as

````math
\exp_p X = Q.
````
"""
exp(::Grassmann, ::Any...)

function exp!(M::Grassmann, q, p, X)
    norm(M, p, X) ≈ 0 && return copyto!(q, p)
    d = svd(X)
    z = p * d.V * Diagonal(cos.(d.S)) * d.Vt + d.U * Diagonal(sin.(d.S)) * d.Vt
    return copyto!(q, Array(qr(z).Q))
end

@doc raw"""
    injectivity_radius(M::Grassmann)
    injectivity_radius(M::Grassmann, p)

Return the injectivity radius on the [`Grassmann`](@ref) `M`, which is $\frac{π}{2}$.
"""
injectivity_radius(::Grassmann) = π / 2
injectivity_radius(::Grassmann, ::ExponentialRetraction) = π / 2
injectivity_radius(::Grassmann, ::Any) = π / 2
injectivity_radius(::Grassmann, ::Any, ::ExponentialRetraction) = π / 2
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::Grassmann,
            rm::AbstractRetractionMethod,
        )
    end,
)

@doc raw"""
    inner(M::Grassmann, p, X, Y)

Compute the inner product for two tangent vectors `X`, `Y` from the tangent space
of `p` on the [`Grassmann`](@ref) manifold `M`. The formula reads

````math
g_p(X,Y) = \operatorname{tr}(X^{\mathrm{H}}Y),
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inner(::Grassmann, p, X, Y) = dot(abs.(X), abs.(Y))

@doc raw"""
    inverse_retract(M::Grassmann, p, q, ::PolarInverseRetraction)

Compute the inverse retraction for the [`PolarRetraction`](@ref), on the
[`Grassmann`](@ref) manifold `M`, i.e.,

````math
\operatorname{retr}_p^{-1}q = q*(p^\mathrm{H}q)^{-1} - p,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(::Grassmann, ::Any, ::Any, ::PolarInverseRetraction)

function inverse_retract!(::Grassmann, X, p, q, ::PolarInverseRetraction)
    return copyto!(X, q / (p' * q) - p)
end

@doc raw"""
    inverse_retract(M, p, q, ::QRInverseRetraction)

Compute the inverse retraction for the [`QRRetraction`](@ref), on the
[`Grassmann`](@ref) manifold `M`, i.e.,

````math
\operatorname{retr}_p^{-1}q = q(p^\mathrm{H}q)^{-1} - p,
````
where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(::Grassmann, ::Any, ::Any, ::QRInverseRetraction)

inverse_retract!(::Grassmann, X, p, q, ::QRInverseRetraction) = copyto!(X, q / (p' * q) - p)

function Base.isapprox(M::Grassmann, p, X, Y; kwargs...)
    return isapprox(sqrt(inner(M, p, zero_tangent_vector(M, p), X - Y)), 0; kwargs...)
end
Base.isapprox(M::Grassmann, p, q; kwargs...) = isapprox(distance(M, p, q), 0.0; kwargs...)

@doc raw"""
    log(M::Grassmann, p, q)

Compute the logarithmic map on the [`Grassmann`](@ref) `M`$ = \mathcal M=\mathrm{Gr}(n,k)$,
i.e. the tangent vector `X` whose corresponding [`geodesic`](@ref) starting from `p`
reaches `q` after time 1 on `M`. The formula reads

````math
\log_p q = V\cdot \operatorname{atan}(S) \cdot U^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
The matrices $U$ and $V$ are the unitary matrices, and $S$ is the diagonal matrix
containing the singular values of the SVD-decomposition

````math
USV = (q^\mathrm{H}p)^{-1} ( q^\mathrm{H} - q^\mathrm{H}pp^\mathrm{H}).
````

In this formula the $\operatorname{atan}$ is meant elementwise.
"""
log(::Grassmann, ::Any...)

function log!(::Grassmann{n,k}, X, p, q) where {n,k}
    z = q' * p
    At = q' - z * p'
    Bt = z \ At
    d = svd(Bt')
    return copyto!(X, d.U[:, 1:k] * Diagonal(atan.(d.S[1:k])) * d.Vt[1:k, :])
end

@doc raw"""
    manifold_dimension(M::Grassmann)

Return the dimension of the [`Grassmann(n,k,𝔽)`](@ref) manifold `M`, i.e.

````math
\dim \operatorname{Gr}(n,k) = k(n-k) \dim_ℝ 𝔽,
````

where $\dim_ℝ 𝔽$ is the [`real_dimension`](@ref) of `𝔽`.
"""
manifold_dimension(::Grassmann{n,k,𝔽}) where {n,k,𝔽} = k * (n - k) * real_dimension(𝔽)

"""
    mean(
        M::Grassmann,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/4);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::Grassmann{n,k} where {n,k}, ::Any...)

function Statistics.mean!(
    M::Grassmann{n,k},
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {n,k}
    return mean!(M, p, x, w, GeodesicInterpolationWithinRadius(π / 4); kwargs...)
end

@doc raw"""
    project(M::Grassmann, p, X)

Project the `n`-by-`k` `X` onto the tangent space of `p` on the [`Grassmann`](@ref) `M`,
which is computed by

````math
\operatorname{proj_p}(X) = X - pp^{\mathrm{H}}X,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
project(::Grassmann, ::Any...)

project!(::Grassmann, Y, p, X) = copyto!(Y, X - p * p' * X)

@doc raw"""
    representation_size(M::Grassmann{n,k})

Return the represenation size or matrix dimension of a point on the [`Grassmann`](@ref)
`M`, i.e. $(n,k)$ for both the real-valued and the complex value case.
"""
@generated representation_size(::Grassmann{n,k}) where {n,k} = (n, k)

@doc raw"""
    retract(M::Grassmann, p, X, ::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@ref) on the
[`Grassmann`](@ref) `M`. With $USV = p + X$ the retraction reads
````math
\operatorname{retr}_p X = UV^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
retract(::Grassmann, ::Any, ::Any, ::PolarRetraction)

@doc raw"""
    retract(M::Grassmann, p, X, ::QRRetraction )

Compute the QR-based retraction [`QRRetraction`](@ref) on the
[`Grassmann`](@ref) `M`. With $QR = p + X$ the retraction reads
````math
\operatorname{retr}_p X = QD,
````
where D is a $m × n$ matrix with
````math
D = \operatorname{diag}( \operatorname{sgn}(R_{ii}+0,5)_{i=1}^n ).
````
"""
retract(::Grassmann, ::Any, ::Any, ::QRRetraction)

function retract!(::Grassmann, q, p, X, ::PolarRetraction)
    s = svd(p + X)
    return mul!(q, s.U, s.Vt)
end
function retract!(::Grassmann{N,K}, q, p, X, ::QRRetraction) where {N,K}
    qrfac = qr(p + X)
    d = diag(qrfac.R)
    D = Diagonal(sign.(sign.(d .+ 0.5)))
    q .= zeros(N, K)
    q[1:K, 1:K] .= D
    return copyto!(q, Array(qrfac.Q) * D)
end

function Base.show(io::IO, ::Grassmann{n,k,𝔽}) where {n,k,𝔽}
    return print(io, "Grassmann($(n), $(k), $(𝔽))")
end

"""
    uniform_distribution(M::Grassmann{n,k,ℝ}, p)

Uniform distribution on given (real-valued) [`Grassmann`](@ref) `M`.
Specifically, this is the normalized Haar measure on `M`.
Generated points will be of similar type as `p`.

The implementation is based on Section 2.5.1 in [^Chikuse2003];
see also Theorem 2.2.2(iii) in [^Chikuse2003].

[^Chikuse2003]:
    > Y. Chikuse: "Statistics on Special Manifolds", Springer New York, 2003,
    > doi: [10.1007/978-0-387-21540-2](https://doi.org/10.1007/978-0-387-21540-2).
"""
function uniform_distribution(M::Grassmann{n,k,ℝ}, p) where {n,k}
    μ = Distributions.Zeros(n, k)
    σ = one(eltype(p))
    Σ1 = Distributions.PDMats.ScalMat(n, σ)
    Σ2 = Distributions.PDMats.ScalMat(k, σ)
    d = MatrixNormal(μ, Σ1, Σ2)

    return ProjectedPointDistribution(M, d, (M, q, p) -> (q .= svd(p).U), p)
end

@doc raw"""
    vector_transport_to(M::Grassmann,p,X,q,::ProjectionTransport)

compute the projection based transport on the [`Grassmann`](@ref) `M` by
interpreting `X` from the tangent space at `p` as a point in the embedding and
projecting it onto the tangent space at q.
"""
vector_transport_to(::Grassmann, ::Any, ::Any, ::Any, ::ProjectionTransport)

function vector_transport_to!(M::Grassmann, Y, p, X, q, ::ProjectionTransport)
    project!(M, Y, q, X)
    return Y
end

@doc raw"""
    zero_tangent_vector(M::Grassmann, p)

Return the zero tangent vector from the tangent space at `p` on the [`Grassmann`](@ref) `M`,
which is given by a zero matrix the same size as `p`.
"""
zero_tangent_vector(::Grassmann, ::Any...)

zero_tangent_vector!(::Grassmann, X, p) = fill!(X, 0)
