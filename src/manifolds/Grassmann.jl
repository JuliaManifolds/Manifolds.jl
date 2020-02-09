@doc raw"""
    Grassmann{n,k,𝔽} <: Manifold

The Grassmann manifold $\operatorname{Gr}(n,k)$ consists of all subspaces spanned by $k$ linear independent
vectors $𝔽^n$, where $𝔽  ∈ \{ℝ, ℂ\}$ is either the real- (or complex-) valued vectors.
This yields all $k$-dimensional subspaces of $ℝ^n$ for the real-valued case and all $2k$-dimensional subspaces
of $ℂ^n$ for the second.

The manifold can be represented as

````math
\operatorname{Gr}(n,k) := \bigl\{ \operatorname{span}(p) : p ∈ 𝔽^{n × k}, p^\mathrm{H}p = I_k\},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian and
$I_k$ is the $k × k$ identity matrix. This means, that the columns of $x$
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

where $0_{k}$ denotes the $k × k$ zero matrix.

Note that a point $p ∈ \operatorname{Gr}(n,k)$ might be represented by
different matrices (i.e. matrices with unitary column vectors that span
the same subspace). Different representations of $p$ also lead to different
representation matrices for the tangent space $T_p\mathrm{Gr}(n,k)$

The manifold is named after
[Hermann G. Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).

# Constructor

    Grassmann(n,k,field=ℝ)

Generate the Grassmann manifold $\operatorname{Gr}(n,k)$, where the real-valued
case $field = ℝ$ is the default.
"""
struct Grassmann{n,k,𝔽} <: Manifold end

Grassmann(n::Int, k::Int, field::AbstractNumbers = ℝ) = Grassmann{n,k,field}()

@doc raw"""
    check_manifold_point(M::Grassmann{n,k,𝔽}, p)

Check whether `p` is representing a point on the [`Grassmann`](@ref) `M`, i.e. its
a `n`-by-`k` matrix of unitary column vectors and of correct `eltype` with respect to `𝔽`.
"""
function check_manifold_point(M::Grassmann{n,k,𝔽}, p; kwargs...) where {n,k,𝔽}
    if (𝔽 === ℝ) && !(eltype(p) <: Real)
        return DomainError(
            eltype(p),
            "The matrix $(p) is not a real-valued matrix, so it does not lie on the Grassmann manifold of dimension ($(n),$(k)).",
        )
    end
    if (𝔽 === ℂ) && !(eltype(p) <: Real) && !(eltype(p) <: Complex)
        return DomainError(
            eltype(p),
            "The matrix $(p) is neither a real- nor complex-valued matrix, so it does not lie on the complex Grassmann manifold of dimension ($(n),$(k)).",
        )
    end
    if size(p) != representation_size(M)
        return DomainError(
            size(p),
            "The matrix $(p) does not lie on the Grassmann manifold of dimension ($(n),$(k)), since its dimensions are wrong.",
        )
    end
    c = p' * p
    if !isapprox(c, one(c); kwargs...)
        return DomainError(
            norm(c - one(c)),
            "The point $(p) does not lie on the Grassmann manifold of dimension ($(n),$(k)), because x'x is not the unit matrix.",
        )
    end
end

@doc raw"""
    check_tangent_vector(M::Grassmann{n,k,𝔽}, p, X)

Check whether `X` is a tangent vector in the tangent space of `p` on
the [`Grassmann`](@ref) `M`, i.e. that `X` is of size and type as well as that

````math
    p^{\mathrm{H}}X + X^{\mathrm{H}}p = 0_k,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian and $0_k$
denotes the $k × k$ zero natrix.
"""
function check_tangent_vector(G::Grassmann{n,k,𝔽}, p, X; kwargs...) where {n,k,𝔽}
    t = check_manifold_point(G, p)
    t === nothing || return t
    if (𝔽 === ℝ) && !(eltype(X) <: Real)
        return DomainError(
            eltype(X),
            "The matrix $(X) is not a real-valued matrix, so it can not be a tangent vector to the Grassmann manifold of dimension ($(n),$(k)).",
        )
    end
    if (𝔽 === ℂ) && !(eltype(X) <: Real) && !(eltype(X) <: Complex)
        return DomainError(
            eltype(X),
            "The matrix $(X) is neither a real- nor complex-valued matrix, so it can not be a tangent vector to the complex Grassmann manifold of dimension ($(n),$(k)).",
        )
    end
    if size(X) != representation_size(G)
        return DomainError(
            size(X),
            "The matrix $(X) does not lie in the tangent space of $(p) on the Grassmann manifold of dimension ($(n),$(k)), since its dimensions are wrong.",
        )
    end
    if !isapprox(p' * X + X' * p, zeros(k, k); kwargs...)
        return DomainError(
            norm(p' * X + X' * p),
            "The matrix $(X) does not lie in the tangent space of $(p) on the Grassmann manifold of dimension ($(n),$(k)), since x'v + v'x is not the zero matrix.",
        )
    end
end

@doc raw"""
    distance(M::Grassmann, p, q)

Compute the Riemannian distance on [`Grassmann`](@ref) manifold `M`$= \mathrm{Gr}(n,k)$.

Let $USV = p^\mathrm{H}q$ denote the SVD decomposition of
$p^\mathrm{H}q$, where $\cdot^{\mathrm{H}}$ denotes the complex
conjugate transposed or Hermitian. Then the distance is given by
````math
d_{\mathrm{GR}(n,k)}(p,q) = \operatorname{norm}(\operatorname{Re}(b)).
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
    a[a.>1] .= 1
    return sqrt(sum((acos.(a)) .^ 2))
end

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
injectivity_radius(::Grassmann, ::Any...) = π / 2

@doc raw"""
    inner(M::Grassmann, p, X, Y)

Compute the inner product for two tangent vectors `X`, `Y` from the tangent space
of `p` on the [`Grassmann`](@ref) manifold `M`. The formula reads

````math
g_p(X,Y) = \operatorname{tr}(X^{\mathrm{H}}Y),
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inner(::Grassmann, p, X, Y) = dot(X, Y)

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
\operatorname{retr}_p^{-1}q = q*(p^\mathrm{H}q)^{-1} - p,
````
where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(::Grassmann, ::Any, ::Any, ::QRInverseRetraction)

inverse_retract!(::Grassmann, X, p, q, ::QRInverseRetraction) = copyto!(X, q / (p' * q) - p)

function isapprox(M::Grassmann, p, X, Y; kwargs...)
    return isapprox(sqrt(inner(M, p, zero_tangent_vector(M, p), X - Y)), 0; kwargs...)
end
isapprox(M::Grassmann, p, q; kwargs...) = isapprox(distance(M, p, q), 0.0; kwargs...)

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

function log!(M::Grassmann, X, p, q)
    z = q' * p
    At = q' - z * p'
    Bt = z \ At
    d = svd(Bt')
    return copyto!(X, d.U * Diagonal(atan.(d.S)) * d.Vt)
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
mean(::Grassmann{n,k,ℝ} where {n,k}, ::Any...)

function mean!(
    M::Grassmann{n,k,ℝ},
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {n,k}
    return mean!(M, p, x, w, GeodesicInterpolationWithinRadius(π / 4); kwargs...)
end

@doc raw"""
    project_tangent(M::Grassmann, p, X)

Project the `n`-by-`k` `X` onto the tangent space of `p` on the [`Grassmann`](@ref) `M`,
which is computed by

````math
\operatorname{proj_p}(X) = X - pp^{\mathrm{H}}X,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
project_tangent(::Grassmann, ::Any...)

project_tangent!(M::Grassmann, v, x, w) = copyto!(v, w - x * x' * w)

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

show(io::IO, ::Grassmann{n,k,𝔽}) where {n,k,𝔽} = print(io, "Grassmann($(n), $(k), $(𝔽))")

@doc raw"""
    zero_tangent_vector(M::Grassmann, p)

Return the zero tangent vector from the tangent space at `p` on the [`Grassmann`](@ref) `M`,
which is given by a zero matrix the same size as `p`.
"""
zero_tangent_vector(::Grassmann, ::Any...)

zero_tangent_vector!(::Grassmann, X, p) = fill!(X, 0)
