@doc doc"""
    Grassmann{n,k,F} <: Manifold

The Grassmann manifold $\operatorname{Gr}(n,k)$ consists of all subspaces spanned
by $k$ linear independent vectors $\mathbb F^n$, where
$\mathbb F \in \{\mathbb R, \mathbb C\}$ is either the real- (or complex-) valued vectors.
This yields all $k$-dimensional subspaces of $\mathbb R^n$ for the real-valued case and all
$2k$-dimensional subspaces of $\mathbb C^n$ for the second.

The manifold can be represented as

````math
\operatorname{Gr}(n,k) \coloneqq \bigl\{ \operatorname{span}(x)
: x \in \mathbb F^{n\times k}, \bar{x}^\mathrm{T}x = I_k\},
````

where ${\bar\cdot}^{\mathrm{T}}$ denotes the complex conjugate transpose and
$I_k$ is the $k\times k$ identity matrix. This means, that the columns of $x$
form an orthonormal basis of the subspace, that is a point on
$\operatorname{Gr}(n,k)$, and hence the subspace can actually be represented by
a whole equivalence class of representers.
Another interpretation is, that

````math
\operatorname{Gr}(n,k) = \operatorname{St}(n,k) / \operatorname{O}(k),
````

i.e the Grassmann manifold is the quotient of the [`Stiefel`](@ref) manifold and
the orthogonal group $\operatorname{O}(k)$ of orthogonal $k\times k$ matrices.

The tangent space at a point (subspace) $x$ is given by

````math
T_x\mathrm{Gr}(n,k) = \bigl\{
v \in \mathbb{F}^{n\times k} :
{\bar v}^{\mathrm{T}}x + {\bar x}^{\mathrm{T}}v = 0_{k} \bigr\},
````

where $0_{k}$ denotes the $k\times k$ zero matrix.

Note that a point $x\in \operatorname{Gr}(n,k)$ might be represented by
different matrices (i.e. matrices with orthonormal column vectors that span
the same subspace). Different representations of $x$ also lead to different
representation matrices for the tangent space $T_x\mathrm{Gr}(n,k)$

The manifold is named after
[Hermann G. Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).

# Constructor

    Grassmann(n,k,F=ℝ)

Generate the Grassmann manifold $\operatorname{Gr}(n,k)$, where the real-valued
case $\mathbb F = \mathbb R$ is the default.
"""
struct Grassmann{n,k,F} <: Manifold end

Grassmann(n::Int, k::Int, F::AbstractNumbers = ℝ) = Grassmann{n,k,F}()

@doc doc"""
    check_manifold_point(M::Grassmann{n,k,F}, x)

Check whether `x` is representing a point on the [`Grassmann`](@ref) `M`, i.e. its
a `n`-by-`k` matrix of unitary column vectors and of correct `eltype` with respect to `F`.
"""
function check_manifold_point(M::Grassmann{n,k,F}, x; kwargs...) where {n,k,F}
    if (F === ℝ) && !(eltype(x) <: Real)
        return DomainError(
            eltype(x),
            "The matrix $(x) is not a real-valued matrix, so it does noe lie on the Grassmann manifold of dimension ($(n),$(k)).",
        )
    end
    if (F === ℂ) && !(eltype(x) <: Real) && !(eltype(x) <: Complex)
        return DomainError(
            eltype(x),
            "The matrix $(x) is neiter real- nor complex-valued matrix, so it does noe lie on the complex Grassmann manifold of dimension ($(n),$(k)).",
        )
    end
    if size(x) != representation_size(M)
        return DomainError(
            size(x),
            "The matrix $(x) is does not lie on the Grassmann manifold of dimension ($(n),$(k)), since its dimensions are wrong.",
        )
    end
    c = x' * x
    if !isapprox(c, one(c); kwargs...)
        return DomainError(
            norm(c - one(c)),
            "The point $(x) does not lie on the Grassmann manifold of dimension ($(n),$(k)), because x'x is not the unit matrix.",
        )
    end
end

@doc doc"""
    check_tangent_vector(M::Grassmann{n,k,F}, x, v)

Check whether `v` is a tangent vector in the tangent space of `x` on the [`Grassmann`](@ref)
`M`, i.e. that `v` is of size and type as well as that

````math
    x^{\mathrm{H}}v + v^{\mathrm{H}}x = 0_k,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian and $0_k$
denotes the $k\times k$ zero natrix.
"""
function check_tangent_vector(G::Grassmann{n,k,F}, x, v; kwargs...) where {n,k,F}
    t = check_manifold_point(G, x)
    t === nothing || return t
    if (F === ℝ) && !(eltype(v) <: Real)
        return DomainError(
            eltype(v),
            "The matrix $(v) is not a real-valued matrix, so it can not be a tangent vector to the Grassmann manifold of dimension ($(n),$(k)).",
        )
    end
    if (F === ℂ) && !(eltype(v) <: Real) && !(eltype(v) <: Complex)
        return DomainError(
            eltype(v),
            "The matrix $(v) is neiter real- nor complex-valued matrix, so it can not bea tangent vector to the complex Grassmann manifold of dimension ($(n),$(k)).",
        )
    end
    if size(v) != representation_size(G)
        return DomainError(
            size(v),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Grassmann manifold of dimension ($(n),$(k)), since its dimensions are wrong.",
        )
    end
    if !isapprox(x' * v + v' * x, zeros(k, k); kwargs...)
        return DomainError(
            norm(x' * v + v' * x),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Grassmann manifold of dimension ($(n),$(k)), since x'v + v'x is not the zero matrix.",
        )
    end
end

@doc doc"""
    distance(M::Grassmann, x, y)

Compute the Riemannian distance on [`Grassmann`](@ref) manifold `M`$= \mathrm{Gr}(n,k)$.

Let $USV = x^\mathrm{H}y$ denote the SVD decomposition of
$x^\mathrm{H}y$, where $\cdot^{\mathrm{H}}$ denotes the complex
conjugate transposed or Hermitian. Then the distance is given by
````math
d_{\mathrm{GR}(n,k)}(x,y) = \operatorname{norm}(\operatorname{Re}(b)).
````
where

$b_{i}=\begin{cases} 0 & \text{if} \; S_i \geq 1\\ \operatorname{acos}(S_i) & \, \text{if} \; S_i<1 \end{cases}.$
"""
function distance(M::Grassmann, x, y)
    x ≈ y && return zero(real(eltype(x)))
    a = svd(x' * y).S
    a[a.>1] .= 1
    return sqrt(sum((acos.(a)) .^ 2))
end

@doc doc"""
    exp(M::Grassmann, x, v)

Compute the exponential map on the [`Grassmann`](@ref) `M`$= \mathrm{Gr}(n,k)$ starting in
`x` with tangent vector (direction) `v`. Let $v = USV$ denote the SVD decomposition of $v$.
Then the exponential map is written using

````math
z = x V\cos(S)V^\mathrm{H} + U\sin(S)V^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
The cosine and sine are applied element wise to the diagonal entries of $S$.
A final QR decomposition $z=QR$ is performed for numerical stability reasons,
yielding the result as
````math
\exp_x v = Q.
````
"""
exp(::Grassmann, ::Any...)

function exp!(M::Grassmann, y, x, v)
    norm(M, x, v) ≈ 0 && return copyto!(y, x)
    d = svd(v)
    z = x * d.V * Diagonal(cos.(d.S)) * d.Vt + d.U * Diagonal(sin.(d.S)) * d.Vt
    # reorthonormalize
    return copyto!(y, Array(qr(z).Q))
end

injectivity_radius(::Grassmann) = π / 2

@doc doc"""
    inner(M::Grassmann, x, v, w)

Compute the inner product for two tangent vectors `v`, `w` from the
tangent space of `x` on the [`Grassmann`](@ref) manifold `M`.
The formula reads

````math
g_x(v,w) = \operatorname{trace}(v^{\mathrm{H}}w),
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inner(::Grassmann, x, v, w) = dot(v, w)

@doc doc"""
    inverse_retract(M::Grassmann, x, y, ::PolarInverseRetraction)

Compute the inverse retraction for the [`PolarRetraction`](@ref), on the
[`Grassmann`](@ref), i.e.,

````math
\operatorname{retr}_x^{-1}y = y*(x^\mathrm{H}y)^{-1} - x,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(M::Grassmann, ::Any, ::Any, ::PolarInverseRetraction)

function inverse_retract!(::Grassmann, v, x, y, ::PolarInverseRetraction)
    return copyto!(v, y / (x' * y) - x)
end

@doc doc"""
    inverse_retract(M, x, y, ::QRInverseRetraction)

Compute the inverse retraction valid of the [`QRRetraction`](@ref)

````math
\operatorname{retr}_x^{-1}y = y*(x^\mathrm{H}y)^{-1} - x,
````
where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(::Grassmann, ::Any, ::Any, ::QRInverseRetraction)

inverse_retract!(::Grassmann, v, x, y, ::QRInverseRetraction) = copyto!(v, y / (x' * y) - x)

function isapprox(M::Grassmann, x, v, w; kwargs...)
    return isapprox(sqrt(inner(M, x, zero_tangent_vector(M, x), v - w)), 0; kwargs...)
end
isapprox(M::Grassmann, x, y; kwargs...) = isapprox(distance(M, x, y), 0.0; kwargs...)

@doc doc"""
    log(M::Grassmann, x, y)

Compute the logarithmic map on the [`Grassmann`](@ref) `M`$ = \mathcal M=\mathrm{Gr}(n,k)$,
i.e. the tangent vector `v` whose corresponding [`geodesic`](@ref) starting from `x`
reaches `y` after time 1 on `M`. The formula reads

````math
\log_xy = V\cdot \operatorname{atan}(S) \cdot U^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
$U$ and $V$ are the unitary matrices, and $S$ is a diagonal matrix containing
the singular values of the SVD-decomposition of
````math
USV = (y^\mathrm{H}x)^{-1} ( y^\mathrm{H} - y^\mathrm{H}xx^\mathrm{H} ).
````
In this formula the $\operatorname{atan}$ is meant elementwise.
"""
log(::Grassmann, ::Any...)

function log!(M::Grassmann, v, x, y)
    z = y' * x
    At = y' - z * x'
    Bt = z \ At
    d = svd(Bt')
    return copyto!(v, d.U * Diagonal(atan.(d.S)) * d.Vt)
end

@doc doc"""
    manifold_dimension(M::Grassmann)

Return the dimension of the [`Grassmann(n,k,𝔽)`](@ref) manifold `M`, i.e.

````math
\dim \operatorname{Gr}(n,k) = k(n-k) \dim_ℝ 𝔽,
````

where $\dim_ℝ 𝔽$ is the [`real_dimension`](@ref) of `𝔽`.
"""
manifold_dimension(M::Grassmann{n,k,𝔽}) where {n,k,𝔽} = k * (n - k) * real_dimension(𝔽)

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
    y,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {n,k}
    return mean!(M, y, x, w, GeodesicInterpolationWithinRadius(π / 4); kwargs...)
end

@doc doc"""
    project_tangent(M::Grassmann, x, w)

Project the `n`-by-`k` `w` onto the tangent space of `x` on the [`Grassmann`](@ref) `M`,
which is computed by

````math
\operatorname{proj_x}(w) = w - xx^{\mathrm{H}}w,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
project_tangent(::Grassmann, ::Any...)

project_tangent!(M::Grassmann, v, x, w) = copyto!(v, w - x * x' * w)

@doc doc"""
    representation_size(M::Grassmann{n,k,F})

Return the represenation size or matrix dimension of a point on the [`Grassmann`](@ref)
`M`, i.e. $(n,k)$ for both the real-valued and the complex value case.
"""
@generated representation_size(::Grassmann{n,k}) where {n,k} = (n, k)

@doc doc"""
    retract(M::Grassmann, x, v, ::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@ref) on the
[`Grassmann`](@ref) `M`. With $USV = x + v$ the retraction reads
````math
\operatorname{retr}_x v = UV^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.

    retract(M::Grassmann, x, v, ::QRRetraction )

Compute the QR-based retraction [`QRRetraction`](@ref) on the
[`Grassmann`](@ref) `M`. With $QR = x + v$ the retraction reads
````math
\operatorname{retr}_xv = QD,
````
where D is a $m\times n$ matrix with
````math
D = \operatorname{diag}( \operatorname{sgn}(R_{ii}+0,5)_{i=1}^n ).
````
"""
retract(::Grassmann, ::Any...)

function retract!(::Grassmann, y, x, v, ::PolarRetraction)
    s = svd(x + v)
    return mul!(y, s.U, s.Vt)
end
function retract!(::Grassmann{N,K}, y, x, v, ::QRRetraction) where {N,K}
    qrfac = qr(x + v)
    d = diag(qrfac.R)
    D = Diagonal(sign.(sign.(d .+ 0.5)))
    y .= zeros(N, K)
    y[1:K, 1:K] .= D
    return copyto!(y, Array(qrfac.Q) * D)
end

show(io::IO, ::Grassmann{n,k,F}) where {n,k,F} = print(io, "Grassmann($(n), $(k), $(F))")

@doc doc"""
    zero_tangent_vector(M::Grassmann, x)

Return the zero tangent vector from the tangent space at `x` on the [`Grassmann`](@ref) `M`,
which is given by a zero matrix the same size as `x`.
"""
zero_tangent_vector(::Grassmann, ::Any...)

zero_tangent_vector!(::Grassmann, v, x) = fill!(v, 0)
