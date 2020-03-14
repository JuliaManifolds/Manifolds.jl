@doc raw"""
    GeneralizedGrassmann{n,k,𝔽} <: Manifold

The generalized Grassmann manifold $\operatorname{Gr}(n,k,B)$ consists of all subspaces
spanned by $k$ linear independent vectors $𝔽^n$, where $𝔽  ∈ \{ℝ, ℂ\}$ is either the real- (or complex-) valued vectors.
This yields all $k$-dimensional subspaces of $ℝ^n$ for the real-valued case and all $2k$-dimensional subspaces
of $ℂ^n$ for the second.

The manifold can be represented as

````math
\operatorname{Gr}(n, k, B) := \bigl\{ \operatorname{span}(p) : p ∈ 𝔽^{n × k}, p^\mathrm{H}Bp = I_k\},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian and
$I_k$ is the $k × k$ identity matrix. This means, that the columns of $p$
form an unitary basis of the subspace  with respect to the scaled inner product, that is a
point on $\operatorname{Gr}(n,k,B)$, and hence the subspace can actually be represented by
a whole equivalence class of representers. For $B=I_n$ this simplifies to the [`Grassmann`](@ref) manifold.

The tangent space at a point (subspace) $x$ is given by

````math
T_x\mathrm{Gr}(n,k) = \bigl\{
X ∈ 𝔽^{n × k} :
X^{\mathrm{H}}Bp + p^{\mathrm{H}}BX = 0_{k} \bigr\},
````

where $0_{k}$ denotes the $k × k$ zero matrix.

Note that a point $p ∈ \operatorname{Gr}(n,k,B)$ might be represented by
different matrices (i.e. matrices with $B$-unitary column vectors that span
the same subspace). Different representations of $p$ also lead to different
representation matrices for the tangent space $T_p\mathrm{Gr}(n,k)$

The manifold is named after
[Hermann G. Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).

# Constructor

    Grassmann(n,k,B=Matrixfield=ℝ)

Generate the Grassmann manifold $\operatorname{Gr}(n,k)$, where the real-valued
case $field = ℝ$ is the default.
"""
struct GeneralizedGrassmann{n,k,TB<:AbstractMatrix,𝔽} <:
       AbstractEmbeddedManifold{DefaultIsometricEmbeddingType}
       B::TB
end

function GeneralizedGrassmann(
    n::Int,
    k::Int,
    B::AbstractMatrix = Matrix{Float64}(I, n, n),
    field::AbstractNumbers = ℝ,
)
    return GeneralizedGrassmann{n,k,typeof(B),field}(B)
end

@doc raw"""
    check_manifold_point(M::GeneralizedGrassmann{n,k,B,𝔽}, p)

Check whether `p` is representing a point on the [`GeneralizedGrassmann`](@ref) `M`, i.e. its
a `n`-by-`k` matrix of unitary column vectors with respect to the B inner prudct and
of correct `eltype` with respect to `𝔽`.
"""
function check_manifold_point(M::GeneralizedGrassmann{n,k,B,𝔽}, p; kwargs...) where {n,k,B,𝔽}
    mpv = invoke(check_manifold_point, Tuple{typeof(get_embedding(M)), typeof(p)}, get_embedding(M), p; kwargs...)
    mpv === nothing || return mpv
    c = p' * M.B * p
    if !isapprox(c, one(c); kwargs...)
        return DomainError(
            norm(c - one(c)),
            "The point $(p) does not lie on $(M), because x'Bx is not the unit matrix.",
        )
    end
end

@doc raw"""
    check_tangent_vector(M::GeneralizedGrassmann{n,k,B,𝔽}, p, X; check_base_point = true, kwargs...)

Check whether `X` is a tangent vector in the tangent space of `p` on
the [`GeneralizedGrassmann`](@ref) `M`, i.e. that `X` is of size and type as well as that

````math
    p^{\mathrm{H}}X + X^{\mathrm{H}}p = 0_k,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transpose or Hermitian and $0_k$
denotes the $k × k$ zero natrix.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
"""
function check_tangent_vector(M::GeneralizedGrassmann{n,k,𝔽}, p, X; check_base_point = true, kwargs...) where {n,k,𝔽}
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{typeof(get_embedding(M)), typeof(p), typeof(X)},
        get_embedding(M),
        p,
        X;
        check_base_point = check_base_point,
        kwargs...
    )
    if !isapprox(p' * M.B * X + X' * M.B * p, zeros(k, k); kwargs...)
        return DomainError(
            norm(p' * M.B * X + X' * M.B * p),
            "The matrix $(X) does not lie in the tangent space of $(p) on $(M), since x'Bv + v'Bx is not the zero matrix.",
        )
    end
end

@doc raw"""
    distance(M::GeneralizedGrassmann, p, q)

Compute the Riemannian distance on [`GeneralizedGrassmann`](@ref) manifold `M`$= \mathrm{Gr}(n,k)$.

Let $USV = p^\mathrm{H}Bq$ denote the SVD decomposition of
$p^\mathrm{H}Bq$, where $\cdot^{\mathrm{H}}$ denotes the complex
conjugate transposed or Hermitian. Then the distance is given by
````math
d_{\mathrm{GR}(n,k,B)}(p,q) = \operatorname{norm}(\operatorname{Re}(b)).
````
where

````math
b_{i}=\begin{cases}
0 & \text{if} \; S_i ≥ 1\\
\arccos(S_i) & \, \text{if} \; S_i<1.
\end{cases}
````
"""
function distance(M::GeneralizedGrassmann, p, q)
    p ≈ q && return zero(real(eltype(p)))
    a = svd(p' * M.B * q).S
    a[a.>1] .= 1
    return sqrt(sum((acos.(a)) .^ 2))
end

@doc raw"""
    exp(M::GeneralizedGrassmann, p, X)

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
exp(::GeneralizedGrassmann, ::Any...)

function exp!(M::GeneralizedGrassmann, q, p, X)
    norm(M, p, X) ≈ 0 && return copyto!(q, p)
    d = svd(X'*M.B*X)
    V = d.Vt
    S = abs(sqrt(d.S))
    U = X*(V/Diagonal(S))
    z = p * V * Diagonal(cos.(S)) * V + U * Diagonal(sin.(S)) * V
    return copyto!(q, Array(qr(z).Q))
end

@doc raw"""
    injectivity_radius(M::GeneralizedGrassmann)
    injectivity_radius(M::GeneraizedGrassmann, p)

Return the injectivity radius on the [`GeneralizedGrassmann`](@ref) `M`, which is $\frac{π}{2}$.
"""
injectivity_radius(::GeneralizedGrassmann, ::Any...) = π / 2

@doc raw"""
    inner(M::GeneralizedGrassmann, p, X, Y)

Compute the inner product for two tangent vectors `X`, `Y` from the tangent space
of `p` on the [`GeneralizedGrassmann`](@ref) manifold `M`. The formula reads

````math
g_p(X,Y) = \operatorname{tr}(X^{\mathrm{H}}BY),
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inner(::GeneralizedGrassmann{n,k,B}, p, X, Y) where {n,k,B} = dot(X, M.B * Y)

@doc raw"""
    inverse_retract(M::Grassmann, p, q, ::PolarInverseRetraction)

Compute the inverse retraction for the [`PolarRetraction`](@ref), on the
[`Grassmann`](@ref) manifold `M`, i.e.,

````math
\operatorname{retr}_p^{-1}q = q*(p^\mathrm{H}q)^{-1} - p,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
inverse_retract(::GeneralizedGrassmann, ::Any, ::Any, ::PolarInverseRetraction)

function inverse_retract!(::GeneralizedGrassmann, X, p, q, ::PolarInverseRetraction)
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
inverse_retract(::GeneralizedGrassmann, ::Any, ::Any, ::QRInverseRetraction)

inverse_retract!(::GeneralizedGrassmann, X, p, q, ::QRInverseRetraction) = copyto!(X, q / (p' * q) - p)

function isapprox(M::GeneralizedGrassmann, p, X, Y; kwargs...)
    return isapprox(sqrt(inner(M, p, zero_tangent_vector(M, p), X - Y)), 0; kwargs...)
end
isapprox(M::GeneralizedGrassmann, p, q; kwargs...) = isapprox(distance(M, p, q), 0.0; kwargs...)

@doc raw"""
    log(M::GeneralizedGrassmann, p, q)

Compute the logarithmic map on the [`GeneralizedGrassmann`](@ref) `M`$ = \mathcal M=\mathrm{Gr}(n,k,B)$,
i.e. the tangent vector `X` whose corresponding [`geodesic`](@ref) starting from `p`
reaches `q` after time 1 on `M`. The formula reads

````math
\log_p q = V\cdot \operatorname{atan}(S) \cdot U^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
The matrices $U$ and $V$ are the unitary matrices, and $S$ is the diagonal matrix
containing the singular values of the SVD-decomposition

````math
USV = (q^\mathrm{H}Bp)^{-1} ( q^\mathrm{H} - q^\mathrm{H}Bpp^\mathrm{H}).
````

In this formula the $\operatorname{atan}$ is meant elementwise.
"""
log(::GeneralizedGrassmann, ::Any...)

function log!(M::GeneralizedGrassmann, X, p, q)
    z = q' * M.B * p
    At = q' - z * p'
    Bt = z \ At
    d = svd(Bt')
    return copyto!(X, d.U * Diagonal(atan.(d.S)) * d.Vt)
end

@doc raw"""
    manifold_dimension(M::GeneralizedGrassmann)

Return the dimension of the [`Grassmann(n,k,𝔽)`](@ref) manifold `M`, i.e.

````math
\dim \operatorname{Gr}(n,k) = k(n-k) \dim_ℝ 𝔽,
````

where $\dim_ℝ 𝔽$ is the [`real_dimension`](@ref) of `𝔽`.
"""
manifold_dimension(::GeneralizedGrassmann{n,k,B,𝔽}) where {n,k,B,𝔽} = k * (n - k) * real_dimension(𝔽)

"""
    mean(
        M::GeneralizedGrassmann,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/4);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::GeneralizedGrassmann{n,k} where {n,k}, ::Any...)

function mean!(
    M::GeneralizedGrassmann{n,k},
    p,
    x::AbstractVector,
    w::AbstractVector;
    kwargs...,
) where {n,k}
    return mean!(M, p, x, w, GeodesicInterpolationWithinRadius(π / 4); kwargs...)
end

@doc doc"""
    project_point(M::GeneralizedGrassmann, p)

Project `p` from the embedding onto the [`GeneralizedGrassmann`](@ref) `M`, i.e. compute `q`
as the polar decomposition of $p$ such that $q^{\mathrm{H}}Bq$ is the identity,
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project_point(::GeneralizedGrassmann, ::Any...)

function project_point!(M::GeneralizedGrassmann, q, p)
    s = svd(p)
    e = eigen(s.U' * M.B * s.U)
    qsinv = e.vectors * Diagonal(1 ./ sqrt.(e.values))
    q .= s.U * qsinv * e.vectors' * s.V'
    return q
end

@doc raw"""
    project_tangent(M::GeneralizedGrassmann, p, X)

Project the `n`-by-`k` `X` onto the tangent space of `p` on the [`Grassmann`](@ref) `M`,
which is computed by

````math
\operatorname{proj_p}(X) = X - pp^{\mathrm{H}}X,
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
project_tangent(::GeneralizedGrassmann, ::Any...)

function project_tangent!(M::GeneralizedGrassmann, Y, p, X)
    A = p' * M.B' * X
    copyto!(Y, X - p * Hermitian((A + A') / 2))
    return Y
end

@doc raw"""
    representation_size(M::GeneralizedGrassmann{n,k})

Return the represenation size or matrix dimension of a point on the [`GeneralizedGrassmann`](@ref)
`M`, i.e. $(n,k)$ for both the real-valued and the complex value case.
"""
@generated representation_size(::GeneralizedGrassmann{n,k}) where {n,k} = (n, k)

@doc raw"""
    retract(M::GeneralizedGrassmann, p, X, ::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@ref) on the
[`Grassmann`](@ref) `M`. With $USV = p + X$ the retraction reads
````math
\operatorname{retr}_p X = UV^\mathrm{H},
````

where $\cdot^{\mathrm{H}}$ denotes the complex conjugate transposed or Hermitian.
"""
retract(::GeneralizedGrassmann, ::Any, ::Any, ::PolarRetraction)

function retract!(M::GeneralizedGrassmann, q, p, X, ::PolarRetraction)
    project_point!(M,q, p + X)
    return q
end
function retract!(M::GeneralizedGrassmann, q, p, X, ::ProjectionRetraction)
    project_point!(M,q, p + X)
    return q
end

show(io::IO, ::GeneralizedGrassmann{n,k,B,𝔽}) where {n,k,B,𝔽} = print(io, "GeneralizedGrassmann($(n), $(k), $(B), $(𝔽))")

@doc raw"""
    zero_tangent_vector(M::Grassmann, p)

Return the zero tangent vector from the tangent space at `p` on the [`Grassmann`](@ref) `M`,
which is given by a zero matrix the same size as `p`.
"""
zero_tangent_vector(::GeneralizedGrassmann, ::Any...)

zero_tangent_vector!(::GeneralizedGrassmann, X, p) = fill!(X, 0)
