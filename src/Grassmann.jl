using LinearAlgebra: svd, qr, diag, Diagonal, det
import LinearAlgebra: norm
@doc doc"""
    Grassmann{n,k,T} <: Manifold

The Grassmann manifold $\operatorname{Gr}(n,k)$ consists of all subspaces spanned
by $k$ linear independent vectors $\mathbb F^n$, where $\mathbb F \in \{\mathbb R, \mathbb C\}$ is
either the real- (or complex-) valued vectors. This yields all $k$-dimensional
subspaces of $\mathbb R^n$ for the real-valued case and all $2k$-dimensional
subspaces of $\mathbb C^n$ for the second.

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

    Grassmann(n,k,T=Real)

generate the Grassmann manifold $\operatorname{Gr}(n,k)$, where the real-valued
case $\mathbb F = \mathbb R$ is the default.
"""
struct Grassmann{N,K,T} <: Manifold end
Grassmann(n::Int, k::Int, T::Type = Real) = Grassmann{n, k, T}()

function check_manifold_point(G::Grassmann{N,K,T},x; kwargs...) where {N,K,T}
    if (T <: Real) && !(eltype(x) <: Real)
        return DomainError(eltype(x),
            "The matrix $(x) is not a real-valued matrix, so it does noe lie on the Grassmann manifold of dimension ($(N),$(K)).")
    end
    if (T <: Complex) && !(eltype(x) <: Real) && !(eltype(x) <: Complex)
        return DomainError(eltype(x),
            "The matrix $(x) is neiter real- nor complex-valued matrix, so it does noe lie on the complex Grassmann manifold of dimension ($(N),$(K)).")
    end
    if size(x) != representation_size(G)
        return DomainError(size(x),
            "The matrix $(x) is does not lie on the Grassmann manifold of dimension ($(N),$(K)), since its dimensions are wrong.")
    end
    c = x'*x
    if !isapprox(c, one(c); kwargs...)
        return DomainError(norm(c-one(c)),
            "The point $(x) does not lie on the Grassmann manifold of dimension ($(N),$(K)), because x'x is not the unit matrix.")
    end
end
function check_tangent_vector(G::Grassmann{N,K,T},x,v; kwargs...) where {N,K,T}
    t = check_manifold_point(G,x)
    if (t !== nothing)
        return t
    end
    if (T <: Real) && !(eltype(v) <: Real)
        return DomainError(eltype(v),
            "The matrix $(v) is not a real-valued matrix, so it can not be a tangent vector to the Grassmann manifold of dimension ($(N),$(K)).")
    end
    if (T <: Complex) && !(eltype(v) <: Real) && !(eltype(v) <: Complex)
        return DomainError(eltype(v),
            "The matrix $(v) is neiter real- nor complex-valued matrix, so it can not bea tangent vector to the complex Grassmann manifold of dimension ($(N),$(K)).")
    end
    if size(v) != representation_size(G)
        return DomainError(size(v),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Grassmann manifold of dimension ($(N),$(K)), since its dimensions are wrong.")
    end
    if !isapprox(x'*v + v'*x, zeros(K,K); kwargs...)
        return DomainError(norm(x'*v + v'*x),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Grassmann manifold of dimension ($(N),$(K)), since x'v + v'x is not the zero matrix.")
    end
end

@doc doc"""
    distance(M,x,y)

computes the Riemannian distance on [`Grassmann`](@ref) manifold `M`$= \mathrm{Gr}(n,k)$.

Let $USV = {\bar x}^\mathrm{T}y$ denote the SVD decomposition of
${\bar x}^\mathrm{T}y$, where ${\bar\cdot}^{\mathrm{T}}$ denotes the complex
conjugate transposed. Then the distance is given by
````math
d_{\mathrm{GR}(n,k)}(x,y) = \operatorname{norm}(\operatorname{Re}(b)).
````
where

$b_{i}=\begin{cases} 0 & \text{if} \; S_i \geq 1\\ \operatorname{acos}(S_i) & \, \text{if} \; S_i<1 \end{cases}.$
"""
function distance(M::Grassmann{N,K,T}, x, y) where {N,K,T}
    if x ≈ y
        return 0.
    else
        a = svd(x'*y).S
        a[a .> 1] .= 1
        return sqrt(sum( (acos.(a)).^2 ))
    end
end

@doc doc"""
    exp!(M, y, x, v)

compute the exponential map on the [`Grassmann`](@ref) manifold
`M`$= \mathrm{Gr}(n,k)$ starting in `x` with tangent vector (direction) `v`
and store the result in `y`.
Let $v = USV$ denote the SVD decomposition of $v$.
Then the exponential map is written using
````math
z = x V\cos(S){\bar V}^\mathrm{T} + U\sin(S){\bar V}^\mathrm{T},
````
where ${\bar\cdot}^{\mathrm{T}}$ denotes the complex conjugate transposed.
The cosine and sine are applied element wise to the diagonal entries of $S$.
A final QR decomposition $z=QR$ is performed for numerical stability reasons,
yielding the result as
````math
y = Q.
````
"""
function exp!(M::Grassmann{N,K,T},y, x, v) where {N,K,T}
    if norm(M,x,v) ≈ 0
        return (y .= x)
    end
    d = svd(v)
    z =  x * d.V * Diagonal(cos.(d.S)) * d.Vt + d.U * Diagonal(sin.(d.S)) * d.Vt
    # reorthonormalize
    copyto!(y, Array(qr(z).Q) )
    return y
end

@doc doc"""
    inner(M,x,v,w)

compute the inner product for two tangent vectors `v`, `w` from the
tangent space of `x` on the [`Grassmann`](@ref) manifold `M`.
The formula reads
````math
(v,w)_x = \operatorname{trace}({\bar v}^{\mathrm{T}}w),
````
where ${\bar\cdot}^{\mathrm{T}}$ denotes the complex conjugate transposed.
"""
inner(::Grassmann{N,K,T}, x, v, w) where {N,K,T} = real(dot(v,w))

@doc doc"""
    inverse_retract!(M, v, x, y, ::PolarInverseRetraction)

compute the inverse retraction for the [`PolarRetraction`](@ref)

````math
\operatorname{retr}_x^{-1}y = y*(\bar{x}^\mathrm{T}y)^{-1} - x,
````
where ${\bar\cdot}^{\mathrm{T}}$ denotes the complex conjugate transposed.
"""
inverse_retract!(::Grassmann{N,K,T}, v, x, y, ::PolarInverseRetraction) where {N,K,T} = ( v .= y/(x'*y) - x)

@doc doc"""
    inverse_retract!(M, v, x, y, ::QRInverseRetraction)

compute the inverse retraction valid of the [`QRRetraction`](@ref)

````math
\operatorname{retr}_x^{-1}y = y*(\bar{x}^\mathrm{T}y)^{-1} - x,
````
where ${\bar\cdot}^{\mathrm{T}}$ denotes the complex conjugate transposed.
"""
inverse_retract!(::Grassmann{N,K,T}, v, x, y, ::QRInverseRetraction) where {N,K,T} = ( v .= y/(x'*y) - x)

isapprox(M::Grassmann{N,K,T}, x, y; kwargs...) where {N,K,T} = isapprox(distance(M,x,y),0.; kwargs...)

@doc doc"""
    log!(M, v, x, y)

compute the logarithmic map on the [`Grassmann`](@ref) manifold
$\mathcal M=\mathrm{Gr}(n,k)$, i.e. the tangent vector `v` whose corresponding
[`geodesic`](@ref) starting from `x` reaches `y` after time 1 on `M`. The formula reads
````math
v = V\cdot \operatorname{atan}(S) \cdot {\bar U}^\mathrm{T},
````
where ${\bar\cdot}^{\mathrm{T}}$ denotes the complex conjugate transposed,
$U$ and $V$ are the unitary matrices, and $S$ is a diagonal matrix containing
the singular values of the SVD-decomposition of
````math
USV = ({\bar y}^\mathrm{T}x)^{-1} ( {\bar y}^\mathrm{T} - {\bar y}^\mathrm{T}x{\bar x}^\mathrm{T} ).
````
In this formula the $\operatorname{atan}$ is meant elementwise.
"""
function log!(M::Grassmann{N,K,T}, v, x, y) where {N,K,T}
    z = y'*x
    At = y' - z*x'
    Bt = z\At
    d = svd(Bt')
    v .= d.U * Diagonal(atan.(d.S)) * d.Vt
    return v
end

@doc doc"""
    manifold_dimension(M)

return the dimension of the real-valued [`Grassmann`](@ref)`(n,k)` manifold `M`,
i.e.
````math
k(n-k)
````
"""
manifold_dimension(M::Grassmann{N,K,Real}) where {N,K} = K*(N-K)

@doc doc"""
    manifold_dimension(M)

return the dimension of the complex-valued [`Grassmann`](@ref)`(n,k)` manifold `M`,
i.e.
````math
2k(n-k)
````
"""
manifold_dimension(M::Grassmann{N,K,Complex}) where {N,K} = 2*K*(N-K)

project_tangent!(M::Grassmann{N,K,T},v, x, w) where {N,K,T} = ( v .= w - x*x'*w )

@doc doc"""
    retract!(M, y, x, v, ::PolarRetraction)

compute the SVD-based retraction [`PolarRetraction`](@ref) on the
[`Grassmann`](@ref) manifold `M`. With $USV = x + v$ the retraction reads
````math
y = \operatorname{retr}_x v = U\bar{V}^\mathrm{T},
````
where ${\bar\cdot}^{\mathrm{T}}$ denotes the complex conjugate transposed.
"""
function retract!(::Grassmann{N,K,T}, y, x, v, ::PolarRetraction) where {N,K,T}
    s = svd(x+v)
    mul!(y, s.U, s.V')
   return y
end

@doc doc"""
    retract!(M, y, x, v, ::QRRetraction )

compute the QR-based retraction [`QRRetraction`](@ref) on the
[`Grassmann`](@ref) manifold `M`. With $QR = x + v$ the retraction reads
````math
y = \operatorname{retr}_xv = QD,
````
where D is a $m\times n$ matrix with
````math
D = \operatorname{diag}( \operatorname{sgn}(R_{ii}+0,5)_{i=1}^n ).
````
"""
function retract!(::Grassmann{N,K,T}, y, x, v, ::QRRetraction) where {N,K,T}
    qrfac = qr(x+v)
    d = diag(qrfac.R)
    D = Diagonal( sign.( sign.(d .+ convert(T, 0.5))) )
    y .= zeros(N,K)
    y[1:K,1:K] .= D
    y .= Array(qrfac.Q) * D
    return y
end

@generated representation_size(::Grassmann{N,K,T}) where {N, K, T} = (N,K)
zero_tangent_vector!(::Grassmann{N,K,T},v,x) where {N,K,T} = fill!(v,0)

@doc doc"""
    mean(M::Grassmann, x::AbstractVector[, w::AbstractWeights]; shuffle_rng=nothing, kwargs...)

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
[`GeodesicInterpolation`](@ref). If any `x` are not within
$\frac{\pi}{4}$ of the estimated mean, then the estimate is used to initialize
mean computation using the [`GradientDescent`](@ref).
"""
mean(::Grassmann, args...)

function mean!(M::Grassmann, y, x::AbstractVector, w::AbstractWeights; shuffle_rng = nothing, kwargs...)
    mean!(M, y, x, w, GeodesicInterpolation(); shuffle_rng = shuffle_rng, kwargs...)
    for i in eachindex(x)
        @inbounds if distance(M, y, x[i]) ≥ π/4
            return mean!(M, y, x, w, GradientDescent(); x0 = y, kwargs...)
        end
    end
    return y
end
