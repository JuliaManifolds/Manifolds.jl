using LinearAlgebra: svd, qr, Diagonal, det
import LinearAlgebra: norm
@doc doc"""
    Grassmann{n,k,T} <: Manifold

The Grassmann manifold $\operatorname{Gr}(n,k)$ consists of all $k$-dimensional
subspaces of $\mathbb F^n$, where $\mathbb F \in \{\mathbb R, \mathbb C\}$ is
either the real- (or complex-) valued subspaces of the ($2$)$n$-dimensional
(complex) space.

The manifold can be represented as
````math
\operatorname{Gr}(n,k) \coloneqq \bigl\{ \operatorname{span}(x)
: x \in \mathbb F^{n\times k}, \bar{x}^\mathrm{T}x = I_k\},
````
where $\bar\cdot$ denotes the complex conjugate and $I_k$ is the $k\times k$
identity matrix. This means, that the columns of $x$ form an orthonormal basis
of the subspace, that is a point on $\operatorname{Gr}(n,k)$, and hence the
subspace can actually be represented by a whole equivalence class of representers.
Another interpretation is, that
````math
\operatorname{Gr}(n,k) = \operatorname{St}(n,k) / \operatorname{O}(k),
````
i.e the Grassmann manifold is the quotient of the [`Stiefel`](@ref) manifold and
the orthogonal group $\operatorname{O}(k)$ of orthogonal $k\times k$ matrices.

The tangent space at a point (subspace) $x$ is given by
````math
T_x\mathrm{Gr}(n,k) = \bigl\{
v \in \mathbb{F^{n\times k} : 
{\bar v}^\mathrm{T}x + {\bar x}^\mathrm{T}v = 0_{k} \bigr\},
````
where $0_{k}$ denotes the $k\times k$ zero matrix.

The manifold is named after
[Hermann G. Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).

# Constructor

    Grassmann(n,k,T=Real)

generate the Grassmann manifold $\operatorname{Gr}(n,k)$ (on $\mathbb R^n$).
"""
struct Grassmann{N,K,T} <: Manifold end
Grassmann(n::Int, k::Int, T::Type = Real) = Grassmann{n, k, T}()

function check_manifold_point(G::Grassmann{M,N,T},x; kwargs...) where {M,N,T}
    if (T <: Real) && !(eltype(x) <: Real)
        return DomainError(eltype(x),
            "The matrix $(x) is not a real-valued matrix, so it does noe lie on the Grassmann manifold of dimension ($(M),$(N)).")
    end
    if (T <: Complex) && !(eltype(x) <: Real) && !(eltype(x) <: Complex)
        return DomainError(eltype(x),
            "The matrix $(x) is neiter real- nor complex-valued matrix, so it does noe lie on the complex Grassmann manifold of dimension ($(M),$(N)).")
    end
    if any( size(x) != representation_size(G) )
        return DomainError(size(x),
            "The matrix $(x) is does not lie on the Grassmann manifold of dimension ($(M),$(N)), since its dimensions are wrong.")
    end
    c = x'*x
    if !isapprox(c, one(c); kwargs...)
        return DomainError(norm(c-one(c)),
            "The point $(x) does not lie on the Grassmann manifold of dimension ($(M),$(N)), because x'x is not the unit matrix.")
    end
end
function check_tangent_vector(G::Grassmann{M,N,T},x,v; kwargs...) where {M,N,T}
    t = check_manifold_point(G,x)
    if (t != nothing)
        return t
    end
    if (T <: Real) && !(eltype(v) <: Real)
        return DomainError(eltype(v),
            "The matrix $(v) is not a real-valued matrix, so it can not be a tangent vector to the Grassmann manifold of dimension ($(M),$(N)).")
    end
    if (T <: Complex) && !(eltype(v) <: Real) && !(eltype(v) <: Complex)
        return DomainError(eltype(v),
            "The matrix $(v) is neiter real- nor complex-valued matrix, so it can not bea tangent vector to the complex Grassmann manifold of dimension ($(M),$(N)).")
    end
    if any( size(v) != representation_size(G) )
        return DomainError(size(v),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Grassmann manifold of dimension ($(M),$(N)), since its dimensions are wrong.")
    end
    if !isapprox(x'*v + v'*x, zeros(N,N); kwargs...)
        return DomainError(norm(x'*v + v'*x),
            "The matrix $(v) is does not lie in the tangent space of $(x) on the Grassmann manifold of dimension ($(M),$(N)), since x'v + v'x is not the zero matrix.")
    end
end

@doc doc"""
    distance(M,x,y)

compute the Riemannian distance on [`Grassmann`](@ref) manifold `M`$= \mathrm{Gr}(k,n)$.
Let $USV = {\bar x}^\mathrm{T}y$ denote the SVD decomposition of
$x'y$. Then the distance is given by
````math
d_{\mathrm{GR}(k,n)}(x,y) = \operatorname{norm}(\operatorname{Re}(b)).
````
where

$b_{i}=\begin{cases} 0 & \text{if} \; S_i≧1 \\ \operatorname{acos}(S_i) & \, \text{if} \; S_i<1 \end{cases}.$
"""
function distance(M::Grassmann{N,K,T}, x, y) where {N,K,T}
    if x==y
		return 0
  	else
    	a = svd(x'*y).S
    	b = zero(a)
    	b[a.<1] = (acos.(a[a.<1])).^2
		return sqrt(sum(b))
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
where cosine and sine are applied element wise to the diagonal entries of $S$,
as
````math
y = Q,
````
where $Q$ of the QR decomposition $z=QR$ of $z$. This last step is for numerical
stability reasons.
"""
function exp!(M::Grassmann{N,K,T},y, x, v) where {N,K,T}
    d = svd(v)
    z = x * d.V * cos.(Diagonal(d.S)) * (d.V)' + (d.U) * sin.(Diagonal(d.S)) * (d.V)'
    # reorthonormalize 
    y = qr(z).Q
end

@doc doc"""
    inner(M,x,v,w)

compute the inner product for two tangent vectors `v`, `w` from the
tangent space of `x` on the [`Grassmann`](@ref) manifold `M`.
The formula reads
````math
(v,w)_x = \operatorname{trace}({\bar v}^{\mathrm{T}}w).
````
"""
inner(::Grassmann{M,N,T}, x, v, w) where {M,N,T} = real(dot(v,w))

@doc doc"""
    inverse_retract!(M, v, x, y, ::PolarInverseRetraction)

compute the inverse retraction valid for the [`PolarRetraction`](@ref).

````math
\operatorname{retr}_x^{-1}y = y*(\bar{x}^\mathrm{T}y)^{-1} - x
````
"""
inverse_retract!(::Grassmann{M,N,T}, v, x, y, ::PolarInverseRetraction) where {M,N,T} = ( v .= y/(x'*y) - x)

@doc doc"""
    inverse_retract!(M, v, x, y, ::QRInverseRetraction)

compute the inverse retraction valid for the [`QRRetraction`](@ref) as

````math
\operatorname{retr}_x^{-1}y = y*(\bar{x}^\mathrm{T}y)^{-1} - x.
````
"""
inverse_retract!(::Grassmann{M,N,T}, v, x, y, ::QRInverseRetraction) where {M,N,T} = ( v .= y/(x'*y) - x)

@doc doc"""
    log!(M, v, x, y)

compute the logarithmic map on the [`Grassmann`](@ref) manifold
$\mathcal M=\mathrm{Gr}(n,k)$, i.e. the tangent vector `v` whose corresponding
[`geodesic`](@ref) starting from `x` reaches `y` after time 1 on `M`. The formula reads
````math
v = V\cdot \operatorname{atan}(S) \cdot {\bar U}^\mathrm{T},
````
where $U$ and $V$ are the unitary matrices and $S$ is a diagonal matrix containing the
singular values of the SVD-decomposition of

$USV = ({\bar y}^\mathrm{T}x)^{-1} ( {\bar y}^\mathrm{T} - {\bar y}^\mathrm{T}x{\bar x}^\mathrm{T} )$

and the $\operatorname{atan}$ is meant elementwise.
"""
function log!(M::Grassmann{N,K,T}, v, x, y) where {N,K,T}
    z = y'*x
  	if det(z)≠0
        d = svd( z\(y' - z*x'), full = false)
        v .= d.V * atan.(Diagonal(d.S)) * (d.U')
        return v   
  	else
   		throw( DomainError(rank(y'x),"The points x=$x and y=$y are antipodal (y'x has no full rank), thus these input parameters are invalid.") )
  	end
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
y = \operatorname{retr}_x v = U\bar{V}^\mathrm{T}.
````
"""
 function retract!(::Grassmann{M,N,T}, y, x, v, ::PolarRetraction) where {M,N,T}
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
 D = \operatorname{diag}( \operatorname{sgn}(R_{ii}+0,5)_{i=1}^n )
 ````
 """
 function retract!(::Grassmann{M,N,T}, y, x, v, ::QRRetraction) where {M,N,T}
     qrfac = qr(x+v)
     d = diag(qrfac.R)
     D = Diagonal( sign.( sign.(d .+ convert(T, 0.5))) )
     y .= zeros(M,N)
     y[1:N,1:N] .= D
     y .= qrfac.Q * D
     return y
 end

@generated representation_size(::Grassmann{N,K,T}) where {N, K, T} = (N,K)
zero_tangent_vector!(::Grassmann{N,K,T},v,x) where {N,K,T} = fill!(v,0)