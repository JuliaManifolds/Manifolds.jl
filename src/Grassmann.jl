using LinearAlgebra: svd, qr
import LinearAlgebra: norm
@doc doc"""
    Grassmann{n,k} <: Manifold

The Grassmann manifold $\operatorname{Gr}(n,k)$ consists of all $k$-dimensional
subspaces of $\mathbb F^n$, where $\mathbb F \in \{\mathbb R, \mathbb C\}$ is
either the real valued or complex valued $n$-dimensional vectors.

The manifold can be represented as
````math
\operatorname{Gr}(n,k) \colonegg \bigl\{ \operatorname{span}(x)
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
v \in \mathbb{K}^{n\times k} : 
{\bar v}^\mathrm{T}x + {\bar x}^\mathrm{T}v = 0_{k} \bigr\},
````
where $0_{k}$ denotes the $k\times k$ zero matrix.
is named after
[Hermann G. Graßmann](https://en.wikipedia.org/wiki/Hermann_Grassmann) (1809-1877).

# Constructor

    Grassmann(n,k,T=Real)

generate the Grassmann manifold $\operatorname{Gr}(n,k)$ (on $\mathbb R^n$).
"""
struct Grassmann{N,K,T} <: Manifold end
Grassmann(m::Int, n::Int,T::Type = Real) = Stiefel{m,n,T}()

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
    exp!(M, y, x, v)

compute the exponential map on the [`Grassmann`](@ref) manifold
`M`$= \mathrm{Gr}(n,k)$ starting in `x` with tangent vector (direction) `v`
and store the result in `y`.
Let $USV = v$ denote the SVD decomposition of $v$.
Then the exponential map is written using
````math
z = x V\cos(S){\bar V}^\mathrm{T} + U\sin(S){\bar V}^\mathrm{T},
````
where cosine and sine are applied element wise to the diagonal entries of $S$,
which yields that $y$ is the matrix $Q$ of the QR decomposition $z=QR$ of $z$.
"""
function exp(M::Grassmannian{N,K,T},y, x, v) where {N,K,T}
    d = svd(v)
    z = x * d.V * cos.(Diagonal(d.S)) * (d.V)' + (d.U) * sin.(Diagonal(d.S)) * (d.V)'
    y .= qr(z).Q
end

@doc doc"""
    inner(M,x,ξ,ν)

compute the inner product for two tangent vectors `v`, `w` from the
tangent space of `x` on the [`Grassmann`](@ref) manifold `M`.
The formula reads
````math
(v,w)_x = \operatorname{trace}({\bar v}^{\mathrm{T}}w).
````
"""
inner(::Grassmann{M,N,T}, x, v, w) where {M,N,T} = real(dot(v,w))

injectivity_radius(::Graddmann{M,K,T},x) where {N,K,T} = sqrt(K)
injectivity_radius(::Graddmann{M,K,T}) where {N,K,T} = sqrt(K)
end
