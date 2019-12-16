using LinearAlgebra: diag, Diagonal, svd, SVD, rank, dot
import Base: \, /, +, -, *
@doc doc"""
    FixedRankMatrices{M,N,K} <: Manifold

The manifold of $m\times n$ real-valued matrices of fixed rank $k$, i.e.
````math
\mathcal M = \{ x \in \mathbb R^{m\times n} : \operatorname{rank}(x) = k \}.
````

# Representation with 3 matrix factors

A point $x\in\mathcal M$ can be stored using orthonormal matrices
$U\in\mathbb R^{m\times k}$, $V\in\mathbb R^{n\times k}$ as well as the $k$ singular
values of $x = USV^\mathrm{T}$. In other words, $U$ and $V$ are from the manifolds
[`Stiefel`](@ref)`(n,k)` and [`Stiefel`](@ref)`(m,k)`, respectively; see
[`SVDMPoint`](@ref) for details

The tangent space $T_x\mathcal M$ at a point $x\in\mathcal M$ with $x=USV^\mathrm{T}$
is given by
````math
T_x\mathcal M = \bigl\{ UMV^\mathrm{T} + U_xV^\mathrm{T} + UV_x^\mathrm{T} : 
    M \in \mathbb R^{k\times k},
    U_x \in \mathbb R^{m\times k},
    V_x \in \mathbb R^{n\times k}
    \text{ s.t. }
    U_x^\mathrm{T}U = 0_k,
    V_x^\mathrm{T}V = 0_k
\bigr\},
````
where $0_k$ is the $k\times k$ zero matrix. See [`UMVTVector`](@ref) for details.

The (default) metric of this manifold is obtained by restricting the metric
on $\mathbb R^{m\times n}$ to the tangent bundle.

This representation follows
> Bart Vandereycken: "Low-rank matrix completion by Riemannian Optimization,
> SIAM Journal on Optimization, 23(2), pp. 1214–1236, 2013.
> doi: [10.1137/110845768](https://doi.org/10.1137/110845768),
> arXiv: [1209.3834](https://arxiv.org/abs/1209.3834).
"""
struct FixedRankMatrices{M,N,K,T} <: Manifold end
FixedRankMatrices(m::Int, n::Int, k::Int, T::Type = Real) = FixedRankMatrices{m,n,k,T}()

@doc doc"""
    SVDMPoint <: MPoint

A point on a certain manifold, where the data is stored in a svd like fashion,
i.e. in the form $USV^\mathrm{T}$, where this structure stores $U$, $S$ and
$V^\mathrm{T}$. The storage might also be shortened to just $k$ singular values
and accordingly shortened $U$ (columns) and $V^\mathrm{T}$ (rows)

# Constructors
* `SVDMPoint(A)` for a matrix `A`, stores its svd factors (i.e. implicitly $k=\min\{m,n\}$)
* `SVDMPoint(S)` for an `SVD` object, stores its svd factors (i.e. implicitly $k=\min\{m,n\}$)
* `SVDMPoint(U,S,Vt)` for the svd factors to initialize the `SVDMPoint`` (i.e. implicitly $k=\min\{m,n\}$)
* `SVDMPoint(A,k)` for a matrix `A`, stores its svd factors shortened to the
  best rank $k$ approximation
* `SVDMPoint(S,k)` for an `SVD` object, stores its svd factors shortened to the
  best rank $k$ approximation
* `SVDMPoint(U,S,Vt,k)` for the svd factors to initialize the `SVDMPoint`,
  stores its svd factors shortened to the best rank $k$ approximation
"""
struct SVDMPoint{TU<:AbstractMatrix, TS<:AbstractVector, TVt<:AbstractMatrix} <: MPoint
    U::TU
    S::TS
    Vt::TVt
end
SVDMPoint(A::AbstractMatrix) = SVDMPoint(svd(A))
SVDMPoint(S::SVD) = SVDMPoint(S.U,S.S,S.Vt)
SVDMPoint(U,S,Vt) = SVDMPoint{eltype(U)}(U,S,Vt)
SVDMPoint(A::Matrix,k::Int) = SVDMPoint(svd(A),k)
SVDMPoint(S::SVD,k::Int) = SVDMPoint(S.U,S.S,S.Vt,k)
SVDMPoint(U,S,Vt,k::Int) = SVDMPoint(U[:,1:k],S[1:k],Vt[1:k,:])

@doc doc"""
    UMVTVector <: TVector

A tangent vector that can be described as a product $UMV^\mathrm{T}$, at least
together with its base point, see for example [`FixedRankMatrices`](@ref)

# Constructors
* `UMVTVector(U,S,Vt)` store umv factors to initialize the `UMVTVector`
* `UMVTVector(U,S,Vt,k)` store the umv factors after shortening them down to
  inner dimensions $k$, i.e. in $UMV^\mathrm{T}$, $M\in\mathbb R^{k\times k}$
"""
struct UMVTVector{TU<:AbstractMatrix, TM<:AbstractMatrix, TVt<:AbstractMatrix} <: TVector
    U::TU
    M::TM
    Vt::TVt
end
UMVTVector(U,M,Vt,k::Int) = UMVTVector(U[:,1:k],M[1:k,1:k],Vt[1:k,:])
UMVTVector(U,M,Vt) = UMVTVector{eltype(U)}(U,M,Vt)

*(v::UMVTVector, s::Number) = UMVTVector(v.U*s, v.M*s,  v.Vpt*s)
*(s::Number, v::UMVTVector) = UMVTVector(s*v.U, s*v.M, s*v.Vpt) 
/(v::UMVTVector, s::Number) = UMVTVector(v.U/s, v.M/s, v.Vt/s)
\(s::Number, v::UMVTVector) = UMVTVector(s\v.U, s\v.M, s\v.Vt)
+(v::UMVTVector, w::UMVTVector) = UMVTVector(v.U + w.U, v.M + w.M, v.Vt + w.Vt)
-(v::UMVTVector, w::UMVTVector) = UMVTVector(v.U - w.U, v.M - w.M, v.Vt - w.Vt)
-(v::UMVTVector) = UMVTVector(-v.U, -v.M, -v.Vt)
+(v::UMVTVector) = UMVTVector(v.U, v.M, v.Vt)


function check_manifold_point(F::FixedRankMatrices{M,N,k,T},x; kwargs...) where {M,N,k,T}
    r = rank(x; kwargs...)
    s = "The point $(x) does not lie on the manifold of fixed rank matrices of size ($(M),$(N)) witk rank $(k), "
    if size(x) != (M,N)
        return DomainError(size(x), string(s,"since its size is wrong."))
    end  
    if r > k
        return DomainError(r, string(s, "since its rank is too large."))
    end 
    return check_manifold_point(F,SVDMPoint(x,k))
end

function check_manifold_point(F::FixedRankMatrices{M,N,k,T}, x::SVDMPoint) where {M,N,k,T}
    s = "The point $(x) does not lie on the manifold of fixed rank matrices of size ($(M),$(N)) witk rank $(k), "
    if (size(x.U) != (M,k)) || (length(x.S) != k) || (size(x.Vt) != (k,N))
        return DomainError([size(x.U)...,length(x.S),size(x.Vt)], string(s, "since the dimensions do not fit (expected $([N,k,k,k,M]))."))
    end
    if !isapprox(x.U'*x.U,one(zeros(M,M)))
        return DomainError(norm(x.U'*x.U-one(zeros(M,M))), string(s," since U is not orthonormal/unitary."))
    end
    if !isapprox(x.Vt'*x.Vt, one(zeros(N,N)))
        return DomainError(norm(x.Vt'*x.Vt-one(zeros(N,N))), string(s," since V is not orthonormal/unitary."))
    end
end

function check_tangent_vector(F::FixedRankMatrices{M,N,k,T}, x::SVDMPoint, v::UMVTVector) where {M,N,k,T}
    c = check_manifold_point(x)
    if c != nothing
        return c
    end
    if (size(v.U) != (M,k)) || (size(v.Vt) != (k,N)) || (size(v.M) != (k,k))
        return DomainError(cat(size(v.U),size(v.M),size(v.Vt),dims=1), "The tangent vector $(v) is not a tangent vector to $(x) on the fixed rank matrices since the matrix dimensions to not fit (expected $([M,k,k,k,k,N])).")
    end
    if !isapprox(v.U'*x.U, zeros(k,k))
        return DomainError(norm(v.U'*x.U-zeros(k,k)), "The tangent vector $(v) is not a tangent vector to $(x) on the fixed rank matrices since v.U'x.U is not zero. ")
    end
    if !isapprox(v.Vt*x.Vt', zeros(k,k))
        return DomainError(norm(v.Vt*x.Vt-zeros(k,k)), "The tangent vector $(v) is not a tangent vector to $(x) on the fixed rank matrices since v.V'x.V is not zero.")
    end
end

function inner(::FixedRankMatrices{M,N,k,T}, x::SVDMPoint, v::UMVTVector, w::UMVTVector) where {M,N,k,T}
    return dot(v.U,w.U) + dot(v.M,w.M) + dot(v.Vt,w.Vt)
end

@doc doc"""
    manifold_dimension(M::FixedRankMatrices{M,N,k,Real})

returns the manifold dimension for the real-valued matrices of dimension `M`x`N`
    of rank `k`, namely

````math
k(M+N-k)
````
"""
manifold_dimension(::FixedRankMatrices{M,N,k,Real}) where {M,N,k} = (M+N-k)*k

@doc doc"""
    manifold_dimension(M::FixedRankMatrices{M,N,k,Real})

returns the manifold dimension for the complex-valued matrices of dimension `M`x`N`
    of rank `k`, namely

````math
2k(M+N-k)
````
"""
manifold_dimension(::FixedRankMatrices{M,N,k,Complex}) where {M,N,k} = 2*(M+N-k)*k

@doc doc"""
    project_tangent!(M,vto,x,A)

project the matrix $A\in\mathbb R^{m,n}$ or a [`UMVTVector`](@ref) (e.g. from
another tangent space) onto the tangent space at $x$, further
decomposing the result into $v=UMV$, i.e. a [`UMVTVector`](@ref) following
Section 3 in 
> Bart Vandereycken: "Low-rank matrix completion by Riemannian Optimization,
> SIAM Journal on Optiomoization, 23(2), pp. 1214–1236, 2013.
> doi: [10.1137/110845768](https://doi.org/10.1137/110845768),
> arXiv: [1209.3834](https://arxiv.org/abs/1209.3834).
"""
function project_tangent!(::FixedRankMatrices{M,N,k,T}, vto::UMVTVector, x::SVDMPoint, A::AbstractMatrix) where {M,N,k,T}
    vto.M .= x.U * A * x.Vt'
    vto.U .= A * x.Vt' - x.U
    vto.Vt .= x.U' * A - x.U' * A * x.Vt' * x.Vt
    return vto
end
project_tangent!(F::FixedRankMatrices{M,N,k,T}, vto::UMVTVector, x::SVDMPoint, v::UMVTVector) where {M,N,k,T} = project_tangent!(F,vto,x, v.U*v.M.v.Vt)

@doc doc"""
    retract!(M, y, x, v, ::PolarRetraction)

compute an SVD-based retraction on the [`FixedRankMatrices`](@ref) manifold
by computing
````math
    y = U_kS_kV_k^\mathrm{T},
````
where $U_k S_k V_k^\mathrm{T}$ is the shortened singular value decomposition $USV=x+v$,
in the sense that $S_k$ is the diagonal matrix of size $k\times k$ with the $k$ largest
singular values and $U$ and $V$ are shortened accordingly.
"""
function retract!(::FixedRankMatrices{M,N,k,T}, y::SVDMPoint, x::SVDMPoint, v::UMVTVector, ::PolarRetraction) where {M,N,k,T}
    s = svd( x.U * Diagonal(x.S) * x.Vt + (x.U * v.M * x.Vt + v.U*x.Vt + v.U*v.Vt) )
    y.U .= s.U[:,1:k]
    y.S .= s.S[1:k]
    y.Vt .= s.Vt[1:k,:]
    return y
end

function zero_tangent_vector!(::FixedRankMatrices{m,n,k,T},v::UMVTVector, x::SVDMPoint) where {m,n,k,T}
    v.U .= zeros(T,n,k)
    v.M .= zeros(T,k,k)
    v.Vt = zeros(T,k,m)
    return v
end
