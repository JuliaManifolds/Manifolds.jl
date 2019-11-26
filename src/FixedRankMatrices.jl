using LinearAlgebra: diag, Diagonal, svd, SVD

@doc doc"""
    FixedRankMatrices{M,N,K} <: Manifold

The manifold of $m\times n$ real-valued matrices of fixed rank $k$, i.e.
````math
\mathcal M = \{ x \in \mathbb R^{m\times n} : \operatorname{rank}(x) = k \}
````

# Representation with 3 matrix factors
A point $x\in\mathcal M$ can be stored using orthonormal matrices
$U\in\mathbb R^{m\times k}$, $V\in\mathbb R^{n,k}$ as well as the $k$ singular
values of $x = USV^\mathrm{T}$. To be precise, $U$ and $V$ are from the manifolds
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

This representation follows
> Bart Vandereycken: "Low-rank matrix completion by Riemannian Optimization,
> SIAM Journal on Optiomoization, 23(2), pp. 1214–1236, 2013.
> doi: [10.1137/110845768](https://doi.org/10.1137/110845768),
> arXiv: [1209.3834](https://arxiv.org/abs/1209.3834).
"""
struct FixedRankMatrices{M,N,K} <: Manifold end
FixedRankMatrices(m::Int, n::Int, k::Int) = FixedRankMatrices{m,n,k}()

@doc doc"""
    SVDMPoint <: MPoint

A point on a certain manifold, where the data is stored in a svd like fashion,
i.e. in the form $USV^\mathrm{T}$, where this structre stores $U$, $S$ and
$V^\mathrm{T}$. The storage might also be shortened to just $k$ singular values
and accordingly shortened $U$ (columns) and $V^\mathrm{T}$ (rows)

# Constructors
* `SVDMPoint(A)` for a matrix `A`, stores its svd factors
* `SVDMPoint(S)` for an `SVD` object, stores its svd factors
* `SVDMPoint(U,S,Vt)` for the svd factors to initialize the `SVDMPoint``
* `SVDMPoint(A,k)` for a matrix `A`, stores its svd factors shortened to the
  best rank $k$ approximation
* `SVDMPoint(S,k)` for an `SVD` object, stores its svd factors shortened to the
  best rank $k$ approximation
* `SVDMPoint(U,S,Vt,k)` for the svd factors to initialize the `SVDMPoint`,
  stores its svd factors shortened to the best rank $k$ approximation
"""
struct SVDMPoint{T} <: MPoint
    U::Matrix{T}
    S::Vector{T}
    Vt::Matrix{T}
end
SVDMPoint(A::Matrix,args...) = SVDMPoint(svd(A),args...)
SVDMPoint(S::SVD,args...) = SVDMPoint(S.U,S.S,S.Vt,args...)
SVDMPoint(U,S,Vt,k::Int) = SVDMPoint(U[:,1:k],S[1:k],Vt[1:k,:])
SVDMPoint(U,S,Vt) = SVDMPoint{eltype(U)}(U,S,Vt)

@doc doc"""
    UMVTVector <: TVector

A tangent vector that can be described as a product $UMV^\mathrm{T}$, at least
together with its base point, see for example [`FixedRankMatrices`](@ref)

# Constructors
* `UMVTVector(U,S,Vt)` store umv factors to initialize the `UMVTVector`
* `UMVTVector(U,S,Vt,k)` store the umv factors after shortening them down to
  inner dimensions $k$, i.e. in $UMV^\mathrm{T}$, $M\in\mathbb R^{k\times k}$
"""
struct UMVTVector{T} <: TVector
    U::Matrix{T}
    M::Matrix{T}
    Vt::Matrix{T}
end
UMVTVector(U,M,Vt,k::Int) = UMVTVector(U[:,1:k],S[1:k],Vt[1:k,:])
UMVTVector(U,M,Vt) = UMVTVector{eltype(U)}(U,M,Vt)

function check_manifold_point(F::FixedRankMatrices{M,N,k},x; kwargs...) where {M,N,k}
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
function check_manifold_point(F::FixedRankMatrices{M,N,k},x::SVDMPoint{T}) where {M,N,k,T}
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

function check_tangent_vector(F::FixedRankMatrices{M,N,k},x::SVDMPoint{T},v::UMVTVector{T}) where {M,N,k,T}
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