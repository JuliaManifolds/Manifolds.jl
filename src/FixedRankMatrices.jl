using LinearAlgebra: diag, Diagonal, svd, SVD, rank, dot
import Base: \, /, +, -, *, ==, similar, one, copyto!
@doc doc"""
    FixedRankMatrices{M,N,K,T} <: Manifold

The manifold of $m\times n$ real-valued (complex-valued) matrices of fixed rank $k$, i.e.
````math
\mathcal M = \{ x \in \mathbb R^{m\times n} : \operatorname{rank}(x) = k \}.
````
# Representation with 3 matrix factors

A point $x\in\mathcal M$ can be stored using orthonormal matrices
$U\in\mathbb R^{m\times k}$, $V\in\mathbb R^{n\times k}$ as well as the $k$ singular
values of $x = USV^\mathrm{T}$. In other words, $U$ and $V$ are from the manifolds
[`Stiefel`](@ref)`(m,k)` and [`Stiefel`](@ref)`(n,k)`, respectively; see
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
on $\mathbb R^{m\times n}$ to the tangent bundle. This implementation follows[^Vandereycken2013].

# Constructor
    FixedRankMatrics(m,n,k,t=ℝ)

generate the manifold of `m`-by-`n` real-valued matrices of rank `k`.

[^Vandereycken2013]:
    > Bart Vandereycken: "Low-rank matrix completion by Riemannian Optimization,
    > SIAM Journal on Optiomoization, 23(2), pp. 1214–1236, 2013.
    > doi: [10.1137/110845768](https://doi.org/10.1137/110845768),
    > arXiv: [1209.3834](https://arxiv.org/abs/1209.3834).
"""
struct FixedRankMatrices{M,N,K,T} <: Manifold end
FixedRankMatrices(m::Int, n::Int, k::Int, t::AbstractField=ℝ) = FixedRankMatrices{m,n,k,t}()

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
SVDMPoint(A::Matrix,k::Int) = SVDMPoint(svd(A),k)
SVDMPoint(S::SVD,k::Int) = SVDMPoint(S.U,S.S,S.Vt,k)
SVDMPoint(U,S,Vt,k::Int) = SVDMPoint(U[:,1:k],S[1:k],Vt[1:k,:])
==(x::SVDMPoint, y::SVDMPoint) = (x.U==y.U) && (x.S==y.S) && (x.Vt==y.Vt)

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

# here the division in M corrects for the first factor in UMV + x.U*Vt + U*x.Vt, where x is the base point to v.  
*(v::UMVTVector, s::Number) = UMVTVector(v.U*s, v.M*s,  v.Vt*s)
*(s::Number, v::UMVTVector) = UMVTVector(s*v.U, s*v.M, s*v.Vt)
/(v::UMVTVector, s::Number) = UMVTVector(v.U/s, v.M/s, v.Vt/s)
\(s::Number, v::UMVTVector) = UMVTVector(s\v.U, s\v.M, s\v.Vt)
+(v::UMVTVector, w::UMVTVector) = UMVTVector(v.U + w.U, v.M + w.M, v.Vt + w.Vt)
-(v::UMVTVector, w::UMVTVector) = UMVTVector(v.U - w.U, v.M - w.M, v.Vt - w.Vt)
-(v::UMVTVector) = UMVTVector(-v.U, -v.M, -v.Vt)
+(v::UMVTVector) = UMVTVector(v.U, v.M, v.Vt)
==(v::UMVTVector,w::UMVTVector) = (v.U==w.U) && (v.M==w.M) && (v.Vt==w.Vt)

function check_manifold_point(F::FixedRankMatrices{M,N,k},x; kwargs...) where {M,N,k}
    r = rank(x; kwargs...)
    s = "The point $(x) does not lie on the manifold of fixed rank matrices of size ($(M),$(N)) witk rank $(k), "
    if size(x) != (M,N)
        return DomainError(size(x), string(s,"since its size is wrong."))
    end
    if r > k
        return DomainError(r, string(s, "since its rank is too large ($(r))."))
    end
    return check_manifold_point(F,SVDMPoint(x,k))
end

function check_manifold_point(F::FixedRankMatrices{M,N,k}, x::SVDMPoint; kwargs...) where {M,N,k}
    s = "The point $(x) does not lie on the manifold of fixed rank matrices of size ($(M),$(N)) witk rank $(k), "
    if (size(x.U) != (M,k)) || (length(x.S) != k) || (size(x.Vt) != (k,N))
        return DomainError([size(x.U)...,length(x.S),size(x.Vt)...], string(s, "since the dimensions do not fit (expected $(N)x$(M) rank $(k) got $(size(x.U,1))x$(size(x.Vt,2)) rank $(size(x.S))."))
    end
    if !isapprox(x.U'*x.U,one(zeros(N,N)); kwargs...)
        return DomainError(norm(x.U'*x.U-one(zeros(N,N))), string(s," since U is not orthonormal/unitary."))
    end
    if !isapprox(x.Vt'*x.Vt, one(zeros(N,N)); kwargs...)
        return DomainError(norm(x.Vt'*x.Vt-one(zeros(N,N))), string(s," since V is not orthonormal/unitary."))
    end
end

function check_tangent_vector(F::FixedRankMatrices{M,N,k}, x::SVDMPoint, v::UMVTVector; kwargs...) where {M,N,k}
    c = check_manifold_point(F,x)
    if c !== nothing
        return c
    end
    if (size(v.U) != (M,k)) || (size(v.Vt) != (k,N)) || (size(v.M) != (k,k))
        return DomainError(cat(size(v.U),size(v.M),size(v.Vt),dims=1), "The tangent vector $(v) is not a tangent vector to $(x) on the fixed rank matrices since the matrix dimensions to not fit (expected $(M)x$(k), $(k)x$(k), $(k)x$(N)).")
    end
    if !isapprox(v.U'*x.U, zeros(k,k); kwargs...)
        return DomainError(norm(v.U'*x.U-zeros(k,k)), "The tangent vector $(v) is not a tangent vector to $(x) on the fixed rank matrices since v.U'x.U is not zero. ")
    end
    if !isapprox(v.Vt*x.Vt', zeros(k,k); kwargs...)
        return DomainError(norm(v.Vt*x.Vt-zeros(k,k)), "The tangent vector $(v) is not a tangent vector to $(x) on the fixed rank matrices since v.V'x.V is not zero.")
    end
end

function inner(::FixedRankMatrices, x::SVDMPoint, v::UMVTVector, w::UMVTVector)
    return dot(v.U,w.U) + dot(v.M,w.M) + dot(v.Vt,w.Vt)
end

isapprox(::FixedRankMatrices, x::SVDMPoint, y::SVDMPoint; kwargs...) = isapprox( x.U*Diagonal(x.S)*x.Vt, y.U*Diagonal(y.S)*y.Vt; kwargs...)
isapprox(::FixedRankMatrices, x::SVDMPoint, v::UMVTVector, w::UMVTVector; kwargs...) = isapprox(x.U*v.M*x.Vt + v.U*x.Vt + x.U*v.Vt, x.U*w.M*x.Vt + w.U*x.Vt + x.U*w.Vt; kwargs...)

@doc doc"""
    manifold_dimension(M::FixedRankMatrices{M,N,k,𝔽})

returns the manifold dimension for the real-valued matrices of dimension `M`x`N`
of rank `k`, namely

````math
\dim(𝔽)k(M+N-k),
````

where $\dim(𝔽)$ is the [`field_dimension`](@ref).
"""
function manifold_dimension(::FixedRankMatrices{M,N,k,𝔽}) where {M,N,k,𝔽}
    return field_dimension(𝔽) * (M+N-k)*k
end

@doc doc"""
    project_tangent!(M,vto,x,A)
    project_tangent!(M,vto,x,v)

project the matrix $A\in\mathbb R^{m,n}$ or a [`UMVTVector`](@ref) `v`from the embedding or
another tangent spaceonto the tangent space at $x$, further decomposing the result into
$v=UMV$, i.e. a [`UMVTVector`](@ref) following Section 3 in [^Vandereycken2013].

[^Vandereycken2013]:
    > Bart Vandereycken: "Low-rank matrix completion by Riemannian Optimization,
    > SIAM Journal on Optiomoization, 23(2), pp. 1214–1236, 2013.
    > doi: [10.1137/110845768](https://doi.org/10.1137/110845768),
    > arXiv: [1209.3834](https://arxiv.org/abs/1209.3834).
"""
function project_tangent!(
    ::FixedRankMatrices,
    vto::UMVTVector,
    x::SVDMPoint,
    A::AbstractMatrix
)
    av = A*(x.Vt')
    uTav = x.U'*av
    aTu = A'*x.U
    vto.M .= uTav
    vto.U .= A * x.Vt' - x.U*uTav
    vto.Vt .= (aTu - x.Vt'*uTav')'
    return vto
end
function project_tangent!(F::FixedRankMatrices, vto::UMVTVector, x::SVDMPoint, v::UMVTVector)
    return project_tangent!(F,vto,x, v.U * v.M * v.Vt)
end
representation_size(F::FixedRankMatrices{M,N}) where {M,N} = (M,N)

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
function retract!(::FixedRankMatrices{M,N,k}, y::SVDMPoint, x::SVDMPoint, v::UMVTVector, ::PolarRetraction) where {M,N,k}
    s = svd( x.U * Diagonal(x.S) * x.Vt + (x.U * v.M * x.Vt + v.U*x.Vt + v.U*v.Vt) )
    y.U .= s.U[:,1:k]
    y.S .= s.S[1:k]
    y.Vt .= s.Vt[1:k,:]
    return y
end

similar(x::SVDMPoint) = SVDMPoint(similar(x.U), similar(x.S), similar(x.Vt))
similar(x::SVDMPoint, ::Type{T}) where T = SVDMPoint(similar(x.U,T), similar(x.S,T), similar(x.Vt,T))
similar(v::UMVTVector) = UMVTVector(similar(v.U), similar(v.M), similar(v.Vt))
similar(v::UMVTVector, ::Type{T}) where T = UMVTVector(similar(v.U,T), similar(v.M,T), similar(v.Vt,T))

eltype(x::SVDMPoint) = typeof(one(eltype(x.U)) + one(eltype(x.S)) + one(eltype(x.Vt)))
eltype(v::UMVTVector) = typeof(one(eltype(v.U)) + one(eltype(v.M)) + one(eltype(v.Vt)))

one(x::SVDMPoint) = SVDMPoint(one(zeros(size(x.U,1),size(x.U,1))), ones(length(x.S)), one(zeros(size(x.Vt,2),size(x.Vt,2))), length(x.S))
one(v::UMVTVector) = UMVTVector(one(zeros(size(v.U,1),size(v.U,1))), one(zeros(size(v.M))), one(zeros(size(v.Vt,2),size(v.Vt,2))), size(v.M,1))

function copyto!(x::SVDMPoint, y::SVDMPoint)
    copyto!(x.U, y.U)
    copyto!(x.S, y.S)
    copyto!(x.Vt, y.Vt)
end
function copyto!(v::UMVTVector, w::UMVTVector)
    copyto!(v.U, w.U)
    copyto!(v.M, w.M)
    copyto!(v.Vt, w.Vt)
end

function zero_tangent_vector!(::FixedRankMatrices{m,n,k}, v::UMVTVector, x::SVDMPoint) where {m,n,k}
    v.U .= zeros(eltype(v.U),m,k)
    v.M .= zeros(eltype(v.M),k,k)
    v.Vt .= zeros(eltype(v.Vt),k,n)
    return v
end

function zero_tangent_vector(::FixedRankMatrices{m,n,k}, x::SVDMPoint) where {m,n,k}
    v = UMVTVector( zeros(eltype(x.U),m,k), zeros(eltype(x.S),k,k), zeros(eltype(x.Vt),k,n))
    return v
end
