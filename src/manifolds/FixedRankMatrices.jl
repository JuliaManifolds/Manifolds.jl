@doc doc"""
    FixedRankMatrices{m,n,k,T} <: Manifold

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

Generate the manifold of `m`-by-`n` real-valued matrices of rank `k`.

[^Vandereycken2013]:
    > Bart Vandereycken: "Low-rank matrix completion by Riemannian Optimization,
    > SIAM Journal on Optiomoization, 23(2), pp. 1214–1236, 2013.
    > doi: [10.1137/110845768](https://doi.org/10.1137/110845768),
    > arXiv: [1209.3834](https://arxiv.org/abs/1209.3834).
"""
struct FixedRankMatrices{M,N,K,T} <: Manifold end
function FixedRankMatrices(m::Int, n::Int, k::Int, t::AbstractNumbers = ℝ)
    return FixedRankMatrices{m,n,k,t}()
end

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
struct SVDMPoint{TU<:AbstractMatrix,TS<:AbstractVector,TVt<:AbstractMatrix} <: MPoint
    U::TU
    S::TS
    Vt::TVt
end
SVDMPoint(A::AbstractMatrix) = SVDMPoint(svd(A))
SVDMPoint(S::SVD) = SVDMPoint(S.U, S.S, S.Vt)
SVDMPoint(A::Matrix, k::Int) = SVDMPoint(svd(A), k)
SVDMPoint(S::SVD, k::Int) = SVDMPoint(S.U, S.S, S.Vt, k)
SVDMPoint(U, S, Vt, k::Int) = SVDMPoint(U[:, 1:k], S[1:k], Vt[1:k, :])
==(x::SVDMPoint, y::SVDMPoint) = (x.U == y.U) && (x.S == y.S) && (x.Vt == y.Vt)

@doc doc"""
    UMVTVector <: TVector

A tangent vector that can be described as a product $UMV^\mathrm{T}$, at least
together with its base point, see for example [`FixedRankMatrices`](@ref)

# Constructors
* `UMVTVector(U,M,Vt)` store umv factors to initialize the `UMVTVector`
* `UMVTVector(U,M,Vt,k)` store the umv factors after shortening them down to
  inner dimensions $k$, i.e. in $UMV^\mathrm{T}$, $M\in\mathbb R^{k\times k}$
"""
struct UMVTVector{TU<:AbstractMatrix,TM<:AbstractMatrix,TVt<:AbstractMatrix} <: TVector
    U::TU
    M::TM
    Vt::TVt
end

UMVTVector(U, M, Vt, k::Int) = UMVTVector(U[:, 1:k], M[1:k, 1:k], Vt[1:k, :])

# here the division in M corrects for the first factor in UMV + x.U*Vt + U*x.Vt, where x is the base point to v.
*(v::UMVTVector, s::Number) = UMVTVector(v.U * s, v.M * s, v.Vt * s)
*(s::Number, v::UMVTVector) = UMVTVector(s * v.U, s * v.M, s * v.Vt)
/(v::UMVTVector, s::Number) = UMVTVector(v.U / s, v.M / s, v.Vt / s)
\(s::Number, v::UMVTVector) = UMVTVector(s \ v.U, s \ v.M, s \ v.Vt)
+(v::UMVTVector, w::UMVTVector) = UMVTVector(v.U + w.U, v.M + w.M, v.Vt + w.Vt)
-(v::UMVTVector, w::UMVTVector) = UMVTVector(v.U - w.U, v.M - w.M, v.Vt - w.Vt)
-(v::UMVTVector) = UMVTVector(-v.U, -v.M, -v.Vt)
+(v::UMVTVector) = UMVTVector(v.U, v.M, v.Vt)
==(v::UMVTVector, w::UMVTVector) = (v.U == w.U) && (v.M == w.M) && (v.Vt == w.Vt)

@doc doc"""
    check_manifold_point(M::FixedRankMatrices{m,n,k},x; kwargs...)

Check whether the matrix or [`SVDMPoint`](@ref) `x` ids a valid point on the
[`FixedRankMatrices`](@ref)`{m,n,k}` `M`, i.e. is (or represents) an `m`-by`n` matrix of
rank `k`. For the [`SVDMPoint`](@ref) the internal representation also has to have the right
shape, i.e. `x.U` and `x.Vt` have to be unitary.
"""
function check_manifold_point(M::FixedRankMatrices{m,n,k}, x; kwargs...) where {m,n,k}
    r = rank(x; kwargs...)
    s = "The point $(x) does not lie on the manifold of fixed rank matrices of size ($(m),$(n)) witk rank $(k), "
    if size(x) != (m, n)
        return DomainError(size(x), string(s, "since its size is wrong."))
    end
    if r > k
        return DomainError(r, string(s, "since its rank is too large ($(r))."))
    end
    return nothing
end
function check_manifold_point(
    F::FixedRankMatrices{m,n,k},
    x::SVDMPoint;
    kwargs...,
) where {m,n,k}
    s = "The point $(x) does not lie on the manifold of fixed rank matrices of size ($(m),$(n)) witk rank $(k), "
    if (size(x.U) != (m, k)) || (length(x.S) != k) || (size(x.Vt) != (k, n))
        return DomainError(
            [size(x.U)..., length(x.S), size(x.Vt)...],
            string(
                s,
                "since the dimensions do not fit (expected $(n)x$(m) rank $(k) got $(size(x.U,1))x$(size(x.Vt,2)) rank $(size(x.S)).",
            ),
        )
    end
    if !isapprox(x.U' * x.U, one(zeros(n, n)); kwargs...)
        return DomainError(
            norm(x.U' * x.U - one(zeros(n, n))),
            string(s, " since U is not orthonormal/unitary."),
        )
    end
    if !isapprox(x.Vt' * x.Vt, one(zeros(n, n)); kwargs...)
        return DomainError(
            norm(x.Vt' * x.Vt - one(zeros(n, n))),
            string(s, " since V is not orthonormal/unitary."),
        )
    end
    return nothing
end

@doc doc"""
    check_tangent_vector(M:FixedRankMatrices{m,n,k}, x, v)

Check whether the tangent [`UMVTVector`](@ref) `v` is from the tangent space of
the [`SVDMPoint`](@ref) `x` on the [`FixedRankMatrices`](@ref) `M`, i.e. that
`v.U` and `v.Vt` are (columnwise) orthogonal to `x.U` and `x.Vt`, respectively,
and its dimensions are consistent with `x` and `M`, i.e. correspond to `m`-by-`n`
matrices of rank `k`.
"""
function check_tangent_vector(
    M::FixedRankMatrices{m,n,k},
    x::SVDMPoint,
    v::UMVTVector;
    kwargs...,
) where {m,n,k}
    c = check_manifold_point(M, x)
    c === nothing || return c
    if (size(v.U) != (m, k)) || (size(v.Vt) != (k, n)) || (size(v.M) != (k, k))
        return DomainError(
            cat(size(v.U), size(v.M), size(v.Vt), dims = 1),
            "The tangent vector $(v) is not a tangent vector to $(x) on the fixed rank matrices since the matrix dimensions to not fit (expected $(m)x$(k), $(k)x$(k), $(k)x$(n)).",
        )
    end
    if !isapprox(v.U' * x.U, zeros(k, k); kwargs...)
        return DomainError(
            norm(v.U' * x.U - zeros(k, k)),
            "The tangent vector $(v) is not a tangent vector to $(x) on the fixed rank matrices since v.U'x.U is not zero. ",
        )
    end
    if !isapprox(v.Vt * x.Vt', zeros(k, k); kwargs...)
        return DomainError(
            norm(v.Vt * x.Vt - zeros(k, k)),
            "The tangent vector $(v) is not a tangent vector to $(x) on the fixed rank matrices since v.V'x.V is not zero.",
        )
    end
end

@doc doc"""
    inner(M::FixedRankMatrices, x::SVDMPoint, v::UMVTVector, w::UMVTVector)

Compute the inner product of `v` and `w` in the tangent space of `x` on the
[`FixedRankMatrices`](@ref) `M`, which is inherited from the embedding, i.e. can be computed
using `dot` on the elements (`U`, `Vt`, `M`) of `v` and `w`.
"""
function inner(::FixedRankMatrices, x::SVDMPoint, v::UMVTVector, w::UMVTVector)
    return dot(v.U, w.U) + dot(v.M, w.M) + dot(v.Vt, w.Vt)
end

function isapprox(::FixedRankMatrices, x::SVDMPoint, y::SVDMPoint; kwargs...)
    return isapprox(x.U * Diagonal(x.S) * x.Vt, y.U * Diagonal(y.S) * y.Vt; kwargs...)
end
function isapprox(
    ::FixedRankMatrices,
    x::SVDMPoint,
    v::UMVTVector,
    w::UMVTVector;
    kwargs...,
)
    return isapprox(
        x.U * v.M * x.Vt + v.U * x.Vt + x.U * v.Vt,
        x.U * w.M * x.Vt + w.U * x.Vt + x.U * w.Vt;
        kwargs...,
    )
end

@doc doc"""
    manifold_dimension(M::FixedRankMatrices{m,n,k,𝔽})

Return the manifold dimension for the `𝔽`-valued [`FixedRankMatrices`](@ref) `M`
of dimension `m`x`n` of rank `k`, namely

````math
k(m + n - k) \dim_ℝ 𝔽,
````

where $\dim_ℝ 𝔽$ is the [`real_dimension`](@ref) of `𝔽`.
"""
function manifold_dimension(::FixedRankMatrices{m,n,k,𝔽}) where {m,n,k,𝔽}
    return (m + n - k) * k * real_dimension(𝔽)
end

@doc doc"""
    project_tangent(M, x, A)
    project_tangent(M, x, v)

Project the matrix $A\in\mathbb R^{m,n}$ or a [`UMVTVector`](@ref) `v` from the embedding or
another tangent space onto the tangent space at $x$ on the [`FixedRankMatrices`](@ref) `M`,
further decomposing the result into $v=UMV$, i.e. a [`UMVTVector`](@ref) following
Section 3 in [^Vandereycken2013].

[^Vandereycken2013]:
    > Bart Vandereycken: "Low-rank matrix completion by Riemannian Optimization,
    > SIAM Journal on Optiomoization, 23(2), pp. 1214–1236, 2013.
    > doi: [10.1137/110845768](https://doi.org/10.1137/110845768),
    > arXiv: [1209.3834](https://arxiv.org/abs/1209.3834).
"""
project_tangent(::FixedRankMatrices, ::Any...)

function project_tangent!(
    ::FixedRankMatrices,
    vto::UMVTVector,
    x::SVDMPoint,
    A::AbstractMatrix,
)
    av = A * (x.Vt')
    uTav = x.U' * av
    aTu = A' * x.U
    vto.M .= uTav
    vto.U .= A * x.Vt' - x.U * uTav
    vto.Vt .= (aTu - x.Vt' * uTav')'
    return vto
end
function project_tangent!(
    M::FixedRankMatrices,
    vto::UMVTVector,
    x::SVDMPoint,
    v::UMVTVector,
)
    return project_tangent!(M, vto, x, v.U * v.M * v.Vt)
end

@doc doc"""
    representation_size(M::FixedRankMatrices{m,n,k})

Return the element size of a point on the [`FixedRankMatrices`](@ref) `M`, i.e.
the size of matrices on this manifold $(m,n)$.
"""
@generated representation_size(::FixedRankMatrices{m,n}) where {m,n} = (m, n)

@doc doc"""
    retract(M, x, v, ::PolarRetraction)

Compute an SVD-based retraction on the [`FixedRankMatrices`](@ref) `M` by computing
````math
    y = U_kS_kV_k^\mathrm{T},
````
where $U_k S_k V_k^\mathrm{T}$ is the shortened singular value decomposition $USV=x+v$,
in the sense that $S_k$ is the diagonal matrix of size $k\times k$ with the $k$ largest
singular values and $U$ and $V$ are shortened accordingly.
"""
retract(::FixedRankMatrices, ::Any, ::Any, ::PolarRetraction)

function retract!(
    ::FixedRankMatrices{M,N,k},
    y::SVDMPoint,
    x::SVDMPoint,
    v::UMVTVector,
    ::PolarRetraction,
) where {M,N,k}
    s = svd(x.U * Diagonal(x.S) * x.Vt + (x.U * v.M * x.Vt + v.U * x.Vt + v.U * v.Vt))
    y.U .= s.U[:, 1:k]
    y.S .= s.S[1:k]
    y.Vt .= s.Vt[1:k, :]
    return y
end

function show(io::IO, ::FixedRankMatrices{M,N,K,T}) where {M,N,K,T}
    print(io, "FixedRankMatrices($(M), $(N), $(K), $(T))")
end
function show(io::IO, mime::MIME"text/plain", x::SVDMPoint)
    summary(io, x)
    println(io, "\nU factor:")
    show(io, mime, x.U)
    println(io, "\nsingular values:")
    show(io, mime, x.S)
    println(io, "\nVt factor:")
    show(io, mime, x.Vt)
end
function show(io::IO, mime::MIME"text/plain", v::UMVTVector)
    summary(io, v)
    println(io, "\nU factor:")
    show(io, mime, v.U)
    println(io, "\nM factor:")
    show(io, mime, v.M)
    println(io, "\nVt factor:")
    show(io, mime, v.Vt)
end

allocate(x::SVDMPoint) = SVDMPoint(allocate(x.U), allocate(x.S), allocate(x.Vt))
function allocate(x::SVDMPoint, ::Type{T}) where {T}
    return SVDMPoint(allocate(x.U, T), allocate(x.S, T), allocate(x.Vt, T))
end
allocate(v::UMVTVector) = UMVTVector(allocate(v.U), allocate(v.M), allocate(v.Vt))
function allocate(v::UMVTVector, ::Type{T}) where {T}
    return UMVTVector(allocate(v.U, T), allocate(v.M, T), allocate(v.Vt, T))
end

function number_eltype(x::SVDMPoint)
    return typeof(one(eltype(x.U)) + one(eltype(x.S)) + one(eltype(x.Vt)))
end
function number_eltype(v::UMVTVector)
    return typeof(one(eltype(v.U)) + one(eltype(v.M)) + one(eltype(v.Vt)))
end

one(x::SVDMPoint) = SVDMPoint(
    one(zeros(size(x.U, 1), size(x.U, 1))),
    ones(length(x.S)),
    one(zeros(size(x.Vt, 2), size(x.Vt, 2))),
    length(x.S),
)
one(v::UMVTVector) = UMVTVector(
    one(zeros(size(v.U, 1), size(v.U, 1))),
    one(zeros(size(v.M))),
    one(zeros(size(v.Vt, 2), size(v.Vt, 2))),
    size(v.M, 1),
)

function copyto!(x::SVDMPoint, y::SVDMPoint)
    copyto!(x.U, y.U)
    copyto!(x.S, y.S)
    copyto!(x.Vt, y.Vt)
    return x
end
function copyto!(v::UMVTVector, w::UMVTVector)
    copyto!(v.U, w.U)
    copyto!(v.M, w.M)
    copyto!(v.Vt, w.Vt)
    return v
end

@doc doc"""
    zero_tangent_vector(M::FixedRankMatrices, x::SVDMPoint)

Return a [`UMVTVector`](@ref) representing the zero tangent vector in the tangent space of
`x` on the [`FixedRankMatrices`](@ref) `M`, for example all three elements of the resulting
structure are zero matrices.
"""
function zero_tangent_vector(::FixedRankMatrices{m,n,k}, x::SVDMPoint) where {m,n,k}
    v = UMVTVector(
        zeros(eltype(x.U), m, k),
        zeros(eltype(x.S), k, k),
        zeros(eltype(x.Vt), k, n),
    )
    return v
end

function zero_tangent_vector!(
    ::FixedRankMatrices{m,n,k},
    v::UMVTVector,
    x::SVDMPoint,
) where {m,n,k}
    v.U .= zeros(eltype(v.U), m, k)
    v.M .= zeros(eltype(v.M), k, k)
    v.Vt .= zeros(eltype(v.Vt), k, n)
    return v
end
