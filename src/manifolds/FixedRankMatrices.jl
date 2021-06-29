@doc raw"""
    FixedRankMatrices{m,n,k,𝔽} <: AbstractManifold{𝔽}

The manifold of ``m × n`` real-valued or complex-valued matrices of fixed rank ``k``, i.e.
````math
\bigl\{ p ∈ 𝔽^{m × n}\ \big|\ \operatorname{rank}(p) = k \bigr\},
````
where ``𝔽 ∈ \{ℝ,ℂ\}`` and the rank is the number of linearly independent columns of a matrix.

# Representation with 3 matrix factors

A point ``p ∈ \mathcal M`` can be stored using unitary matrices ``U ∈ 𝔽^{m × k}``, ``V ∈ 𝔽^{n × k}`` as well as the ``k``
singular values of ``p = U_p S V_p^\mathrm{H}``, where ``\cdot^{\mathrm{H}}`` denotes the complex conjugate transpose or
Hermitian. In other words, ``U`` and ``V`` are from the manifolds [`Stiefel`](@ref)`(m,k,𝔽)` and [`Stiefel`](@ref)`(n,k,𝔽)`,
respectively; see [`SVDMPoint`](@ref) for details.

The tangent space ``T_p \mathcal M`` at a point ``p ∈ \mathcal M`` with ``p=U_p S V_p^\mathrm{H}``
is given by
````math
T_p\mathcal M = \bigl\{ U_p M V_p^\mathrm{T} + U_X V_p^\mathrm{H} + U_p V_X^\mathrm{H} :
    M  ∈ 𝔽^{k × k},
    U_X  ∈ 𝔽^{m × k},
    V_X  ∈ 𝔽^{n × k}
    \text{ s.t. }
    U_p^\mathrm{H}U_X = 0_k,
    V_p^\mathrm{H}V_X = 0_k
\bigr\},
````
where ``0_k`` is the ``k × k`` zero matrix. See [`UMVTVector`](@ref) for details.

The (default) metric of this manifold is obtained by restricting the metric
on ``ℝ^{m × n}`` to the tangent bundle[^Vandereycken2013].

# Constructor
    FixedRankMatrics(m, n, k[, field=ℝ])

Generate the manifold of `m`-by-`n` (`field`-valued) matrices of rank `k`.

[^Vandereycken2013]:
    > Bart Vandereycken: "Low-rank matrix completion by Riemannian Optimization,
    > SIAM Journal on Optiomoization, 23(2), pp. 1214–1236, 2013.
    > doi: [10.1137/110845768](https://doi.org/10.1137/110845768),
    > arXiv: [1209.3834](https://arxiv.org/abs/1209.3834).
"""
struct FixedRankMatrices{M,N,K,𝔽} <: AbstractEmbeddedManifold{𝔽,DefaultEmbeddingType} end
function FixedRankMatrices(m::Int, n::Int, k::Int, field::AbstractNumbers=ℝ)
    return FixedRankMatrices{m,n,k,field}()
end

@doc raw"""
    SVDMPoint <: AbstractManifoldPoint

A point on a certain manifold, where the data is stored in a svd like fashion,
i.e. in the form ``USV^\mathrm{H}``, where this structure stores ``U``, ``S`` and
``V^\mathrm{H}``. The storage might also be shortened to just ``k`` singular values
and accordingly shortened ``U`` (columns) and ``V^\mathrm{T}`` (rows).

# Constructors
* `SVDMPoint(A)` for a matrix `A`, stores its svd factors (i.e. implicitly ``k=\min\{m,n\}``)
* `SVDMPoint(S)` for an `SVD` object, stores its svd factors (i.e. implicitly ``k=\min\{m,n\}``)
* `SVDMPoint(U,S,Vt)` for the svd factors to initialize the `SVDMPoint`` (i.e. implicitly ``k=\min\{m,n\}``)
* `SVDMPoint(A,k)` for a matrix `A`, stores its svd factors shortened to the
  best rank ``k`` approximation
* `SVDMPoint(S,k)` for an `SVD` object, stores its svd factors shortened to the
  best rank ``k`` approximation
* `SVDMPoint(U,S,Vt,k)` for the svd factors to initialize the `SVDMPoint`,
  stores its svd factors shortened to the best rank ``k`` approximation
"""
struct SVDMPoint{TU<:AbstractMatrix,TS<:AbstractVector,TVt<:AbstractMatrix} <:
       AbstractManifoldPoint
    U::TU
    S::TS
    Vt::TVt
end
SVDMPoint(A::AbstractMatrix) = SVDMPoint(svd(A))
SVDMPoint(S::SVD) = SVDMPoint(S.U, S.S, S.Vt)
SVDMPoint(A::Matrix, k::Int) = SVDMPoint(svd(A), k)
SVDMPoint(S::SVD, k::Int) = SVDMPoint(S.U, S.S, S.Vt, k)
SVDMPoint(U, S, Vt, k::Int) = SVDMPoint(U[:, 1:k], S[1:k], Vt[1:k, :])
Base.:(==)(x::SVDMPoint, y::SVDMPoint) = (x.U == y.U) && (x.S == y.S) && (x.Vt == y.Vt)

@doc raw"""
    UMVTVector <: TVector

A tangent vector that can be described as a product `U_p M V_p^\mathrm{H} + U_X V_p^\mathrm{H} + U_p V_X^\mathrm{H}`,
where `X = U_X S V_X^\mathrm{H}` is its base point, see for example [`FixedRankMatrices`](@ref).
This vector structure stores the additionally (to the point) required fields.

# Constructors
* `UMVTVector(U,M,Vt)` store umv factors to initialize the `UMVTVector`
* `UMVTVector(U,M,Vt,k)` store the umv factors after shortening them down to
  inner dimensions `k`.
"""
struct UMVTVector{TU<:AbstractMatrix,TM<:AbstractMatrix,TVt<:AbstractMatrix} <: TVector
    U::TU
    M::TM
    Vt::TVt
end

UMVTVector(U, M, Vt, k::Int) = UMVTVector(U[:, 1:k], M[1:k, 1:k], Vt[1:k, :])

# here the division in M corrects for the first factor in UMV + x.U*Vt + U*x.Vt, where x is the base point to v.
Base.:*(v::UMVTVector, s::Number) = UMVTVector(v.U * s, v.M * s, v.Vt * s)
Base.:*(s::Number, v::UMVTVector) = UMVTVector(s * v.U, s * v.M, s * v.Vt)
Base.:/(v::UMVTVector, s::Number) = UMVTVector(v.U / s, v.M / s, v.Vt / s)
Base.:\(s::Number, v::UMVTVector) = UMVTVector(s \ v.U, s \ v.M, s \ v.Vt)
Base.:+(v::UMVTVector, w::UMVTVector) = UMVTVector(v.U + w.U, v.M + w.M, v.Vt + w.Vt)
Base.:-(v::UMVTVector, w::UMVTVector) = UMVTVector(v.U - w.U, v.M - w.M, v.Vt - w.Vt)
Base.:-(v::UMVTVector) = UMVTVector(-v.U, -v.M, -v.Vt)
Base.:+(v::UMVTVector) = UMVTVector(v.U, v.M, v.Vt)
Base.:(==)(v::UMVTVector, w::UMVTVector) = (v.U == w.U) && (v.M == w.M) && (v.Vt == w.Vt)

allocate(p::SVDMPoint) = SVDMPoint(allocate(p.U), allocate(p.S), allocate(p.Vt))
function allocate(p::SVDMPoint, ::Type{T}) where {T}
    return SVDMPoint(allocate(p.U, T), allocate(p.S, T), allocate(p.Vt, T))
end
allocate(X::UMVTVector) = UMVTVector(allocate(X.U), allocate(X.M), allocate(X.Vt))
function allocate(X::UMVTVector, ::Type{T}) where {T}
    return UMVTVector(allocate(X.U, T), allocate(X.M, T), allocate(X.Vt, T))
end

function allocate_result(::FixedRankMatrices{m,n,k}, ::typeof(embed), vals...) where {m,n,k}
    #note that vals is (p,) or (X,p) but both first entries have a U of correct type
    return similar(typeof(vals[1].U), m, n)
end
function allocate_result_type(
    ::FixedRankMatrices{m,n,k},
    ::typeof(project),
    ::Tuple{SVDMPoint,<:AbstractMatrix},
) where {m,n,k}
    return UMVTVector
end

Base.copy(v::UMVTVector) = UMVTVector(copy(v.U), copy(v.M), copy(v.Vt))

# Tuple-like broadcasting of UMVTVector

function Broadcast.BroadcastStyle(::Type{<:UMVTVector})
    return Broadcast.Style{UMVTVector}()
end
function Broadcast.BroadcastStyle(
    ::Broadcast.AbstractArrayStyle{0},
    b::Broadcast.Style{UMVTVector},
)
    return b
end

Broadcast.instantiate(bc::Broadcast.Broadcasted{Broadcast.Style{UMVTVector},Nothing}) = bc
function Broadcast.instantiate(bc::Broadcast.Broadcasted{Broadcast.Style{UMVTVector}})
    Broadcast.check_broadcast_axes(bc.axes, bc.args...)
    return bc
end

Broadcast.broadcastable(v::UMVTVector) = v

@inline function Base.copy(bc::Broadcast.Broadcasted{Broadcast.Style{UMVTVector}})
    return UMVTVector(
        @inbounds(Broadcast._broadcast_getindex(bc, Val(:U))),
        @inbounds(Broadcast._broadcast_getindex(bc, Val(:M))),
        @inbounds(Broadcast._broadcast_getindex(bc, Val(:Vt))),
    )
end

Base.@propagate_inbounds function Broadcast._broadcast_getindex(
    v::UMVTVector,
    ::Val{I},
) where {I}
    return getfield(v, I)
end

Base.axes(::UMVTVector) = ()

@inline function Base.copyto!(
    dest::UMVTVector,
    bc::Broadcast.Broadcasted{Broadcast.Style{UMVTVector}},
)
    # Performance optimization: broadcast!(identity, dest, A) is equivalent to copyto!(dest, A) if indices match
    if bc.f === identity && bc.args isa Tuple{UMVTVector} # only a single input argument to broadcast!
        A = bc.args[1]
        return copyto!(dest, A)
    end
    bc′ = Broadcast.preprocess(dest, bc)
    copyto!(dest.U, Broadcast._broadcast_getindex(bc′, Val(:U)))
    copyto!(dest.M, Broadcast._broadcast_getindex(bc′, Val(:M)))
    copyto!(dest.Vt, Broadcast._broadcast_getindex(bc′, Val(:Vt)))
    return dest
end

@doc raw"""
    check_point(M::FixedRankMatrices{m,n,k}, p; kwargs...)

Check whether the matrix or [`SVDMPoint`](@ref) `x` ids a valid point on the
[`FixedRankMatrices`](@ref)`{m,n,k,𝔽}` `M`, i.e. is an `m`-by`n` matrix of
rank `k`. For the [`SVDMPoint`](@ref) the internal representation also has to have the right
shape, i.e. `p.U` and `p.Vt` have to be unitary. The keyword arguments are passed to the
`rank` function that verifies the rank of `p`.
"""
function check_point(M::FixedRankMatrices{m,n,k}, p; kwargs...) where {m,n,k}
    r = rank(p; kwargs...)
    s = "The point $(p) does not lie on $(M), "
    if size(p) != (m, n)
        return DomainError(size(p), string(s, "since its size is wrong."))
    end
    if r > k
        return DomainError(r, string(s, "since its rank is too large ($(r))."))
    end
    return nothing
end
function check_point(M::FixedRankMatrices{m,n,k}, x::SVDMPoint; kwargs...) where {m,n,k}
    s = "The point $(x) does not lie on $(M), "
    if (size(x.U) != (m, k)) || (length(x.S) != k) || (size(x.Vt) != (k, n))
        return DomainError(
            [size(x.U)..., length(x.S), size(x.Vt)...],
            string(
                s,
                "since the dimensions do not fit (expected $(n)x$(m) rank $(k) got $(size(x.U,1))x$(size(x.Vt,2)) rank $(size(x.S)).",
            ),
        )
    end
    if !isapprox(x.U' * x.U, one(zeros(k, k)); kwargs...)
        return DomainError(
            norm(x.U' * x.U - one(zeros(k, k))),
            string(s, " since U is not orthonormal/unitary."),
        )
    end
    if !isapprox(x.Vt * x.Vt', one(zeros(k, k)); kwargs...)
        return DomainError(
            norm(x.Vt * x.Vt' - one(zeros(k, k))),
            string(s, " since V is not orthonormal/unitary."),
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M:FixedRankMatrices{m,n,k}, p, X; kwargs...)

Check whether the tangent [`UMVTVector`](@ref) `X` is from the tangent space of the [`SVDMPoint`](@ref) `p` on the
[`FixedRankMatrices`](@ref) `M`, i.e. that `v.U` and `v.Vt` are (columnwise) orthogonal to `x.U` and `x.Vt`,
respectively, and its dimensions are consistent with `p` and `X.M`, i.e. correspond to `m`-by-`n` matrices of rank `k`.
"""
function check_vector(
    M::FixedRankMatrices{m,n,k},
    p::SVDMPoint,
    X::UMVTVector;
    kwargs...,
) where {m,n,k}
    if (size(X.U) != (m, k)) || (size(X.Vt) != (k, n)) || (size(X.M) != (k, k))
        return DomainError(
            cat(size(X.U), size(X.M), size(X.Vt), dims=1),
            "The tangent vector $(X) is not a tangent vector to $(p) on $(M), since matrix dimensions do not agree (expected $(m)x$(k), $(k)x$(k), $(k)x$(n)).",
        )
    end
    if !isapprox(X.U' * p.U, zeros(k, k); kwargs...)
        return DomainError(
            norm(X.U' * p.U - zeros(k, k)),
            "The tangent vector $(X) is not a tangent vector to $(p) on $(M) since v.U'x.U is not zero. ",
        )
    end
    if !isapprox(X.Vt * p.Vt', zeros(k, k); kwargs...)
        return DomainError(
            norm(X.Vt * p.Vt - zeros(k, k)),
            "The tangent vector $(X) is not a tangent vector to $(p) on $(M) since v.V'x.V is not zero.",
        )
    end
    return nothing
end

function Base.copyto!(p::SVDMPoint, q::SVDMPoint)
    copyto!(p.U, q.U)
    copyto!(p.S, q.S)
    copyto!(p.Vt, q.Vt)
    return p
end
function Base.copyto!(X::UMVTVector, Y::UMVTVector)
    copyto!(X.U, Y.U)
    copyto!(X.M, Y.M)
    copyto!(X.Vt, Y.Vt)
    return X
end

@doc raw"""
    embed(::FixedRankMatrices, p::SVDMPoint)

Embed the point `p` from its `SVDMPoint` representation into the set of ``m×n`` matrices
by computing ``USV^{\mathrm{T}}``.
"""
embed(::FixedRankMatrices, p)

function embed!(::FixedRankMatrices, q, p)
    return mul!(q, p.U * Diagonal(p.S), p.Vt)
end

@doc raw"""
    embed(M::FixedRankMatrices, p, X)

Embed the tangent vector `X` at point `p` in `M` from
its [`UMVTVector`](@ref) representation  into the set of ``m×n`` matrices.

The formula reads
```math
U_pMV_p^{\mathrm{T}} + U_XV_p^{\mathrm{T}} + U_pV_X^{mathrm{T}}
```
"""
embed(::FixedRankMatrices, p, X)

function embed!(::FixedRankMatrices, Y, p, X)
    tmp = p.U * X.M
    tmp .+= X.U
    mul!(Y, tmp, p.Vt)
    return mul!(Y, p.U, X.Vt, true, true)
end

get_embedding(::FixedRankMatrices{m,n,k,𝔽}) where {m,n,k,𝔽} = Euclidean(m, n; field=𝔽)

@doc raw"""
    inner(M::FixedRankMatrices, p::SVDMPoint, X::UMVTVector, Y::UMVTVector)

Compute the inner product of `X` and `Y` in the tangent space of `p` on the [`FixedRankMatrices`](@ref) `M`,
which is inherited from the embedding, i.e. can be computed using `dot` on the elements (`U`, `Vt`, `M`) of `X` and `Y`.
"""
function inner(::FixedRankMatrices, x::SVDMPoint, v::UMVTVector, w::UMVTVector)
    return dot(v.U, w.U) + dot(v.M, w.M) + dot(v.Vt, w.Vt)
end

function Base.isapprox(::FixedRankMatrices, p::SVDMPoint, q::SVDMPoint; kwargs...)
    return isapprox(p.U * Diagonal(p.S) * p.Vt, q.U * Diagonal(q.S) * q.Vt; kwargs...)
end
function Base.isapprox(
    ::FixedRankMatrices,
    p::SVDMPoint,
    X::UMVTVector,
    Y::UMVTVector;
    kwargs...,
)
    return isapprox(
        p.U * X.M * p.Vt + X.U * p.Vt + p.U * X.Vt,
        p.U * Y.M * p.Vt + Y.U * p.Vt + p.U * Y.Vt;
        kwargs...,
    )
end

function number_eltype(p::SVDMPoint)
    return typeof(one(eltype(p.U)) + one(eltype(p.S)) + one(eltype(p.Vt)))
end
function number_eltype(X::UMVTVector)
    return typeof(one(eltype(X.U)) + one(eltype(X.M)) + one(eltype(X.Vt)))
end

@doc raw"""
    manifold_dimension(M::FixedRankMatrices{m,n,k,𝔽})

Return the manifold dimension for the `𝔽`-valued [`FixedRankMatrices`](@ref) `M`
of dimension `m`x`n` of rank `k`, namely

````math
\dim(\mathcal M) = k(m + n - k) \dim_ℝ 𝔽,
````

where ``\dim_ℝ 𝔽`` is the [`real_dimension`](@ref) of `𝔽`.
"""
function manifold_dimension(::FixedRankMatrices{m,n,k,𝔽}) where {m,n,k,𝔽}
    return (m + n - k) * k * real_dimension(𝔽)
end

function Base.one(p::SVDMPoint)
    return SVDMPoint(
        one(zeros(size(p.U, 1), size(p.U, 1))),
        ones(length(p.S)),
        one(zeros(size(p.Vt, 2), size(p.Vt, 2))),
        length(p.S),
    )
end
function Base.one(X::UMVTVector)
    return UMVTVector(
        one(zeros(size(X.U, 1), size(X.U, 1))),
        one(zeros(size(X.M))),
        one(zeros(size(X.Vt, 2), size(X.Vt, 2))),
        size(X.M, 1),
    )
end

@doc raw"""
    project(M, p, A)

Project the matrix ``A ∈ ℝ^{m,n}`` or from the embedding the tangent space at ``p`` on the [`FixedRankMatrices`](@ref) `M`,
further decomposing the result into ``X=UMV``, i.e. a [`UMVTVector`](@ref).
"""
project(::FixedRankMatrices, ::Any, ::Any)

function project!(::FixedRankMatrices, Y::UMVTVector, p::SVDMPoint, A::AbstractMatrix)
    av = A * (p.Vt')
    uTav = p.U' * av
    aTu = A' * p.U
    Y.M .= uTav
    Y.U .= A * p.Vt' - p.U * uTav
    Y.Vt .= (aTu - p.Vt' * uTav')'
    return Y
end

@doc raw"""
    representation_size(M::FixedRankMatrices{m,n,k})

Return the element size of a point on the [`FixedRankMatrices`](@ref) `M`, i.e.
the size of matrices on this manifold ``(m,n)``.
"""
@generated representation_size(::FixedRankMatrices{m,n}) where {m,n} = (m, n)

@doc raw"""
    retract(M, p, X, ::PolarRetraction)

Compute an SVD-based retraction on the [`FixedRankMatrices`](@ref) `M` by computing
````math
    q = U_kS_kV_k^\mathrm{H},
````
where ``U_k S_k V_k^\mathrm{H}`` is the shortened singular value decomposition ``USV=p+X``,
in the sense that ``S_k`` is the diagonal matrix of size ``k × k`` with the ``k`` largest
singular values and ``U`` and ``V`` are shortened accordingly.
"""
retract(::FixedRankMatrices, ::Any, ::Any, ::PolarRetraction)

function retract!(
    ::FixedRankMatrices{m,n,k},
    q::SVDMPoint,
    p::SVDMPoint,
    X::UMVTVector,
    ::PolarRetraction,
) where {m,n,k}
    QU, RU = qr([p.U X.U])
    QV, RV = qr([p.Vt' X.Vt'])

    # Compute T = svd(RU * [diagm(p.S) + X.M I; I zeros(k, k)] * RV')
    @views begin
        RU11 = RU[:, 1:k]
        RU12 = RU[:, (k + 1):(2 * k)]
        RV11 = RV[:, 1:k]
        RV12 = RV[:, (k + 1):(2 * k)]
    end
    tmp = RU11 .* p.S' .+ RU12
    mul!(tmp, RU11, X.M, true, true)
    tmp2 = tmp * RV11'
    mul!(tmp2, RU11, RV12', true, true)
    T = svd(tmp2)

    mul!(q.U, QU, @view(T.U[:, 1:k]))
    q.S .= @view(T.S[1:k])
    copyto!(q.Vt, @view(T.Vt[1:k, :]) * QV')

    return q
end

function Base.show(io::IO, ::FixedRankMatrices{M,N,K,𝔽}) where {M,N,K,𝔽}
    return print(io, "FixedRankMatrices($(M), $(N), $(K), $(𝔽))")
end
function Base.show(io::IO, ::MIME"text/plain", p::SVDMPoint)
    pre = " "
    summary(io, p)
    println(io, "\nU factor:")
    su = sprint(show, "text/plain", p.U; context=io, sizehint=0)
    su = replace(su, '\n' => "\n$(pre)")
    println(io, pre, su)
    println(io, "singular values:")
    ss = sprint(show, "text/plain", p.S; context=io, sizehint=0)
    ss = replace(ss, '\n' => "\n$(pre)")
    println(io, pre, ss)
    println(io, "Vt factor:")
    sv = sprint(show, "text/plain", p.Vt; context=io, sizehint=0)
    sv = replace(sv, '\n' => "\n$(pre)")
    return print(io, pre, sv)
end
function Base.show(io::IO, ::MIME"text/plain", X::UMVTVector)
    pre = " "
    summary(io, X)
    println(io, "\nU factor:")
    su = sprint(show, "text/plain", X.U; context=io, sizehint=0)
    su = replace(su, '\n' => "\n$(pre)")
    println(io, pre, su)
    println(io, "M factor:")
    sm = sprint(show, "text/plain", X.M; context=io, sizehint=0)
    sm = replace(sm, '\n' => "\n$(pre)")
    println(io, pre, sm)
    println(io, "Vt factor:")
    sv = sprint(show, "text/plain", X.Vt; context=io, sizehint=0)
    sv = replace(sv, '\n' => "\n$(pre)")
    return print(io, pre, sv)
end

@doc raw"""
    zero_vector(M::FixedRankMatrices, p::SVDMPoint)

Return a [`UMVTVector`](@ref) representing the zero tangent vector in the tangent space of
`p` on the [`FixedRankMatrices`](@ref) `M`, for example all three elements of the resulting
structure are zero matrices.
"""
function zero_vector(::FixedRankMatrices{m,n,k}, p::SVDMPoint) where {m,n,k}
    v = UMVTVector(
        zeros(eltype(p.U), m, k),
        zeros(eltype(p.S), k, k),
        zeros(eltype(p.Vt), k, n),
    )
    return v
end

function zero_vector!(::FixedRankMatrices{m,n,k}, X::UMVTVector, p::SVDMPoint) where {m,n,k}
    X.U .= zeros(eltype(X.U), m, k)
    X.M .= zeros(eltype(X.M), k, k)
    X.Vt .= zeros(eltype(X.Vt), k, n)
    return X
end
