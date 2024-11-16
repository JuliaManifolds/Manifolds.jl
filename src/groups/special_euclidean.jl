@doc raw"""
    SpecialEuclidean(
        n::Int;
        vectors::AbstractGroupVectorRepresentation=LeftInvariantRepresentation()
    )

Special Euclidean group ``\mathrm{SE}(n)``, the group of rigid motions.

``\mathrm{SE}(n)`` is the semidirect product of the [`TranslationGroup`](@ref) on ``ℝ^n`` and
[`SpecialOrthogonal`](@ref)`(n)`

````math
\mathrm{SE}(n) ≐ \mathrm{T}(n) ⋊_θ \mathrm{SO}(n),
````

where ``θ`` is the canonical action of ``\mathrm{SO}(n)`` on ``\mathrm{T}(n)`` by vector rotation.

This constructor is equivalent to calling

```julia
Tn = TranslationGroup(n)
SOn = SpecialOrthogonal(n)
SemidirectProductGroup(Tn, SOn, RotationAction(Tn, SOn), vectors)
```

Points on ``\mathrm{SE}(n)`` may be represented as points on the underlying product manifold
``\mathrm{T}(n) × \mathrm{SO}(n)``. For group-specific functions, they may also be
represented as affine matrices with size `(n + 1, n + 1)` (see [`affine_matrix`](@ref)), for
which the group operation is [`MultiplicationOperation`](@ref).

There are two supported conventions for tangent vector storage, which can be selected
using the `vectors` keyword argument:
* [`LeftInvariantRepresentation`](@ref) (default one), which corresponds to left-invariant 
  storage commonly used in other Lie groups.
* [`HybridTangentRepresentation`](@ref) which corresponds to the representation implied by
  product manifold structure of underlying groups.
"""
const SpecialEuclidean{T} = SemidirectProductGroup{
    ℝ,
    TranslationGroup{T,ℝ},
    SpecialOrthogonal{T},
    RotationAction{LeftAction,TranslationGroup{T,ℝ},SpecialOrthogonal{T}},
}

const SpecialEuclideanManifold{N} =
    ProductManifold{ℝ,Tuple{TranslationGroup{N,ℝ},SpecialOrthogonal{N}}}

function SpecialEuclidean(
    n::Int;
    vectors::AbstractGroupVectorRepresentation=LeftInvariantRepresentation(),
    parameter::Symbol=:type,
)
    Tn = TranslationGroup(n; parameter=parameter)
    SOn = SpecialOrthogonal(n; parameter=parameter)
    A = RotationAction(Tn, SOn)
    return SemidirectProductGroup(Tn, SOn, A, vectors)
end

const SpecialEuclideanOperation{N} = SemidirectProductOperation{
    RotationAction{LeftAction,TranslationGroup{N,ℝ},SpecialOrthogonal{N}},
}
const SpecialEuclideanIdentity{N} = Identity{SpecialEuclideanOperation{N}}

function Base.show(io::IO, G::SpecialEuclidean{TypeParameter{Tuple{n}}}) where {n}
    if vector_representation(G) isa LeftInvariantRepresentation
        return print(io, "SpecialEuclidean($(n))")
    else
        return print(io, "SpecialEuclidean($(n); vectors=$(G.vectors))")
    end
end
function Base.show(io::IO, G::SpecialEuclidean{Tuple{Int}})
    n = _get_parameter(G)
    if vector_representation(G) isa LeftInvariantRepresentation
        return print(io, "SpecialEuclidean($(n); parameter=:field)")
    else
        return print(io, "SpecialEuclidean($(n); parameter=:field, vectors=$(G.vectors))")
    end
end

@inline function active_traits(f, M::SpecialEuclidean, args...)
    return merge_traits(IsGroupManifold(M.op, M.vectors), IsExplicitDecorator())
end

"""
    _get_parameter(M::AbstractManifold)

Similar to `get_parameter` but it can be specialized for manifolds without breaking
manifolds being parametrized by other manifolds.
"""
_get_parameter(::AbstractManifold)

_get_parameter(::SpecialEuclidean{TypeParameter{Tuple{N}}}) where {N} = N
_get_parameter(M::SpecialEuclidean{Tuple{Int}}) = _get_parameter(M.manifold)
_get_parameter(::SpecialEuclideanManifold{TypeParameter{Tuple{N}}}) where {N} = N
_get_parameter(M::SpecialEuclideanManifold{Tuple{Int}}) = manifold_dimension(M.manifolds[1])

Base.@propagate_inbounds function Base.getindex(
    p::AbstractMatrix,
    M::Union{SpecialEuclidean,SpecialEuclideanManifold},
    i::Union{Integer,Val},
)
    return submanifold_component(M, p, i)
end

Base.@propagate_inbounds function Base.setindex!(
    q::AbstractMatrix,
    p,
    M::Union{SpecialEuclidean,SpecialEuclideanManifold},
    i::Union{Integer,Val},
)
    copyto!(submanifold_component(M, q, i), p)
    return p
end

Base.@propagate_inbounds function submanifold_component(
    G::Union{SpecialEuclidean,SpecialEuclideanManifold},
    p::AbstractMatrix,
    ::Val{1},
)
    n = _get_parameter(G)
    return view(p, 1:n, n + 1)
end
Base.@propagate_inbounds function submanifold_component(
    G::Union{SpecialEuclidean,SpecialEuclideanManifold},
    p::AbstractMatrix,
    ::Val{2},
)
    n = _get_parameter(G)
    return view(p, 1:n, 1:n)
end

function submanifold_components(
    G::Union{SpecialEuclidean,SpecialEuclideanManifold},
    p::AbstractMatrix,
)
    n = _get_parameter(G)
    @assert size(p) == (n + 1, n + 1)
    @inbounds t = submanifold_component(G, p, Val(1))
    @inbounds R = submanifold_component(G, p, Val(2))
    return (t, R)
end

Base.@propagate_inbounds function _padpoint!(
    G::Union{SpecialEuclidean,SpecialEuclideanManifold},
    q::AbstractMatrix,
)
    n = _get_parameter(G)
    for i in 1:n
        q[n + 1, i] = 0
    end
    q[n + 1, n + 1] = 1
    return q
end

Base.@propagate_inbounds function _padvector!(
    G::Union{SpecialEuclidean,SpecialEuclideanManifold},
    X::AbstractMatrix,
)
    n = _get_parameter(G)
    for i in 1:(n + 1)
        X[n + 1, i] = 0
    end
    return X
end

@doc raw"""
    adjoint_action(
        ::SpecialEuclidean{TypeParameter{Tuple{3}},<:HybridTangentRepresentation},
        p,
        fX::TFVector{<:Any,VeeOrthogonalBasis{ℝ}},
    )

Adjoint action of the [`SpecialEuclidean`](@ref) group on the vector with coefficients `fX`
tangent at point `p`.

The formula for the coefficients reads ``t×(R⋅ω) + R⋅r`` for the translation part and
``R⋅ω`` for the rotation part, where `t` is the translation part of `p`, `R` is the rotation
matrix part of `p`, `r` is the translation part of `fX` and `ω` is the rotation part of `fX`,
``×`` is the cross product and ``⋅`` is the matrix product.
"""
function adjoint_action(
    ::SpecialEuclidean{TypeParameter{Tuple{3}},<:HybridTangentRepresentation},
    p,
    fX::TFVector{<:Any,VeeOrthogonalBasis{ℝ}},
)
    t, R = submanifold_components(p)
    r = fX.data[SA[1, 2, 3]]
    ω = fX.data[SA[4, 5, 6]]
    Rω = R * ω
    return TFVector([cross(t, Rω) + R * r; Rω], fX.basis)
end

@doc raw"""
    adjoint_matrix(::SpecialEuclidean{TypeParameter{Tuple{2}}}, p)

Compute the adjoint matrix for the group [`SpecialEuclidean`](@ref)`(2)` at point `p`
in default coordinates. The formula follows Section 10.6.2 in [Chirikjian:2012](@cite)
but with additional scaling by ``\sqrt{2}`` due to a different choice of inner product.
The formula reads
````math
\begin{pmatrix}
R_{1,1} & R_{1,2} & t_2 \\
R_{2,1} & R_{2,2} & -t_1 \\
0 & 0 & 1
\end{pmatrix},
````
where ``R`` is the rotation matrix part of `p` and ``[t_1, t_2]`` is the translation part
of `p`.
"""
function adjoint_matrix(::SpecialEuclidean{TypeParameter{Tuple{2}}}, p)
    t, R = submanifold_components(p)
    return @SMatrix [
        R[1, 1] R[1, 2] t[2]/sqrt(2)
        R[2, 1] R[2, 2] -t[1]/sqrt(2)
        0 0 1
    ]
end
@doc raw"""
    adjoint_matrix(::SpecialEuclidean{TypeParameter{Tuple{3}}}, p)

Compute the adjoint matrix for the group [`SpecialEuclidean`](@ref)`(3)` at point `p`
in default coordinates. The formula follows Section 10.6.9 in [Chirikjian:2012](@cite) with
changes due to different conventions. The formula reads
````math
\begin{pmatrix}
R & UR/\sqrt{2} \\
0_{3×3} & R
\end{pmatrix}.
````
where ``R`` is the rotation matrix of `p` and ``U`` is the matrix
````math
\begin{pmatrix}
0 & -t_3 & t_2 \\
t_3 & 0 & -t_1 \\
-t_2 & t_1 & 0
\end{pmatrix}
````
where ``[t_1, t_2, t_3]`` is the translation vector of `p`.
"""
function adjoint_matrix(::SpecialEuclidean{TypeParameter{Tuple{3}}}, p)
    t, R = submanifold_components(p)
    Z = @SMatrix zeros(3, 3)
    c = sqrt(2) \ @SMatrix [0 -t[3] t[2]; t[3] 0 -t[1]; -t[2] t[1] 0]
    U = c * R
    return vcat(hcat(R, U), hcat(Z, R))
end

@doc raw"""
    affine_matrix(G::SpecialEuclidean, p) -> AbstractMatrix

Represent the point ``p ∈ \mathrm{SE}(n)`` as an affine matrix.
For ``p = (t, R) ∈ \mathrm{SE}(n)``, where ``t ∈ \mathrm{T}(n), R ∈ \mathrm{SO}(n)``, the
affine representation is the ``n + 1 × n + 1`` matrix

````math
\begin{pmatrix}
R & t \\
0^\mathrm{T} & 1
\end{pmatrix}.
````

This function embeds ``\mathrm{SE}(n)`` in the general linear group ``\mathrm{GL}(n+1)``.
It is an isometric embedding and group homomorphism [RicoMartinez:1988](@cite).

See also [`screw_matrix`](@ref) for matrix representations of the Lie algebra.

"""
function affine_matrix(G::SpecialEuclidean, p)
    pis = submanifold_components(G, p)
    pmat = allocate_result(G, affine_matrix, pis...)
    map(copyto!, submanifold_components(G, pmat), pis)
    @inbounds _padpoint!(G, pmat)
    return pmat
end
affine_matrix(::SpecialEuclidean, p::AbstractMatrix) = p
function affine_matrix(
    ::SpecialEuclidean{TypeParameter{Tuple{n}}},
    ::SpecialEuclideanIdentity{TypeParameter{Tuple{n}}},
) where {n}
    s = maybesize(Size(n, n))
    s isa Size && return SDiagonal{n,Float64}(I)
    return Diagonal{Float64}(I, n)
end
function affine_matrix(
    G::SpecialEuclidean{Tuple{Int}},
    ::SpecialEuclideanIdentity{Tuple{Int}},
)
    n = _get_parameter(G)
    return Diagonal{Float64}(I, n)
end

function check_point(G::SpecialEuclideanManifold, p::AbstractMatrix; kwargs...)
    n = _get_parameter(G)
    errs = DomainError[]
    # homogeneous
    if !isapprox(p[end, :], [zeros(size(p, 2) - 1)..., 1]; kwargs...)
        push!(
            errs,
            DomainError(
                p[end, :],
                "The last row of $p is not homogeneous, i.e. of form [0,..,0,1].",
            ),
        )
    end
    # translate part
    err2 = check_point(submanifold(G, 1), p[1:n, end]; kwargs...)
    !isnothing(err2) && push!(errs, err2)
    # SOn
    err3 = check_point(submanifold(G, 2), p[1:n, 1:n]; kwargs...)
    !isnothing(err3) && push!(errs, err3)
    if length(errs) > 1
        return CompositeManifoldError(errs)
    end
    return length(errs) == 0 ? nothing : first(errs)
end

function check_size(G::SpecialEuclideanManifold, p::AbstractMatrix; kwargs...)
    n = _get_parameter(G)
    return check_size(Euclidean(n + 1, n + 1), p)
end
function check_size(
    G::SpecialEuclideanManifold,
    p::AbstractMatrix,
    X::AbstractMatrix;
    kwargs...,
)
    n = _get_parameter(G)
    return check_size(Euclidean(n + 1, n + 1), X)
end

function check_vector(
    G::SpecialEuclideanManifold,
    p::AbstractMatrix,
    X::AbstractMatrix;
    kwargs...,
)
    n = _get_parameter(G)
    errs = DomainError[]
    # homogeneous
    if !isapprox(X[end, :], zeros(size(X, 2)); kwargs...)
        push!(
            errs,
            DomainError(
                X[end, :],
                "The last row of $X is not homogeneous, i.e. of form [0,..,0,0].",
            ),
        )
    end
    # translate part
    err2 = check_vector(submanifold(G, 1), p[1:n, end], X[1:n, end]; kwargs...)
    !isnothing(err2) && push!(errs, err2)
    # SOn
    err3 = check_vector(submanifold(G, 2), p[1:n, 1:n], X[1:n, 1:n]; kwargs...)
    !isnothing(err3) && push!(errs, err3)
    (length(errs) > 1) && (return CompositeManifoldError(errs))
    return length(errs) == 0 ? nothing : first(errs)
end

@doc raw"""
    screw_matrix(G::SpecialEuclidean, X) -> AbstractMatrix

Represent the Lie algebra element ``X ∈ 𝔰𝔢(n) = T_e \mathrm{SE}(n)`` as a screw matrix.
For ``X = (b, Ω) ∈ 𝔰𝔢(n)``, where ``Ω ∈ 𝔰𝔬(n) = T_e \mathrm{SO}(n)``, the screw representation is
the ``n + 1 × n + 1`` matrix

````math
\begin{pmatrix}
Ω & b \\
0^\mathrm{T} & 0
\end{pmatrix}.
````

This function embeds ``𝔰𝔢(n)`` in the general linear Lie algebra ``𝔤𝔩(n+1)`` but it's not
a homomorphic embedding (see [`SpecialEuclideanInGeneralLinear`](@ref) for a homomorphic one).

See also [`affine_matrix`](@ref) for matrix representations of the Lie group.
"""
function screw_matrix(G::SpecialEuclidean, X)
    Xis = submanifold_components(G, X)
    Xmat = allocate_result(G, screw_matrix, Xis...)
    map(copyto!, submanifold_components(G, Xmat), Xis)
    @inbounds _padvector!(G, Xmat)
    return Xmat
end
screw_matrix(::SpecialEuclidean, X::AbstractMatrix) = X

function allocate_result(G::SpecialEuclidean, ::typeof(affine_matrix), p...)
    n = _get_parameter(G)
    return allocate(p[1], Size(n + 1, n + 1))
end
function allocate_result(G::SpecialEuclidean, ::typeof(screw_matrix), X...)
    n = _get_parameter(G)
    return allocate(X[1], Size(n + 1, n + 1))
end

compose(::SpecialEuclidean, p::AbstractMatrix, q::AbstractMatrix) = p * q

function compose!(
    ::SpecialEuclidean,
    x::AbstractMatrix,
    p::AbstractMatrix,
    q::AbstractMatrix,
)
    copyto!(x, p * q)
    return x
end

# More generic default was mostly OK but it lacks padding
function exp!(M::SpecialEuclideanManifold, q::AbstractMatrix, p, X)
    map(
        exp!,
        M.manifolds,
        submanifold_components(M, q),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    @inbounds _padpoint!(M, q)
    return q
end
function exp!(M::SpecialEuclideanManifold, q::AbstractMatrix, p, X, t::Number)
    map(
        (N, qc, pc, Xc) -> exp!(N, qc, pc, Xc, t),
        M.manifolds,
        submanifold_components(M, q),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    @inbounds _padpoint!(M, q)
    return q
end

@doc raw"""
    exp_lie(G::SpecialEuclidean{n}, X)

Compute the group exponential of ``X = (b, Ω) ∈ 𝔰𝔢(n)``, where ``b ∈ 𝔱(n)`` and ``Ω ∈ 𝔰𝔬(n)``:

````math
\exp X = (t, R),
````

where ``t ∈ \mathrm{T}(n)`` and ``R = \exp Ω`` is the group exponential on ``\mathrm{SO}(n)``.

In the [`screw_matrix`](@ref) representation, the group exponential is the matrix
exponential (see [`exp_lie`](@ref)).
"""
exp_lie(::SpecialEuclidean, ::Any)

@doc raw"""
    exp_lie(G::SpecialEuclidean{TypeParameter{Tuple{2}}}, X)

Compute the group exponential of ``X = (b, Ω) ∈ 𝔰𝔢(2)``, where ``b ∈ 𝔱(2)`` and ``Ω ∈ 𝔰𝔬(2)``:

````math
\exp X = (t, R) = (U(θ) b, \exp Ω),
````

where ``t ∈ \mathrm{T}(2)``, ``R = \exp Ω`` is the group exponential on ``\mathrm{SO}(2)``,

````math
U(θ) = \frac{\sin θ}{θ} I_2 + \frac{1 - \cos θ}{θ^2} Ω,
````

and ``θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e``
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.
"""
exp_lie(::SpecialEuclidean{TypeParameter{Tuple{2}}}, ::Any)

@doc raw"""
    exp_lie(G::SpecialEuclidean{TypeParameter{Tuple{3}}}, X)

Compute the group exponential of ``X = (b, Ω) ∈ 𝔰𝔢(3)``, where ``b ∈ 𝔱(3)`` and ``Ω ∈ 𝔰𝔬(3)``:

````math
\exp X = (t, R) = (U(θ) b, \exp Ω),
````

where ``t ∈ \mathrm{T}(3)``, ``R = \exp Ω`` is the group exponential on ``\mathrm{SO}(3)``,

````math
U(θ) = I_3 + \frac{1 - \cos θ}{θ^2} Ω + \frac{θ - \sin θ}{θ^3} Ω^2,
````

and ``θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e``
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.
"""
exp_lie(::SpecialEuclidean{TypeParameter{Tuple{3}}}, ::Any)

function exp_lie!(G::SpecialEuclidean, q, X)
    Xmat = screw_matrix(G, X)
    qmat = exp(Xmat)
    map(copyto!, submanifold_components(G, q), submanifold_components(G, qmat))
    _padpoint!(G, q)
    return q
end
function exp_lie!(G::SpecialEuclidean{TypeParameter{Tuple{2}}}, q, X)
    SO2 = submanifold(G, 2)
    b, Ω = submanifold_components(G, X)
    t, R = submanifold_components(G, q)
    @assert size(R) == (2, 2)
    @assert size(t) == (2,)
    @assert size(b) == (2,)

    θ = vee(SO2, identity_element(SO2, R), Ω)[1]
    sinθ, cosθ = sincos(θ)
    if θ ≈ 0
        α = 1 - θ^2 / 6
        β = θ / 2
    else
        α = sinθ / θ
        β = (1 - cosθ) / θ
    end

    @inbounds begin
        R[1] = cosθ
        R[2] = sinθ
        R[3] = -sinθ
        R[4] = cosθ
        t[1] = α * b[1] - β * b[2]
        t[2] = α * b[2] + β * b[1]
        _padpoint!(G, q)
    end
    return q
end
function exp_lie!(G::SpecialEuclidean{TypeParameter{Tuple{3}}}, q, X)
    SO3 = submanifold(G, 2)
    b, Ω = submanifold_components(G, X)
    t, R = submanifold_components(G, q)
    @assert size(R) == (3, 3)
    @assert size(t) == (3,)

    θ = norm(SO3, Identity(SO3), Ω) / sqrt(2)
    θ² = θ^2
    if θ ≈ 0
        α = 1 - θ² / 6
        β = θ / 2
        γ = 1 / 6 - θ² / 120
    else
        sinθ, cosθ = sincos(θ)
        α = sinθ / θ
        β = (1 - cosθ) / θ²
        γ = (1 - α) / θ²
    end

    Ω² = Ω^2
    Jₗ = I + β .* Ω .+ γ .* Ω²
    R .= I + α .* Ω .+ β .* Ω²
    copyto!(t, Jₗ * b)
    @inbounds _padpoint!(G, q)
    return q
end

@doc raw"""
    jacobian_exp_inv_argument(
        M::SpecialEuclidean{TypeParameter{Tuple{2}}},
        p,
        X,
    )

Compute Jacobian matrix of the invariant exponential map on [`SpecialEuclidean`](@ref)`(2)`.
The formula reads
````math
\begin{pmatrix}
\frac{1}{θ}\sin(θ) & \frac{1}{θ} (1-\cos(θ)) & \frac{1}{\sqrt{2} θ^2}(t_1(\sin(θ) - θ) + t_2(\cos(θ) - 1)) \\
\frac{1}{θ}(-1+\cos(θ)) & \frac{1}{θ}\sin(θ) & \frac{1}{\sqrt{2} θ^2}(t_2(\sin(θ) - θ) + t_1(-\cos(θ) + 1)) \\
0 & 0 & 1
\end{pmatrix}.
````
where ``θ`` is the norm of `X` and ``[t_1, t_2]`` is the translation part
of `X`.
It is adapted from [Chirikjian:2012](@cite), Section 10.6.2, to `Manifolds.jl` conventions.
"""
jacobian_exp_inv_argument(M::SpecialEuclidean{TypeParameter{Tuple{2}}}, p, X)
@doc raw"""
    jacobian_exp_inv_argument(
        M::SpecialEuclidean{TypeParameter{Tuple{3}}},
        p,
        X,
    )

Compute Jacobian matrix of the invariant exponential map on [`SpecialEuclidean`](@ref)`(3)`.
The formula reads
````math
\begin{pmatrix}
R & Q \\
0_{3×3} & R
\end{pmatrix},
````
where ``R`` is the Jacobian of exponential map on [`Rotations`](@ref)`(3)` with respect to
the argument, and ``Q`` is
````math
\begin{align*}
Q = &\frac{1}{2} T \\
    &- \frac{θ - \sin(θ)}{θ^3} (X_r T + T X_r + X_r T X_r) \\
    & + \frac{1 - \frac{θ^2}{2} - \cos(θ)}{θ^4} (X_r^2 T + T X_r^2 - 3 X_r T X_r)\\
    & + \frac{1}{2}\left(\frac{1 - \frac{θ^2}{2} - \cos(θ)}{θ^4} - 3 \frac{θ - \sin(θ) - \frac{θ^3}{6}}{θ^5}\right) (X_r T X_r^2 + X_r^2 T X_r)
\end{align*}
````
where ``X_r`` is the rotation part of ``X`` and ``T`` is
````math
\frac{1}{\sqrt{2}}\begin{pmatrix}
0 & -t_3 & t_2 \\
t_3 & 0 & -t_1 \\
-t_2 & t_1 & 0
\end{pmatrix},
````
where ``[t_1, t_2, t_3]`` is the translation part of `X`.
It is adapted from [BarfootFurgale:2014](@cite), Eq. (102), to `Manifolds.jl` conventions.
"""
jacobian_exp_inv_argument(M::SpecialEuclidean{TypeParameter{Tuple{3}}}, p, X)
function jacobian_exp_inv_argument(M::SpecialEuclidean, p, X)
    J = allocate_jacobian(M, M, jacobian_exp_inv_argument, p)
    return jacobian_exp_inv_argument!(M, J, p, X)
end
function jacobian_exp_inv_argument!(
    M::SpecialEuclidean{TypeParameter{Tuple{2}}},
    J::AbstractMatrix,
    p,
    X,
)
    θ = norm(X.x[2]) / sqrt(2)
    t1, t2 = X.x[1]
    copyto!(J, I)
    if θ ≈ 0
        J[1, 3] = -t2 / (sqrt(2) * 2)
        J[2, 3] = t1 / (sqrt(2) * 2)
    else
        J[1, 1] = J[2, 2] = sin(θ) / θ
        J[1, 2] = (cos(θ) - 1) / θ
        J[2, 1] = -J[1, 2]
        J[1, 3] = (t1 * (sin(θ) - θ) + t2 * (cos(θ) - 1)) / (sqrt(2) * θ^2)
        J[2, 3] = (t2 * (sin(θ) - θ) + t1 * (1 - cos(θ))) / (sqrt(2) * θ^2)
    end
    return J
end

function jacobian_exp_inv_argument!(
    M::SpecialEuclidean{TypeParameter{Tuple{3}}},
    J::AbstractMatrix,
    p,
    X,
)
    θ = norm(X.x[2]) / sqrt(2)
    t1, t2, t3 = X.x[1]
    Xr = X.x[2]
    copyto!(J, I)
    if θ ≈ 0
        J[1, 5] = t3 / (sqrt(2) * 2)
        J[1, 6] = -t2 / (sqrt(2) * 2)
        J[2, 6] = t1 / (sqrt(2) * 2)
        J[2, 4] = -t3 / (sqrt(2) * 2)
        J[3, 4] = t2 / (sqrt(2) * 2)
        J[3, 5] = -t1 / (sqrt(2) * 2)
    else
        a = (cos(θ) - 1) / θ^2
        b = (θ - sin(θ)) / θ^3
        # top left block
        view(J, SOneTo(3), SOneTo(3)) .+= a .* Xr .+ b .* (Xr^2)
        # bottom right block
        view(J, 4:6, 4:6) .= view(J, SOneTo(3), SOneTo(3))
        # top right block
        Xr = -Xr
        tx = @SMatrix [
            0 -t3/sqrt(2) t2/sqrt(2)
            t3/sqrt(2) 0 -t1/sqrt(2)
            -t2/sqrt(2) t1/sqrt(2) 0
        ]
        J[1:3, 4:6] .= -tx ./ 2
        J[1:3, 4:6] .-= (θ - sin(θ)) / (θ^3) * (Xr * tx + tx * Xr + Xr * tx * Xr)
        J[1:3, 4:6] .+=
            ((1 - θ^2 / 2 - cos(θ)) / θ^4) * (Xr^2 * tx + tx * Xr^2 - 3 * Xr * tx * Xr)
        J[1:3, 4:6] .+=
            0.5 *
            ((1 - (θ^2) / 2 - cos(θ)) / (θ^4) - 3 * (θ - sin(θ) - (θ^3) / 6) / (θ^5)) *
            (Xr * tx * Xr^2 + Xr^2 * tx * Xr)
    end
    return J
end

@doc raw"""
    log_lie(G::SpecialEuclidean, p)

Compute the group logarithm of ``p = (t, R) ∈ \mathrm{SE}(n)``, where ``t ∈ \mathrm{T}(n)``
and ``R ∈ \mathrm{SO}(n)``:

````math
\log p = (b, Ω),
````

where ``b ∈ 𝔱(n)`` and ``Ω = \log R ∈ 𝔰𝔬(n)`` is the group logarithm on ``\mathrm{SO}(n)``.

In the [`affine_matrix`](@ref) representation, the group logarithm is the matrix logarithm
(see [`log_lie`](@ref)):
"""
log_lie(::SpecialEuclidean, ::Any)

@doc raw"""
    log_lie(G::SpecialEuclidean{TypeParameter{Tuple{2}}}, p)

Compute the group logarithm of ``p = (t, R) ∈ \mathrm{SE}(2)``, where ``t ∈ \mathrm{T}(2)``
and ``R ∈ \mathrm{SO}(2)``:

````math
\log p = (b, Ω) = (U(θ)^{-1} t, \log R),
````

where ``b ∈ 𝔱(2)``, ``Ω = \log R ∈ 𝔰𝔬(2)`` is the group logarithm on ``\mathrm{SO}(2)``,

````math
U(θ) = \frac{\sin θ}{θ} I_2 + \frac{1 - \cos θ}{θ^2} Ω,
````

and ``θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e``
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.
"""
log_lie(::SpecialEuclidean{TypeParameter{Tuple{2}}}, ::Any)

@doc raw"""
    log_lie(G::SpecialEuclidean{TypeParameter{Tuple{3}}}, p)

Compute the group logarithm of ``p = (t, R) ∈ \mathrm{SE}(3)``, where ``t ∈ \mathrm{T}(3)``
and ``R ∈ \mathrm{SO}(3)``:

````math
\log p = (b, Ω) = (U(θ)^{-1} t, \log R),
````

where ``b ∈ 𝔱(3)``, ``Ω = \log R ∈ 𝔰𝔬(3)`` is the group logarithm on ``\mathrm{SO}(3)``,

````math
U(θ) = I_3 + \frac{1 - \cos θ}{θ^2} Ω + \frac{θ - \sin θ}{θ^3} Ω^2,
````

and ``θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e``
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.
"""
log_lie(::SpecialEuclidean{TypeParameter{Tuple{3}}}, ::Any)

function _log_lie!(G::SpecialEuclidean, X, q)
    qmat = affine_matrix(G, q)
    Xmat = real(log_safe(qmat))
    map(copyto!, submanifold_components(G, X), submanifold_components(G, Xmat))
    _padvector!(G, X)
    return X
end
function _log_lie!(G::SpecialEuclidean{TypeParameter{Tuple{2}}}, X, q)
    SO2 = submanifold(G, 2)
    b, Ω = submanifold_components(G, X)
    t, R = submanifold_components(G, q)
    @assert size(b) == (2,)

    log_lie!(SO2, Ω, R)
    @inbounds θ = Ω[2]
    β = θ / 2
    α = θ ≈ 0 ? 1 - β^2 / 3 : β * cot(β)

    @inbounds begin
        b[1] = α * t[1] + β * t[2]
        b[2] = α * t[2] - β * t[1]
        _padvector!(G, X)
    end
    return X
end
function _log_lie!(G::SpecialEuclidean{TypeParameter{Tuple{3}}}, X, q)
    b, Ω = submanifold_components(G, X)
    t, R = submanifold_components(G, q)
    @assert size(Ω) == (3, 3)
    @assert size(b) == (3,)

    trR = tr(R)
    cosθ = (trR - 1) / 2
    θ = acos(clamp(cosθ, -1, 1))
    θ² = θ^2
    if θ ≈ 0
        α = 1 / 2 + θ² / 12
        β = 1 / 12 + θ² / 720
    else
        sinθ = sin(θ)
        α = θ / sinθ / 2
        β = 1 / θ² - (1 + cosθ) / 2 / θ / sinθ
    end

    Ω .= (R .- transpose(R)) .* α
    Jₗ⁻¹ = I - Ω ./ 2 .+ β .* Ω^2
    mul!(b, Jₗ⁻¹, t)
    @inbounds _padvector!(G, X)
    return X
end

# More generic default was mostly OK but it lacks padding
function log!(M::SpecialEuclideanManifold, X::AbstractMatrix, p, q)
    map(
        log!,
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        submanifold_components(M, q),
    )
    @inbounds _padvector!(M, X)
    return X
end

"""
    lie_bracket(G::SpecialEuclidean, X::ArrayPartition, Y::ArrayPartition)
    lie_bracket(G::SpecialEuclidean, X::AbstractMatrix, Y::AbstractMatrix)

Calculate the Lie bracket between elements `X` and `Y` of the special Euclidean Lie
algebra. For the matrix representation (which can be obtained using [`screw_matrix`](@ref))
the formula is ``[X, Y] = XY-YX``, while in the `ArrayPartition` representation the
formula reads ``[X, Y] = [(t_1, R_1), (t_2, R_2)] = (R_1 t_2 - R_2 t_1, R_1 R_2 - R_2 R_1)``.
"""
function lie_bracket(::SpecialEuclidean, X::AbstractMatrix, Y::AbstractMatrix)
    return X * Y - Y * X
end

function lie_bracket!(G::SpecialEuclidean, Z, X, Y)
    nX, hX = submanifold_components(G, X)
    nY, hY = submanifold_components(G, Y)
    nZ, hZ = submanifold_components(G, Z)
    lie_bracket!(G.manifold.manifolds[2], hZ, hX, hY)
    nZ .= hX * nY .- hY * nX
    @inbounds _padvector!(G, Z)
    return Z
end

"""
    translate_diff(G::SpecialEuclidean, p, q, X, ::RightBackwardAction)

Differential of the right action of the [`SpecialEuclidean`](@ref) group on itself.
The formula for the rotation part is the differential of the right rotation action, while
the formula for the translation part reads
````math
R_q⋅X_R⋅t_p + X_t
````
where ``R_q`` is the rotation part of `q`, ``X_R`` is the rotation part of `X`, ``t_p``
is the translation part of `p` and ``X_t`` is the translation part of `X`.
"""
translate_diff(G::SpecialEuclidean, p, q, X, ::RightBackwardAction)

function translate_diff!(
    G::SpecialEuclidean{T,<:HybridTangentRepresentation},
    Y,
    p,
    q,
    X,
    ::RightBackwardAction,
) where {T}
    np, hp = submanifold_components(G, p)
    nq, hq = submanifold_components(G, q)
    nX, hX = submanifold_components(G, X)
    nY, hY = submanifold_components(G, Y)
    hY .= hp' * hX * hp
    copyto!(nY, hq * (hX * np) + nX)
    @inbounds _padvector!(G, Y)
    return Y
end

function adjoint_action!(G::SpecialEuclidean, Y, p, Xₑ, ::LeftAction)
    np, hp = submanifold_components(G, p)
    n, h = submanifold_components(G, Y)
    nX, hX = submanifold_components(G, Xₑ)
    H = submanifold(G, 2)
    adjoint_action!(H, h, hp, hX, LeftAction())
    A = G.op.action
    apply!(A, n, hp, nX)
    LinearAlgebra.axpy!(-1, apply_diff_group(A, Identity(H), h, np), n)
    @inbounds _padvector!(G, Y)
    return Y
end

@doc raw"""
    SpecialEuclideanInGeneralLinear

An explicit isometric and homomorphic embedding of ``\mathrm{SE}(n)`` in ``\mathrm{GL}(n+1)``
and ``𝔰𝔢(n)`` in ``𝔤𝔩(n+1)``.
Note that this is *not* a transparently isometric embedding.

# Constructor

    SpecialEuclideanInGeneralLinear(
        n::Int;
        se_vectors::AbstractGroupVectorRepresentation=LeftInvariantVectorRepresentation(),
    )

Where `se_vectors` is the tangent vector representation of the [`SpecialEuclidean`](@ref)
group to be used.
"""
const SpecialEuclideanInGeneralLinear =
    EmbeddedManifold{ℝ,<:SpecialEuclidean,<:GeneralLinear}

function SpecialEuclideanInGeneralLinear(
    n::Int;
    se_vectors::AbstractGroupVectorRepresentation=LeftInvariantVectorRepresentation(),
)
    return EmbeddedManifold(SpecialEuclidean(n; vectors=se_vectors), GeneralLinear(n + 1))
end

"""
    embed(M::SpecialEuclideanInGeneralLinear, p)

Embed the point `p` on [`SpecialEuclidean`](@ref) in the [`GeneralLinear`](@ref) group.
The embedding is calculated using [`affine_matrix`](@ref).
"""
function embed(M::SpecialEuclideanInGeneralLinear, p)
    G = M.manifold
    return affine_matrix(G, p)
end
"""
    embed(M::SpecialEuclideanInGeneralLinear, p, X)

Embed the tangent vector `X`` at point `p` on [`SpecialEuclidean`](@ref) in the
[`GeneralLinear`](@ref) group. Point `p` can use any representation valid for
`SpecialEuclidean`. The embedding is similar from the one defined by [`screw_matrix`](@ref).
"""
function embed(M::SpecialEuclideanInGeneralLinear, p, X)
    G = M.manifold
    np, hp = submanifold_components(G, p)
    nX, hX = submanifold_components(G, X)
    Y = allocate_result(G, screw_matrix, nX, hX)
    nY, hY = submanifold_components(G, Y)
    copyto!(hY, hX)
    if vector_representation(M.manifold) isa LeftInvariantRepresentation
        copyto!(nY, nX)
    else
        copyto!(nY, hp' * nX)
    end
    @inbounds _padvector!(G, Y)
    return Y
end

function embed!(M::SpecialEuclideanInGeneralLinear, q, p)
    return copyto!(q, embed(M, p))
end
function embed!(M::SpecialEuclideanInGeneralLinear, Y, p, X)
    return copyto!(Y, embed(M, p, X))
end

function project!(M::SpecialEuclideanInGeneralLinear, q, p)
    return copyto!(q, project(M, p))
end
function project!(M::SpecialEuclideanInGeneralLinear, Y, p, X)
    return copyto!(Y, project(M, p, X))
end
