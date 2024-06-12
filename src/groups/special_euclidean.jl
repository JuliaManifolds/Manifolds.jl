@doc raw"""
    SpecialEuclidean(n)

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
SemidirectProductGroup(Tn, SOn, RotationAction(Tn, SOn))
```

Points on ``\mathrm{SE}(n)`` may be represented as points on the underlying product manifold
``\mathrm{T}(n) × \mathrm{SO}(n)``. For group-specific functions, they may also be
represented as affine matrices with size `(n + 1, n + 1)` (see [`affine_matrix`](@ref)), for
which the group operation is [`MultiplicationOperation`](@ref).
"""
const SpecialEuclidean{T} = SemidirectProductGroup{
    ℝ,
    TranslationGroup{T,ℝ},
    SpecialOrthogonal{T},
    RotationAction{LeftAction,TranslationGroup{T,ℝ},SpecialOrthogonal{T}},
}

const SpecialEuclideanManifold{N} =
    ProductManifold{ℝ,Tuple{TranslationGroup{N,ℝ},SpecialOrthogonal{N}}}

function SpecialEuclidean(n; parameter::Symbol=:type)
    Tn = TranslationGroup(n; parameter=parameter)
    SOn = SpecialOrthogonal(n; parameter=parameter)
    A = RotationAction(Tn, SOn)
    return SemidirectProductGroup(Tn, SOn, A)
end

const SpecialEuclideanOperation{N} = SemidirectProductOperation{
    RotationAction{LeftAction,TranslationGroup{N,ℝ},SpecialOrthogonal{N}},
}
const SpecialEuclideanIdentity{N} = Identity{SpecialEuclideanOperation{N}}

function Base.show(io::IO, ::SpecialEuclidean{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "SpecialEuclidean($(n))")
end
function Base.show(io::IO, G::SpecialEuclidean{Tuple{Int}})
    n = _get_parameter(G)
    return print(io, "SpecialEuclidean($(n); parameter=:field)")
end

@inline function active_traits(f, M::SpecialEuclidean, args...)
    return merge_traits(IsGroupManifold(M.op), IsExplicitDecorator())
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
    adjoint_action(::SpecialEuclidean{TypeParameter{Tuple{3}}}, p, fX::TFVector{<:Any,VeeOrthogonalBasis{ℝ}})

Adjoint action of the [`SpecialEuclidean`](@ref) group on the vector with coefficients `fX`
tangent at point `p`.

The formula for the coefficients reads ``t×(R⋅ω) + R⋅r`` for the translation part and
``R⋅ω`` for the rotation part, where `t` is the translation part of `p`, `R` is the rotation
matrix part of `p`, `r` is the translation part of `fX` and `ω` is the rotation part of `fX`,
``×`` is the cross product and ``⋅`` is the matrix product.
"""
function adjoint_action(
    ::SpecialEuclidean{TypeParameter{Tuple{3}}},
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
    lie_bracket(G::SpecialEuclidean, X, Y)

Calculate the Lie bracket between elements `X` and `Y` of the special Euclidean Lie
algebra. For the matrix representation (which can be obtained using [`screw_matrix`](@ref))
the formula is ``[X, Y] = XY-YX``,

which is optimized when using components, then the formula reads
``[X, Y] = [(t_1, R_1), (t_2, R_2)] = (R_1 t_2 - R_2 t_1, R_1 R_2 - R_2 R_1)``.
"""
function lie_bracket(::SpecialEuclidean, X, Y)
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

function translate_diff!(G::SpecialEuclidean, Y, p, q, X, ::RightBackwardAction)
    np, hp = submanifold_components(G, p)
    nq, hq = submanifold_components(G, q)
    nX, hX = submanifold_components(G, X)
    nY, hY = submanifold_components(G, Y)
    hY .= hp' * hX * hp
    copyto!(nY, hq * (hX * np) + nX)
    @inbounds _padvector!(G, Y)
    return Y
end

@doc raw"""
    SpecialEuclideanInGeneralLinear

An explicit isometric and homomorphic embedding of ``\mathrm{SE}(n)`` in ``\mathrm{GL}(n+1)``
and ``𝔰𝔢(n)`` in ``𝔤𝔩(n+1)``.
Note that this is *not* a transparently isometric embedding.

# Constructor

    SpecialEuclideanInGeneralLinear(n)
"""
const SpecialEuclideanInGeneralLinear =
    EmbeddedManifold{ℝ,<:SpecialEuclidean,<:GeneralLinear}

function SpecialEuclideanInGeneralLinear(n)
    return EmbeddedManifold(SpecialEuclidean(n), GeneralLinear(n + 1))
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

Embed the tangent vector X at point `p` on [`SpecialEuclidean`](@ref) in the
[`GeneralLinear`](@ref) group. Point `p` can use any representation valid for
`SpecialEuclidean`. The embedding is similar from the one defined by [`screw_matrix`](@ref)
but the translation part is multiplied by inverse of the rotation part.
"""
function embed(M::SpecialEuclideanInGeneralLinear, p, X)
    G = M.manifold
    np, hp = submanifold_components(G, p)
    nX, hX = submanifold_components(G, X)
    Y = allocate_result(G, screw_matrix, nX, hX)
    nY, hY = submanifold_components(G, Y)
    copyto!(hY, hX)
    copyto!(nY, hp' * nX)
    @inbounds _padvector!(G, Y)
    return Y
end

function embed!(M::SpecialEuclideanInGeneralLinear, q, p)
    return copyto!(q, embed(M, p))
end
function embed!(M::SpecialEuclideanInGeneralLinear, Y, p, X)
    return copyto!(Y, embed(M, p, X))
end

"""
    project(M::SpecialEuclideanInGeneralLinear, p)

Project point `p` in [`GeneralLinear`](@ref) to the [`SpecialEuclidean`](@ref) group.
This is performed by extracting the rotation and translation part as in [`affine_matrix`](@ref).
"""
project(M::SpecialEuclideanInGeneralLinear, p)

"""
    project(M::SpecialEuclideanInGeneralLinear, p, X)

Project tangent vector `X` at point `p` in [`GeneralLinear`](@ref) to the
[`SpecialEuclidean`](@ref) Lie algebra.
This reverses the transformation performed by [`embed`](@ref embed(M::SpecialEuclideanInGeneralLinear, p, X))
"""
project(M::SpecialEuclideanInGeneralLinear, p, X)

function project!(M::SpecialEuclideanInGeneralLinear, q, p)
    return copyto!(q, project(M, p))
end
function project!(M::SpecialEuclideanInGeneralLinear, Y, p, X)
    return copyto!(Y, project(M, p, X))
end
