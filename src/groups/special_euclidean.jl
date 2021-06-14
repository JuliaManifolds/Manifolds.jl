@doc raw"""
    SpecialEuclidean(n)

Special Euclidean group $\mathrm{SE}(n)$, the group of rigid motions.

``\mathrm{SE}(n)`` is the semidirect product of the [`TranslationGroup`](@ref) on $ℝ^n$ and
[`SpecialOrthogonal(n)`](@ref)

````math
\mathrm{SE}(n) ≐ \mathrm{T}(n) ⋊_θ \mathrm{SO}(n),
````

where $θ$ is the canonical action of $\mathrm{SO}(n)$ on $\mathrm{T}(n)$ by vector rotation.

This constructor is equivalent to calling

```julia
Tn = TranslationGroup(n)
SOn = SpecialOrthogonal(n)
SemidirectProductGroup(Tn, SOn, RotationAction(Tn, SOn))
```

Points on $\mathrm{SE}(n)$ may be represented as points on the underlying product manifold
$\mathrm{T}(n) × \mathrm{SO}(n)$. For group-specific functions, they may also be
represented as affine matrices with size `(n + 1, n + 1)` (see [`affine_matrix`](@ref)), for
which the group operation is [`MultiplicationOperation`](@ref).
"""
const SpecialEuclidean{N} = SemidirectProductGroup{
    ℝ,
    TranslationGroup{Tuple{N},ℝ},
    SpecialOrthogonal{N},
    RotationAction{TranslationGroup{Tuple{N},ℝ},SpecialOrthogonal{N},LeftAction},
}

function SpecialEuclidean(n)
    Tn = TranslationGroup(n)
    SOn = SpecialOrthogonal(n)
    A = RotationAction(Tn, SOn)
    return SemidirectProductGroup(Tn, SOn, A)
end

Base.show(io::IO, ::SpecialEuclidean{n}) where {n} = print(io, "SpecialEuclidean($(n))")

Base.@propagate_inbounds function submanifold_component(
    ::SpecialEuclidean{n},
    p::AbstractMatrix,
    ::Val{1},
) where {n}
    return view(p, 1:n, n + 1)
end
Base.@propagate_inbounds function submanifold_component(
    ::SpecialEuclidean{n},
    p::AbstractMatrix,
    ::Val{2},
) where {n}
    return view(p, 1:n, 1:n)
end

function submanifold_components(G::SpecialEuclidean{n}, p::AbstractMatrix) where {n}
    @assert size(p) == (n + 1, n + 1)
    @inbounds t = submanifold_component(G, p, Val(1))
    @inbounds R = submanifold_component(G, p, Val(2))
    return (t, R)
end

Base.@propagate_inbounds function _padpoint!(
    ::SpecialEuclidean{n},
    q::AbstractMatrix,
) where {n}
    for i in 1:n
        q[n + 1, i] = 0
    end
    q[n + 1, n + 1] = 1
    return q
end

Base.@propagate_inbounds function _padvector!(
    ::SpecialEuclidean{n},
    X::AbstractMatrix,
) where {n}
    for i in 1:(n + 1)
        X[n + 1, i] = 0
    end
    return X
end

function adjoint_action(::SpecialEuclidean{3}, p, fX::TFVector{<:Any,VeeOrthogonalBasis{ℝ}})
    t = p.parts[1]
    R = p.parts[2]
    r = fX.data[SA[1, 2, 3]]
    ω = fX.data[SA[4, 5, 6]]
    Rω = R * ω
    return TFVector([cross(Rω, t) + R * r; Rω], fX.basis)
end

@doc raw"""
    affine_matrix(G::SpecialEuclidean, p) -> AbstractMatrix

Represent the point $p ∈ \mathrm{SE}(n)$ as an affine matrix.
For $p = (t, R) ∈ \mathrm{SE}(n)$, where $t ∈ \mathrm{T}(n), R ∈ \mathrm{SO}(n)$, the
affine representation is the $n + 1 × n + 1$ matrix

````math
\begin{pmatrix}
R & t \\
0^\mathrm{T} & 1
\end{pmatrix}.
````

This function embeds $\mathrm{SE}(n)$ in the general linear group $\mathrm{GL}(n+1)$.
It is an isometric embedding and group homomorphism [^RicoMartinez1988].

See also [`screw_matrix`](@ref) for matrix representations of the Lie algebra.

[^RicoMartinez1988]:
    > Rico Martinez, J. M., “Representations of the Euclidean group and its applications
    > to the kinematics of spatial chains,” PhD Thesis, University of Florida, 1988.
"""
function affine_matrix(G::SpecialEuclidean{n}, p) where {n}
    pis = submanifold_components(G, p)
    pmat = allocate_result(G, affine_matrix, pis...)
    map(copyto!, submanifold_components(G, pmat), pis)
    @inbounds _padpoint!(G, pmat)
    return pmat
end
affine_matrix(::SpecialEuclidean{n}, p::AbstractMatrix) where {n} = p
function affine_matrix(::GT, ::Identity{GT}) where {n,GT<:SpecialEuclidean{n}}
    s = maybesize(Size(n, n))
    s isa Size && return SDiagonal{n,Float64}(I)
    return Diagonal{Float64}(I, n)
end

@doc raw"""
    screw_matrix(G::SpecialEuclidean, X) -> AbstractMatrix

Represent the Lie algebra element $X ∈ 𝔰𝔢(n) = T_e \mathrm{SE}(n)$ as a screw matrix.
For $X = (b, Ω) ∈ 𝔰𝔢(n)$, where $Ω ∈ 𝔰𝔬(n) = T_e \mathrm{SO}(n)$, the screw representation is
the $n + 1 × n + 1$ matrix

````math
\begin{pmatrix}
Ω & b \\
0^\mathrm{T} & 0
\end{pmatrix}.
````

This function embeds $𝔰𝔢(n)$ in the general linear Lie algebra $𝔤𝔩(n+1)$.

See also [`affine_matrix`](@ref) for matrix representations of the Lie group.
"""
function screw_matrix(G::SpecialEuclidean{n}, X) where {n}
    Xis = submanifold_components(G, X)
    Xmat = allocate_result(G, screw_matrix, Xis...)
    map(copyto!, submanifold_components(G, Xmat), Xis)
    @inbounds _padvector!(G, Xmat)
    return Xmat
end
screw_matrix(::SpecialEuclidean{n}, X::AbstractMatrix) where {n} = X

function allocate_result(G::SpecialEuclidean{n}, f::typeof(affine_matrix), p...) where {n}
    return allocate(p[1], Size(n + 1, n + 1))
end
function allocate_result(G::SpecialEuclidean{n}, f::typeof(screw_matrix), X...) where {n}
    return allocate(X[1], Size(n + 1, n + 1))
end

compose(::SpecialEuclidean, p::AbstractMatrix, q::AbstractMatrix) = p * q

function compose!(
    ::SpecialEuclidean,
    x::AbstractMatrix,
    p::AbstractMatrix,
    q::AbstractMatrix,
)
    return mul!(x, p, q)
end

@doc raw"""
    group_exp(G::SpecialEuclidean{n}, X)

Compute the group exponential of $X = (b, Ω) ∈ 𝔰𝔢(n)$, where $b ∈ 𝔱(n)$ and $Ω ∈ 𝔰𝔬(n)$:

````math
\exp X = (t, R),
````

where $t ∈ \mathrm{T}(n)$ and $R = \exp Ω$ is the group exponential on $\mathrm{SO}(n)$.

In the [`screw_matrix`](@ref) representation, the group exponential is the matrix
exponential (see [`group_exp`](@ref)).
"""
group_exp(::SpecialEuclidean, ::Any)

@doc raw"""
    group_exp(G::SpecialEuclidean{2}, X)

Compute the group exponential of $X = (b, Ω) ∈ 𝔰𝔢(2)$, where $b ∈ 𝔱(2)$ and $Ω ∈ 𝔰𝔬(2)$:

````math
\exp X = (t, R) = (U(θ) b, \exp Ω),
````

where $t ∈ \mathrm{T}(2)$, $R = \exp Ω$ is the group exponential on $\mathrm{SO}(2)$,

````math
U(θ) = \frac{\sin θ}{θ} I_2 + \frac{1 - \cos θ}{θ^2} Ω,
````

and $θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e$
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.
"""
group_exp(::SpecialEuclidean{2}, ::Any)

@doc raw"""
    group_exp(G::SpecialEuclidean{3}, X)

Compute the group exponential of $X = (b, Ω) ∈ 𝔰𝔢(3)$, where $b ∈ 𝔱(3)$ and $Ω ∈ 𝔰𝔬(3)$:

````math
\exp X = (t, R) = (U(θ) b, \exp Ω),
````

where $t ∈ \mathrm{T}(3)$, $R = \exp Ω$ is the group exponential on $\mathrm{SO}(3)$,

````math
U(θ) = I_3 + \frac{1 - \cos θ}{θ^2} Ω + \frac{θ - \sin θ}{θ^3} Ω^2,
````

and $θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e$
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.
"""
group_exp(::SpecialEuclidean{3}, ::Any)

function group_exp!(G::SpecialEuclidean, q, X)
    Xmat = screw_matrix(G, X)
    qmat = exp(Xmat)
    map(copyto!, submanifold_components(G, q), submanifold_components(G, qmat))
    _padpoint!(G, q)
    return q
end
function group_exp!(G::SpecialEuclidean{2}, q, X)
    SO2 = submanifold(G, 2)
    b, Ω = submanifold_components(G, X)
    t, R = submanifold_components(G, q)
    @assert size(R) == (2, 2)
    @assert size(t) == (2,)
    @assert size(b) == (2,)

    θ = vee(SO2, Identity(SO2, Ω), Ω)[1]
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
function group_exp!(G::SpecialEuclidean{3}, q, X)
    SO3 = submanifold(G, 2)
    b, Ω = submanifold_components(G, X)
    t, R = submanifold_components(G, q)
    @assert size(R) == (3, 3)
    @assert size(t) == (3,)

    θ = norm(SO3, Identity(SO3, Ω), Ω) / sqrt(2)
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
    group_log(G::SpecialEuclidean{n}, p) where {n}

Compute the group logarithm of $p = (t, R) ∈ \mathrm{SE}(n)$, where $t ∈ \mathrm{T}(n)$
and $R ∈ \mathrm{SO}(n)$:

````math
\log p = (b, Ω),
````

where $b ∈ 𝔱(n)$ and $Ω = \log R ∈ 𝔰𝔬(n)$ is the group logarithm on $\mathrm{SO}(n)$.

In the [`affine_matrix`](@ref) representation, the group logarithm is the matrix logarithm
(see [`group_log`](@ref)):
"""
group_log(::SpecialEuclidean, ::Any)

@doc raw"""
    group_log(G::SpecialEuclidean{2}, p)

Compute the group logarithm of $p = (t, R) ∈ \mathrm{SE}(2)$, where $t ∈ \mathrm{T}(2)$
and $R ∈ \mathrm{SO}(2)$:

````math
\log p = (b, Ω) = (U(θ)^{-1} t, \log R),
````

where $b ∈ 𝔱(2)$, $Ω = \log R ∈ 𝔰𝔬(2)$ is the group logarithm on $\mathrm{SO}(2)$,

````math
U(θ) = \frac{\sin θ}{θ} I_2 + \frac{1 - \cos θ}{θ^2} Ω,
````

and $θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e$
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.
"""
group_log(::SpecialEuclidean{2}, ::Any)

@doc raw"""
    group_log(G::SpecialEuclidean{3}, p)

Compute the group logarithm of $p = (t, R) ∈ \mathrm{SE}(3)$, where $t ∈ \mathrm{T}(3)$
and $R ∈ \mathrm{SO}(3)$:

````math
\log p = (b, Ω) = (U(θ)^{-1} t, \log R),
````

where $b ∈ 𝔱(3)$, $Ω = \log R ∈ 𝔰𝔬(3)$ is the group logarithm on $\mathrm{SO}(3)$,

````math
U(θ) = I_3 + \frac{1 - \cos θ}{θ^2} Ω + \frac{θ - \sin θ}{θ^3} Ω^2,
````

and $θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e$
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.
"""
group_log(::SpecialEuclidean{3}, ::Any)

function group_log!(G::SpecialEuclidean, X, q)
    qmat = affine_matrix(G, q)
    Xmat = real(log_safe(qmat))
    map(copyto!, submanifold_components(G, X), submanifold_components(G, Xmat))
    _padvector!(G, X)
    return X
end
function group_log!(G::SpecialEuclidean{2}, X, q)
    SO2 = submanifold(G, 2)
    b, Ω = submanifold_components(G, X)
    t, R = submanifold_components(G, q)
    @assert size(b) == (2,)

    group_log!(SO2, Ω, R)
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
function group_log!(G::SpecialEuclidean{3}, X, q)
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

translate_diff(::SpecialEuclidean, p, q, X, ::LeftAction) = X
function translate_diff(
    ::SpecialEuclidean{N},
    p,
    q,
    X::ProductRepr,
    ::RightAction,
) where {N}
    p_aff = affine_matrix(G, p)
    X_screw = screw_matrix(G, X)
    diff_aff = p_aff \ X_screw * p_aff
    return ProductRepr(diff_aff[1:N, N + 1], diff_aff[1:N, 1:N])
end

function translate_diff!(G::SpecialEuclidean, Y, p, q, X, conv::ActionDirection)
    return copyto!(Y, translate_diff(G, p, q, X, conv))
end
