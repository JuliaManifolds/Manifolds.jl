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

show(io::IO, ::SpecialEuclidean{n}) where {n} = print(io, "SpecialEuclidean($(n))")

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
    for i ∈ 1:n
        q[n+1, i] = 0
    end
    q[n+1, n+1] = 1
    return q
end

Base.@propagate_inbounds function _padvector!(
    ::SpecialEuclidean{n},
    X::AbstractMatrix,
) where {n}
    for i ∈ 1:n+1
        X[n+1, i] = 0
    end
    return X
end

@doc doc"""
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

    affine_matrix(G::SpecialEuclidean, e, X) -> AbstractMatrix

Represent the Lie algebra element $X ∈ 𝔰𝔢(n) = T_e \mathrm{SE}(n)$ as a (screw) matrix.
For $X = (b, Ω) ∈ 𝔰𝔢(n)$, where $Ω ∈ 𝔰𝔬(n) = T_e \mathrm{SO}(n)$, the screw representation is
the $n + 1 × n + 1$ matrix

````math
\begin{pmatrix}
Ω & b \\
0^\mathrm{T} & 0
\end{pmatrix}.
````
"""
function affine_matrix(G::SpecialEuclidean{n}, p) where {n}
    pmat = allocate(p, Size(n + 1, n + 1))
    map(copyto!, submanifold_components(G, pmat), submanifold_components(G, p))
    @inbounds _padpoint!(G, pmat)
    return pmat
end
affine_matrix(::SpecialEuclidean{n}, p::AbstractMatrix) where {n} = p
@generated function affine_matrix(::GT, ::Identity{GT}) where {n,GT<:SpecialEuclidean{n}}
    return SDiagonal{n}(I)
end
function affine_matrix(G::SpecialEuclidean{n}, e, X) where {n}
    Xmat = allocate(X, Size(n + 1, n + 1))
    map(copyto!, submanifold_components(G, Xmat), submanifold_components(G, X))
    @inbounds _padvector!(G, Xmat)
    return Xmat
end
affine_matrix(::SpecialEuclidean{n}, e, X::AbstractMatrix) where {n} = X

compose(::SpecialEuclidean, p::AbstractMatrix, q::AbstractMatrix) = p * q

function compose!(
    ::SpecialEuclidean,
    x::AbstractMatrix,
    p::AbstractMatrix,
    q::AbstractMatrix,
)
    return mul!(x, p, q)
end

@doc doc"""
    group_exp(G::SpecialEuclidean{n}, X)

Compute the group exponential of $X = (b, Ω) ∈ 𝔰𝔢(n)$, where $b ∈ 𝔱(n)$ and $Ω ∈ 𝔰𝔬(n)$:

````math
\exp X = (t, R),
````

where $t ∈ \mathrm{T}(n)$ and $R = \exp Ω$ is the group exponential on $\mathrm{SO}(n)$.

In the [`affine_matrix`](@ref) representation, the group exponential is the matrix
exponential (see [`group_exp`](@ref)).

    group_exp(G::SpecialEuclidean{2}, X)

The group exponential on $\mathrm{SE}(2)$ is

````math
\exp X = (t, R) = (U(θ) b, \exp Ω),
````

where $U(θ)$ is

````math
U(θ) = \frac{\sin θ}{θ} I_2 + \frac{1 - \cos θ}{θ^2} Ω,
````

and $θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e$
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.

    group_exp(G::SpecialEuclidean{3}, X)

The group exponential on $\mathrm{SE}(3)$ is

````math
\exp X = (t, R) = (U(θ) b, \exp Ω),
````

where $U(θ)$ is

````math
U(θ) = I_3 + \frac{1 - \cos θ}{θ^2} Ω + \frac{θ - \sin θ}{θ^3} Ω^2,
````

and $θ$ is the same as above.
"""
group_exp(::SpecialEuclidean, ::Any)

function group_exp!(G::SpecialEuclidean, q, X)
    Xmat = affine_matrix(G, Identity(G), X)
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

    θ = vee(SO2, Identity(SO2), Ω)[1]
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

@doc doc"""
    group_log(G::SpecialEuclidean{n}, p)

Compute the group logarithm of $p = (t, R) ∈ \mathrm{SE}(n)$, where $t ∈ \mathrm{T}(n)$
and $R ∈ \mathrm{SO}(n)$:

````math
\log p = (b, Ω),
````

where $b ∈ 𝔱(n)$ and $Ω = \log R ∈ 𝔰𝔬(n)$ is the group logarithm on $\mathrm{SO}(n)$.

In the [`affine_matrix`](@ref) representation, the group logarithm is the matrix logarithm
(see [`group_log`](@ref)):

    group_log(G::SpecialEuclidean{2}, p)

The group logarithm on $\mathrm{SE}(2)$ is

````math
\log p = (b, Ω) = (U(θ)^{-1} t, \log R),
````

where $U(θ)$ is

````math
U(θ) = \frac{\sin θ}{θ} I_2 + \frac{1 - \cos θ}{θ^2} Ω,
````

and $θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e$
(see [`norm`](@ref norm(M::Rotations, p, X))) is the angle of the rotation.

    group_exp(G::SpecialEuclidean{3}, p)

The group logarithm on $\mathrm{SE}(3)$ is

````math
\log p = (b, Ω) = (U(θ)^{-1} t, \log R),
````

where $U(θ)$ is

````math
U(θ) = I_3 + \frac{1 - \cos θ}{θ^2} Ω + \frac{θ - \sin θ}{θ^3} Ω^2,
````

and $θ$ is the same as above.
"""
group_log(::SpecialEuclidean, ::Any)

function group_log!(G::SpecialEuclidean, X, q)
    qmat = affine_matrix(G, q)
    Xmat = real(log(qmat))
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
