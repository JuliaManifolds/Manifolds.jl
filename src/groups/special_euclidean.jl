@doc doc"""
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
$\mathrm{T}(n) \times \mathrm{SO}(n)$. For group-specific functions, they may also be
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
    x::AbstractMatrix,
    ::Val{1},
) where {n}
    return view(x, 1:n, n + 1)
end
Base.@propagate_inbounds function submanifold_component(
    ::SpecialEuclidean{n},
    x::AbstractMatrix,
    ::Val{2},
) where {n}
    return view(x, 1:n, 1:n)
end

function submanifold_components(G::SpecialEuclidean{n}, x::AbstractMatrix) where {n}
    @assert size(x) == (n + 1, n + 1)
    @inbounds t = submanifold_component(G, x, Val(1))
    @inbounds R = submanifold_component(G, x, Val(2))
    return (t, R)
end

Base.@propagate_inbounds function _padpoint!(
    ::SpecialEuclidean{n},
    y::AbstractMatrix,
) where {n}
    for i = 1:n
        y[n+1, i] = 0
    end
    y[n+1, n+1] = 1
    return y
end

Base.@propagate_inbounds function _padvector!(
    ::SpecialEuclidean{n},
    v::AbstractMatrix,
) where {n}
    for i = 1:n+1
        v[n+1, i] = 0
    end
    return v
end

@doc doc"""
    affine_matrix(G::SpecialEuclidean, x) -> AbstractMatrix

Represent the point $x ∈ \mathrm{SE}(n)$ as an affine matrix.
For $x = (t, R) ∈ \mathrm{SE}(n)$, where $t ∈ \mathrm{T}(n), R ∈ \mathrm{SO}(n)$, the
affine representation is the $n + 1 × n + 1$ matrix

````math
\begin{pmatrix}
R & t \\
0^\mathrm{T} & 1
\end{pmatrix}.
````

    affine_matrix(G::SpecialEuclidean, e, v) -> AbstractMatrix

Represent the Lie algebra element $v ∈ 𝔰𝔢(n) = T_e \mathrm{SE}(n)$ as a (screw) matrix.
For $v = (b, Ω) ∈ 𝔰𝔢(n)$, where $Ω ∈ 𝔰𝔬(n) = T_e \mathrm{SO}(n)$, the screw representation is
the $n + 1 × n + 1$ matrix

````math
\begin{pmatrix}
Ω & b \\
0^\mathrm{T} & 0
\end{pmatrix}.
````
"""
function affine_matrix(G::SpecialEuclidean{n}, x) where {n}
    y = allocate(x, Size(n + 1, n + 1))
    map(copyto!, submanifold_components(G, y), submanifold_components(G, x))
    @inbounds _padpoint!(G, y)
    return y
end
affine_matrix(::SpecialEuclidean{n}, x::AbstractMatrix) where {n} = x
@generated function affine_matrix(::GT, ::Identity{GT}) where {n,GT<:SpecialEuclidean{n}}
    return SDiagonal{n}(I)
end
function affine_matrix(G::SpecialEuclidean{n}, e, v) where {n}
    w = allocate(v, Size(n + 1, n + 1))
    map(copyto!, submanifold_components(G, w), submanifold_components(G, v))
    @inbounds _padvector!(G, w)
    return w
end
affine_matrix(::SpecialEuclidean{n}, e, v::AbstractMatrix) where {n} = v

compose(::SpecialEuclidean, x::AbstractMatrix, y::AbstractMatrix) = x * y

function compose!(
    ::SpecialEuclidean,
    z::AbstractMatrix,
    x::AbstractMatrix,
    y::AbstractMatrix,
)
    return mul!(z, x, y)
end

@doc doc"""
    group_exp(G::SpecialEuclidean, v)

Compute the group exponential of $v ∈ 𝔰𝔢(n)$. In the [`affine_matrix`](@ref) representation,
the group exponential is the matrix exponential (see [`group_exp`](@ref)).

    group_exp(G::SpecialEuclidean{2}, v)

The group exponential on $\mathrm{SE}(2)$ for $v = (b, Ω) ∈ 𝔰𝔢(2)$, where $R ∈ 𝔰𝔬(2)$ is

````math
\exp v = (U(θ) b, \exp Ω) = (t, R),
````

where $R = \exp Ω$ is the group exponential on $\mathrm{SO}(3)$, and $U(θ)$ is

````math
U(θ) = \frac{\sin θ}{θ} I_2 + \frac{1 - \cos θ}{θ^2} Ω,
````

where $θ = \frac{1}{\sqrt{2}} \lVert Ω \rVert_e$
(see [`norm`](@ref norm(M::Rotations, x, v))) is the angle of the rotation.

    group_exp(G::SpecialEuclidean{3}, v)

The group exponential on $\mathrm{SE}(3)$ for $v = (b, Ω) ∈ 𝔰𝔢(3)$, where $R ∈ 𝔰𝔬(3)$ is

````math
\exp v = (U(θ) b, \exp Ω) = (t, R),
````

where $R = \exp Ω$ is the group exponential on $\mathrm{SO}(3)$, and $U(θ)$ is

````math
U(θ) = I_3 + \frac{1 - \cos θ}{θ^2} Ω + \frac{θ - \sin θ}{θ^3} Ω^2,
````

where $θ$ is the same as above.
"""
group_exp(::SpecialEuclidean, ::Any)

function group_exp!(G::SpecialEuclidean, y, v)
    vmat = affine_matrix(G, Identity(G), v)
    expv = exp(vmat)
    map(copyto!, submanifold_components(G, y), submanifold_components(G, expv))
    _padpoint!(G, y)
    return y
end
function group_exp!(G::SpecialEuclidean{2}, y, v)
    SO2 = submanifold(G, 2)
    b, Ω = submanifold_components(G, v)
    t, R = submanifold_components(G, y)
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
        _padpoint!(G, y)
    end
    return y
end
function group_exp!(G::SpecialEuclidean{3}, y, v)
    SO3 = submanifold(G, 2)
    b, Ω = submanifold_components(G, v)
    t, R = submanifold_components(G, y)
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
    @inbounds _padpoint!(G, y)
    return y
end

function group_log!(G::SpecialEuclidean, v, y)
    ymat = affine_matrix(G, y)
    logy = real(log(ymat))
    map(copyto!, submanifold_components(G, v), submanifold_components(G, logy))
    _padvector!(G, v)
    return v
end
function group_log!(G::SpecialEuclidean{2}, v, y)
    SO2 = submanifold(G, 2)
    b, Ω = submanifold_components(G, v)
    t, R = submanifold_components(G, y)
    @assert size(b) == (2,)

    group_log!(SO2, Ω, R)
    @inbounds θ = Ω[2]
    β = θ / 2
    α = θ ≈ 0 ? 1 - β^2 / 3 : β * cot(β)

    @inbounds begin
        b[1] = α * t[1] + β * t[2]
        b[2] = α * t[2] - β * t[1]
        _padvector!(G, v)
    end
    return v
end
function group_log!(G::SpecialEuclidean{3}, v, y)
    b, Ω = submanifold_components(G, v)
    t, R = submanifold_components(G, y)
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
    @inbounds _padvector!(G, v)
    return v
end
