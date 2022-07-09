@doc raw"""
    Orthogonal{n} <: GroupManifold{ℝ,OrthogonalMatrices{n},MultiplicationOperation}

Orthogonal group $\mathrm{O}(n)$ represented by [`OrthogonalMatrices`](@ref).

# Constructor

    Orthogonal(n)
"""
const Orthogonal{n} = GeneralUnitaryMultiplicationGroup{n,ℝ,AbsoluteDeterminantOneMatrices}

Orthogonal(n) = Orthogonal{n}(OrthogonalMatrices(n))

@doc raw"""
     exp_lie(G::Orthogonal{2}, X)

Compute the Lie group exponential map on the [`Orthogonal(2)`] group.
Given ``X = \begin{pmatrix} 0 & -θ \\ θ & 0 \end{pmatrix}``, the group exponential is

```math
\exp_e \colon X ↦ \begin{pmatrix} \cos θ & -\sin θ \\ \sin θ & \cos θ \end{pmatrix}.
```
"""
exp_lie(::Orthogonal{2}, X)

@doc raw"""
     exp_lie(G::Orthogonal{4}, X)

Compute the group exponential map on the [`Orthogonal`](@ref)`(4)` group.
The algorithm used is a more numerically stable form of those proposed in [^Gallier2002], [^Andrica2013].

[^Gallier2002]:
    > Gallier J.; Xu D.; Computing exponentials of skew-symmetric matrices
    > and logarithms of orthogonal matrices.
    > International Journal of Robotics and Automation (2002), 17(4), pp. 1-11.
    > [pdf](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.35.3205).
[^Andrica2013]:
    > Andrica D.; Rohan R.-A.; Computing the Rodrigues coefficients of the
    > exponential map of the Lie groups of matrices.
    > Balkan Journal of Geometry and Its Applications (2013), 18(2), pp. 1-2.
    > [pdf](https://www.emis.de/journals/BJGA/v18n2/B18-2-an.pdf).
"""
exp_lie(::Orthogonal{4}, X)

function exp_lie!(::Orthogonal{2}, q, X)
    @assert size(X) == (2, 2)
    @inbounds θ = (X[2, 1] - X[1, 2]) / 2
    sinθ, cosθ = sincos(θ)
    @inbounds begin
        q[1, 1] = cosθ
        q[2, 1] = sinθ
        q[1, 2] = -sinθ
        q[2, 2] = cosθ
    end
    return q
end
function exp_lie!(::Orthogonal{3}, q, X)
    θ = norm(X) / sqrt(2)
    if θ ≈ 0
        a = 1 - θ^2 / 6
        b = θ / 2
    else
        a = sin(θ) / θ
        b = (1 - cos(θ)) / θ^2
    end
    copyto!(q, I)
    q .+= a .* X
    mul!(q, X, X, b, true)
    return q
end
function exp_lie!(::Orthogonal{4}, q, X)
    T = eltype(X)
    α, β = angles_4d_skew_sym_matrix(X)
    sinα, cosα = sincos(α)
    sinβ, cosβ = sincos(β)
    α² = α^2
    β² = β^2
    Δ = β² - α²
    if !isapprox(Δ, 0; atol=1e-6)  # Case α > β ≥ 0
        sincα = sinα / α
        sincβ = β == 0 ? one(T) : sinβ / β
        a₀ = (β² * cosα - α² * cosβ) / Δ
        a₁ = (β² * sincα - α² * sincβ) / Δ
        a₂ = (cosα - cosβ) / Δ
        a₃ = (sincα - sincβ) / Δ
    elseif α == 0 # Case α = β = 0
        a₀ = a₁ = one(T)
        a₂ = inv(T(2))
        a₃ = inv(T(6))
    else  # Case α ⪆ β ≥ 0, α ≠ 0
        sincα = sinα / α
        r = β / α
        c = 1 / (1 + r)
        d = α * (α - β) / 2
        if α < 1e-2
            e = evalpoly(α², (inv(T(3)), inv(T(-30)), inv(T(840)), inv(T(-45360))))
        else
            e = (sincα - cosα) / α²
        end
        a₀ = (α * sinα + (1 + r - d) * cosα) * c
        a₁ = ((3 - d) * sincα - (2 - r) * cosα) * c
        a₂ = (sincα - (1 - r) / 2 * cosα) * c
        a₃ = (e + (1 - r) * (e - sincα / 2)) * c
    end

    X² = X * X
    copyto!(q, a₀ * I)
    q .+= a₂ .* X²
    mul!(q, X², X, a₃, true)
    return q
end

function group_log!(G::Orthogonal, X::AbstractMatrix, q::AbstractMatrix)
    log_safe!(X, q)
    return project!(G, X, Identity(G, q), X)
end
function group_log!(G::Orthogonal{2}, X::AbstractMatrix, q::AbstractMatrix)
    @assert size(q) == (2, 2)
    @inbounds θ = atan(q[2, 1], q[1, 1])
    return get_vector!(G, X, Identity(G, q), θ, DefaultOrthogonalBasis())
end
function group_log!(G::Orthogonal{3}, X::AbstractMatrix, q::AbstractMatrix)
    e = Identity(G, q)
    cosθ = (tr(q) - 1) / 2
    if cosθ ≈ -1
        eig = eigen_safe(q)
        ival = findfirst(λ -> isapprox(λ, 1), eig.values)
        inds = SVector{3}(1:3)
        ax = eig.vectors[inds, ival]
        return get_vector!(G, X, e, π * ax, DefaultOrthogonalBasis())
    end
    X .= q ./ usinc_from_cos(cosθ)
    return project!(G, X, e, X)
end
function group_log!(G::Orthogonal{4}, X::AbstractMatrix, q::AbstractMatrix)
    cosα, cosβ = cos_angles_4d_rotation_matrix(q)
    α = acos(clamp(cosα, -1, 1))
    β = acos(clamp(cosβ, -1, 1))
    if α ≈ π && β ≈ 0
        A² = Symmetric((q - I) ./ 2)
        P = eigvecs(A²)
        E = similar(q)
        fill!(E, 0)
        α = acos(clamp(cosα, -1, 1))
        E[2, 1] = -α
        E[1, 2] = α
        copyto!(X, P * E * transpose(P))
    else
        log_safe!(X, q)
    end
    return project!(G, X, Identity(G, q), X)
end

show(io::IO, ::Orthogonal{n}) where {n} = print(io, "Orthogonal($(n))")
