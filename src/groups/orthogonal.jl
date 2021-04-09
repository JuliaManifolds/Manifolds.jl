@doc raw"""
    Orthogonal(n)

The group of (real) orthogonal matrices ``\mathrm{O}(n)``.

This constructor is equivalent to calling [`Unitary(n,ℝ)`](@ref).
"""
const Orthogonal{n} = Unitary{n,ℝ}

Orthogonal(n) = Orthogonal{n}()

@doc raw"""
    get_coordinates(G::Orthogonal, p, X)

Extract the unique tangent vector components $X^i$ at point `p` on [`Orthogonal`](@ref)
$\mathrm{O}(n)$ from the matrix representation `X` of the tangent vector.

The basis on the Lie algebra $𝔬(n)$ is chosen such that
for $\mathrm{O}(2)$, $X^1 = θ = X_{21}$ is the angle of rotation, and
for $\mathrm{O}(3)$, $(X^1, X^2, X^3) = (X_{32}, X_{13}, X_{21}) = θ u$ is the
angular velocity and axis-angle representation, where $u$ is the unit vector
along the axis of rotation.

For $\mathrm{O}(n)$ where $n ≥ 4$, the additional elements of $X^i$ are
$X^{j (j - 3)/2 + k + 1} = X_{jk}$, for $j ∈ [4,n], k ∈ [1,j)$.
"""
get_coordinates(::Orthogonal, ::Any...)
get_coordinates(::Orthogonal{2}, p, X, ::DefaultOrthogonalBasis) = [X[2]]

function get_coordinates!(::Orthogonal{2}, Xⁱ, p, X, ::DefaultOrthogonalBasis)
    Xⁱ[1] = X[2]
    return Xⁱ
end
function get_coordinates!(G::Orthogonal{n}, Xⁱ, p, X, B::DefaultOrthogonalBasis) where {n}
    @inbounds begin
        Xⁱ[1] = X[3, 2]
        Xⁱ[2] = X[1, 3]
        Xⁱ[3] = X[2, 1]

        k = 4
        for i in 4:n, j in 1:(i - 1)
            Xⁱ[k] = X[i, j]
            k += 1
        end
    end
    return Xⁱ
end
function get_coordinates!(G::Orthogonal, Xⁱ, p, X, ::DefaultOrthonormalBasis)
    get_coordinates!(G, Xⁱ, p, X, DefaultOrthogonalBasis())
    T = eltype(Xⁱ)
    Xⁱ .*= sqrt(T(2))
    return Xⁱ
end

@doc raw"""
    get_vector(G::Orthogonal, p, Xⁱ, B:: DefaultOrthogonalBasis)

Convert the unique tangent vector components `Xⁱ` at point `p` on [`Orthogonal`](@ref)
group $\mathrm{O}(n)$ to the matrix representation $X$ of the tangent vector. See
[`get_coordinates`](@ref get_coordinates(::Orthogonal, ::Any...)) for the conventions used.
"""
get_vector(::Orthogonal, ::Any...)

function get_vector!(G::Orthogonal{2}, X, p, Xⁱ, B::DefaultOrthogonalBasis)
    return get_vector!(G, X, p, Xⁱ[1], B)
end
function get_vector!(::Orthogonal{2}, X, p, Xⁱ::Real, ::DefaultOrthogonalBasis)
    @assert length(X) == 4
    @inbounds begin
        X[1] = 0
        X[2] = Xⁱ
        X[3] = -Xⁱ
        X[4] = 0
    end
    return X
end
function get_vector!(G::Orthogonal{n}, X, p, Xⁱ, ::DefaultOrthogonalBasis) where {n}
    @assert size(X) == (n, n)
    @assert length(Xⁱ) == manifold_dimension(G)
    @inbounds begin
        X[1, 1] = 0
        X[1, 2] = -Xⁱ[3]
        X[1, 3] = Xⁱ[2]
        X[2, 1] = Xⁱ[3]
        X[2, 2] = 0
        X[2, 3] = -Xⁱ[1]
        X[3, 1] = -Xⁱ[2]
        X[3, 2] = Xⁱ[1]
        X[3, 3] = 0
        k = 4
        for i in 4:n
            for j in 1:(i - 1)
                X[i, j] = Xⁱ[k]
                X[j, i] = -Xⁱ[k]
                k += 1
            end
            X[i, i] = 0
        end
    end
    return X
end
function get_vector!(G::Orthogonal, X, p, Xⁱ, B::DefaultOrthonormalBasis)
    get_vector!(G, X, p, Xⁱ, DefaultOrthogonalBasis())
    T = eltype(X)
    X .*= inv(sqrt(T(2)))
    return X
end

@doc raw"""
    group_exp(G::Orthogonal{2}, X)

Compute the group exponential map on the [`Orthogonal(2)`] group.

Given ``X = \begin{pmatrix} 0 & -θ \\ θ & 0 \end{pmatrix}``, the group exponential is

````math
\exp_e \colon X ↦ \begin{pmatrix} \cos θ & -\sin θ \\ \sin θ & \cos θ \end{pmatrix}.
````
"""
group_exp(::Orthogonal{2}, X)

@doc raw"""
    group_exp(G::Orthogonal{4}, X)

Compute the group exponential map on the [`Orthogonal(4)`] group.

The algorithm used is a more numerically stable form of those proposed in
[^Gallier2002] and [^Andrica2013].

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
group_exp(::Orthogonal{4}, X)

function group_exp!(::Orthogonal{2}, q, X)
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
function group_exp!(::Orthogonal{3}, q, X)
    θ = norm(X) / sqrt(2)
    if θ ≈ 0
        a = 1 - θ^2 / 6
        b = θ / 2
    else
        a = sin(θ) / θ
        b = (1 - cos(θ)) / θ^2
    end
    q .= a .* X .+ b .* X^2
    for i in 1:3
        q[i, i] += 1
    end
    return q
end
function group_exp!(::Orthogonal{4}, q, X)
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
        a₀ = one(T)
        a₁ = one(T)
        a₂ = T(1 / 2)
        a₃ = T(1 / 6)
    else  # Case α ⪆ β ≥ 0, α ≠ 0
        sincα = sinα / α
        r = β / α
        c = 1 / (1 + r)
        d = α * (α - β) / 2
        if α < 1e-2
            e = @evalpoly(α², T(1 / 3), T(-1 / 30), T(1 / 840), T(-1 / 45360))
        else
            e = (sincα - cosα) / α²
        end
        a₀ = (α * sinα + (1 + r - d) * cosα) * c
        a₁ = ((3 - d) * sincα - (2 - r) * cosα) * c
        a₂ = (sincα - (1 - r) / 2 * cosα) * c
        a₃ = (e + (1 - r) * (e - sincα / 2)) * c
    end

    X² = X * X
    X³ = X² * X
    q .= a₁ .* X .+ a₂ .* X² .+ a₃ .* X³
    for i in 1:4
        q[i, i] += a₀
    end
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
        @inbounds begin
            E[2, 1] = -α
            E[1, 2] = α
        end
        copyto!(X, P * E * transpose(P))
    else
        log_safe!(X, q)
    end
    return project!(G, X, Identity(G, q), X)
end

@doc raw"""
    injectivity_radius(G::Orthogonal)
    injectivity_radius(G::Orthogonal, p)

Return the injectivity radius on the [`Orthogonal`](@ref) group `G`, which is globally
``π \sqrt{2}``
"""
function injectivity_radius(::Orthogonal, p)
    T = float(real(eltype(p)))
    return T(sqrt(2)) * π
end
function injectivity_radius(::Orthogonal, p, ::ExponentialRetraction)
    T = float(real(eltype(p)))
    return T(sqrt(2)) * π
end

"""
    mean(
        G::Orthogonal,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/2/√2);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(G::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::Orthogonal, ::Any)

function Statistics.mean!(G::Orthogonal, q, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(G, q, x, w, GeodesicInterpolationWithinRadius(π / 2 / √2); kwargs...)
end

Base.show(io::IO, ::Orthogonal{n}) where {n} = print(io, "Orthogonal($n)")

vector_transport_to(::Orthogonal, p, X, q, ::ParallelTransport) = X

vector_transport_to!(::Orthogonal, Y, p, X, q, ::ParallelTransport) = copyto!(Y, X)
