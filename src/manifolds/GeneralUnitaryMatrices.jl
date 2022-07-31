@doc """
    AbstractMatrixType

A plain type to distinguish different types of matrices, for example [`DeterminantOneMatrices`](@ref)
and [`AbsoluteDeterminantOneMatrices`](@ref)
"""
abstract type AbstractMatrixType end

@doc """
    DeterminantOneMatrices <: AbstractMatrixType

A type to indicate that we require special (orthogonal / unitary) matrices, i.e. of determinant 1.
"""
struct DeterminantOneMatrices <: AbstractMatrixType end

@doc """
    AbsoluteDeterminantOneMatrices <: AbstractMatrixType

A type to indicate that we require (orthogonal / unitary) matrices with normed determinant,
i.e. that the absolute value of the determinant is 1.
"""
struct AbsoluteDeterminantOneMatrices <: AbstractMatrixType end

@doc raw"""
    GeneralUnitaryMatrices{n,𝔽,S<:AbstractMatrixType} <: AbstractDecoratorManifold

A common parametric type for matrices with a unitary property of size ``n×n`` over the field ``\mathbb F``
which additionally have the `AbstractMatrixType`, e.g. are `DeterminantOneMatrices`.
"""
struct GeneralUnitaryMatrices{n,𝔽,S<:AbstractMatrixType} <: AbstractDecoratorManifold{𝔽} end

function active_traits(f, ::GeneralUnitaryMatrices, args...)
    return merge_traits(IsEmbeddedManifold(), IsDefaultMetric(EuclideanMetric()))
end

@doc raw"""
    check_point(M::UnitaryMatrices, p; kwargs...)
    check_point(M::OrthogonalMatrices, p; kwargs...)
    check_point(M::GeneralUnitaryMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid point on the [`UnitaryMatrices`](@ref) or [`OrthogonalMatrices`] `M`,
i.e. that ``p`` has an determinante of absolute value one

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(
    M::GeneralUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices},
    p;
    kwargs...,
) where {n,𝔽}
    if !isapprox(abs(det(p)), 1; kwargs...)
        return DomainError(
            abs(det(p)),
            "The absolute value of the determinant of $p has to be 1 but it is $(abs(det(p)))",
        )
    end
    if !isapprox(p' * p, one(p); kwargs...)
        return DomainError(
            norm(p' * p - one(p)),
            "$p must be orthogonal but it's not at kwargs $kwargs",
        )
    end
    return nothing
end

@doc raw"""
    check_point(M::Rotations, p; kwargs...)

Check whether `p` is a valid point on the [`UnitaryMatrices`](@ref) `M`,
i.e. that ``p`` has an determinante of absolute value one, i.e. that ``p^{\mathrm{H}}p``

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(
    M::GeneralUnitaryMatrices{n,𝔽,DeterminantOneMatrices},
    p;
    kwargs...,
) where {n,𝔽}
    if !isapprox(det(p), 1; kwargs...)
        return DomainError(det(p), "The determinant of $p has to be +1 but it is $(det(p))")
    end
    if !isapprox(p' * p, one(p); kwargs...)
        return DomainError(
            norm(p' * p - one(p)),
            "$p must be orthogonal but it's not at kwargs $kwargs",
        )
    end
    return nothing
end

function check_size(::GeneralUnitaryMatrices{n}, p) where {n}
    m = size(p)
    if length(m) != 2
        return DomainError(
            size(p),
            "The point $p is not a matrix (expected a length of size to be 2, got $(length(size(p))))",
        )
    end
    if m != (n, n)
        return DomainError(
            size(p),
            "The point $p is not a matrix of size $((n,n)), but $(size(p)).",
        )
    end
    return nothing
end
function check_size(::GeneralUnitaryMatrices{n}, p, X) where {n}
    m = size(X)
    if length(size(X)) != 2
        return DomainError(
            size(X),
            "The tangent vector $X is not a matrix (expected a length of size to be 2, got $(length(size(X))))",
        )
    end
    if m != (n, n)
        return DomainError(
            size(X),
            "The tangent vector $X is not a matrix of size $((n,n)), but $(size(X)).",
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M::UnitaryMatrices{n}, p, X; kwargs... )
    check_vector(M::OrthogonalMatrices{n}, p, X; kwargs... )
    check_vector(M::Rotations{n}, p, X; kwargs... )
    check_vector(M::GeneralUnitaryMatrices{n,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`UnitaryMatrices`](@ref)
space `M`, i.e. after [`check_point`](@ref)`(M,p)`, `X` has to be skew symmetric (Hermitian)
and orthogonal to `p`.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(M::GeneralUnitaryMatrices{n,𝔽}, p, X; kwargs...) where {n,𝔽}
    return check_point(SkewHermitianMatrices(n, 𝔽), X; kwargs...)
end

@doc raw"""
    cos_angles_4d_rotation_matrix(R)

4D rotations can be described by two orthogonal planes that are unchanged by
the action of the rotation (vectors within a plane rotate only within the
plane). The cosines of the two angles $α,β$ of rotation about these planes may be
obtained from the distinct real parts of the eigenvalues of the rotation
matrix. This function computes these more efficiently by solving the system

```math
\begin{aligned}
\cos α + \cos β &= \frac{1}{2} \operatorname{tr}(R)\\
\cos α \cos β &= \frac{1}{8} \operatorname{tr}(R)^2
                 - \frac{1}{16} \operatorname{tr}((R - R^T)^2) - 1.
\end{aligned}
```

By convention, the returned values are sorted in decreasing order.
See also [`angles_4d_skew_sym_matrix`](@ref).
"""
function cos_angles_4d_rotation_matrix(R)
    a = tr(R)
    b = sqrt(clamp(2 * dot(transpose(R), R) - a^2 + 8, 0, Inf))
    return ((a + b) / 4, (a - b) / 4)
end

function default_estimation_method(
    ::GeneralUnitaryMatrices{n,ℝ},
    ::typeof(mean),
) where {n,𝔽}
    return GeodesicInterpolationWithinRadius(π / 2 / √2)
end

embed(::GeneralUnitaryMatrices, p) = p

@doc raw"""
    embed(M::GeneralUnitaryMatrices{n,𝔽}, p, X)

Embed the tangent vector `X` at point `p` in `M` from
its Lie algebra representation (set of skew matrices) into the
Riemannian submanifold representation

The formula reads
```math
X_{\text{embedded}} = p * X
```
"""
embed(::GeneralUnitaryMatrices, p, X)

function embed!(::GeneralUnitaryMatrices, Y, p, X)
    return mul!(Y, p, X)
end

@doc raw"""
    exp(M::Rotations, p, X)
    exp(M::OrthogonalMatrices, p, X)
    exp(M::UnitaryMatrices, p, X)

Compute the exponential map, that is, since ``X`` is represented in the Lie algebra,

```
exp_p(X) = p\mathrm{e}^X
```

For different sizes, like ``n=2,3,4`` there is specialised implementations

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
exp(::GeneralUnitaryMatrices, p, X)

function exp!(M::GeneralUnitaryMatrices, q, p, X)
    return copyto!(M, q, p * exp(X))
end

function exp(M::GeneralUnitaryMatrices{2,ℝ}, p::SMatrix, X::SMatrix)
    θ = get_coordinates(M, p, X, DefaultOrthogonalBasis())[1]
    sinθ, cosθ = sincos(θ)
    return p * SA[cosθ -sinθ; sinθ cosθ]
end
function exp!(M::GeneralUnitaryMatrices{2,ℝ}, q, p, X)
    @assert size(q) == (2, 2)
    θ = get_coordinates(M, p, X, DefaultOrthogonalBasis())[1]
    sinθ, cosθ = sincos(θ)
    return copyto!(q, p * SA[cosθ -sinθ; sinθ cosθ])
end
function exp!(M::GeneralUnitaryMatrices{3,ℝ}, q, p, X)
    θ = norm(M, p, X) / sqrt(2)
    if θ ≈ 0
        a = 1 - θ^2 / 6
        b = θ / 2
    else
        a = sin(θ) / θ
        b = (1 - cos(θ)) / θ^2
    end
    pinvq = I + a .* X .+ b .* (X^2)
    return copyto!(q, p * pinvq)
end
function exp!(::GeneralUnitaryMatrices{4,ℝ}, q, p, X)
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
    pinvq = a₀ * I + a₁ .* X .+ a₂ .* X² .+ a₃ .* X³
    return copyto!(q, p * pinvq)
end

@doc raw"""
    get_coordinates(M::Rotations, p, X)
    get_coordinates(M::OrthogonalMatrices, p, X)
    get_coordinates(M::UnitaryMatrices, p, X)

Extract the unique tangent vector components $X^i$ at point `p` on [`Rotations`](@ref)
$\mathrm{SO}(n)$ from the matrix representation `X` of the tangent
vector.

The basis on the Lie algebra $𝔰𝔬(n)$ is chosen such that
for $\mathrm{SO}(2)$, $X^1 = θ = X_{21}$ is the angle of rotation, and
for $\mathrm{SO}(3)$, $(X^1, X^2, X^3) = (X_{32}, X_{13}, X_{21}) = θ u$ is the
angular velocity and axis-angle representation, where $u$ is the unit vector
along the axis of rotation.

For $\mathrm{SO}(n)$ where $n ≥ 4$, the additional elements of $X^i$ are
$X^{j (j - 3)/2 + k + 1} = X_{jk}$, for $j ∈ [4,n], k ∈ [1,j)$.
"""
get_coordinates(::GeneralUnitaryMatrices{n,ℝ}, ::Any...) where {n}
function get_coordinates(
    ::GeneralUnitaryMatrices{2,ℝ},
    p,
    X,
    ::DefaultOrthogonalBasis{ℝ,TangentSpaceType},
)
    return [X[2]]
end
function get_coordinates(
    ::GeneralUnitaryMatrices{2,ℝ},
    p::SMatrix,
    X::SMatrix,
    ::DefaultOrthogonalBasis{ℝ,TangentSpaceType},
)
    return SA[X[2]]
end

function get_coordinates_orthogonal(M::GeneralUnitaryMatrices{n,ℝ}, p, X, N) where {n}
    Y = allocate_result(M, get_coordinates, p, X, DefaultOrthogonalBasis(N))
    return get_coordinates_orthogonal!(M, Y, p, X, N)
end

function get_coordinates_orthogonal!(::GeneralUnitaryMatrices{2,ℝ}, Xⁱ, p, X, ::RealNumbers)
    Xⁱ[1] = X[2]
    return Xⁱ
end
function get_coordinates_orthogonal!(
    ::GeneralUnitaryMatrices{n,ℝ},
    Xⁱ,
    p,
    X,
    ::RealNumbers,
) where {n}
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
function get_coordinates_orthonormal!(
    M::GeneralUnitaryMatrices{n,ℝ},
    Xⁱ,
    p,
    X,
    num::RealNumbers,
) where {n}
    T = Base.promote_eltype(p, X)
    get_coordinates_orthogonal!(M, Xⁱ, p, X, num)
    Xⁱ .*= sqrt(T(2))
    return Xⁱ
end

@doc raw"""
    get_embedding(M::OrthogonalMatrices{n})
    get_embedding(M::Rotations{n})
    get_embedding(M::UnitaryMatrices{n})

Return the embedding, i.e. The ``\mathbb F^{n×n}``, where ``\mathbb F = \mathbb R`` for the
first two and ``\mathbb F = \mathbb C`` for the unitary matrices.
"""
get_embedding(::GeneralUnitaryMatrices{n,𝔽}) where {n,𝔽} = Euclidean(n, n; field=𝔽)

@doc raw"""
    get_vector(M::OrthogonalMatrices, p, Xⁱ, B::DefaultOrthogonalBasis)
    get_vector(M::Rotations, p, Xⁱ, B::DefaultOrthogonalBasis)

Convert the unique tangent vector components `Xⁱ` at point `p` on [`Rotations`](@ref)
or [`OrthogonalMatrices`](@ref)
to the matrix representation $X$ of the tangent vector. See
[`get_coordinates`](@ref get_coordinates(::GeneralUnitaryMatrices{n,ℝ} where {n}, ::Any...)) for the conventions used.
"""
get_vector(::GeneralUnitaryMatrices{n,ℝ}, ::Any...) where {n}

function get_vector_orthogonal(
    M::GeneralUnitaryMatrices{n,ℝ},
    p,
    c,
    N::RealNumbers,
) where {n}
    Y = allocate_result(M, get_vector, p, c)
    return get_vector_orthogonal!(M, Y, p, c, N)
end

function get_vector_orthogonal(::GeneralUnitaryMatrices{2,ℝ}, p::SMatrix, Xⁱ, ::RealNumbers)
    return @SMatrix [0 -Xⁱ[]; Xⁱ[] 0]
end

function get_vector_orthogonal!(M::GeneralUnitaryMatrices{2,ℝ}, X, p, Xⁱ, N::RealNumbers)
    return get_vector_orthogonal!(M, X, p, Xⁱ[1], N)
end
function get_vector_orthogonal!(
    ::GeneralUnitaryMatrices{2,ℝ},
    X,
    p,
    Xⁱ::Real,
    ::RealNumbers,
)
    @assert length(X) == 4
    @inbounds begin
        X[1] = 0
        X[2] = Xⁱ
        X[3] = -Xⁱ
        X[4] = 0
    end
    return X
end
function get_vector_orthogonal!(
    M::GeneralUnitaryMatrices{n,ℝ},
    X,
    p,
    Xⁱ,
    ::RealNumbers,
) where {n}
    @assert size(X) == (n, n)
    @assert length(Xⁱ) == manifold_dimension(M)
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
function get_vector_orthonormal(
    M::GeneralUnitaryMatrices{n,ℝ},
    p,
    Xⁱ,
    N::RealNumbers,
) where {n}
    return get_vector_orthogonal(M, p, Xⁱ, N) ./ sqrt(eltype(Xⁱ)(2))
end

function get_vector_orthonormal!(
    M::GeneralUnitaryMatrices{n,ℝ},
    X,
    p,
    Xⁱ,
    N::RealNumbers,
) where {n}
    T = Base.promote_eltype(p, X)
    get_vector_orthogonal!(M, X, p, Xⁱ, N)
    X ./= sqrt(T(2))
    return X
end

@doc raw"""
    injectivity_radius(G::Orthogonal)
    injectivity_radius(M::Rotations)

Return the injectivity radius on the [`Rotations`](@ref) and the [`OrthogonalMatrices`](@ref) `M`,
which is globally

````math
    \operatorname{inj}_{\mathrm{O}(n)}(p) = \operatorname{inj}_{\mathrm{SO}(n)}(p) = π\sqrt{2}.
````
"""
injectivity_radius(::GeneralUnitaryMatrices{n,ℝ}) where {n} = π * sqrt(2.0)
function _injectivity_radius(
    ::GeneralUnitaryMatrices{n,ℝ},
    ::ExponentialRetraction,
) where {n}
    return π * sqrt(2.0)
end
function _injectivity_radius(::GeneralUnitaryMatrices{n,ℝ}, ::PolarRetraction) where {n}
    return π / sqrt(2.0)
end
@doc raw"""
    injectivity_radius(G::UnitaryMatrices)

Return the injectivity radius on the [`UnitaryMatrices`](@ref), which is ``π``.
"""
injectivity_radius(::GeneralUnitaryMatrices{n,ℂ}) where {n} = π

inner(::GeneralUnitaryMatrices, p, X, Y) = dot(X, Y)

@doc raw"""
    log(M::Rotations, p, X)
    log(M::OrthogonalMatrices, p, X)
    log(M::UnitaryMatrices, p, X)

Compute the logarithmic map, that is, since the resulting ``X`` is represented in the Lie algebra,

```
log_p q = \log(p^{\mathrm{H}q)
```
which is projected onto the skew symmetric matrices for numerical stability.
"""
log(::GeneralUnitaryMatrices, p, q)

@doc raw"""
    log(M::Rotations, p, q)

Compute the logarithmic map on the [`Rotations`](@ref) manifold
`M` which is given by

```math
\log_p q = \operatorname{log}(p^{\mathrm{T}}q)
```

where $\operatorname{Log}$ denotes the matrix logarithm. For numerical stability,
the result is projected onto the set of skew symmetric matrices.

For antipodal rotations the function returns deterministically one of the tangent vectors
that point at `q`.
"""
log(::GeneralUnitaryMatrices{n,ℝ}, ::Any...) where {n}
function ManifoldsBase.log(M::GeneralUnitaryMatrices{2,ℝ}, p, q)
    U = transpose(p) * q
    @assert size(U) == (2, 2)
    @inbounds θ = atan(U[2], U[1])
    return get_vector(M, p, θ, DefaultOrthogonalBasis())
end
function log!(::GeneralUnitaryMatrices{n,ℝ}, X, p, q) where {n}
    U = transpose(p) * q
    X .= real(log_safe(U))
    return project!(SkewSymmetricMatrices(n), X, p, X)
end
function log!(M::GeneralUnitaryMatrices{2,ℝ}, X, p, q)
    U = transpose(p) * q
    @assert size(U) == (2, 2)
    @inbounds θ = atan(U[2], U[1])
    return get_vector!(M, X, p, θ, DefaultOrthogonalBasis())
end
function log!(M::GeneralUnitaryMatrices{3,ℝ}, X, p, q)
    U = transpose(p) * q
    cosθ = (tr(U) - 1) / 2
    if cosθ ≈ -1
        eig = eigen_safe(U)
        ival = findfirst(λ -> isapprox(λ, 1), eig.values)
        inds = SVector{3}(1:3)
        ax = eig.vectors[inds, ival]
        return get_vector!(M, X, p, π * ax, DefaultOrthogonalBasis())
    end
    X .= U ./ usinc_from_cos(cosθ)
    return project!(SkewSymmetricMatrices(3), X, p, X)
end
function log!(::GeneralUnitaryMatrices{4,ℝ}, X, p, q)
    U = transpose(p) * q
    cosα, cosβ = Manifolds.cos_angles_4d_rotation_matrix(U)
    α = acos(clamp(cosα, -1, 1))
    β = acos(clamp(cosβ, -1, 1))
    if α ≈ 0 && β ≈ π
        A² = Symmetric((U - I) ./ 2)
        P = eigvecs(A²)
        E = similar(U)
        fill!(E, 0)
        @inbounds begin
            E[2, 1] = -β
            E[1, 2] = β
        end
        copyto!(X, P * E * transpose(P))
    else
        det(U) < 0 && throw(
            DomainError(
                "The logarithm is not defined for $p and $q with a negative determinant of p'q) ($(det(U)) < 0).",
            ),
        )
        copyto!(X, real(Manifolds.log_safe(U)))
    end
    return project!(SkewSymmetricMatrices(4), X, p, X)
end

function log!(::GeneralUnitaryMatrices{n,𝔽}, X, p, q) where {n,𝔽}
    log_safe!(X, adjoint(p) * q)
    project!(SkewHermitianMatrices(n, 𝔽), q, q)
    return q
end

norm(::GeneralUnitaryMatrices, p, X) = norm(X)

@doc raw"""
    manifold_dimension(M::Rotations)
    manifold_dimension(M::OrthogonalMatrices)

Return the dimension of the manifold orthogonal matrices and of the manifold of rotations
```math
\dim_{\mathrm{O}(n)} = \dim_{\mathrm{SO}(n)} = \frac{n(n-1)}{2}.
```
"""
manifold_dimension(::GeneralUnitaryMatrices{n,ℝ}) where {n} = div(n * (n - 1), 2)

"""
    mean(
        M::Rotations,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/2/√2);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::GeneralUnitaryMatrices{n,ℝ}, ::Any) where {n}

@doc raw"""
     project(G::UnitaryMatrices{n}, p)
     project(G::OrthogonalMatrices{n}, p)

Project the point ``p ∈ 𝔽^{n × n}`` to the nearest point in
``\mathrm{U}(n,𝔽)=``[`Unitary(n,𝔽)`](@ref) under the Frobenius norm.
If ``p = U S V^\mathrm{H}`` is the singular value decomposition of ``p``, then the projection
is

````math
  \operatorname{proj}_{\mathrm{U}(n,𝔽)} \colon p ↦ U V^\mathrm{H}.
````
"""
project(::GeneralUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices}, p) where {n,𝔽}

function project!(
    ::GeneralUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices},
    q,
    p,
) where {n,𝔽}
    F = svd(p)
    mul!(q, F.U, F.Vt)
    return q
end

@doc raw"""
     project(M::OrthogonalMatrices{n}, p, X)
     project(M::Rotations{n}, p, X)
     project(M::UnitaryMatrices{n}, p, X)

Orthogonally project the tangent vector ``X ∈ 𝔽^{n × n}``, ``\mathbb F ∈ \{\mathbb R, \mathbb C\}``
to the tangent space of `M` at `p`,
and change the representer to use the corresponding Lie algebra, i.e. we compute

```math
    \operatorname{proj}_p(X) = \frac{pX-(pX)^{\mathrm{T}}}{2},
```
"""
project(::GeneralUnitaryMatrices, p, X)

function project!(::GeneralUnitaryMatrices{n,𝔽}, Y, p, X) where {n,𝔽}
    project!(SkewHermitianMatrices(n, 𝔽), Y, p \ X)
    return Y
end

@doc raw"""
    retract(M::Rotations, p, X, ::PolarRetraction)
    retract(M::OrthogonalMatrices, p, X, ::PolarRetraction)

Compute the SVD-based retraction on the [`Rotations`](@ref) and [`OrthogonalMatrices`](@ref) `M` from `p` in direction `X`
(as an element of the Lie group) and is a second-order approximation of the exponential map.
Let

````math
USV = p + pX
````

be the singular value decomposition, then the formula reads

````math
\operatorname{retr}_p X = UV^\mathrm{T}.
````
"""
retract(::GeneralUnitaryMatrices{n,𝔽}, ::Any, ::Any, ::PolarRetraction) where {n,𝔽}

@doc raw"""
    retract(M::Rotations, p, X, ::QRRetraction)
    retract(M::OrthogonalMatrices, p. X, ::QRRetraction)

Compute the QR-based retraction on the [`Rotations`](@ref) and [`OrthogonalMatrices`](@ref) `M` from `p` in direction `X`
(as an element of the Lie group), which is a first-order approximation of the exponential map.

This is also the default retraction on these manifolds.
"""
retract(::GeneralUnitaryMatrices{n,𝔽}, ::Any, ::Any, ::QRRetraction) where {n,𝔽}

function retract_qr!(::GeneralUnitaryMatrices{n,𝔽}, q::AbstractArray{T}, p, X) where {n,𝔽,T}
    A = p + p * X
    qr_decomp = qr(A)
    d = diag(qr_decomp.R)
    D = Diagonal(sign.(d .+ convert(T, 0.5)))
    return copyto!(q, qr_decomp.Q * D)
end
function retract_polar!(M::GeneralUnitaryMatrices{n,𝔽}, q, p, X) where {n,𝔽}
    A = p + p * X
    return project!(M, q, A; check_det=false)
end

@doc raw"""
    riemann_tensor(::GeneralUnitaryMatrices, p, X, Y, Z)

Compute the value of Riemann tensor on the [`GeneralUnitaryMatrices`](@ref) manifold.
The formula reads[^Rentmeesters2011] ``R(X,Y)Z=\frac{1}{4}[Z, [X, Y]]``.

[^Rentmeesters2011]:
    > Q. Rentmeesters, “A gradient method for geodesic data fitting on some symmetric
    > Riemannian manifolds,” in 2011 50th IEEE Conference on Decision and Control and
    > European Control Conference, Dec. 2011, pp. 7141–7146. doi: 10.1109/CDC.2011.6161280.
"""
riemann_tensor(::GeneralUnitaryMatrices, p, X, Y, Z)

function riemann_tensor!(::GeneralUnitaryMatrices, Xresult, p, X, Y, Z)
    Xtmp = X * Y - Y * X
    Xresult .= 1 // 4 .* (Z * Xtmp .- Xtmp * Z)
    return Xresult
end
