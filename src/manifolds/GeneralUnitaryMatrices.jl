@doc """
    AbstractMatrixType

A plain type to distinguish different types of matrices, for example [`DeterminantOneMatrices`](@ref)
and [`UnitMatrices`](@ref)
"""
abstract type AbstractMatrixType end

@doc """
    SoecialMatrices <: AbstractMatrixType

A type to indicate that we require special (orthogonal / unitary) matrices, i.e. of determinant 1.
"""
struct DeterminantOneMatrices <: AbstractMatrixType end

@doc """
    AbsoluteDeterminantOneMatrices <: AbstractMatrixType

A type to indicate that we require (orthogonal / unitary) matrices with normed determinant,
i.e. that the absolute value of the determinant is one of determinant 1.
"""
struct AbsoluteDeterminantOneMatrices <: AbstractMatrixType end

@doc raw"""
    GeneralUnitaryMatrices{n,𝔽,S<:AbstractMatrixType} <: AbstractDecoratorManifold

A common parametric type for matrices with a unitary property of size ``n×n`` over the field ``\mathbb F``
which additionally have the `AbstractMatrixType`, e.g. are `DeterminantOneMatrices`.
"""
struct GeneralUnitaryMatrices{n,𝔽,S<:AbstractMatrixType} <: AbstractDecoratorManifold{𝔽} end

function active_traits(f, ::GeneralUnitaryMatrices, args...)
    return merge_traits(IsIsometricEmbeddedManifold(), IsDefaultMetric(EuclideanMetric()))
end

function allocation_promotion_function(::GeneralUnitaryMatrices{n,ℂ}, f, ::Tuple) where {n}
    return complex
end

@doc raw"""
    check_point(M::UnitaryMatrices, p; kwargs...)
    check_point(M::OrthogonalMatrices, p; kwargs...)
    check_point(M::GeneralUnitaryMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid point on the [`UnitaryMatrices`](@ref) or [`OrthogonalMatrices`] `M`,
i.e. that ``p`` has an determinante of absolute value one

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_point(M::GeneralUnitaryMatrices{n,𝔽}, p; kwargs...) where {n,𝔽}
    if !isapprox(abs(det(p)), 1; kwargs...)
        return DomainError(
            det(p),
            "The absolute value of the determinant of $p has to be 1 but it is $(abs(det(p)))",
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
    return nothing
end
@doc raw"""
    check_vector(M::UnitaryMatrices{n}, p, X; kwargs... )
    check_vector(M::OrthogonalMatrices{n}, p, X; kwargs... )
    check_vector(M::Rotations{n}, p, X; kwargs... )
    check_vector(M::GeneralUnitaryMatrices{n,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to `p` on the [`UnitaryMatrices`](@ref)
space `M`, i.e. after [`check_point`](@ref)`(M,p)`, `X` has to be skew symmetric (hermitian)
dimension and orthogonal to `p`.

The tolerance for the last test can be set using the `kwargs...`.
"""
function check_vector(M::GeneralUnitaryMatrices{n,𝔽}, p, X; kwargs...) where {n,𝔽}
    return check_point(SkewHermitianMatrices(n, 𝔽), X; kwargs...)
end

function default_estimation_method(
    ::GeneralUnitaryMatrices{n,ℝ},
    ::typeof(mean),
) where {n,𝔽}
    return GeodesicInterpolationWithinRadius(π / 2 / √2)
end

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
    get_coordinates(M::Rotations, p, X)

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
[`get_coordinates`](@ref get_coordinates(::GeneralUnitaryMatrices{n,ℝ}, ::Any...)) for the conventions used.
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
        for i in 4:N
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
_injectivity_radius(::GeneralUnitaryMatrices{n,ℝ}, ::ExponentialRetraction) where {n} = π * sqrt(2.0)
_injectivity_radius(::GeneralUnitaryMatrices{n,ℝ}, ::PolarRetraction) where {n} = π / sqrt(2.0)

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

function project_no_rep_change!(::GeneralUnitaryMatrices{n,𝔽}, Y, p, X) where {n,𝔽}
    project!(SkewSymmetricMatrices(n, 𝔽), Y, X)
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
