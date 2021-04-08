@doc raw"""
    Rotations{N} <: Manifold{ℝ}

The special orthogonal manifold $\mathrm{SO}(n)$ represented by $n × n$
real-valued orthogonal matrices with determinant $+1$ is the manifold of `Rotations`,
since these matrices represent all rotations of points in $ℝ^n$.

# Constructor

    Rotations(n)

Generate the $\mathrm{SO}(n) \subset ℝ^{n × n}$
"""
struct Rotations{N} <: Manifold{ℝ} end

Rotations(n::Int) = Rotations{n}()

"""
    NormalRotationDistribution(M::Rotations, d::Distribution, x::TResult)

Distribution that returns a random point on the manifold [`Rotations`](@ref)
`M`. Random point is generated using base distribution `d` and the type
of the result is adjusted to `TResult`.

See [`normal_rotation_distribution`](@ref) for details.
"""
struct NormalRotationDistribution{TResult,TM<:Rotations,TD<:Distribution} <:
       MPointDistribution{TM}
    manifold::TM
    distr::TD
end

function NormalRotationDistribution(
    M::Rotations,
    d::Distribution,
    x::TResult,
) where {TResult}
    return NormalRotationDistribution{TResult,typeof(M),typeof(d)}(M, d)
end

@doc raw"""
    angles_4d_skew_sym_matrix(A)

The Lie algebra of [`Rotations(4)`](@ref) in $ℝ^{4 × 4}$, $𝔰𝔬(4)$, consists of $4 × 4$
skew-symmetric matrices. The unique imaginary components of their eigenvalues are the
angles of the two plane rotations. This function computes these more efficiently than
`eigvals`.

By convention, the returned values are sorted in decreasing order
(corresponding to the same ordering of _angles_ as
[`cos_angles_4d_rotation_matrix`](@ref)).
"""
function angles_4d_skew_sym_matrix(A)
    @assert size(A) == (4, 4)
    @inbounds begin
        halfb = (A[1, 2]^2 + A[1, 3]^2 + A[2, 3]^2 + A[1, 4]^2 + A[2, 4]^2 + A[3, 4]^2) / 2
        c = (A[1, 2] * A[3, 4] - A[1, 3] * A[2, 4] + A[1, 4] * A[2, 3])^2
    end
    sqrtdisc = sqrt(halfb^2 - c)
    return sqrt(halfb + sqrtdisc), sqrt(halfb - sqrtdisc)
end

"""
    check_manifold_point(M, p; kwargs...)

Check whether `p` is a valid point on the [`Rotations`](@ref) `M`,
i.e. is an array of size [`manifold_dimension`](@ref)`(M)` and represents a
valid rotation.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::Rotations{N}, p; kwargs...) where {N}
    if size(p) != (N, N)
        return DomainError(
            size(p),
            "The point $(p) does not lie on $M, since its size is not $((N, N)).",
        )
    end
    if !isapprox(det(p), 1; kwargs...)
        return DomainError(det(p), "The determinant of $p has to be +1 but it is $(det(p))")
    end
    if !isapprox(transpose(p) * p, one(p); kwargs...)
        return DomainError(
            norm(transpose(p) * p - one(p)),
            "$p must be orthogonal but it's not at kwargs $kwargs",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to `p` on the [`Rotations`](@ref)
space `M`, i.e. after [`check_manifold_point`](@ref)`(M,p)`, `X` has to be of same
dimension and orthogonal to `p`.
The optional parameter `check_base_point` indicates, whether to call [`check_manifold_point`](@ref)  for `p`.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(
    M::Rotations{N},
    p,
    X;
    check_base_point=true,
    kwargs...,
) where {N}
    if check_base_point
        perr = check_manifold_point(M, p)
        perr === nothing || return perr
    end
    if size(X) != (N, N)
        return DomainError(
            size(X),
            "The array $(X) is not a tangent to a point on $M since its size does not match $((N, N)).",
        )
    end
    if !isapprox(transpose(X) + X, zero(X); kwargs...)
        return DomainError(
            size(X),
            "The array $(X) is not a tangent to a point on $M since it is not skew-symmetric.",
        )
    end
    return nothing
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
\cos α + \cos β &= \frac{1}{8} \operatorname{tr}(R)^2
                 - \frac{1}{16} \operatorname{tr}((R - R^T)^2) - 1.
\end{aligned}
```

By convention, the returned values are sorted in increasing order. See
[`angles_4d_skew_sym_matrix`](@ref).
"""
function cos_angles_4d_rotation_matrix(R)
    trR = tr(R)
    a = trR / 4
    b = sqrt(clamp(tr((R .- transpose(R))^2) / 16 - a^2 + 1, 0, Inf))
    return (a + b, a - b)
end

@doc raw"""
    exp(M::Rotations, p, X)

Compute the exponential map on the [`Rotations`](@ref) from `p` into direction
`X`, i.e.

````math
\exp_p X = p \operatorname{Exp}(X),
````

where $\operatorname{Exp}(X)$  denotes the matrix exponential of $X$.

    exp(M::Rotations{4}, p, X)

Compute the exponential map of tangent vector `X` at point `p` from $\mathrm{SO}(4)$
manifold `M`.

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
exp(::Rotations, ::Any...)

exp!(M::Rotations, q, p, X) = copyto!(q, p * exp(X))
function exp!(M::Rotations{2}, q, p, X)
    @assert size(q) == (2, 2)
    θ = get_coordinates(M, p, X, DefaultOrthogonalBasis())[1]
    sinθ, cosθ = sincos(θ)
    @inbounds begin
        q[1] = cosθ
        q[2] = sinθ
        q[3] = -sinθ
        q[4] = cosθ
    end
    return copyto!(q, p * q)
end
function exp!(M::Rotations{3}, q, p, X)
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
function exp!(M::Rotations{4}, q, p, X)
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

flat!(M::Rotations, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

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
get_coordinates(::Rotations, ::Any...)
get_coordinates(::Rotations{2}, p, X, ::DefaultOrthogonalBasis) = [X[2]]

function get_coordinates!(::Rotations{2}, Xⁱ, p, X, ::DefaultOrthogonalBasis)
    Xⁱ[1] = X[2]
    return Xⁱ
end
function get_coordinates!(M::Rotations{N}, Xⁱ, p, X, B::DefaultOrthogonalBasis) where {N}
    @inbounds begin
        Xⁱ[1] = X[3, 2]
        Xⁱ[2] = X[1, 3]
        Xⁱ[3] = X[2, 1]

        k = 4
        for i in 4:N, j in 1:(i - 1)
            Xⁱ[k] = X[i, j]
            k += 1
        end
    end
    return Xⁱ
end
function get_coordinates!(M::Rotations{N}, Xⁱ, p, X, ::DefaultOrthonormalBasis) where {N}
    T = Base.promote_eltype(p, X)
    get_coordinates!(M, Xⁱ, p, X, DefaultOrthogonalBasis())
    Xⁱ .*= sqrt(T(2))
    return Xⁱ
end

@doc raw"""
    get_vector(M::Rotations, p, Xⁱ, B:: DefaultOrthogonalBasis)

Convert the unique tangent vector components `Xⁱ` at point `p` on [`Rotations`](@ref)
group $\mathrm{SO}(n)$ to the matrix representation $X$ of the tangent vector. See
[`get_coordinates`](@ref get_coordinates(::Rotations, ::Any...)) for the conventions used.
"""
get_vector(::Rotations, ::Any...)

function get_vector!(M::Rotations{2}, X, p, Xⁱ, B::DefaultOrthogonalBasis)
    return get_vector!(M, X, p, Xⁱ[1], B)
end
function get_vector!(::Rotations{2}, X, p, Xⁱ::Real, ::DefaultOrthogonalBasis)
    @assert length(X) == 4
    @inbounds begin
        X[1] = 0
        X[2] = Xⁱ
        X[3] = -Xⁱ
        X[4] = 0
    end
    return X
end
function get_vector!(M::Rotations{N}, X, p, Xⁱ, ::DefaultOrthogonalBasis) where {N}
    @assert size(X) == (N, N)
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
function get_vector!(M::Rotations, X, p, Xⁱ, B::DefaultOrthonormalBasis)
    T = Base.promote_eltype(p, X)
    get_vector!(M, X, p, Xⁱ, DefaultOrthogonalBasis())
    X ./= sqrt(T(2))
    return X
end

@doc raw"""
    injectivity_radius(M::Rotations)
    injectivity_radius(M::Rotations, p)

Return the injectivity radius on the [`Rotations`](@ref) `M`, which is globally

````math
    \operatorname{inj}_{\mathrm{SO}(n)}(p) = π\sqrt{2}.
````

    injectivity_radius(M::Rotations, p, ::PolarRetraction)

Return the radius of injectivity for the [`PolarRetraction`](@ref) on the
[`Rotations`](@ref) `M` which is $\frac{π}{\sqrt{2}}$.
"""
injectivity_radius(::Rotations) = π * sqrt(2.0)
injectivity_radius(::Rotations, ::ExponentialRetraction) = π * sqrt(2.0)
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::Rotations,
            rm::AbstractRetractionMethod,
        )
    end,
)
injectivity_radius(::Rotations, ::Any) = π * sqrt(2.0)
injectivity_radius(::Rotations, ::Any, ::ExponentialRetraction) = π * sqrt(2.0)
injectivity_radius(::Rotations, ::PolarRetraction) = π / sqrt(2.0)
injectivity_radius(::Rotations, p, ::PolarRetraction) = π / sqrt(2.0)

@doc raw"""
    inner(M::Rotations, p, X, Y)

Compute the inner product of the two tangent vectors `X`, `Y` from the tangent
plane at `p` on the special orthogonal space `M=`$\mathrm{SO}(n)$ using the
restriction of the metric from the embedding, i.e.

````math
g_p(X, Y) = \operatorname{tr}(X^\mathrm{T} Y),
````

Tangent vectors are represented by matrices.
"""
inner(M::Rotations, p, X, Y) = dot(X, Y)

@doc raw"""
    inverse_retract(M, p, q, ::PolarInverseRetraction)

Compute a vector from the tangent space $T_p\mathrm{SO}(n)$
of the point `p` on the [`Rotations`](@ref) manifold `M`
with which the point `q` can be reached by the
[`PolarRetraction`](@ref) from the point `p` after time 1.

The formula reads
````math
\operatorname{retr}^{-1}_p(q)
= -\frac{1}{2}(p^{\mathrm{T}}qs - (p^{\mathrm{T}}qs)^{\mathrm{T}})
````

where $s$ is the solution to the Sylvester equation

$p^{\mathrm{T}}qs + s(p^{\mathrm{T}}q)^{\mathrm{T}} + 2I_n = 0.$
"""
inverse_retract(::Rotations, ::Any, ::Any, ::PolarInverseRetraction)

@doc raw"""
    inverse_retract(M::Rotations, p, q, ::QRInverseRetraction)

Compute a vector from the tangent space $T_p\mathrm{SO}(n)$ of the point `p` on the
[`Rotations`](@ref) manifold `M` with which the point `q` can be reached by the
[`QRRetraction`](@ref) from the point `q` after time 1.
"""
inverse_retract(::Rotations, ::Any, ::Any, ::QRInverseRetraction)

function inverse_retract!(M::Rotations, X, p, q, method::PolarInverseRetraction)
    A = transpose(p) * q
    Amat = A isa StaticMatrix ? A : convert(Matrix, A)
    H = copyto!(allocate(Amat), 2I)
    try
        B = lyap(A, A, H)
    catch e
        if isa(e, LinearAlgebra.LAPACKException)
            throw(OutOfInjectivityRadiusError())
        else
            rethrow()
        end
    end
    mul!(X, A, B)
    return project!(M, p, X)
end
function inverse_retract!(M::Rotations{N}, X, p, q, ::QRInverseRetraction) where {N}
    A = transpose(p) * q
    R = zero(X)
    for i in 1:N
        b = zeros(i)
        b[end] = 1
        b[1:(end - 1)] = -transpose(R[1:(i - 1), 1:(i - 1)]) * A[i, 1:(i - 1)]
        R[1:i, i] = A[1:i, 1:i] \ b
    end
    C = A * R
    X .= (C .- transpose(C)) ./ 2
    return X
end

@doc raw"""
    log(M::Rotations, p, q)

Compute the logarithmic map on the [`Rotations`](@ref) manifold
`M`$=\mathrm{SO}(n)$, which is given by
```math
\log_p q =
  \frac{1}{2} \bigl(\operatorname{Log}(p^{\mathrm{T}}q)
  - (\operatorname{Log}(p^{\mathrm{T}}q)^{\mathrm{T}}),
```
where $\operatorname{Log}$ denotes the matrix logarithm.

For antipodal rotations the function returns deterministically one of the tangent vectors
that point at `q`.
"""
log(::Rotations, ::Any...)

function log!(M::Rotations, X, p, q)
    U = transpose(p) * q
    X .= real(log_safe(U))
    return project!(M, X, p, X)
end
function log!(M::Rotations{2}, X, p, q)
    U = transpose(p) * q
    @assert size(U) == (2, 2)
    @inbounds θ = atan(U[2], U[1])
    return get_vector!(M, X, p, θ, DefaultOrthogonalBasis())
end
function log!(M::Rotations{3}, X, p, q)
    U = transpose(p) * q
    cosθ = (tr(U) - 1) / 2
    if cosθ ≈ -1
        eig = eigen_safe(U)
        ival = findfirst(λ -> isapprox(λ, 1), eig.values)
        inds = SVector{3}(1:3)
        ax = eig.vectors[inds, ival]
        return get_vector!(M, X, p, π * ax, DefaultOrthogonalBasis())
    end
    X .= ((U .- transpose(U)) ./ (2 * usinc_from_cos(cosθ)))
    return X
end
function log!(M::Rotations{4}, X, p, q)
    U = transpose(p) * q
    cosα, cosβ = cos_angles_4d_rotation_matrix(U)
    α = acos(clamp(cosα, -1, 1))
    β = acos(clamp(cosβ, -1, 1))
    if α ≈ π && β ≈ 0
        A² = Symmetric((U - I) ./ 2)
        P = eigvecs(A²)
        E = similar(U)
        fill!(E, 0)
        α = acos(clamp(cosα, -1, 1))
        @inbounds begin
            E[2, 1] = -α
            E[1, 2] = α
        end
        copyto!(X, P * E * transpose(P))
    else
        copyto!(X, real(log_safe(U)))
    end
    return project!(M, X, p, X)
end

@doc raw"""
    manifold_dimension(M::Rotations)

Return the dimension of the manifold $\mathrm{SO}(n)$, i.e.
```math
\dim_{\mathrm{SO}(n)} = \frac{n(n-1)}{2}.
```
"""
manifold_dimension(::Rotations{N}) where {N} = div(N * (N - 1), 2)

"""
    mean(
        M::Rotations,
        x::AbstractVector,
        [w::AbstractWeights,]
        method = GeodesicInterpolationWithinRadius(π/2/√2);
        kwargs...,
    )

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` using
[`GeodesicInterpolationWithinRadius`](@ref).
"""
mean(::Rotations, ::Any)

function Statistics.mean!(M::Rotations, q, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(M, q, x, w, GeodesicInterpolationWithinRadius(π / 2 / √2); kwargs...)
end

@doc raw"""
    norm(M::Rotations, p, X)

Compute the norm of a tangent vector `X` from the tangent space at `p` on the
[`Rotations`](@ref) `M`. The formula reads

````math
\lVert X \rVert_p = \lVert X \rVert,
````

i.e. the Frobenius norm of `X`, where tangent vectors are represented by
elements from the Lie algebra.
"""
LinearAlgebra.norm(M::Rotations, p, X) = norm(X)

@doc raw"""
    normal_rotation_distribution(M::Rotations, p, σ::Real)

Return a random point on the manifold [`Rotations`](@ref) `M`
by generating a (Gaussian) random orthogonal matrix with determinant $+1$. Let

$QR = A$

be the QR decomposition of a random matrix $A$, then the formula reads

$p = QD$

where $D$ is a diagonal matrix with the signs of the diagonal entries of $R$,
i.e.

$D_{ij}=\begin{cases} \operatorname{sgn}(R_{ij}) & \text{if} \; i=j \\ 0 & \, \text{otherwise} \end{cases}.$

It can happen that the matrix gets -1 as a determinant. In this case, the first
and second columns are swapped.

The argument `p` is used to determine the type of returned points.
"""
function normal_rotation_distribution(M::Rotations{N}, p, σ::Real) where {N}
    d = Distributions.MvNormal(zeros(N * N), σ)
    return NormalRotationDistribution(M, d, p)
end

@doc raw"""
    project(M::Rotations, p; check_det = true)

Project `p` to the nearest point on manifold `M`.

Given the singular value decomposition $p = U Σ V^\mathrm{T}$, with the
singular values sorted in descending order, the projection is

````math
\operatorname{proj}_{\mathrm{SO}(n)}(p) =
U\operatorname{diag}\left[1,1,…,\det(U V^\mathrm{T})\right] V^\mathrm{T}
````

The diagonal matrix ensures that the determinant of the result is $+1$.
If `p` is expected to be almost special orthogonal, then you may avoid this
check with `check_det = false`.
"""
project(::Rotations, ::Any)

function project!(::Rotations{N}, q, p; check_det=true) where {N}
    F = svd(p)
    copyto!(q, F.U * F.Vt)
    if check_det && det(q) < 0
        d = similar(F.S)
        @inbounds fill!(view(d, 1:(N - 1)), 1)
        @inbounds d[N] = -1
        copyto!(q, F.U * Diagonal(d) * F.Vt)
    end
    return q
end

@doc raw"""
    project(M::Rotations, p, X)

Project the matrix `X` onto the tangent space by making `X` skew symmetric,

````math
\operatorname{proj}_p(X) = \frac{X-X^{\mathrm{T}}}{2},
````

where tangent vectors are represented by elements from the Lie group
"""
project(::Rotations, ::Any, ::Any)

project!(M::Rotations, Y, p, X) = (Y .= (X .- transpose(X)) ./ 2)

@doc raw"""
    representation_size(M::Rotations)

Return the `size()` of a point on the [`Rotations`](@ref) `M`, i.e. for the
$\mathrm{SO}(n)$ it's `(n,n)`.
"""
@generated representation_size(::Rotations{N}) where {N} = (N, N)

sharp!(M::Rotations, X::TFVector, p, ξ::CoTFVector) = copyto!(X, ξ)

function Random.rand(
    rng::AbstractRNG,
    d::NormalRotationDistribution{TResult,Rotations{N}},
) where {TResult,N}
    return if N == 1
        convert(TResult, ones(1, 1))
    else
        A = reshape(rand(rng, d.distr), (N, N))
        convert(TResult, _fix_random_rotation(A))
    end
end

function Distributions._rand!(
    rng::AbstractRNG,
    d::NormalRotationDistribution{TResult,Rotations{N}},
    x::AbstractArray{<:Real},
) where {TResult,N}
    return copyto!(x, rand(rng, d))
end

function _fix_random_rotation(A::AbstractMatrix)
    s = diag(sign.(qr(A).R))
    D = Diagonal(s)
    C = qr(A).Q * D
    if det(C) < 0
        C[:, [1, 2]] = C[:, [2, 1]]
    end
    return C
end

@doc raw"""
    retract(M::Rotations, p, X, ::PolarRetraction)

Compute the SVD-based retraction on the [`Rotations`](@ref) `M` from `p` in direction `X`
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
retract(::Rotations, ::Any, ::Any, ::PolarRetraction)

@doc raw"""
    retract(M, p, X, ::QRRetraction)

Compute the QR-based retraction on the [`Rotations`](@ref) `M` from `p` in direction `X`
(as an element of the Lie group), which is a first-order approximation of the exponential map.

This is also the default retraction on the [`Rotations`](@ref)
"""
retract(::Rotations, ::Any, ::Any, ::QRRetraction)

function retract!(M::Rotations, q::AbstractArray{T}, p, X, method::QRRetraction) where {T}
    A = p + p * X
    qr_decomp = qr(A)
    d = diag(qr_decomp.R)
    D = Diagonal(sign.(d .+ convert(T, 0.5)))
    return copyto!(q, qr_decomp.Q * D)
end
function retract!(M::Rotations, q, p, X, method::PolarRetraction)
    A = p + p * X
    return project!(M, q, A; check_det=false)
end

Base.show(io::IO, ::Rotations{N}) where {N} = print(io, "Rotations($(N))")

Distributions.support(d::NormalRotationDistribution) = MPointSupport(d.manifold)

@doc raw"""
    zero_tangent_vector(M::Rotations, p)

Return the zero tangent vector from the tangent space art `p` on the [`Rotations`](@ref)
as an element of the Lie group, i.e. the zero matrix.
"""
zero_tangent_vector(M::Rotations, p) = zero(p)

zero_tangent_vector!(M::Rotations, X, p) = fill!(X, 0)
