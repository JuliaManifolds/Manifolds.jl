@doc doc"""
    Rotations{N} <: Manifold

Special orthogonal manifold $\mathrm{SO}(n)$ represented by $n\times n$
real-valued orthogonal matrices with determinant $+1$.

# Constructor

    Rotations(n)

Generate the $\mathrm{SO}(n) \subset \mathbb R^{n\times n}$
"""
struct Rotations{N} <: Manifold end

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

@doc doc"""
    angles_4d_skew_sym_matrix(A)

The Lie algebra of $\mathrm{SO}(4)$ consists of 4x4 skew-symmetric matrices.
The unique imaginary components of their eigenvalues are the angles of the two
plane rotations. This function computes these more efficiently than
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
    check_manifold_point(M,x; kwargs...)

Check whether `x` is a valid point on the [`Rotations`](@ref) `M`,
i.e. is an array of size [`manifold_dimension`](@ref)`(M)` and represents a
valid rotation.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_manifold_point(M::Rotations{N}, x; kwargs...) where {N}
    if size(x) != (N, N)
        return DomainError(
            size(x),
            "The point $(x) does not lie on $M, since its size is not $((N, N)).",
        )
    end
    if !isapprox(det(x), 1; kwargs...)
        return DomainError(det(x), "The determinant of $x has to be +1 but it is $(det(x))")
    end
    if !isapprox(transpose(x) * x, one(x); kwargs...)
        return DomainError(norm(x), "$x has to be orthogonal but it's not")
    end
    return nothing
end

"""
    check_tangent_vector(M,x,v; kwargs... )

Check whether `v` is a tangent vector to `x` on the [`Rotations`](@ref)
space `M`, i.e. after [`check_manifold_point`](@ref)`(M,x)`, `v` has to be of same
dimension as `x` and orthogonal to `x`.
The tolerance for the last test can be set using the `kwargs...`.
"""
function check_tangent_vector(M::Rotations{N}, x, v; kwargs...) where {N}
    perr = check_manifold_point(M, x)
    perr === nothing || return perr
    if size(v) != (N, N)
        return DomainError(
            size(v),
            "The array $(v) is not a tangent to a point on $M since its size does not match $((N, N)).",
        )
    end
    if !isapprox(transpose(v) + v, zero(v); kwargs...)
        return DomainError(
            size(v),
            "The array $(v) is not a tangent to a point on $M since it is not skew-symmetric.",
        )
    end
    return nothing
end

@doc doc"""
    cos_angles_4d_rotation_matrix(R)

4D rotations can be described by two orthogonal planes that are unchanged by
the action of the rotation (vectors within a plane rotate only within the
plane). The cosines of the two angles of rotation about these planes may be
obtained from the distinct real parts of the eigenvalues of the rotation
matrix. This function computes these more efficiently by solving the system

```math
\begin{aligned}
\cos\alpha + \cos\beta &= \frac{1}{2} \operatorname{tr}(R)\\
\cos\alpha + \cos\beta &= \frac{1}{8} \operatorname{tr}(R)^2
                           - \frac{1}{16} \operatorname{tr}((R - R^T)^2) - 1.
\end{aligned}
```

By convention, the returned values are sorted in increasing order. See
[`angles_4d_skew_sym_matrix`](@ref). For derivation of the above, see
[[Gallier, 2013]](#Gallier2003).
"""
function cos_angles_4d_rotation_matrix(R)
    trR = tr(R)
    a = trR / 4
    b = sqrt(clamp(tr((R .- transpose(R))^2) / 16 - a^2 + 1, 0, Inf))
    return (a + b, a - b)
end

@doc doc"""
    exp(M::Rotations, x, v)

Compute the exponential map on the [`Rotations`](@ref) from `x` into direction
`v`, i.e.
````math
\exp_xv = x \operatorname{Exp}(v),
````

where $\operatorname{Exp}(v)$  denotes the matrix exponential of $v$.

    exp(M::Rotations{4}, x, v)

Compute the exponential map of tangent vector `v` at point `x` from $\mathrm{SO}(4)$
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

exp!(M::Rotations, y, x, v) = copyto!(y, x * exp(v))
function exp!(M::Rotations{2}, y, x, v)
    @assert size(y) == (2, 2)
    θ = vee(M, x, v)[1]
    sinθ, cosθ = sincos(θ)
    @inbounds begin
        y[1] = cosθ
        y[2] = sinθ
        y[3] = -sinθ
        y[4] = cosθ
    end
    return copyto!(y, x * y)
end
function exp!(M::Rotations{3}, y, x, v)
    θ = norm(M, x, v) / sqrt(2)
    if θ ≈ 0
        a = 1 - θ^2 / 6
        b = θ / 2
    else
        a = sin(θ) / θ
        b = (1 - cos(θ)) / θ^2
    end
    xinvy = I + a .* v .+ b .* (v^2)
    return copyto!(y, x * xinvy)
end
function exp!(M::Rotations{4}, y, x, v)
    T = eltype(v)
    α, β = angles_4d_skew_sym_matrix(v)
    sinα, cosα = sincos(α)
    sinβ, cosβ = sincos(β)
    α² = α^2
    β² = β^2
    Δ = β² - α²
    if !isapprox(Δ, 0; atol = 1e-6)  # Case α > β ≥ 0
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

    v² = v * v
    xinvy = a₀ * I + a₁ .* v .+ a₂ .* v² .+ a₃ .* (v² * v)
    return copyto!(y, x * xinvy)
end

flat!(M::Rotations, v::CoTFVector, x, w::TFVector) = copyto!(v, w)

function get_coordinates(M::Rotations, x, v, B::ArbitraryOrthonormalBasis) where {N}
    T = Base.promote_eltype(x, v)
    return vee(M, x, v) .* sqrt(T(2))
end

function get_vector(M::Rotations, x, v, B::ArbitraryOrthonormalBasis) where {N}
    T = Base.promote_eltype(x, v)
    return hat(M, x, v) ./ sqrt(T(2))
end

@doc doc"""
    hat(M::Rotations, x, ω)

Convert the unique tangent vector components $\omega$ at point $x$ on rotations
group $\mathrm{SO}(n)$ to the matrix representation $\Omega$ of the tangent
vector. See [`vee`](@ref) for the conventions used.
"""
hat(M::Rotations, ::Any...)

function hat!(M::Rotations{N}, Ω, x, ω) where {N}
    @assert size(Ω) == (N, N)
    @assert length(ω) == manifold_dimension(M)
    @inbounds begin
        Ω[1, 1] = 0
        Ω[1, 2] = -ω[3]
        Ω[1, 3] = ω[2]
        Ω[2, 1] = ω[3]
        Ω[2, 2] = 0
        Ω[2, 3] = -ω[1]
        Ω[3, 1] = -ω[2]
        Ω[3, 2] = ω[1]
        Ω[3, 3] = 0
        k = 4
        for i = 4:N
            for j = 1:i-1
                Ω[i, j] = ω[k]
                Ω[j, i] = -ω[k]
                k += 1
            end
            Ω[i, i] = 0
        end
    end
    return Ω
end
function hat!(M::Rotations{2}, Ω, x, θ::Real)
    @assert length(Ω) == 4
    @inbounds begin
        Ω[1] = 0
        Ω[2] = θ
        Ω[3] = -θ
        Ω[4] = 0
    end
    return Ω
end
hat!(M::Rotations{2}, Ω, x, ω) = hat!(M, Ω, x, ω[1])

@doc doc"""
    injectivity_radius(M::Rotations)
    injectivity_radius(M::Rotations, x)

Return the injectivity radius on the [`Rotations`](@ref) `M`, which is globally

````math
    \operatorname{inj}_{\mathrm{SO}(n)}(x) = \pi\sqrt{2}.
````

    injectivity_radius(M::Rotations, x, ::PolarRetraction)

Return the radius of injectivity for the [`PolarRetraction`](@ref) on the
[`Rotations`](@ref) `M` which is $\frac{\pi}{\sqrt{2}}$.
"""
injectivity_radius(::Rotations) = π * sqrt(2.0)
injectivity_radius(::Rotations, x, ::PolarRetraction) = π / sqrt(2.0)

@doc doc"""
    inner(M::Rotations, x, w, v)

Compute the inner product of the two tangent vectors `w, v` from the tangent
plane at `x` on the special orthogonal space `M=`$\mathrm{SO}(n)$ using the
restriction of the metric from the embedding, i.e.

$(v, w)_x = \operatorname{tr}(v^T w)$.

Tangent vectors are represented by matrices.
"""
inner(M::Rotations, x, w, v) = dot(w, v)

@doc doc"""
    inverse_retract(M, x, y, ::PolarInverseRetraction)

Compute a vector from the tangent space $T_x\mathrm{SO}(n)$
of the point `x` on the [`Rotations`](@ref) manifold `M`
with which the point `y` can be reached by the
[`PolarRetraction`](@ref) from the point `x` after time 1.

The formula reads
````math
\operatorname{retr}^{-1}_x(y)
= -\frac{1}{2}(x^{\mathrm{T}}ys - (x^{\mathrm{T}}ys)^{\mathrm{T}})
````

where $s$ is the solution to the Sylvester equation

$x^{\mathrm{T}}ys + s(x^{\mathrm{T}}y)^{\mathrm{T}} + 2\mathrm{I}_n = 0.$

    inverse_retract(M::Rotations, x, y, ::QRInverseRetraction)

Compute a vector from the tangent space $T_x\mathrm{SO}(n)$ of the point `x` on the
[`Rotations`](@ref) manifold `M` with which the point `y` can be reached by the
[`QRRetraction`](@ref) from the point `x` after time 1.
"""
inverse_retract(::Rotations, ::Any, ::Any, ::PolarInverseRetraction)

function inverse_retract!(M::Rotations, v, x, y, method::PolarInverseRetraction)
    A = transpose(x) * y
    H = 2 * one(x)
    try
        B = sylvester(collect(A), collect(transpose(A)), collect(H))
        C = A * B
        v .= (transpose(C) .- C) ./ 2
    catch e
        if isa(e, LinearAlgebra.LAPACKException)
            throw(OutOfInjectivityRadiusError())
        else
            rethrow()
        end
    end
    return v
end
function inverse_retract!(M::Rotations{N}, v, x, y, ::QRInverseRetraction) where {N}
    A = transpose(x) * y
    R = zero(v)
    for i = 1:N
        b = zeros(i)
        b[end] = 1
        b[1:(end-1)] = -transpose(R[1:(i-1), 1:(i-1)]) * A[i, 1:(i-1)]
        R[1:i, i] = A[1:i, 1:i] \ b
    end
    C = A * R
    v .= (C .- transpose(C)) ./ 2
    return v
end

@doc doc"""
    log(M::Rotations, x, y)

Compute the logarithmic map on the [`Rotations`](@ref) manifold
`M`$=\mathrm{SO}(n)$, which is given by
```math
\log_{x} y =
  \frac{1}{2} \bigl(\operatorname{Log}(x^{\mathrm{T}}y)
  - (\operatorname{Log} x^{\mathrm{T}}y)^{\mathrm{T}}),
```
where $\operatorname{Log}$ denotes the matrix logarithm.

For antipodal rotations the function returns deterministically one of the tangent vectors
that point at `y`.
"""
log(::Rotations, ::Any...)

function log!(M::Rotations, v, x, y)
    U = transpose(x) * y
    v .= real(log_safe(U))
    return project_tangent!(M, v, x, v)
end
function log!(M::Rotations{2}, v, x, y)
    U = transpose(x) * y
    @assert size(U) == (2, 2)
    @inbounds θ = atan(U[2], U[1])
    return hat!(M, v, x, θ)
end
function log!(M::Rotations{3}, v, x, y)
    U = transpose(x) * y
    cosθ = (tr(U) - 1) / 2
    if cosθ ≈ -1
        eig = eigen_safe(U)
        ival = findfirst(λ -> isapprox(λ, 1), eig.values)
        vi = SVector{3}(1:3)
        ax = eig.vectors[vi, ival]
        return hat!(M, v, x, π * ax)
    end
    v .= ((U .- transpose(U)) ./ (2 * usinc_from_cos(cosθ)))
    return v
end
function log!(M::Rotations{4}, v, x, y)
    U = transpose(x) * y
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
        copyto!(v, P * E * transpose(P))
    else
        copyto!(v, real(log_safe(U)))
    end
    return project_tangent!(M, v, x, v)
end

@doc doc"""
    manifold_dimension(M::Rotations)

Return the dimension of the manifold $\mathrm{SO}(n)$, i.e. $\frac{n(n-1)}{2}$.
"""
manifold_dimension(M::Rotations{N}) where {N} = div(N * (N - 1), 2)

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

function mean!(M::Rotations, y, x::AbstractVector, w::AbstractVector; kwargs...)
    return mean!(M, y, x, w, GeodesicInterpolationWithinRadius(π / 2 / √2); kwargs...)
end

@doc doc"""
    norm(M::Rotations, x, v)

Compute the norm of a tangent vector `v` from the tangent space at `x` on the
[`Rotations`](@ref) `M`. The formula reads

````math
\lVert v \rVert_x = \lVert v \rVert,
````

i.e. the Frobenius norm of `v`, where tangent vectors are represented by
elements from the Lie group.
"""
norm(M::Rotations, x, v) = norm(v)

@doc doc"""
    normal_rotation_distribution(M::Rotations, x, σ::Real)

Return a random point on the manifold [`Rotations`](@ref) `M`
by generating a (Gaussian) random orthogonal matrix with determinant $+1$. Let

$QR = A$

be the QR decomposition of a random matrix $A$, then the formula reads

$x = QD$

where $D$ is a diagonal matrix with the signs of the diagonal entries of $R$,
i.e.

$D_{ij}=\begin{cases} \operatorname{sgn}(R_{ij}) & \text{if} \; i=j \\ 0 & \, \text{otherwise} \end{cases}.$

It can happen that the matrix gets -1 as a determinant. In this case, the first
and second columns are swapped.

The argument `x` is used to determine the type of returned points.
"""
function normal_rotation_distribution(M::Rotations{N}, x, σ::Real) where {N}
    d = Distributions.MvNormal(zeros(N * N), σ)
    return NormalRotationDistribution(M, d, x)
end

"""
    normal_tvector_distribution(M::Rotations, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(M::Rotations, x, σ)
    d = Distributions.MvNormal(reshape(zero(x), :), σ)
    return ProjectedFVectorDistribution(TangentBundleFibers(M), x, d, project_vector!, x)
end

@doc doc"""
    project_point(M::Rotations, x; check_det = true)

Project `x` to the nearest point on manifold `M`.

Given the singular value decomposition $x = U \Sigma V^\mathrm{T}$, with the
singular values sorted in descending order, the projection is

````math
\operatorname{proj}_{\mathrm{SO}(n)}(x) =
U\operatorname{diag}\left[1,1,\dots,\det(U V^\mathrm{T})\right] V^\mathrm{T}
````

The diagonal matrix ensures that the determinant of the result is $+1$.
If `x` is expected to be almost special orthogonal, then you may avoid this
check with `check_det = false`.
"""
project_point(::Rotations, ::Any...)

function project_point!(M::Rotations{N}, y, x; check_det = true) where {N}
    F = svd(x)
    copyto!(y, F.U * F.Vt)
    if check_det && det(y) < 0
        d = similar(F.S)
        @inbounds fill!(view(d, 1:N-1), 1)
        @inbounds d[N] = -1
        copyto!(y, F.U * Diagonal(d) * F.Vt)
    end
    return y
end

@doc doc"""
    project_tangent(M::Rotations, x, v)

Project the matrix `v` onto the tangent space by making `v` skew symmetric,

````math
\operatorname{proj}_x(v) = \frac{v-v^{\mathrm{T}}}{2},
````
where tangent vectors are represented by elements from the Lie group
"""
project_tangent(::Rotations, ::Any...)

project_tangent!(M::Rotations, w, x, v) = (w .= (v .- transpose(v)) ./ 2)

@doc doc"""
    representation_size(M::Rotations)

Return the `size()` of a point on the [`Rotations`](@ref) `M`, i.e. for the
$\mathrm{SO}(n)$ it's `(n,n)`.
"""
@generated representation_size(::Rotations{N}) where {N} = (N, N)

sharp!(M::Rotations, v::TFVector, x, w::CoTFVector) = copyto!(v, w)

function rand(
    rng::AbstractRNG,
    d::NormalRotationDistribution{TResult,Rotations{N}},
) where {TResult,N}
    if N == 1
        return convert(TResult, ones(1, 1))
    else
        A = reshape(rand(rng, d.distr), (N, N))
        return convert(TResult, _fix_random_rotation(A))
    end
end

function _rand!(
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

@doc doc"""
    retract(M, x, v)
    retract(M, x, v, ::QRRetraction)

Compute the SVD-based retraction on the [`Rotations`](@ref) `M` from `x` in direction `v`
(as an element of the Lie group) and is a first-order approximation of the exponential map.

This is also the default retraction on the [`Rotations`](@ref)

    retract(M::Rotations, x, v, ::PolarRetraction)

Compute the SVD-based retraction on the [`Rotations`](@ref) `M` from `x` in direction `v`
(as an element of the Lie group) and is a second-order approximation of the exponential map.
Let

````math
USV = x + xv
````

be the singular value decomposition, then the formula reads

````math
\operatorname{retr}_x v = UV^\mathrm{T}.
````
"""
retract(::Rotations, ::Any...)

function retract!(M::Rotations, y::AbstractArray{T}, x, v, method::QRRetraction) where {T}
    A = x + x * v
    qr_decomp = qr(A)
    d = diag(qr_decomp.R)
    D = Diagonal(sign.(d .+ convert(T, 0.5)))
    return copyto!(y, qr_decomp.Q * D)
end
retract!(M::Rotations, y, x, v) = retract!(M, y, x, v, QRRetraction())
function retract!(M::Rotations, y, x, v, method::PolarRetraction)
    A = x + x * v
    return project_point!(M, y, A; check_det = false)
end

show(io::IO, ::Rotations{N}) where {N} = print(io, "Rotations($(N))")

@doc doc"""
    vee(M::Rotations, x, Ω)

Extract the unique tangent vector components $\omega$ at point $x$ on rotations
group $\mathrm{SO}(n)$ from the matrix representation $\Omega$ of the tangent
vector.

The basis on the Lie algebra $\mathfrak{so}(n)$ is chosen such that for
$\mathrm{SO}(2)$, $\omega=\theta=\Omega_{21}$ is the angle of rotation, and
for $\mathrm{SO}(3)$,
$\omega = (\Omega_{32}, \Omega_{13}, \Omega_{21}) = \theta u$ is the
angular velocity and axis-angle representation, where $u$ is the unit vector
along the axis of rotation.

For $\mathrm{SO}(n)$ where $n \ge 4$, the additional elements of $\omega$ are
$\omega_{i (i - 3)/2 + j + 1} = \Omega_{ij}$, for $i \in [4, n], j \in [1,i)$.
"""
vee(::Rotations, ::Any...)
vee(M::Rotations{2}, x, Ω) = [Ω[2]]

function vee!(M::Rotations{N}, ω, x, Ω) where {N}
    @assert size(Ω) == (N, N)
    @assert length(ω) == manifold_dimension(M)
    @inbounds begin
        ω[1] = Ω[3, 2]
        ω[2] = Ω[1, 3]
        ω[3] = Ω[2, 1]

        k = 4
        for i = 4:N, j = 1:i-1
            ω[k] = Ω[i, j]
            k += 1
        end
    end
    return ω
end
function vee!(M::Rotations{2}, ω, x, Ω)
    ω[1] = Ω[2]
    return ω
end

@doc doc"""
    zero_tangent_vector(M::Rotations, x)

Return the zero tangent vector from the tangent space art `x` on the [`Rotations`](@ref)
as an element of the Lie group, i.e. the zero matrix.
"""
zero_tangent_vector(M::Rotations, x) = zero(x)

zero_tangent_vector!(M::Rotations, v, x) = fill!(v, 0)
