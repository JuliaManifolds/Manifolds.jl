@doc doc"""
    Rotations{N} <: Manifold

Special orthogonal manifold $\mathrm{SO}(n)$ represented by $n\times n$
real-valued orthogonal matrices with determinant $+1$.

# Constructor

    Rotations(n)

generates the $\mathrm{SO}(n) \subset \mathbb R^{n\times n}$
"""
struct Rotations{N} <: Manifold end
Rotations(n::Int) = Rotations{n}()

function representation_size(::Rotations{N}, ::Type{T}) where {N,T<:Union{MPoint, TVector, CoTVector}}
    return (N, N)
end

@doc doc"""
    manifold_dimension(S::Rotations)

returns the dimension of the manifold $\mathrm{SO}(n)$, i.e. $\frac{n(n-1)}{2}$.
"""
manifold_dimension(S::Rotations{N}) where {N} = div(N*(N-1), 2)

@doc doc"""
    inner(S, x, w, v)

compute the inner product of the two tangent vectors `w, v` from the tangent
plane at `x` on the special orthogonal space `S=`$\mathrm{SO}(n)$ using the
restriction of the metric from the embedding, i.e.

$(v, w)_x = \operatorname{tr}(v^T w)$.

Tangent vectors are represented by matrices.
"""
inner(S::Rotations, x, w, v) = dot(w, v)

norm(S::Rotations, x, v) = norm(v)

project_tangent!(S::Rotations, w, x, v) = w .= (v .- transpose(v))./2

function exp!(S::Rotations, y, x, v)
    y .= x * exp(v)
    return y
end

function exp!(S::Rotations{2}, y, x, v)
    θ = vee(S, x, v)[1]
    @assert size(y) == (2, 2)
    @assert size(x) == (2, 2)
    @inbounds begin
        sinθ, cosθ = sincos(θ)
        y[1] = x[1] * cosθ + x[3] * sinθ
        y[2] = x[2] * cosθ + x[4] * sinθ
        y[3] = x[3] * cosθ - x[1] * sinθ
        y[4] = x[4] * cosθ - x[2] * sinθ
    end
    return y
end

function exp!(S::Rotations{3}, y, x, v)
    θ = norm(S, x, v) / sqrt(2)
    if θ ≈ 0
        a = 1 - θ^2 / 6
        b = θ / 2
    else
        a = sin(θ) / θ
        b = (1 - cos(θ)) / θ^2
    end
    y .= x .+ x * (a .* v .+ b .* (v^2))
    return y
end

@doc doc"""
    log!(M, v, x, y)

compute the logarithmic map on the [`Rotations`](@ref) manifold
`M`$=\mathrm{SO}(n)$, which is given by
```math
\operatorname{log}_{x} y =
  \frac{1}{2} \bigl(\operatorname{Log}(x^{\mathrm{T}}y)
  - (\operatorname{Log} x^{\mathrm{T}}y)^{\mathrm{T}}),
```
where $\operatorname{Log}$ denotes the matrix logarithm
and save the result to `v`.

For antipodal rotations the function returns one of the tangent vectors that
point at `y`.
"""
function log!(S::Rotations, v, x, y)
    U = transpose(x) * y
    v .= real(log_safe(U))
    project_tangent!(S, v, x, v)
    return v
end

function log!(S::Rotations{2}, v, x, y)
    U = transpose(x) * y
    @assert size(U) == (2, 2)
    @inbounds θ = atan(U[2], U[1])
    hat!(S, v, x, θ)
    return v
end

function log!(S::Rotations{3}, v, x, y)
    U = transpose(x) * y
    cosθ = (tr(U) - 1) / 2
    if cosθ ≈ -1
        eig = eigen_safe(U)
        ival = findfirst(λ -> isapprox(λ, 1), eig.values)
        vi = SVector{3}(1:3)
        ax = eig.vectors[vi, ival]
        hat!(S, v, x, π * ax)
    else
        v .= (U .- transpose(U)) ./ (2 * usinc_from_cos(cosθ))
    end
    return v
end

@doc doc"""
    cos_angle_4d_rotation_matrix(R)

4D rotations can be described by two orthogonal planes that are unchanged by
the action of the rotation (vectors within a plane rotate only within the
plane). The cosines of the two angles of rotation about these planes may be
obtained from the distinct real parts of the eigenvalues of the rotation
matrix. This function computes these more efficiently by solving the system

```math
\begin{aligned}
\cos\theta_1 + \cos\theta_2 &= \frac{1}{2} \operatorname{tr}(R)\\
\cos\theta_1 \cos\theta_2 &= \frac{1}{8} \operatorname{tr}(R)^2
                           - \frac{1}{16} \operatorname{tr}((R - R^T)^2) - 1.
\end{aligned}
```

By convention, the returned values are sorted in increasing order.
"""
function cos_angle_4d_rotation_matrix(R)
    trR = tr(R)
    a = trR / 4
    b = sqrt(clamp(tr((R .- transpose(R))^2) / 16 - a^2 + 1, 0, Inf))
    return (a + b, a - b)
end

function log!(S::Rotations{4}, v, x, y)
    U = transpose(x) * y
    cosθ₁, cosθ₂ = cos_angle_4d_rotation_matrix(U)
    θ₁ = acos(clamp(cosθ₁, -1, 1))
    θ₂ = acos(clamp(cosθ₂, -1, 1))
    if θ₁ ≈ π && θ₂ ≈ 0
        B₁² = Symmetric((U - I) ./ 2)
        P = eigvecs(B₁²)
        E = similar(U)
        fill!(E, 0)
        θ₁ = acos(clamp(cosθ₁, -1, 1))
        @inbounds begin
            E[2, 1] = -θ₁
            E[1, 2] = θ₁
        end
        v .= P * E * transpose(P)
    else
        v .= real(log_safe(U))
    end
    project_tangent!(S, v, x, v)
    return v
end


injectivity_radius(M::Rotations, x) = π*sqrt(2.0)

@doc doc"""
    PolarRetraction

Retraction on the rotations manifold using the polar method.
This SVD-based retraction is a second-order approximation of the
exponential map. Let

$USV = x + xv$

be the singular value decomposition, then the formula reads

$\operatorname{retr}_x v = UV^\mathrm{T}$

Retraction is performed by the function [`retract!(::Rotations, y, x, v, ::PolarRetraction)`](@ref)
"""
struct PolarRetraction <: AbstractRetractionMethod end

@doc doc"""
    retract_polar!(M::Rotations, y, x, v, method::PolarRetraction)

Compute the SVD-based retraction [`PolarRetraction`](@ref), a second-order
approximation of the exponential map.
"""
function retract!(M::Rotations, y, x, v, method::PolarRetraction)
    A = x + x*v
    S = svd(A)
    y .= S.U * transpose(S.V)
    return y
end

"""
    QRRetraction

Retraction on the rotations manifold using the QR method, a first order
approximation of the exponential map.
"""
struct QRRetraction <: AbstractRetractionMethod end

@doc doc"""
    retract!(M, y, x, v, method::QRRetraction)

This QR-based retraction is a first-order approximation of the exponential map.
"""
function retract!(M::Rotations, y::AbstractArray{T}, x, v, method::QRRetraction) where T
    A = x + x*v
    qr_decomp = qr(A)
    d = diag(qr_decomp.R)
    D = Diagonal( sign.(d .+ convert(T, 0.5)) )
    y .= qr_decomp.Q * D
    return y
end

retract!(M::Rotations, y, x, v) = retract!(M, y, x, v, QRRetraction())

"""
    PolarInverseRetraction

Inverse retraction on the rotations manifold using the polar method.
"""
struct PolarInverseRetraction <: AbstractInverseRetractionMethod end

injectivity_radius(::Rotations, x, ::PolarRetraction) = π*sqrt(2.0)/2

@doc doc"""
    inverse_retract!(M, v, x, y, ::PolarInverseRetraction)

Compute a vector from the tagent space $T_x\mathrm{SO}(n)$
of the point `x` on the [`Rotations`](@ref) manifold `M`
with which the point `y` can be reached by the
[`PolarRetraction`](@ref) from the point `x` after time 1.

The formula reads
$v = -\frac{1}{2}(x^{\mathrm{T}}ys - (x^{\mathrm{T}}ys)^{\mathrm{T}})$

where $s$ is the solution to the Sylvester equation

$x^{\mathrm{T}}ys + s(x^{\mathrm{T}}y)^{\mathrm{T}} + 2\mathrm{I}_n = 0.$
"""
function inverse_retract!(M::Rotations, v, x, y, method::PolarInverseRetraction)
    A = transpose(x) * y
    H = 2 * one(x)
    try
        B = sylvester(collect(A), collect(transpose(A)), collect(H))
        C = A * B
        v .= (transpose(C) .- C)./2
    catch e
        if isa(e, LinearAlgebra.LAPACKException)
            throw(OutOfInjectivityRadiusError())
        else
            rethrow()
        end
    end
    return v
end

"""
    QRInverseRetraction

Inverse retraction on the rotations manifold using the QR method.
"""
struct QRInverseRetraction <: AbstractInverseRetractionMethod end

@doc doc"""
    inverse_retract!(M::Rotations, x, y, ::QRInverseRetraction)

Compute a vector from the tagent space $T_x\mathrm{SO}(n)$
of the point `x` on the [`Rotations`](@ref) manifold `M`
with which the point `y` can be reached by the
[`QRRetraction`](@ref) from the point `x` after time 1.
"""
function inverse_retract!(M::Rotations{N}, v, x, y, ::QRInverseRetraction) where N
    A = transpose(x) * y
    R = zero(v)
    for i = 1:N
        b = zeros(i)
        b[end] = 1
        b[1:(end-1)] = - transpose(R[1:(i-1), 1:(i-1)]) * A[i, 1:(i-1)]
        R[1:i, i] = A[1:i, 1:i] \ b
    end
    C =  A * R
    v .= (C .- transpose(C))./2
    return v
end

zero_tangent_vector(S::Rotations, x) = zero(x)
zero_tangent_vector!(S::Rotations, v, x) = (v .= zero(x))

function hat!(S::Rotations{2}, Ω, x, θ::Real)
    @assert length(Ω) == 4
    @inbounds begin
        Ω[1] = 0;  Ω[3] = -θ
        Ω[2] = θ;  Ω[4] = 0
    end
    return Ω
end

hat!(S::Rotations{2}, Ω, x, ω) = hat!(S, Ω, x, ω[1])

@doc doc"""
    hat(S::Rotations, x, ω)

Convert the tangent vector $\omega$ at point $x$ on rotations group
$\mathrm{SO}(n)$ to the equivalent matrix representation. See [`vee`](@ref)
for the conventions used.
"""
function hat end

function hat!(S::Rotations{N}, Ω, x, ω) where {N}
    @assert size(Ω) == (N, N)
    @assert length(ω) == manifold_dimension(S)
    @inbounds begin
        Ω[1,1] = 0;      Ω[1,2] = -ω[3];  Ω[1,3] = ω[2]
        Ω[2,1] = ω[3];   Ω[2,2] = 0;      Ω[2,3] = -ω[1]
        Ω[3,1] = -ω[2];  Ω[3,2] = ω[1];   Ω[3,3] = 0
        k = 4
        for i=4:N
            for j=1:i-1
                Ω[i,j] = ω[k]
                Ω[j,i] = -ω[k]
                k += 1
            end
            Ω[i,i] = 0
        end
    end
    return Ω
end

@doc doc"""
    vee(S::Rotations, x, Ω)

Convert the tangent vector $\Omega$ at point $x$ on rotations group
$\mathrm{SO}(n)$ represented as a matrix to the equivalent vector
representation $\omega=\Omega^\vee$. The basis on the Lie algebra
$\mathfrak{so}(n)$ is chosen such that for $\mathrm{SO}(2)$,
$\omega=\theta=\Omega_{21}$ is the angle of rotation, and for $\mathrm{SO}(3)$,
$\omega = (\Omega_{32}, \Omega_{13}, \Omega_{21}) = \theta u$ is the
angular velocity and axis-angle representation, where $u$ is the unit vector
along the axis of rotation.

For $\mathrm{SO}(n)$ where $n \ge 4$, the additional elements of $\omega$ are
$\omega_{i (i - 3)/2 + j + 1} = \Omega_{ij}$, for $i \in [4, n], j \in [1,i)$.
"""
function vee end

vee!(S::Rotations{2}, ω, x, Ω) = (ω[1] = Ω[2])

function vee!(S::Rotations{N}, ω, x, Ω) where {N}
    @assert size(Ω) == (N, N)
    @assert length(ω) == manifold_dimension(S)
    @inbounds begin
        ω[1] = Ω[3,2]
        ω[2] = Ω[1,3]
        ω[3] = Ω[2,1]

        k = 4
        for i=4:N
            for j=1:i-1
                ω[k] = Ω[i,j]
                k += 1
            end
        end
    end
    return ω
end

"""
    is_manifold_point(S,x; kwargs...)

checks, whether `x` is a valid point on the [`Rotations`](@ref) `S`,
i.e. is an array of size [`manifold_dimension`](@ref)`(S)` and represents a
valid rotation.
The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_manifold_point(S::Rotations{N},x; kwargs...) where {N}
    if size(x) != (N, N)
        throw(DomainError(size(x),
            "The point $(x) does not lie on $S, since its size is not $((N, N))."))
    end
    if !isapprox(det(x), 1; kwargs...)
        throw(DomainError(norm(x), "The determinant of $x has to be +1 but it is $(det(x))"))
    end
    if !isapprox(transpose(x)*x, one(x); kwargs...)
        throw(DomainError(norm(x), "$x has to be orthogonal but it's not"))
    end
    return true
end

"""
    is_tangent_vector(S,x,v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`Rotations`](@ref)
space `S`, i.e. after [`is_manifold_point`](@ref)`(S,x)`, `v` has to be of same
dimension as `x` and orthogonal to `x`.
The tolerance for the last test can be set using the ´kwargs...`.
"""
function is_tangent_vector(S::Rotations{N},x,v; kwargs...) where N
    is_manifold_point(S,x)
    if size(v) != (N, N)
        throw(DomainError(size(v),
            "The array $(v) is not a tangent to a point on $S since its size does not match $((N, N))."))
    end
    if !isapprox(transpose(v)+v, zero(v); kwargs...)
        throw(DomainError(size(v),
            "The array $(v) is not a tangent to a point on $S since it is not skew-symmetric."))
    end
    return true
end

"""
    gaussian_tvector_distribution(S::Rotations, x, σ)

Normal distribution in ambient space with standard deviation `σ`
projected to tangent space at `x`.
"""
function normal_tvector_distribution(S::Rotations, x, σ)
    d = Distributions.MvNormal(reshape(zero(x), :), σ)
    return ProjectedTVectorDistribution(S, x, d, project_tangent!, x)
end

"""
    NormalRotationDistribution(M::Rotations, d::Distribution, x::TResult)

Distribution that returns a random point on the manifold [`Rotations`](@ref)
`M`. Random point is generated using base distribution `d` and the type
of the result is adjusted to `TResult`.

See [`normal_rotation_distribution`](@ref) for details.
"""
struct NormalRotationDistribution{TResult, TM<:Rotations, TD<:Distribution} <: MPointDistribution{TM}
    manifold::TM
    distr::TD
end

function NormalRotationDistribution(M::Rotations, d::Distribution, x::TResult) where TResult
    return NormalRotationDistribution{TResult, typeof(M), typeof(d)}(M, d)
end

function _fix_random_rotation(A::AbstractMatrix)
    s=diag(sign.(qr(A).R))
    D=Diagonal(s)
    C = qr(A).Q*D
    if det(C) < 0
        C[:,[1,2]] = C[:,[2,1]]
    end
    return C
end

function rand(rng::AbstractRNG, d::NormalRotationDistribution{TResult,Rotations{N}}) where {TResult,N}
    if N==1
        return convert(TResult, ones(1,1))
    else
        A = reshape(rand(rng, d.distr), (N, N))
        return convert(TResult, _fix_random_rotation(A))
    end
end

function _rand!(rng::AbstractRNG, d::NormalRotationDistribution{TResult,Rotations{N}}, x::AbstractArray{<:Real}) where {TResult,N}
    if N==1
        x .= ones(1,1)
    else
        rand!(rng, d.distr, x)
        x .= _fix_random_rotation(x)
    end
    return x
end

@doc doc"""
    normal_rotation_distribution(M::Rotations, x, σ::Real)

return a random point on the manifold [`Rotations`](@ref) `M`
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
function normal_rotation_distribution(M::Rotations{N}, x, σ::Real) where N
    d = Distributions.MvNormal(zeros(N*N), σ)
    return NormalRotationDistribution(M, d, x)
end
