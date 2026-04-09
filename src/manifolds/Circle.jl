@doc raw"""
    Circle{𝔽} <: AbstractManifold{𝔽}

The circle ``𝕊^1`` is a manifold here represented by
real-valued points in ``[-π,π)`` or complex-valued points ``z ∈ ℂ`` of absolute value
``\lvert z\rvert = 1``.
# Constructor

    Circle(𝔽=ℝ)

Generate the `ℝ`-valued Circle represented by angles, which
alternatively can be set to use the [`AbstractNumbers`](@extref ManifoldsBase number-system) `𝔽=ℂ` to obtain the circle
represented by `ℂ`-valued circle of unit numbers.
"""
struct Circle{𝔽} <: AbstractManifold{𝔽} end

Circle(𝔽::AbstractNumbers = ℝ) = Circle{𝔽}()

function adjoint_Jacobi_field(::Circle{ℝ}, p, q, t, X, β::Tβ) where {Tβ}
    return X
end

@doc raw"""
    check_point(M::Circle, p)

Check whether `p` is a point on the [`Circle`](@ref) `M`.
For the real-valued case, `p` is an angle and hence it checks that ``p ∈ [-π,π)``.
for the complex-valued case, it is a unit number, ``p ∈ ℂ`` with ``\lvert p \rvert = 1``.
"""
check_point(::Circle, ::Any...)

function check_point(M::Circle{ℝ}, p; kwargs...)
    if !isapprox(sym_rem(p), p; kwargs...)
        return DomainError(
            p,
            "The point $(p) does not lie on $(M), since its is not in [-π,π).",
        )
    end
    return nothing
end
function check_point(M::Circle{ℂ}, p; kwargs...)
    if !isapprox(sum(abs.(p)), 1.0; kwargs...)
        return DomainError(
            sum(abs.(p)),
            "The point $(p) does not lie on the $(M) since its norm is not 1.",
        )
    end
    return nothing
end
check_size(::Circle, ::Number) = nothing
function check_size(M::Circle, p)
    (size(p) === () || size(p) === (1,)) && return nothing
    return DomainError(
        size(p),
        "The point $p can not belong to the $M, since it is not a number nor a vector of size (1,).",
    )
end
check_size(::Circle, ::Number, ::Number) = nothing
function check_size(M::Circle, p, X)
    (size(X) === () || size(X) === (1,)) && return nothing
    return DomainError(
        size(X),
        "The vector $X is not a tangent vector to $p on $M, since it is not a number nor a vector of size (1,).",
    )
end

"""
    check_vector(M::Circle, p, X; kwargs...)

Check whether `X` is a tangent vector in the tangent space of `p` on the
[`Circle`](@ref) `M`.
For the real-valued case represented by angles, all `X` are valid, since the tangent space is the whole real line.
For the complex-valued case `X` has to lie on the line parallel to the tangent line at `p`
in the complex plane, i.e. their inner product has to be zero.
"""
check_vector(::Circle{ℝ}, ::Any...; ::Any...)

function check_vector(M::Circle{ℝ}, p, X; kwargs...)
    return nothing
end
function check_vector(
        M::Circle{ℂ},
        p,
        X::T;
        atol::Real = sqrt(eps(real(float(number_eltype(T))))),
        kwargs...,
    ) where {T}
    if !isapprox(abs(complex_dot(p, X)), 0; atol = atol, kwargs...)
        return DomainError(
            abs(complex_dot(p, X)),
            "The value $(X) is not a tangent vector to $(p) on $(M), since it is not orthogonal in the embedding.",
        )
    end
    return nothing
end

@doc raw"""
    complex_dot(a, b)

Compute the inner product of two (complex) numbers with in the complex plane.
"""
complex_dot(a, b) = dot(map(real, a), map(real, b)) + dot(map(imag, a), map(imag, b))
complex_dot(a::Number, b::Number) = (real(a) * real(b) + imag(a) * imag(b))

function diagonalizing_projectors(M::Circle{ℝ}, p, X)
    sbv = sign(X[])
    proj = ProjectorOntoVector(M, p, @SVector [sbv == 0 ? one(sbv) : sbv])
    return ((zero(number_eltype(p)), proj),)
end

@doc raw"""
    distance(M::Circle, p, q)

Compute the distance on the [`Circle`](@ref) `M`, which is
the absolute value of the symmetric remainder of `p` and `q` for the real-valued
case and the angle between both complex numbers in the Gaussian plane for the
complex-valued case.
"""
distance(::Circle, ::Any...)
distance(::Circle{ℝ}, p::Real, q::Real) = abs(sym_rem(p - q))
distance(::Circle{ℝ}, p, q) = abs(sum(sym_rem.(p - q)))
function distance(::Circle{ℂ}, p, q)
    cosθ = complex_dot(p, q)
    T = float(real(Base.promote_eltype(p, q)))
    # abs and relative error of acos is less than sqrt(eps(T))
    -1 < cosθ < 1 - sqrt(eps(T)) / 8 && return acos(cosθ)
    # improved accuracy for q close to p or -p
    return 2 * abs(atan(norm(p - q), norm(p + q)))
end

@doc raw"""
    embed(M::Circle{ℂ}, p)

Embed a point `p` on the complex [`Circle`](@ref) `M` in the ambient space.
It returns `p`.
"""
embed(::Circle{ℂ}, p) = p

@doc raw"""
    embed(M::Circle{ℂ}, p, X)

Embed a tangent vector `X` at `p` on the complex [`Circle`](@ref) `M` in the ambient space.
It returns `X`.
"""
embed(::Circle{ℂ}, p, X) = X

@doc raw"""
    exp(M::Circle, p, X)

Compute the exponential map on the [`Circle`](@ref).
````math
\exp_p X = (p+X)_{2π},
````
where ``(⋅)_{2π}`` is the (symmetric) remainder with respect to division by ``2π``, i.e. in ``[-π,π)``.

For the complex-valued case, the same formula as for the [`Sphere`](@ref) ``𝕊^1`` is applied to values in the
complex plane.
"""
exp(::Circle, ::Any...)
Base.exp(::Circle{ℝ}, p::Real, X::Real) = sym_rem(p + X)
exp_fused(::Circle{ℝ}, p::Real, X::Real, t::Real) = sym_rem(p + t * X)
function Base.exp(M::Circle{ℂ}, p::Number, X::Number)
    θ = norm(M, p, X)
    return cos(θ) * p + usinc(θ) * X
end
function exp_fused(M::Circle{ℂ}, p::Number, X::Number, t::Number)
    θ = abs(t) * norm(M, p, X)
    return cos(θ) * p + usinc(θ) * t * X
end

exp!(::Circle{ℝ}, q, p, X) = (q .= sym_rem(p + X))
exp_fused!(::Circle{ℝ}, q, p, X, t::Number) = (q .= sym_rem(p[] + t * X[]))
function exp!(M::Circle{ℂ}, q, p, X)
    θ = norm(M, p, X)
    q .= cos(θ) * p + usinc(θ) * X
    return q
end
function exp_fused!(M::Circle{ℂ}, q, p, X, t::Number)
    θ = abs(t) * norm(M, p, X)
    q .= cos(θ) * p + usinc(θ) * t * X
    return q
end

function get_basis_diagonalizing(::Circle{ℝ}, p, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.frame_direction[])
    vs = @SVector [@SVector [sbv == 0 ? one(sbv) : sbv]]
    return CachedBasis(B, (@SVector [0]), vs)
end

get_coordinates_orthonormal(::Circle{ℝ}, p, X, ::RealNumbers) = @SVector [X[]]
get_coordinates_orthonormal(::Circle{ℝ}, p, X::AbstractArray, ::RealNumbers) = copy(vec(X))
get_coordinates_orthonormal!(::Circle{ℝ}, c, p, X, ::RealNumbers) = (c .= X)
function get_coordinates_diagonalizing(::Circle{ℝ}, p, X, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.frame_direction[])
    return X * (sbv == 0 ? one(sbv) : sbv)
end
function get_coordinates_diagonalizing!(
        M::Circle{ℝ},
        Y,
        p,
        X,
        B::DiagonalizingOrthonormalBasis,
    )
    Y[] = get_coordinates_diagonalizing(M, p, X, B)[]
    return Y
end

"""
    get_coordinates(M::Circle{ℂ}, p, X, B::DefaultOrthonormalBasis)

Return tangent vector coordinates in the Lie algebra of the [`Circle`](@ref).
"""
get_coordinates(::Circle{ℂ}, p, X, ::DefaultOrthonormalBasis{<:Any, TangentSpaceType})
function get_coordinates_orthonormal!(
        M::Circle{ℂ},
        Y,
        p,
        X,
        n::Union{RealNumbers, ComplexNumbers},
    )
    Y[] = get_coordinates_orthonormal(M, p, X, n)[]
    return Y
end
function get_coordinates_orthonormal(::Circle{ℂ}, p, X, ::Union{RealNumbers, ComplexNumbers})
    X, p = X[1], p[1]
    Xⁱ = imag(X) * real(p) - real(X) * imag(p)
    return @SVector [Xⁱ]
end

"""
    get_embedding(M::Circle{ℂ})

Get the ambient space of the complex [`Circle`](@ref) `M`, which is `ℂ`.
"""
ManifoldsBase.get_embedding(::Circle{ℂ}) = Euclidean(; field = ℂ)

get_vector_orthonormal(::Circle{ℝ}, p::StaticArray, c, ::RealNumbers) = Scalar(c[])
get_vector_orthonormal(::Circle{ℝ}, p, c, ::RealNumbers) = fill(c[])
# the method below is required for FD and AD differentiation in ManifoldDiff.jl
# if changed, make sure no tests in that repository get broken
get_vector_orthonormal(::Circle{ℝ}, p::AbstractVector, c, ::RealNumbers) = c
get_vector_orthonormal!(::Circle{ℝ}, X, p, c, ::RealNumbers) = (X .= c[])
function get_vector_diagonalizing!(M::Circle{ℝ}, X, p, c, B::DiagonalizingOrthonormalBasis)
    X[] = get_vector_diagonalizing(M, p, c, B)[]
    return X
end
function get_vector_diagonalizing(::Circle{ℝ}, p, c, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.frame_direction[])
    return c .* (sbv == 0 ? one(sbv) : sbv)
end
"""
    get_vector(M::Circle{ℂ}, p, X, B::DefaultOrthonormalBasis)

Return tangent vector from the coordinates in the Lie algebra of the [`Circle`](@ref).
"""
function get_vector_orthonormal(
        ::Circle{ℂ},
        p::StaticArray,
        c,
        ::Union{RealNumbers, ComplexNumbers},
    )
    return @SArray fill(1im * c[1] * p[1])
end
function get_vector_orthonormal(::Circle{ℂ}, p, c, ::Union{RealNumbers, ComplexNumbers})
    return fill(1im * c[1] * p[1])
end
function get_vector_orthonormal!(::Circle{ℂ}, X, p, c, ::Union{RealNumbers, ComplexNumbers})
    X .= 1im * c[1] * p[1]
    return X
end

@doc raw"""
    injectivity_radius(M::Circle[, p])

Return the injectivity radius on the [`Circle`](@ref) `M`, i.e. ``π``.
"""
injectivity_radius(::Circle) = π

@doc raw"""
    inner(M::Circle, p, X, Y)

Compute the inner product of the two tangent vectors `X,Y` from the tangent plane at `p` on
the [`Circle`](@ref) `M` using the restriction of the metric from the embedding,
i.e.

````math
g_p(X,Y) = X*Y
````

for the real case and

````math
g_p(X,Y) = Y^\mathrm{T}X
````

for the complex case interpreting complex numbers in the Gaussian plane.
"""
inner(::Circle, ::Any...)
@inline inner(::Circle{ℝ}, p, X, Y) = dot(X, Y)
@inline inner(::Circle{ℝ}, p::Real, X::Real, Y::Real) = X * Y
@inline inner(::Circle{ℂ}, p, X, Y) = complex_dot(X, Y)

# these methods make sure that we allow for checking mixed bare number and number wrapped in array
_isapprox(::Circle, x, y; kwargs...) = isapprox(x[], y[]; kwargs...)
_isapprox(::Circle, p, X, Y; kwargs...) = isapprox(X[], Y[]; kwargs...)

"""
    is_flat(::Circle)

Return true. [`Circle`](@ref) is a flat manifold.
"""
is_flat(M::Circle) = true

function jacobi_field(::Circle{ℝ}, p, q, t, X, β::Tβ) where {Tβ}
    return X
end

@doc raw"""
    log(M::Circle, p, q)

Compute the logarithmic map on the [`Circle`](@ref) `M`.
````math
\log_p q = (q-p)_{2π},
````
where ``(⋅)_{2π}`` is the (symmetric) remainder with respect to division by ``2π``, i.e. in ``[-π,π)``.

For the complex-valued case, the same formula as for the [`Sphere`](@ref) ``𝕊^1`` is applied to values in the
complex plane.
"""
log(::Circle, ::Any...)
Base.log(::Circle{ℝ}, p::Real, q::Real) = sym_rem(q - p)
function Base.log(M::Circle{ℂ}, p::Number, q::Number)
    cosθ = complex_dot(p, q)
    if cosθ ≈ -1  # appr. opposing points, return deterministic choice from set-valued log
        X = real(p) ≈ 1 ? 1im : 1 + 0im
        X = X - complex_dot(p, X) * p
        X *= π / norm(X)
    else
        cosθ = cosθ > 1 ? one(cosθ) : cosθ
        θ = acos(cosθ)
        X = (q - cosθ * p) / usinc(θ)
    end
    return project(M, p, X)
end

log!(::Circle{ℝ}, X, p, q) = (X .= sym_rem(q - p))
function log!(M::Circle{ℂ}, X, p, q)
    cosθ = complex_dot(p, q)
    if cosθ ≈ -1
        X .= sum(real.(p)) ≈ 1 ? 1.0im : 1.0 + 0.0im
        X .= X - complex_dot(p, X) * p
        X .*= π / norm(X)
    else
        cosθ = cosθ > 1 ? one(cosθ) : cosθ
        θ = acos(cosθ)
        X .= (q - cosθ * p) / usinc(θ)
    end
    return project!(M, X, p, X)
end

@doc raw"""
    manifold_dimension(M::Circle)

Return the dimension of the [`Circle`](@ref) `M`,
i.e. ``\dim(𝕊^1) = 1``.
"""
manifold_dimension(::Circle) = 1

@doc raw"""
    manifold_volume(M::Circle)

Return the volume of the [`Circle`](@ref) `M`, i.e. ``2π``.
"""
manifold_volume(::Circle) = 2 * π

@doc raw"""
    mean(M::Circle{ℝ}, x::AbstractVector[, w::AbstractWeights])

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` of points on
the [`Circle`](@ref) ``𝕊^1``, represented by real numbers, i.e. the angular mean
````math
\operatorname{atan}\Bigl( \sum_{i=1}^n w_i\sin(x_i),  \sum_{i=1}^n w_i\sin(x_i) \Bigr).
````
"""
mean(::Circle{ℝ}, ::Any)
function Statistics.mean(::Circle{ℝ}, x::AbstractVector{<:Real}; kwargs...)
    return atan(1 / length(x) * sum(sin, x), 1 / length(x) * sum(cos, x))
end
function Statistics.mean(
        ::Circle{ℝ},
        x::AbstractVector{<:Real},
        w::AbstractVector;
        kwargs...,
    )
    return atan(sum(w .* sin.(x)), sum(w .* cos.(x)))
end
@doc raw"""
    mean(M::Circle{ℂ}, x::AbstractVector[, w::AbstractWeights])

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` of points on
the [`Circle`](@ref) ``𝕊^1``, represented by complex numbers, i.e. embedded in the complex plane.
Computing the sum

````math
s = \sum_{i=1}^n x_i
````

the mean is the angle of the complex number ``s``, so represented in the complex plane as
``\frac{s}{\lvert s \rvert}`` whenever ``s \neq 0``.

If the sum ``s=0``, the mean is not unique. For example for opposite points or equally spaced angles.
"""
mean(::Circle{ℂ}, ::Any)
function Statistics.mean(M::Circle{ℂ}, x::AbstractVector{<:Complex}; kwargs...)
    s = sum(x)
    abs(s) == 0 && return error(
        "The mean for $(x) on $(M) is not defined/unique, since the sum of the complex numbers is zero",
    )
    return s / abs(s)
end
function Statistics.mean(
        M::Circle{ℂ},
        x::AbstractVector{<:Complex},
        w::AbstractVector;
        kwargs...,
    )
    s = sum(w .* x)
    abs(s) == 0 && error(
        "The mean for $(x) on $(M) is not defined/unique, since the sum of the complex numbers is zero",
    )
    return s /= abs(s)
end

mid_point(M::Circle{ℝ}, p1, p2) = exp(M, p1, 0.5 * log(M, p1, p2))
mid_point(::Circle{ℂ}, p1::Complex, p2::Complex) = exp(im * (angle(p1) + angle(p2)) / 2)
mid_point(M::Circle{ℂ}, p1::StaticArray, p2::StaticArray) = Scalar(mid_point(M, p1[], p2[]))

@inline LinearAlgebra.norm(::Circle, p, X) = sum(abs, X)

number_of_coordinates(::Circle, ::AbstractBasis) = 1

@doc raw"""
    project(M::Circle{ℂ}, p)

Project a point `p` onto the complex [`Circle`](@ref) `M`, i.e. the unit circle in the complex plane.
"""
project(::Circle, ::Any)
project(::Circle{ℂ}, p::Number) = p / abs(p)
project!(::Circle{ℂ}, q, p) = copyto!(q, p / sum(abs.(p)))

@doc raw"""
    project(M::Circle, p, X)

Project a value `X` onto the tangent space of the point `p` on the [`Circle`](@ref) `M`.

For the complex valued case `X` is projected onto the line in the complex plane
that is parallel to the tangent to `p` on the unit circle and contains `0`.
"""
project(::Circle, ::Any, ::Any)
project(::Circle{ℂ}, p::Number, X::Number) = X - complex_dot(p, X) * p
project!(::Circle{ℂ}, Y, p, X) = (Y .= X - complex_dot(p, X) * p)

@doc raw"""
    Random.rand(M::Circle{ℝ}; vector_at = nothing, σ::Real=1.0)

If `vector_at` is `nothing`, return a random point on the [`Circle`](@ref) ``\mathbb S^1``
by picking a random element from ``[-\pi,\pi)`` uniformly.

If `vector_at` is not `nothing`, return a random tangent vector from the tangent space of
the point `vector_at` on the [`Circle`](@ref) by using a normal distribution with
mean 0 and standard deviation `σ`.
"""
function Random.rand(M::Circle; vector_at = nothing, σ::Real = 1.0)
    return rand(Random.default_rng(), M; vector_at = vector_at, σ = σ)
end
function Random.rand(rng::AbstractRNG, ::Circle{ℝ}; vector_at = nothing, σ::Real = 1.0)
    if vector_at === nothing
        return sym_rem(rand(rng) * 2 * π)
    else
        return map(_ -> σ * randn(rng), vector_at)
    end
end
function Random.rand(rng::AbstractRNG, M::Circle{ℂ}; vector_at = nothing, σ::Real = 1.0)
    if vector_at === nothing
        return sign(randn(rng, ComplexF64))
    else
        # written like that to properly handle `vector_at` being a number or a one-element array
        return map(p -> project(M, p, σ * randn(rng, complex(typeof(p)))), vector_at)
    end
end

function Random.rand!(
        rng::AbstractRNG,
        M::Circle,
        pX;
        vector_at = nothing,
        σ::Real = one(real(eltype(pX))),
    )
    pX .= rand(rng, M; vector_at, σ)
    return pX
end

retract(M::Circle, p, q) = retract(M, p, q, ExponentialRetraction())
retract(M::Circle, p, q, m::ExponentialRetraction) = exp(M, p, q)

representation_size(::Circle) = ()

Base.show(io::IO, ::Circle{𝔽}) where {𝔽} = print(io, "Circle($(𝔽))")

@doc raw"""
    sym_rem(x,[T=π])

Compute symmetric remainder of `x` with respect to the interall 2*`T`, i.e.
`(x+T)%2T`, where the default for `T` is ``π``
"""
function sym_rem(x::N, T = π) where {N <: Number}
    return (x ≈ T ? convert(N, -T) : rem(x, convert(N, 2 * T), RoundNearest))
end
sym_rem(x, T = π) = map(sym_rem, x, Ref(T))

@doc raw"""
     parallel_transport_to(M::Circle, p, X, q)

Compute the parallel transport of `X` from the tangent space at `p` to the tangent space at
`q` on the [`Circle`](@ref) `M`.
For the real-valued case this results in the identity.
For the complex-valued case, the formula is the same as for the [`Sphere`](@ref)`(1)` in the
complex plane.
````math
\mathcal P_{q←p} X = X - \frac{⟨\log_p q,X⟩_p}{d^2_{ℂ}(p,q)}
\bigl(\log_p q + \log_q p \bigr),
````
where [`log`](@extref `Base.log-Tuple{AbstractManifold, Any, Any}`) denotes the logarithmic map on `M`.
"""
parallel_transport_to(::Circle, ::Any, ::Any, ::Any)

parallel_transport_to(::Circle{ℝ}, p::Real, X::Real, q::Real) = X
function parallel_transport_to(M::Circle{ℂ}, p::Number, X::Number, q::Number)
    X_pq = log(M, p, q)
    Xnorm = norm(M, p, X_pq)
    Y = X
    if Xnorm > 0
        factor = 2 * complex_dot(X, q) / (abs(p + q)^2)
        Y -= factor .* (p + q)
    end
    return Y
end

parallel_transport_to!(::Circle{ℝ}, Y, p, X, q) = (Y .= X[])
function parallel_transport_to!(M::Circle{ℂ}, Y, p, X, q)
    X_pq = log(M, p, q)
    Xnorm = norm(M, p, X_pq)
    Y .= X
    if Xnorm > 0
        factor = 2 * complex_dot(X, q) / (sum(abs.(p + q) .^ 2))
        Y .-= factor .* (p + q)
    end
    return Y
end

# dispatch before allocation
function _vector_transport_direction(M::Circle, p, X, d, ::ParallelTransport)
    return parallel_transport_to(M, p, X, exp(M, p, d))
end

"""
    volume_density(::Circle, p, X)

Return volume density of [`Circle`](@ref), i.e. 1.
"""
volume_density(::Circle, p, X) = one(eltype(X))

zero_vector(::Circle, p::T) where {T <: Number} = zero(p)
zero_vector!(::Circle, X, p) = fill!(X, 0)
