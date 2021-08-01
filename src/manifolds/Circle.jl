@doc raw"""
    Circle{𝔽} <: AbstractManifold{𝔽}

The circle $𝕊^1$ is a manifold here represented by
real-valued points in $[-π,π)$ or complex-valued points $z ∈ ℂ$ of absolute value
$\lvert z\rvert = 1$.
# Constructor

    Circle(𝔽=ℝ)

Generate the `ℝ`-valued Circle represented by angles, which
alternatively can be set to use the [`AbstractNumbers`](@ref) `𝔽=ℂ` to obtain the circle
represented by `ℂ`-valued circle of unit numbers.
"""
struct Circle{𝔽} <: AbstractManifold{𝔽} end

Circle(𝔽::AbstractNumbers=ℝ) = Circle{𝔽}()

@doc raw"""
    check_point(M::Circle, p)

Check whether `p` is a point on the [`Circle`](@ref) `M`.
For the real-valued case, `p` is an angle and hence it checks that $p  ∈ [-π,π)$.
for the complex-valued case, it is a unit number, $p ∈ ℂ$ with $\lvert p \rvert = 1$.
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
            abs(p),
            "The point $(p) does not lie on the $(M) since its norm is not 1.",
        )
    end
    return nothing
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
function check_vector(M::Circle{ℂ}, p, X; kwargs...)
    if !isapprox(abs(complex_dot(p, X)), 0.0; kwargs...)
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
distance(::Circle{ℂ}, p, q) = acos(clamp(complex_dot(p, q), -1, 1))

@doc raw"""
    exp(M::Circle, p, X)

Compute the exponential map on the [`Circle`](@ref).
````math
\exp_p X = (p+X)_{2π},
````
where $(\cdot)_{2π}$ is the (symmetric) remainder with respect to division by $2π$, i.e. in $[-π,π)$.

For the complex-valued case, the same formula as for the [`Sphere`](@ref) $𝕊^1$ is applied to values in the
complex plane.
"""
exp(::Circle, ::Any...)
Base.exp(::Circle{ℝ}, p::Real, X::Real) = sym_rem(p + X)
function Base.exp(M::Circle{ℂ}, p::Number, X::Number)
    θ = norm(M, p, X)
    return cos(θ) * p + usinc(θ) * X
end

exp!(::Circle{ℝ}, q, p, X) = (q .= sym_rem(p + X))
function exp!(M::Circle{ℂ}, q, p, X)
    θ = norm(M, p, X)
    q .= cos(θ) * p + usinc(θ) * X
    return q
end

function get_basis(::Circle{ℝ}, p, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.frame_direction[])
    vs = @SVector [@SVector [sbv == 0 ? one(sbv) : sbv]]
    return CachedBasis(B, (@SVector [0]), vs)
end

get_coordinates(::Circle{ℝ}, p, X, ::AbstractBasis{<:Any,TangentSpaceType}) = X
get_coordinates(::Circle{ℝ}, p, X, ::DefaultOrthonormalBasis{<:Any,TangentSpaceType}) = X
function get_coordinates(M::Circle{ℝ}, p, X, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.frame_direction[])
    return X .* (sbv == 0 ? one(sbv) : sbv)
end
"""
    get_coordinates(M::Circle{ℂ}, p, X, B::DefaultOrthonormalBasis)

Return tangent vector coordinates in the Lie algebra of the [`Circle`](@ref).
"""
function get_coordinates(
    ::Circle{ℂ},
    p,
    X,
    ::DefaultOrthonormalBasis{<:Any,TangentSpaceType},
)
    X, p = X[1], p[1]
    Xⁱ = imag(X) * real(p) - real(X) * imag(p)
    return @SVector [Xⁱ]
end

function get_coordinates!(
    M::Circle,
    Y::AbstractArray,
    p,
    X,
    B::DefaultOrthonormalBasis{<:Any,TangentSpaceType},
)
    Y[] = get_coordinates(M, p, X, B)[]
    return Y
end
function get_coordinates!(
    M::Circle,
    Y::AbstractArray,
    p,
    X,
    B::DiagonalizingOrthonormalBasis,
)
    Y[] = get_coordinates(M, p, X, B)[]
    return Y
end

eval(
    quote
        @invoke_maker 1 AbstractManifold get_coordinates!(
            M::Circle,
            Y::AbstractArray,
            p,
            X,
            B::VeeOrthogonalBasis,
        )
    end,
)

get_vector(::Circle{ℝ}, p, X, ::AbstractBasis{ℝ,TangentSpaceType}) = X
get_vector(::Circle{ℝ}, p, X, ::DefaultOrthonormalBasis{ℝ,TangentSpaceType}) = X
function get_vector(::Circle{ℝ}, p, X, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.frame_direction[])
    return X .* (sbv == 0 ? one(sbv) : sbv)
end
"""
    get_vector(M::Circle{ℂ}, p, X, B::DefaultOrthonormalBasis)

Return tangent vector from the coordinates in the Lie algebra of the [`Circle`](@ref).
"""
function get_vector(::Circle{ℂ}, p, X, ::AbstractBasis{<:Any,TangentSpaceType})
    @SVector [1im * X[1] * p[1]]
end
eval(
    quote
        @invoke_maker 4 AbstractBasis get_vector(
            M::Circle{ℂ},
            p,
            X,
            B::DefaultOrthonormalBasis{<:Any,TangentSpaceType},
        )
    end,
)

for BT in [AbstractBasis{<:Any,TangentSpaceType}]
    eval(quote
        function get_vector!(::Circle{ℝ}, Y::AbstractArray, p, X, ::$BT)
            Y[] = X[]
            return Y
        end
    end)
    eval(quote
        function get_vector!(::Circle{ℂ}, Y::AbstractArray, p, X, ::$BT)
            Y[] = 1im * X[1] * p[1]
            return Y
        end
    end)
end
for BT in ManifoldsBase.DISAMBIGUATION_BASIS_TYPES, CT in [Circle, Circle{ℝ}, Circle{ℂ}]
    eval(
        quote
            @invoke_maker 5 $(supertype(BT)) get_vector!(
                M::$CT,
                Y::AbstractArray,
                p,
                X,
                B::$BT,
            )
        end,
    )
end

@doc raw"""
    injectivity_radius(M::Circle[, p])

Return the injectivity radius on the [`Circle`](@ref) `M`, i.e. $π$.
"""
injectivity_radius(::Circle) = π
injectivity_radius(::Circle, ::ExponentialRetraction) = π
injectivity_radius(::Circle, ::Any) = π
injectivity_radius(::Circle, ::Any, ::ExponentialRetraction) = π
eval(
    quote
        @invoke_maker 1 AbstractManifold injectivity_radius(
            M::Circle,
            rm::AbstractRetractionMethod,
        )
    end,
)

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

function inverse_retract(M::Circle, p::Number, q::Number)
    return inverse_retract(M, p, q, LogarithmicInverseRetraction())
end
function inverse_retract(M::Circle, p::Number, q::Number, ::LogarithmicInverseRetraction)
    return log(M, p, q)
end

@doc raw"""
    log(M::Circle, p, q)

Compute the logarithmic map on the [`Circle`](@ref) `M`.
````math
\log_p q = (q-p)_{2π},
````
where $(\cdot)_{2π}$ is the (symmetric) remainder with respect to division by $2π$, i.e. in $[-π,π)$.

For the complex-valued case, the same formula as for the [`Sphere`](@ref) $𝕊^1$ is applied to values in the
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
i.e. $\dim(𝕊^1) = 1$.
"""
manifold_dimension(::Circle) = 1

@doc raw"""
    mean(M::Circle{ℝ}, x::AbstractVector[, w::AbstractWeights])

Compute the Riemannian [`mean`](@ref mean(M::AbstractManifold, args...)) of `x` of points on
the [`Circle`](@ref) $𝕊^1$, reprsented by real numbers, i.e. the angular mean
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
the [`Circle`](@ref) $𝕊^1$, reprsented by complex numbers, i.e. embedded in the complex plade.
Comuting the sum
````math
s = \sum_{i=1}^n x_i
````
the mean is the angle of the complex number $s$, so represented in the complex plane as
$\frac{s}{\lvert s \rvert}$, whenever $s \neq 0$.

If the sum $s=0$, the mean is not unique. For example for opposite points or equally spaced
angles.
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
mid_point(M::Circle{ℂ}, p1::StaticArray, p2::StaticArray) = SA[mid_point(M, p1[], p2[])]

@inline LinearAlgebra.norm(::Circle, p, X) = sum(abs, X)

number_of_coordinates(::Circle, ::AbstractBasis) = 1

@doc raw"""
    project(M::Circle, p)

Project a point `p` onto the [`Circle`](@ref) `M`.
For the real-valued case this is the remainder with respect to modulus $2π$.
For the complex-valued case the result is the projection of `p` onto the unit circle in the
complex plane.
"""
project(::Circle, ::Any)
project(::Circle{ℝ}, p::Real) = sym_rem(p)
project(::Circle{ℂ}, p::Number) = p / abs(p)

project!(::Circle{ℝ}, q, p) = copyto!(q, sym_rem(p))
project!(::Circle{ℂ}, q, p) = copyto!(q, p / sum(abs.(p)))

@doc raw"""
    project(M::Circle, p, X)

Project a value `X` onto the tangent space of the point `p` on the [`Circle`](@ref) `M`.

For the real-valued case this is just the identity.
For the complex valued case `X` is projected onto the line in the complex plane
that is parallel to the tangent to `p` on the unit circle and contains `0`.
"""
project(::Circle, ::Any, ::Any)
project(::Circle{ℝ}, p::Real, X::Real) = X
project(::Circle{ℂ}, p::Number, X::Number) = X - complex_dot(p, X) * p

project!(::Circle{ℝ}, Y, p, X) = (Y .= X)
project!(::Circle{ℂ}, Y, p, X) = (Y .= X - complex_dot(p, X) * p)

retract(M::Circle, p, q) = retract(M, p, q, ExponentialRetraction())
retract(M::Circle, p, q, m::ExponentialRetraction) = exp(M, p, q)

representation_size(::Circle) = ()

Base.show(io::IO, ::Circle{𝔽}) where {𝔽} = print(io, "Circle($(𝔽))")

@doc raw"""
    sym_rem(x,[T=π])

Compute symmetric remainder of `x` with respect to the interall 2*`T`, i.e.
`(x+T)%2T`, where the default for `T` is $π$
"""
function sym_rem(x::N, T=π) where {N<:Number}
    return (x ≈ T ? convert(N, -T) : rem(x, convert(N, 2 * T), RoundNearest))
end
sym_rem(x, T=π) where {N} = map(sym_rem, x, Ref(T))

@doc raw"""
    vector_transport_to(M::Circle, p, X, q, ::ParallelTransport)

Compute the parallel transport of `X` from the tangent space at `p` to the tangent space at
`q` on the [`Circle`](@ref) `M`.
For the real-valued case this results in the identity.
For the complex-valud case, the formula is the same as for the [`Sphere`](@ref)`(1)` in the
complex plane.
````math
\mathcal P_{q←p} X = X - \frac{⟨\log_p q,X⟩_p}{d^2_{ℂ}(p,q)}
\bigl(\log_p q + \log_q p \bigr),
````
where [`log`](@ref) denotes the logarithmic map on `M`.
"""
vector_transport_to(::Circle, ::Any, ::Any, ::Any, ::ParallelTransport)
vector_transport_to(::Circle{ℝ}, p::Real, X::Real, q::Real, ::ParallelTransport) = X
function vector_transport_to(
    M::Circle{ℂ},
    p::Number,
    X::Number,
    q::Number,
    ::ParallelTransport,
)
    X_pq = log(M, p, q)
    Xnorm = norm(M, p, X_pq)
    Y = X
    if Xnorm > 0
        factor = 2 * complex_dot(X, q) / (abs(p + q)^2)
        Y -= factor .* (p + q)
    end
    return Y
end

vector_transport_to!(::Circle{ℝ}, Y, p, X, q, ::ParallelTransport) = (Y .= X)
function vector_transport_to!(M::Circle{ℂ}, Y, p, X, q, ::ParallelTransport)
    X_pq = log(M, p, q)
    Xnorm = norm(M, p, X_pq)
    Y .= X
    if Xnorm > 0
        factor = 2 * complex_dot(X, q) / (sum(abs.(p + q) .^ 2))
        Y .-= factor .* (p + q)
    end
    return Y
end

function vector_transport_direction(
    M::Circle,
    p::Number,
    X::Number,
    Y::Number,
    m::AbstractVectorTransportMethod,
)
    q = exp(M, p, Y)
    return vector_transport_to(M, p, X, q, m)
end

zero_vector(::Circle, p::Number) = zero(p)
zero_vector!(::Circle, X, p) = fill!(X, 0)
