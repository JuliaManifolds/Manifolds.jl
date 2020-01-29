@doc raw"""
    Circle{F} <: Manifold

The circle $𝕊^1$ as a manifold ere manifold represented by
real-valued data in $[-\pi,\pi)$ or complex-valued data $z ∈ ℂ$ of absolute value
$\lvert z\rvert = 1$.
# Constructor

    Circle(f=ℝ)

Generate the `ℝ`-valued Circle represented by angles, which
alternatively can be set to use the [`AbstractNumbers`](@ref) `f=ℂ` to obtain the `Circle`
represented by `ℂ`-valued `Circle` of unit numbers.
"""
struct Circle{F} <: Manifold where {F<:AbstractNumbers} end

Circle(f::AbstractNumbers = ℝ) = Circle{f}()

@doc raw"""
    check_manifold_point(M::Circle, p)

Check whether `p` is a point on the [`Circle`](@ref) `M`.
For the real-valued case, `x` is an angle and hence it checks that $p  ∈ [-\pi,\pi)$.
for the complex-valued case its a unit number, $p  ∈ ℂ$ with $\lvert p \rvert = 1$.
"""
check_manifold_point(::Circle, ::Any...)

function check_manifold_point(M::Circle{ℝ}, p; kwargs...)
    if !isapprox(sym_rem(p), p; kwargs...)
        return DomainError(
            p,
            "The point $(p) does not lie on $(M), since its is not in [-π,π).",
        )
    end
    return nothing
end
function check_manifold_point(M::Circle{ℂ}, p; kwargs...)
    if !isapprox(sum(abs.(p)), 1.0; kwargs...)
        return DomainError(
            abs(p),
            "The point $(p) does not lie on the $(M) since its norm is not 1.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::Circle, p, X)

Check whether `X` is a tangent vector in the tangent space of `p` on the
[`Circle`](@ref) `M`.
For the real-valued case represented by angles all `X` are valid, since the tangent space is the whole real line.
For the complex-valued case `X` has to lie on the line parallel to the tangent line at `p`
in the complex plane, i.e. their inner product has to be zero.
"""
check_tangent_vector(::Circle{ℝ}, ::Any...; ::Any...)

function check_tangent_vector(M::Circle{ℝ}, p, X; kwargs...)
    perr = check_manifold_point(M, p)
    return perr # if x is valid all v that are real numbers are valid
end
function check_tangent_vector(M::Circle{ℂ}, p, X; kwargs...)
    perr = check_manifold_point(M, p)
    perr === nothing || return perr
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
\exp_pX = (p+X)_{2\pi},
````
where $(\cdot)_{2\pi}$ is the (symmetric) remainder with respect to division by $2\pi$, i.e. in $[-\pi,\pi)$.

For the complex-valued case the formula is the same as for the [`Sphere`](@ref) $𝕊^1$ is applied, to values in the
complex plane.
"""
exp(::Circle, ::Any...)
exp(::Circle{ℝ}, p::Real, X::Real) = sym_rem(p + X)
function exp(M::Circle{ℂ}, x::Number, v::Number)
    θ = norm(M, x, v)
    return cos(θ) * x + usinc(θ) * v
end

exp!(::Circle{ℝ}, q, p, X) = (q .= sym_rem(p + X))
function exp!(M::Circle{ℂ}, q, p, X)
    θ = norm(M, p, X)
    q .= cos(θ) * p + usinc(θ) * X
    return q
end

flat(M::Circle, p::Number, X::TFVector) = FVector(CotangentSpace, X.data)

flat!(::Circle, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

function get_basis(M::Circle{ℝ}, p, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.v[1])
    vs = @SVector [@SVector [sbv == 0 ? one(sbv) : sbv]]
    return PrecomputedDiagonalizingOrthonormalBasis(vs, @SVector [0])
end

get_coordinates(M::Circle{ℝ}, p, X, B::ArbitraryOrthonormalBasis) = X
function get_coordinates(M::Circle{ℝ}, p, X, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.v[1])
    return X .* (sbv == 0 ? 1 : sbv)
end
"""
    get_coordinates(M::Circle{ℂ}, x, v, B::ArbitraryOrthonormalBasis)

Return tangent vector coordinates in the Lie algebra of the circle.
"""
function get_coordinates(M::Circle{ℂ}, p, X, B::ArbitraryOrthonormalBasis)
    X, p = X[1], p[1]
    w = imag(X) * real(p) - real(X) * imag(p)
    return @SVector [w]
end

get_vector(M::Circle{ℝ}, p, X, B::ArbitraryOrthonormalBasis) = X
function get_vector(M::Circle{ℝ}, p, X, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.v[1])
    return X .* (sbv == 0 ? 1 : sbv)
end
"""
    get_vector(M::Circle{ℂ}, p, X, B::ArbitraryOrthonormalBasis)

Return tangent vector from the coordinates in the Lie algebra of the circle.
"""
get_vector(M::Circle{ℂ}, p, X, B::ArbitraryOrthonormalBasis) = @SVector [1im * X[1] * p[1]]

@doc raw"""
    injectivity_radius(M::Circle[, p])

Return the injectivity radius on the [`Circle`](@ref) `M`, i.e. $\pi$.
"""
injectivity_radius(::Circle, args...) = π

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

function inverse_retract(M::Circle, x::Number, y::Number)
    return inverse_retract(M, x, y, LogarithmicInverseRetraction())
end
function inverse_retract(M::Circle, x::Number, y::Number, ::LogarithmicInverseRetraction)
    return log(M, x, y)
end

@doc raw"""
    log(M::Circle, p, q)

Compute the logarithmic map on the [`Circle`](@ref) `M`.
````math
\log_p q = (q-p)_{2\pi},
````
where $(\cdot)_{2\pi}$ is the (symmetric) remainder with respect to division by $2\pi$, i.e. in $[-\pi,\pi)$.

For the complex-valued case the formula is the same as for the [`Sphere`](@ref) $𝕊^1$ is applied, to values in the
complex plane.
"""
log(::Circle, ::Any...)
log(::Circle{ℝ}, p::Real, q::Real) = sym_rem(q - p)
function log(M::Circle{ℂ}, p::Number, q::Number)
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
    return project_tangent(M, p, X)
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
    return project_tangent!(M, X, p, X)
end

@doc raw"""
    manifold_dimension(M::Circle)

Return the dimension of the [`Circle`](@ref) `M`,
i.e. $\operatorname{dim}(𝕊^1) = 1$.
"""
manifold_dimension(::Circle) = 1

@doc raw"""
    mean(M::Circle, x::AbstractVector[, w::AbstractWeights])

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` of points on the [`Circle`](@ref) $𝕊^1$,
which is computed with wrapped mean, i.e. the remainder of the mean modulo 2π.
"""
mean(::Circle, ::Any)
mean(::Circle, x::Array{<:Real}; kwargs...) = sym_rem(sum(x))
mean(::Circle, x::Array{<:Real}, w::AbstractVector; kwargs...) = sym_rem(sum(w .* x))

@inline norm(::Circle, p, X) = sum(abs, X)

@doc raw"""
    project_point(M::Circle, p)

Project a point `p` onto the [`Circle`](@ref) `M`.
For the real-valued case this is the remainder with respect to modulus $2\pi$.
For the complex-valued case the result is the projection of `p` onto the unit circle in the
complex plane.
"""
project_point(::Circle, ::Any)
project_point(::Circle{ℝ}, p::Real) = sym_rem(p)
project_point(::Circle{ℂ}, p::Number) = p / abs(p)

project_point!(::Circle{ℝ}, p) = (p .= sym_rem(p))
project_point!(::Circle{ℂ}, p) = (p .= p / sum(abs.(p)))

@doc raw"""
    project_tangent(M::Circle, p, X)

Project a value `X` onto the tangent space of the point `p` on the [`Circle`](@ref) `M`.

For the real-valued case this is just the identity.
For the complex valued case `X` is projected onto the line in the complex plane
that is parallel to the tangent to `p` on the unit circle and contains `0`.
"""
project_tangent(::Circle{ℝ}, p::Real, X::Real) = X
project_tangent(::Circle{ℂ}, p::Number, X::Number) = X - complex_dot(p, X) * p

project_tangent!(::Circle{ℝ}, Y, p, X) = (Y .= X)
project_tangent!(::Circle{ℂ}, Y, p, X) = (Y .= X - complex_dot(p, X) * p)

retract(M::Circle, p, q) = retract(M, p, q, ExponentialRetraction())
retract(M::Circle, p, q, m::ExponentialRetraction) = exp(M, p, q)

representation_size(::Circle) = ()

sharp(M::Circle, p::Number, ξ::CoTFVector) = FVector(TangentSpace, ξ.data)

sharp!(M::Circle, X::TFVector, p, ξ::CoTFVector) = copyto!(X, ξ)

@doc raw"""
    sym_rem(x,[T=π])

Compute symmetric remainder of `x` with respect to the interall 2*`T`, i.e.
`(x+T)%2T`, where the default for `T` is $\pi$
"""
function sym_rem(x::N, T = π) where {N<:Number}
    return (x ≈ T ? convert(N, -T) : rem(x, convert(N, 2 * T), RoundNearest))
end
sym_rem(x, T=π) where N = map(sym_rem, x, Ref(T))

@doc raw"""
    vector_transport_to(M::Circle, p, X, q, ::ParallelTransport)

Compute the parallel transport of `X` from the tangent space at `p` to the tangent space at
`q` on the [`Circle`](@ref) `M`.
For the real-valued case this results in the identity.
For the complex-valud case, the formula is the same as for the [`Sphere`](@ref)`(1)` in the
complex plane.
````math
𝒫_{q←p}(v) = X - \frac{⟨\log_p q,X⟩_p}{d^2_{ℂ}(p,q)}
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
    v_xy = log(M, p, q)
    vl = norm(M, p, v_xy)
    Y = X
    if vl > 0
        factor = 2 * complex_dot(X, q) / (abs(p + q)^2)
        Y -= factor .* (p + q)
    end
    return Y
end

vector_transport_to!(::Circle{ℝ}, Y, p, X, q, ::ParallelTransport) = (Y .= X)
function vector_transport_to!(M::Circle{ℂ}, Y, p, X, q, ::ParallelTransport)
    v_xy = log(M, p, q)
    vl = norm(M, p, v_xy)
    Y .= X
    if vl > 0
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
    y = exp(M, p, Y)
    return vector_transport_to(M, p, X, y, m)
end

zero_tangent_vector(::Circle, p::Number) = zero(p)
zero_tangent_vector!(::Circle, X, p) = fill!(X, 0)
