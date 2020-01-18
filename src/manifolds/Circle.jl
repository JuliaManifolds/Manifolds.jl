@doc doc"""
    Circle{F} <: Manifold

The circle $\mathbb S^1$ as a manifold ere manifold represented by
real-valued data in $[-\pi,\pi)$ or complex-valued data $z\in \mathbb C$ of absolute value
$\lvert z\rvert = 1$.
# Constructor

    Circle(f=ℝ)

Generate the `ℝ`-valued Circle represented by angles, which
alternatively can be set to use the [`AbstractNumbers`](@ref) `f=ℂ` to obtain the `Circle`
represented by `ℂ`-valued `Circle` of unit numbers.
"""
struct Circle{F} <: Manifold where {F<:AbstractNumbers} end

Circle(f::AbstractNumbers = ℝ) = Circle{f}()

@doc doc"""
    check_manifold_point(M::Circle, x)

Check whether `x` is a point on the [`Circle`](@ref) `M`.
For the real-valued case, `x` is an angle and hence it checks that $x \in [-\pi,\pi)$.
for the complex-valued case its a unit number, $x \in \mathbb C$ with $\lvert x \rvert = 1$.
"""
check_manifold_point(::Circle, ::Any...)

function check_manifold_point(M::Circle{ℝ}, x; kwargs...)
    if !isapprox(sym_rem(x), x; kwargs...)
        return DomainError(
            x,
            "The point $(x) does not lie on $(M), since its is not in [-π,π).",
        )
    end
    return nothing
end
function check_manifold_point(M::Circle{ℂ}, x; kwargs...)
    if !isapprox(sum(abs.(x)), 1.0; kwargs...)
        return DomainError(
            abs(x),
            "The point $(x) does not lie on the $(M) since its norm is not 1.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::Circle, x, v)

Check whether `v` is a tangent vector in the tangent space of `x` on the
[`Circle`](@ref) `M`.
For the real-valued case represented by angles all `v` are valid, since the tangent space is
the whole real line.
For the complex-valued case `v` has to lie on the line parallel to the tangent line at `x`
in the complex plane, i.e. the inner product is zero.
"""
check_tangent_vector(::Circle{ℝ}, ::Any...; ::Any...)

function check_tangent_vector(M::Circle{ℝ}, x, v; kwargs...)
    perr = check_manifold_point(M, x)
    return perr # if x is valid all v that are real numbers are valid
end
function check_tangent_vector(M::Circle{ℂ}, x, v; kwargs...)
    perr = check_manifold_point(M, x)
    perr === nothing || return perr
    if !isapprox(abs(complex_dot(x, v)), 0.0; kwargs...)
        return DomainError(
            abs(complex_dot(x, v)),
            "The value $(v) is not a tangent vector to $(x) on $(M), since it is not orthogonal in the embedding.",
        )
    end
    return nothing
end

@doc doc"""
    complex_dot(a, b)

Compute the inner product of two (complex) numbers with in the complex plane.
"""
complex_dot(a, b) = dot(map(real, a), map(real, b)) + dot(map(imag, a), map(imag, b))
complex_dot(a::Number, b::Number) = (real(a) * real(b) + imag(a) * imag(b))

@doc doc"""
    distance(M::Circle, x, y)

Compute the distance on the [`Circle`](@ref) `M`, which is
the absolute value of the symmetric remainder of `x` and `y` for the real-valued
case and the angle between both complex numbers in the Gaussian plane for the
complex-valued case.
"""
distance(::Circle, ::Any...)
distance(::Circle{ℝ}, x::Real, y::Real) = abs(sym_rem(x - y))
distance(::Circle{ℝ}, x, y) = abs(sum(sym_rem.(x - y)))
distance(::Circle{ℂ}, x, y) = acos(clamp(complex_dot(x, y), -1, 1))

@doc doc"""
    exp(M::Circle, x, v)

Compute the exponential map on the [`Circle`](@ref).
````math
\exp_xv = (x+v)_{2\pi},
````
where $(\cdot)$ is the (symmetric) remainder with respect to division by $2\pi$,
i.e. in $[-\pi,\pi)$.

For the complex-valued case the formula is the same as for the [`Sphere`](@ref)
applied to valuedin the complex plane.
"""
exp(::Circle, ::Any...)
exp(::Circle{ℝ}, x::Real, v::Real) = sym_rem(x + v)
function exp(M::Circle{ℂ}, x::Number, v::Number)
    θ = norm(M, x, v)
    return cos(θ) * x + usinc(θ) * v
end

exp!(::Circle{ℝ}, y, x, v) = (y .= sym_rem(x + v))
function exp!(M::Circle{ℂ}, y, x, v)
    θ = norm(M, x, v)
    y .= cos(θ) * x + usinc(θ) * v
    return y
end

flat(M::Circle, x::Number, w::TFVector) = FVector(CotangentSpace, w.data)

flat!(::Circle, v::CoTFVector, x, w::TFVector) = copyto!(v, w)

function get_basis(M::Circle{ℝ}, x, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.v[1])
    vs = @SVector [@SVector [sbv == 0 ? one(sbv) : sbv]]
    return PrecomputedDiagonalizingOrthonormalBasis(vs, @SVector [0])
end

get_coordinates(M::Circle{ℝ}, x, v, B::ArbitraryOrthonormalBasis) = v
function get_coordinates(M::Circle{ℝ}, x, v, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.v[1])
    return v .* (sbv == 0 ? 1 : sbv)
end
"""
    get_coordinates(M::Circle{ℂ}, x, v, B::ArbitraryOrthonormalBasis)

Return tangent vector coordinates in the Lie algebra of the circle.
"""
function get_coordinates(M::Circle{ℂ}, x, v, B::ArbitraryOrthonormalBasis)
    v, x = v[1], x[1]
    w = imag(v) * real(x) - real(v) * imag(x)
    return @SVector [w]
end

get_vector(M::Circle{ℝ}, x, v, B::ArbitraryOrthonormalBasis) = v
function get_vector(M::Circle{ℝ}, x, v, B::DiagonalizingOrthonormalBasis)
    sbv = sign(B.v[1])
    return v .* (sbv == 0 ? 1 : sbv)
end
"""
    get_vector(M::Circle{ℂ}, x, v, B::ArbitraryOrthonormalBasis)

Return tangent vector from the coordinates in the Lie algebra of the circle.
"""
get_vector(M::Circle{ℂ}, x, v, B::ArbitraryOrthonormalBasis) = @SVector [1im * v[1] * x[1]]

@doc doc"""
    injectivity_radius(M::Circle[, x])

Return the injectivity radius on the [`Circle`](@ref) `M`, i.e. $\pi$.
"""
injectivity_radius(::Circle, args...) = π

@doc doc"""
    inner(M::Circle, x, w, v)

Compute the inner product of the two tangent vectors `w,v` from the tangent plane at `x` on
the [`Circle`](@ref) `M` using the restriction of the metric from the embedding,
i.e.

````math
g_x(v,w) = w*v
````

for the real case and

    ````math
g_x(v,w) = v^\mathrm{T}w
````

for the complex case interpreting complex numbers in the Gaussian plane.
"""
inner(::Circle, ::Any...)
@inline inner(::Circle{ℝ}, x, w, v) = dot(v, w)
@inline inner(::Circle{ℝ}, x::Real, w::Real, v::Real) = v * w
@inline inner(::Circle{ℂ}, x, w, v) = complex_dot(w, v)

function inverse_retract(M::Circle, x::Number, y::Number)
    return inverse_retract(M, x, y, LogarithmicInverseRetraction())
end
function inverse_retract(M::Circle, x::Number, y::Number, ::LogarithmicInverseRetraction)
    return log(M, x, y)
end

@doc doc"""
    log(M::Circle, x, y)

Compute the logarithmic map on the [`Circle`](@ref) `M`.
````math
\exp_xv = (y,x)_{2\pi},
````
where $(\cdot)$ is the (symmetric) remainder with respect to division by $2\pi$,
i.e. in $[-\pi,\pi)$.

For the complex-valued case the formula is the same as for the [`Sphere`](@ref)
applied to valuedin the complex plane.
"""
log(::Circle, ::Any...)
log(::Circle{ℝ}, x::Real, y::Real) = sym_rem(y - x)
function log(M::Circle{ℂ}, x::Number, y::Number)
    cosθ = complex_dot(x, y)
    if cosθ ≈ -1  # appr. opposing points, return deterministic choice from set-valued log
        v = real(x) ≈ 1 ? 1im : 1 + 0im
        v = v - complex_dot(x, v) * x
        v *= π / norm(v)
    else
        cosθ = cosθ > 1 ? one(cosθ) : cosθ
        θ = acos(cosθ)
        v = (y - cosθ * x) / usinc(θ)
    end
    return project_tangent(M, x, v)
end

log!(::Circle{ℝ}, v, x, y) = (v .= sym_rem(y - x))
function log!(M::Circle{ℂ}, v, x, y)
    cosθ = complex_dot(x, y)
    if cosθ ≈ -1
        v .= sum(real.(x)) ≈ 1 ? 1.0im : 1.0 + 0.0im
        v .= v - complex_dot(x, v) * x
        v .*= π / norm(v)
    else
        cosθ = cosθ > 1 ? one(cosθ) : cosθ
        θ = acos(cosθ)
        v .= (y - cosθ * x) / usinc(θ)
    end
    return project_tangent!(M, v, x, v)
end

@doc doc"""
    manifold_dimension(M::Circle)

Return the dimension of the [`Circle`](@ref) `M`,
i.e. $\operatorname{dim}(\mathbb S^1) = 1$.
"""
manifold_dimension(::Circle) = 1

@doc doc"""
    mean(M::Circle, x::AbstractVector[, w::AbstractWeights])

Compute the Riemannian [`mean`](@ref mean(M::Manifold, args...)) of `x` on the
[`Circle`](@ref) $\mathbb S^1$ by the wrapped mean, i.e. the remainder of the
mean modulo 2π.
"""
mean(::Circle, ::Any)
mean(::Circle, x::Array{<:Real}, w::AbstractVector; kwargs...) = sym_rem(sum(w .* x))

@inline norm(::Circle, x, v) = sum(abs, v)

@doc doc"""
    project_point(M::Circle, x)

Project a point `x` onto the [`Circle`](@ref) `M`.
For the real-valued case this is the remainder with respect to modulus $2\pi$.
For the complex-valued case the result is the projection of `x` onto the unit circle in the
complex plane.
"""
project_point(::Circle, ::Any)
project_point(::Circle{ℝ}, x::Real) = sym_rem(x)
project_point(::Circle{ℂ}, x::Number) = x / abs(x)

project_point!(::Circle{ℝ}, x) = (x .= sym_rem(x))
project_point!(::Circle{ℂ}, x) = (x .= x / sum(abs.(x)))

@doc doc"""
    project_tangent(M::Circle, x, v)

Project a value `v` onto the tangent space of the point `x` on the [`Circle`](@ref) `M`.

For the real-valued case this is just the identity.
For the complex valued case `v` is projected onto the line in the complex plane
that is parallel to the tangent to `x` on the unit circle and contains `0`.
"""
project_tangent(::Circle{ℝ}, x::Real, v::Real) = v
project_tangent(::Circle{ℂ}, x::Number, v::Number) = v - complex_dot(x, v) * x

project_tangent!(::Circle{ℝ}, w, x, v) = (w .= v)
project_tangent!(::Circle{ℂ}, w, x, v) = (w .= v - complex_dot(x, v) * x)

retract(M::Circle, x, y) = retract(M, x, y, ExponentialRetraction())
retract(M::Circle, x, y, m::ExponentialRetraction) = exp(M, x, y)

representation_size(::Circle) = (1,)

sharp(M::Circle, x::Number, w::CoTFVector) = FVector(TangentSpace, w.data)

sharp!(M::Circle, v::TFVector, x, w::CoTFVector) = copyto!(v, w)

@doc doc"""
    sym_rem(x,[T=π])

Compute symmetric remainder of `x` with respect to the interall 2*`T`, i.e.
`(x+T)%2T`, where the default for `T` is $\pi$
"""
function sym_rem(x::N, T = π) where {N<:Number}
    return (x ≈ T ? convert(N, -T) : rem(x, convert(N, 2 * T), RoundNearest))
end
sym_rem(x, T = π) where {N} = sym_rem.(x, Ref(T))

@doc doc"""
    vector_transport_to(M::Circle, x, v, y, ::ParallelTransport)

Compute the parallel transport of `v` from the tangent space at `x` to the tangent space at
`y` on the [`Circle`](@ref) `M`.
For the real-valued case this results in the identity.
For the complex-valud case, the formula is the same as for the [`Sphere`](@ref)`(1)` in the
complex plane.
````math
P_{y\gets x}(v) = v - \frac{\langle \log_xy,v\rangle_x}{d^2_{\mathbb C}(x,y)}
\bigl(\log_xy + \log_yx \bigr),
````
where [`log`](@ref) denotes the logarithmic map on `M`.
"""
vector_transport_to(::Circle, ::Any, ::Any, ::Any, ::ParallelTransport)
vector_transport_to(::Circle{ℝ}, x::Real, v::Real, y::Real, ::ParallelTransport) = v
function vector_transport_to(
    M::Circle{ℂ},
    x::Number,
    v::Number,
    y::Number,
    ::ParallelTransport,
)
    v_xy = log(M, x, y)
    vl = norm(M, x, v_xy)
    vto = v
    if vl > 0
        factor = 2 * complex_dot(v, y) / (abs(x + y)^2)
        vto -= factor .* (x + y)
    end
    return vto
end

vector_transport_to!(::Circle{ℝ}, w, x, v, y, ::ParallelTransport) = (w .= v)
function vector_transport_to!(M::Circle{ℂ}, vto, x, v, y, ::ParallelTransport)
    v_xy = log(M, x, y)
    vl = norm(M, x, v_xy)
    vto .= v
    if vl > 0
        factor = 2 * complex_dot(v, y) / (sum(abs.(x + y) .^ 2))
        vto .-= factor .* (x + y)
    end
    return vto
end

function vector_transport_along(M::Circle, x::Number, v::Number, c)
    return vector_transport_along!(M, zero(v), x, v, c)
end

function vector_transport_direction(
    M::Circle,
    x::Number,
    v::Number,
    vdir::Number,
    m::AbstractVectorTransportMethod,
)
    y = exp(M, x, vdir)
    return vector_transport_to(M, x, v, y, m)
end

zero_tangent_vector(::Circle, x::Number) = zero(x)

zero_tangent_vector!(::Circle, v, x) = fill!(v, 0)
