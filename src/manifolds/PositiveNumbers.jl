@doc raw"""
    PositiveNumbers{𝔽} <: Manifold{𝔽}

The hyprebolic manifold of positive numbers $H^1$ is a the hyperbolic manifold either
represented by just positive numbers or the [Poiincaré half-plane model](https://en.wikipedia.org/wiki/Poincaré_half-plane_model).

# Constructor

    PositiveNumbers(𝔽=ℝ)

Generate the `ℝ`-valued hyperbolic model represented by positive numbers, which
alternatively can be set to use the half-plan model `𝔽=ℂ` of complex numbers with positive
imaginary part. The real-valued case can also be performed in 1-element arrays using [`SymmetricPositiveDefinite`](@ref)(1)
"""
struct PositiveNumbers{𝔽} <: Manifold{𝔽} end

PositiveNumbers(𝔽::AbstractNumbers = ℝ) = PositiveNumbers{𝔽}()

@doc raw"""
    check_manifold_point(M::PositiveNumbers, p)

Check whether `p` is a point on the [`PositiveNumbers`](@ref) `M`.
For the real-valued case, `x` is an angle and hence it checks that $p  ∈ [-π,π)$.
for the complex-valued case, it is a unit number, $p ∈ ℂ$ with $\lvert p \rvert = 1$.
"""
check_manifold_point(::PositiveNumbers, ::Any...)

function check_manifold_point(M::PositiveNumbers{ℝ}, p; kwargs...)
    if p <= 0
        return DomainError(
            p,
            "The point $(p) does not lie on $(M), since its is nonpositive.",
        )
    end
    return nothing
end
function check_manifold_point(M::PositiveNumbers{ℂ}, p; kwargs...)
    if imag(p) <= 0
        return DomainError(
            imag(p),
            "The point $(p) does not lie on the $(M) since its imaginary part is nonpositive.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::PositiveNumbers, p, X; check_base_point, kwargs...)

Check whether `X` is a tangent vector in the tangent space of `p` on the
[`PositiveNumbers`](@ref) `M`.
For the real-valued case represented by positive numbers, all `X` are valid, since the tangent space is the whole real line.
For the complex-valued case `X` [...]

"""
check_tangent_vector(::PositiveNumbers, ::Any...; ::Any...)

function check_tangent_vector(M::PositiveNumbers{ℝ}, p, X; check_base_point = true, kwargs...)
    if check_base_point
        perr = check_manifold_point(M, p; kwargs...)
        return perr # if x is valid all v that are real numbers are valid
    end
    return nothing
end
function check_tangent_vector(M::PositiveNumbers{ℂ}, p, X; check_base_point = true, kwargs...)
    if check_base_point
        perr = check_manifold_point(M, p)
        perr === nothing || return perr
    end
    return nothing
end

@doc raw"""
    distance(M::PositiveNumbers, p, q)

Compute the distance on the [`PositiveNumbers`](@ref) `M`, which is
the absolute value of the symmetric remainder of `p` and `q` for the real-valued
case and the angle between both complex numbers in the Gaussian plane for the
complex-valued case.
"""
distance(::PositiveNumbers, ::Any...)
distance(::PositiveNumbers{ℝ}, p::Real, q::Real) = abs(log(p)-log(q))
function distance(::PositiveNumbers{ℂ}, p::Complex, q::Complex)
    return acosh(1 + ( (real(q)-real(p))^2 + (imag(q)-imag(p))^2)/(2*imag(q)*imag(p)) )
end

@doc raw"""
    exp(M::PositiveNumbers, p, X)

Compute the exponential map on the [`PositiveNumbers`](@ref).
```math
\exp_p X = p\operatorname{exp}(X/p),
```

For the complex-valued case, the same formula as for the [`Sphere`](@ref) $𝕊^1$ is applied to values in the
complex plane.
"""
exp(::PositiveNumbers, ::Any...)
Base.exp(::PositiveNumbers{ℝ}, p::Real, X::Real) = p*exp(X/p)
function Base.exp(M::PositiveNumbers{ℂ}, x::Number, v::Number)
    error("Stil Todo")
end

exp!(::PositiveNumbers{ℝ}, q, p, X) = (q .= p*exp(X/p))

flat(::PositiveNumbers, ::Number, X::TFVector) = FVector(CotangentSpace, X.data)

flat!(::PositiveNumbers, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

@doc raw"""
    injectivity_radius(M::PositiveNumbers[, p])

Return the injectivity radius on the [`PositiveNumbers`](@ref) `M`, i.e. $\infty$.
"""
injectivity_radius(::PositiveNumbers) = Inf
injectivity_radius(::PositiveNumbers, ::ExponentialRetraction) = Inf
injectivity_radius(::PositiveNumbers, ::Any) = Inf
injectivity_radius(::PositiveNumbers, ::Any, ::ExponentialRetraction) = Inf
eval(
    quote
        @invoke_maker 1 Manifold injectivity_radius(
            M::PositiveNumbers,
            rm::AbstractRetractionMethod,
        )
    end,
)

@doc raw"""
    inner(M::PositiveNumbers, p, X, Y)

Compute the inner product of the two tangent vectors `X,Y` from the tangent plane at `p` on
the [`PositiveNumbers`](@ref) `M` using the restriction of the metric from the embedding,
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
inner(::PositiveNumbers, ::Any...)
@inline inner(::PositiveNumbers{ℝ}, p, X, Y) = X * Y / p^2
@inline inner(::PositiveNumbers{ℝ}, p::Real, X::Real, Y::Real) = X * Y / p^2
@inline inner(::PositiveNumbers{ℂ}, p, X, Y) = (real(X)*real(Y) + imag(X)*imag(Y))/(imag(p)^2)

@doc raw"""
    log(M::PositiveNumbers, p, q)

Compute the logarithmic map on the [`PositiveNumbers`](@ref) `M`.
````math
\log_p q = p\log\frac{q}{p}
````
"""
log(::PositiveNumbers, ::Any...)
Base.log(::PositiveNumbers{ℝ}, p::Real, q::Real) = p*log(q/p)

log!(::PositiveNumbers{ℝ}, X, p, q) = (X .= p*log(q/p))

@doc raw"""
    manifold_dimension(M::PositiveNumbers)

Return the dimension of the [`PositiveNumbers`](@ref) `M`,
i.e. $\dim(H^1) = 1$ for the real-valued case and
$\dim(H^2) = 2$ for the half-plane model.
"""
manifold_dimension(::PositiveNumbers{ℝ}) = 1
manifold_dimension(::PositiveNumbers{ℂ}) = 2

mid_point(M::PositiveNumbers{ℝ}, p1, p2) = exp(M, p1, 0.5 * log(M, p1, p2))
mid_point(M::PositiveNumbers{ℂ}, p1::Complex, p2::Complex) = exp(M, p1, 0.5 * log(M, p1, p2))

@inline LinearAlgebra.norm(::PositiveNumbers, p, X) = sum(abs, X/p)

@doc raw"""
    project(M::PositiveNumbers, p, X)

Project a value `X` onto the tangent space of the point `p` on the [`PositiveNumbers`](@ref) `M`,
which is just the identity.
"""
project(::PositiveNumbers, ::Any, ::Any)
project(::PositiveNumbers{ℝ}, ::Real, X::Real) = X
project(::PositiveNumbers{ℂ}, ::Number, X::Number) = X

project!(::PositiveNumbers{ℝ}, Y, p, X) = (Y .= X)
project!(::PositiveNumbers{ℂ}, Y, p, X) = (Y .= X)

retract(M::PositiveNumbers, p, q) = retract(M, p, q, ExponentialRetraction())
retract(M::PositiveNumbers, p, q, ::ExponentialRetraction) = exp(M, p, q)

representation_size(::PositiveNumbers) = ()

sharp(::PositiveNumbers, ::Number, ξ::CoTFVector) = FVector(TangentSpace, ξ.data)

sharp!(::PositiveNumbers, X::TFVector, p, ξ::CoTFVector) = copyto!(X, ξ)

Base.show(io::IO, ::PositiveNumbers{𝔽}) where {𝔽} = print(io, "PositiveNumbers($(𝔽))")

@doc raw"""
    vector_transport_to(M::PositiveNumbers, p, X, q, ::ParallelTransport)

Compute the parallel transport of `X` from the tangent space at `p` to the tangent space at
`q` on the [`PositiveNumbers`](@ref) `M`.

````math
\mathcal P_{q\gets p}(X) = X*q/p.
````
"""
vector_transport_to(::PositiveNumbers, ::Any, ::Any, ::Any, ::ParallelTransport)
vector_transport_to(::PositiveNumbers{ℝ}, p::Real, X::Real, q::Real, ::ParallelTransport) = X*q/p
vector_transport_to!(::PositiveNumbers{ℝ}, Y, p, X, q, ::ParallelTransport) = (Y .= X * q / p )

function vector_transport_direction(
    M::PositiveNumbers,
    p::Number,
    X::Number,
    Y::Number,
    m::AbstractVectorTransportMethod,
)
    q = exp(M, p, Y)
    return vector_transport_to(M, p, X, q, m)
end

zero_tangent_vector(::PositiveNumbers, p::Number) = zero(p)
zero_tangent_vector!(::PositiveNumbers, X, p) = fill!(X, 0)
