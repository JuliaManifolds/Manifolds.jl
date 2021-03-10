@doc raw"""
    PositiveNumbers <: Manifold{ℝ}

The hyperbolic manifold of positive numbers $H^1$ is a the hyperbolic manifold represented
by just positive numbers.

# Constructor

    PositiveNumbers()

Generate the `ℝ`-valued hyperbolic model represented by positive positive numbers.
To use this with arrays (1-element arrays),
please use [`SymmetricPositiveDefinite`](@ref)`(1)`.
"""
struct PositiveNumbers <: Manifold{ℝ} end

"""
    PositiveVectors(n)

Generate the manifold of vectors with positive entries.
This manifold is modeled as a [`PowerManifold`](@ref) of [`PositiveNumbers`](@ref).
"""
PositiveVectors(n::Integer) = PositiveNumbers()^n

"""
    PositiveMatrices(m,n)

Generate the manifold of matrices with positive entries.
This manifold is modeled as a [`PowerManifold`](@ref) of [`PositiveNumbers`](@ref).
"""
PositiveMatrices(n::Integer, m::Integer) = PositiveNumbers()^(n, m)

"""
    PositiveArrays(n₁,n₂,...,nᵢ)

Generate the manifold of `i`-dimensional arrays with positive entries.
This manifold is modeled as a [`PowerManifold`](@ref) of [`PositiveNumbers`](@ref).
"""
PositiveArrays(n::Vararg{Int,I}) where {I} = PositiveNumbers()^(n)

@doc raw"""
    check_manifold_point(M::PositiveNumbers, p)

Check whether `p` is a point on the [`PositiveNumbers`](@ref) `M`, i.e. $p>0$.
"""
function check_manifold_point(M::PositiveNumbers, p; kwargs...)
    if any(p .<= 0.0)
        return DomainError(
            p,
            "The point $(p) does not lie on $(M), since it is nonpositive.",
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
function check_tangent_vector(M::PositiveNumbers, p, X; check_base_point=true, kwargs...)
    if check_base_point
        perr = check_manifold_point(M, p; kwargs...)
        return perr # if x is valid all v that are real numbers are valid
    end
    return nothing
end

@doc raw"""
    distance(M::PositiveNumbers, p, q)

Compute the distance on the [`PositiveNumbers`](@ref) `M`, which is

````math
d(p,q) = \Bigl\lvert \log \frac{p}{q} \Bigr\rvert = \lvert \log p - \log q\rvert.
````
"""
distance(::PositiveNumbers, ::Any, ::Any)

distance(::PositiveNumbers, p::Real, q::Real) = abs(log(p / q))

@doc raw"""
    exp(M::PositiveNumbers, p, X)

Compute the exponential map on the [`PositiveNumbers`](@ref) `M`.
```math
\exp_p X = p\operatorname{exp}(X/p).
```
"""
Base.exp(::PositiveNumbers, ::Any, ::Any)

Base.exp(::PositiveNumbers, p::Real, X::Real) = p * exp(X / p)

exp!(::PositiveNumbers, q, p, X) = (q .= p .* exp.(X ./ p))

flat(::PositiveNumbers, ::Real, X::TFVector) = FVector(CotangentSpace, X.data)

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
the [`PositiveNumbers`](@ref) `M`, i.e.

````math
g_p(X,Y) = \frac{XY}{p^2}.
````
"""
inner(::PositiveNumbers, ::Any...)
@inline inner(::PositiveNumbers, p::Real, X::Real, Y::Real) = X * Y / p^2
function inverse_retract(M::PositiveNumbers, x, y)
    return inverse_retract(M, x, y, LogarithmicInverseRetraction())
end
function inverse_retract(M::PositiveNumbers, x, y, ::LogarithmicInverseRetraction)
    return log(M, x, y)
end

@doc raw"""
    log(M::PositiveNumbers, p, q)

Compute the logarithmic map on the [`PositiveNumbers`](@ref) `M`.
````math
\log_p q = p\log\frac{q}{p}.
````
"""
Base.log(::PositiveNumbers, ::Any, ::Any)
Base.log(::PositiveNumbers, p::Real, q::Real) = p * log(q / p)

log!(::PositiveNumbers, X, p, q) = (X .= p .* log.(q ./ p))

@doc raw"""
    manifold_dimension(M::PositiveNumbers)

Return the dimension of the [`PositiveNumbers`](@ref) `M`,
i.e. of the 1-dimensional hyperbolic space,

````math
\dim(H^1) = 1
````
"""
manifold_dimension(::PositiveNumbers) = 1

mid_point(M::PositiveNumbers, p1, p2) = exp(M, p1, log(M, p1, p2) / 2)

@inline LinearAlgebra.norm(::PositiveNumbers, p, X) = sum(abs.(X ./ p))

@doc raw"""
    project(M::PositiveNumbers, p, X)

Project a value `X` onto the tangent space of the point `p` on the [`PositiveNumbers`](@ref) `M`,
which is just the identity.
"""
project(::PositiveNumbers, ::Any, ::Any)
project(::PositiveNumbers, ::Real, X::Real) = X

project!(::PositiveNumbers, Y, p, X) = (Y .= X)

retract(M::PositiveNumbers, p, q) = retract(M, p, q, ExponentialRetraction())
retract(M::PositiveNumbers, p, q, ::ExponentialRetraction) = exp(M, p, q)

representation_size(::PositiveNumbers) = ()

sharp(::PositiveNumbers, ::Real, ξ::CoTFVector) = FVector(TangentSpace, ξ.data)

sharp!(::PositiveNumbers, X::TFVector, p, ξ::CoTFVector) = copyto!(X, ξ)

Base.show(io::IO, ::PositiveNumbers) = print(io, "PositiveNumbers()")

function Base.show(
    io::IO,
    ::PowerManifold{ℝ,PositiveNumbers,TSize,ArrayPowerRepresentation},
) where {TSize}
    s = [TSize.parameters...]
    (length(s) == 1) && return print(io, "PositiveVectors($(s[1]))")
    (length(s) == 2) && return print(io, "PositiveMatrices($(s[1]), $(s[2]))")
    return print(io, "PositiveArrays($(join(s, ", ")))")
end

@doc raw"""
    vector_transport_to(M::PositiveNumbers, p, X, q, ::ParallelTransport)

Compute the parallel transport of `X` from the tangent space at `p` to the tangent space at
`q` on the [`PositiveNumbers`](@ref) `M`.

````math
\mathcal P_{q\gets p}(X) = X\cdot\frac{q}{p}.
````
"""
vector_transport_to(::PositiveNumbers, ::Any, ::Any, ::Any, ::ParallelTransport)
function vector_transport_to(
    ::PositiveNumbers,
    p::Real,
    X::Real,
    q::Real,
    ::ParallelTransport,
)
    return X * q / p
end

function vector_transport_to!(::PositiveNumbers, Y, p, X, q, ::ParallelTransport)
    return (Y .= X .* q ./ p)
end

function vector_transport_direction(
    M::PositiveNumbers,
    p,
    X,
    Y,
    m::AbstractVectorTransportMethod,
)
    q = exp(M, p, Y)
    return vector_transport_to(M, p, X, q, m)
end

zero_tangent_vector(::PositiveNumbers, p::Real) = zero(p)
zero_tangent_vector!(::PositiveNumbers, X, p) = fill!(X, 0)
