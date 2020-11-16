@doc raw"""
    usinc(θ::Real)

Unnormalized version of `sinc` function, i.e. $\operatorname{usinc}(θ) = \frac{\sin(θ)}{θ}$.
This is equivalent to `sinc(θ/π)`.
"""
@inline usinc(θ::Real) = θ == 0 ? one(θ) : isinf(θ) ? zero(θ) : sin(θ) / θ

@doc raw"""
    usinc_from_cos(x::Real)

Unnormalized version of `sinc` function, i.e. $\operatorname{usinc}(θ) = \frac{\sin(θ)}{θ}$,
computed from $x = cos(θ)$.
"""
@inline function usinc_from_cos(x::Real)
    return if x >= 1
        one(x)
    elseif x <= -1
        zero(x)
    else
        sqrt(1 - x^2) / acos(x)
    end
end

@doc raw"""
    nzsign(z[, absz])

Compute a modified `sign(z)` that is always nonzero, i.e. where
````math
\operatorname(nzsign)(z) = \begin{cases}
    1 & \text{if } z = 0\\
    \frac{z}{|z|} & \text{otherwise}
\end{cases}
````
"""
@inline function nzsign(z, absz = abs(z))
    psignz = z / absz
    return ifelse(iszero(absz), one(psignz), psignz)
end

allocate(p, s::Size{S}) where {S} = similar(p, S...)
allocate(p::StaticArray, s::Size{S}) where {S} = similar(p, maybesize(s))
allocate(p, ::Type{T}, s::Size{S}) where {S,T} = similar(p, T, S...)
allocate(p::StaticArray, ::Type{T}, s::Size{S}) where {S,T} = similar(p, T, maybesize(s))

"""
    eigen_safe(x)

Compute the eigendecomposition of `x`. If `x` is a `StaticMatrix`, it is
converted to a `Matrix` before the decomposition.
"""
@inline eigen_safe(x; kwargs...) = eigen(x; kwargs...)
@inline function eigen_safe(x::StaticMatrix; kwargs...)
    s = size(x)
    E = eigen!(Matrix(parent(x)); kwargs...)
    return Eigen(SizedVector{s[1]}(E.values), SizedMatrix{s...}(E.vectors))
end

"""
    log_safe(x)

Compute the matrix logarithm of `x`. If `x` is a `StaticMatrix`, it is
converted to a `Matrix` before computing the log.
"""
@inline log_safe(x) = log(x)
@inline function log_safe(x::StaticMatrix)
    s = Size(x)
    return SizedMatrix{s[1],s[2]}(log(Matrix(parent(x))))
end

"""
    mul!_safe(Y, A, B) -> Y

Call `mul!` safely, that is, `A` and/or `B` are permitted to alias with `Y`.
"""
mul!_safe(Y, A, B) = (Y === A || Y === B) ? copyto!(Y, A * B) : mul!(Y, A, B)

@doc raw"""
    realify(X::AbstractMatrix{T𝔽}, 𝔽::AbstractNumbers) -> Y::AbstractMatrix{<:Real}

Given a matrix $X ∈ 𝔽^{n × n}$, compute $Y ∈ ℝ^{m × m}$, where $m = n \operatorname{dim}_𝔽$,
and $\operatorname{dim}_𝔽$ is the [`real_dimension`](@ref) of the number field $𝔽$, using
the map $ϕ \colon X ↦ Y$, that preserves the matrix product, so that for all
$C,D ∈ 𝔽^{n × n}$,
````math
ϕ(C) ϕ(D) = ϕ(CD).
````
See [`realify!`](@ref) for an in-place version, and [`unrealify!`](@ref) to compute the
inverse of $ϕ$.
"""
function realify(X, 𝔽)
    n = LinearAlgebra.checksquare(X)
    nℝ = real_dimension(𝔽) * n
    Y = allocate(X, real(eltype(X)), nℝ, nℝ)
    return realify!(Y, X, 𝔽, n)
end
realify(X, ::typeof(ℝ)) = X


"""
    realify!(Y::AbstractMatrix{<:Real}, X::AbstractMatrix{T𝔽}, 𝔽::AbstractNumbers)

In-place version of [`realify`](@ref).
"""
realify!(Y, X, 𝔽)

@doc raw"""
    realify!(Y::AbstractMatrix{<:Real}, X::AbstractMatrix{<:Complex}, ::typeof(ℂ))

Given a complex matrix $X = A + iB ∈ ℂ^{n × n}$, compute its realified matrix
$Y ∈ ℝ^{2n × 2n}$, written
where
````math
Y = \begin{pmatrix}A & -B \\ B & A \end{pmatrix}.
````
"""
function realify!(Y, X, ::typeof(ℂ), n = LinearAlgebra.checksquare(X))
    axul, axlr = 1:n, (n + 1):(2n)
    @views begin
        Y[axul, axul] .= Y[axlr, axlr] .= real.(X)
        Y[axlr, axul] .= imag.(X)
        Y[axul, axlr] .= .-imag.(X)
    end
    return Y
end

@doc raw"""
    unrealify!(X::AbstractMatrix{T𝔽}, Y::AbstractMatrix{<:Real}, 𝔽::AbstractNumbers[, n])

Given a real matrix $Y ∈ ℝ^{m × m}$, where $m = n \operatorname{dim}_𝔽$, and
$\operatorname{dim}_𝔽$ is the [`real_dimension`](@ref) of the number field $𝔽$, compute
in-place its equivalent matrix $X ∈ 𝔽^{n × n}$. Note that this function does not check that
$Y$ has a valid structure to be un-realified.

See [`realify!`](@ref) for the inverse of this function.
"""
unrealify!(X, Y, 𝔽)

function unrealify!(X, Y, ::typeof(ℂ), n = LinearAlgebra.checksquare(X))
    axul, axlr = 1:n, (n + 1):(2n)
    @views begin
        X .=
            complex.(
                (Y[axul, axul] .+ Y[axlr, axlr]) ./ 2,
                (Y[axlr, axul] .- Y[axul, axlr]) ./ 2,
            )
    end
    return X
end
unrealify!(Y, X, ::typeof(ℝ), args...) = copyto!(Y, X)

@generated maybesize(s::Size{S}) where {S} = prod(S) > 100 ? S : :(s)

"""
    select_from_tuple(t::NTuple{N, Any}, positions::Val{P})

Selects elements of tuple `t` at positions specified by the second argument.
For example `select_from_tuple(("a", "b", "c"), Val((3, 1, 1)))` returns
`("c", "a", "a")`.
"""
@generated function select_from_tuple(t::NTuple{N,Any}, positions::Val{P}) where {N,P}
    for k in P
        (k < 0 || k > N) && error("positions must be between 1 and $N")
    end
    return Expr(:tuple, [Expr(:ref, :t, k) for k in P]...)
end

"""
    size_to_tuple(::Type{S}) where S<:Tuple

Converts a size given by `Tuple{N, M, ...}` into a tuple `(N, M, ...)`.
"""
Base.@pure size_to_tuple(::Type{S}) where {S<:Tuple} = tuple(S.parameters...)

@doc raw"""
    vec2skew!(X, v, k)

create a skew symmetric matrix inplace in `X` of size $k\times k$ from a vector `v`,
for example for `v=[1,2,3]` and `k=3` this
yields
````julia
[  0  1  2;
  -1  0  3;
  -2 -3  0
]
````
"""
function vec2skew!(X, v)
    k = size(X)[1]
    size(X)[2] != k && error("X is of wrong size, expected ($k,$k) got $(size(X)).")
    n = div(k * (k - 1), 2)
    length(v) < n && error("The vector $(v) is too short, expected $(n) got $(length(v)).")
    m = 0
    X .= [i < j ? (m += 1; v[m]) : zero(eltype(v)) for i in 1:k, j in 1:k]
    X .= X - X'
    return X
end
function vec2skew(v, k)
    X = similar(v, k, k)
    vec2skew!(X, v)
    return X
end

@doc raw"""
    isnormal(x; kwargs...) -> Bool

Check if the matrix or number `x` is normal, that is, if it commutes with its adjoint:
````math
x x^\mathrm{H} = x^\mathrm{H} x.
````
By default, this is an equality check. Provide `kwargs` for `isapprox` to perform an
approximate check.
"""
function isnormal(x; kwargs...)
    isempty(kwargs) && return x * x' == x' * x
    return isapprox(x * x', x' * x; kwargs...)
end
isnormal(::LinearAlgebra.RealHermSymComplexHerm; kwargs...) = true
isnormal(::Diagonal; kwargs...) = true

"""
    ziptuples(a, b[, c[, d[, e]]])

Zips tuples `a`, `b`, and remaining in a fast, type-stable way. If they have different
lengths, the result is trimmed to the length of the shorter tuple.
"""
@generated function ziptuples(a::NTuple{N,Any}, b::NTuple{M,Any}) where {N,M}
    ex = Expr(:tuple)
    for i in 1:min(N, M)
        push!(ex.args, :((a[$i], b[$i])))
    end
    return ex
end
@generated function ziptuples(
    a::NTuple{N,Any},
    b::NTuple{M,Any},
    c::NTuple{L,Any},
) where {N,M,L}
    ex = Expr(:tuple)
    for i in 1:min(N, M, L)
        push!(ex.args, :((a[$i], b[$i], c[$i])))
    end
    return ex
end
@generated function ziptuples(
    a::NTuple{N,Any},
    b::NTuple{M,Any},
    c::NTuple{L,Any},
    d::NTuple{K,Any},
) where {N,M,L,K}
    ex = Expr(:tuple)
    for i in 1:min(N, M, L, K)
        push!(ex.args, :((a[$i], b[$i], c[$i], d[$i])))
    end
    return ex
end
@generated function ziptuples(
    a::NTuple{N,Any},
    b::NTuple{M,Any},
    c::NTuple{L,Any},
    d::NTuple{K,Any},
    e::NTuple{J,Any},
) where {N,M,L,K,J}
    ex = Expr(:tuple)
    for i in 1:min(N, M, L, K, J)
        push!(ex.args, :((a[$i], b[$i], c[$i], d[$i], e[$i])))
    end
    return ex
end

# TODO: make a better implementation for StaticArrays
function LinearAlgebra.eigvals(A::StaticArray, B::StaticArray; kwargs...)
    return eigvals(Array(A), Array(B); kwargs...)
end
