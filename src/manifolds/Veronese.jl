@doc raw"""
    Veronese{N,D} <: AbstractManifold{ℝ}

The manifold of nonzero symmetric rank-one tensors of order ``D`` over
``ℝ^N``,

````math
\mathcal V_{N,D}
=
\left\{
    \lambda x^{\otimes D}
    \;\middle|\;
    \lambda\in\mathbb R\setminus\{0\},\;
    x\in\mathbb S^{N-1}
\right\}.
````

Points are represented by `p = [[λ], x]` and tangent vectors by `X = [[ν], u]`, where ``u`` is
orthogonal to ``x``. The parametrization ``\Phi(\lambda,x)=\lambda x^{\otimes D}`` is two-to-one,
with ``(\lambda,x)\sim\bigl((-1)^D\lambda,-x\bigr).``

This is the radial manifold of nonzero symmetric rank-one tensors, whose dimension is ``N``.
It is distinct from the projective Veronese variety, whose dimension is ``N-1``.

The metric is induced by the Euclidean metric on the full tensor space ``(ℝ^N)^{\otimes D}``.
Its geometry is a special case of the Segre--Veronese geometry described in
[JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).

# Constructor

    Veronese(n::Int, d::Int)

Generate the Veronese manifold of nonzero symmetric rank-one tensors of order
`d` over ``ℝ^n``. Both `n` and `d` must be positive.
"""
struct Veronese{N, D} <: AbstractManifold{ℝ} end

function Veronese(n::Int, d::Int)
    n > 0 || throw(ArgumentError("The vector-space dimension n must be positive."))
    d > 0 || throw(ArgumentError("The tensor order d must be positive."))
    return Veronese{n, d}()
end

function ManifoldsBase.allocate_result(::Veronese{N}, ::typeof(rand)) where {N}
    return [zeros(1), zeros(N)]
end

@doc raw"""
    check_point(M::Veronese, p; kwargs...)

Check whether `p = [[λ], x]` represents a point on [`Veronese`](@ref) `M`.
The scale ``\lambda`` must be finite and nonzero, and ``x`` must lie on the
unit sphere.
"""
function check_point(M::Veronese{N}, p; kwargs...) where {N}
    e = check_size(M, p)
    isnothing(e) || return e

    λ = p[1][1]
    if !isfinite(λ) || iszero(λ)
        return DomainError(λ, "The scale of a point on $M must be finite and nonzero.")
    end
    return check_point(Sphere(N - 1), p[2]; kwargs...)
end

@doc raw"""
    check_size(M::Veronese, p)
    check_size(M::Veronese, p, X)

Check that a point `p` and, optionally, a tangent vector `X` use the nested
representation `[[scalar], vector]` of sizes `(1,)` and `(N,)`.
"""
function check_size(M::Veronese{N}, p) where {N}
    p_size = only.(size.(p))
    M_size = [1, N]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $p can not belong to $M, since its size $p_size is not equal to the manifold's representation size $M_size.",
        )
    end

    return nothing
end
function check_size(M::Veronese{N}, p, X) where {N}
    p_size = only.(size.(p))
    X_size = only.(size.(X))
    M_size = [1, N]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $p can not belong to $M, since its size $p_size is not equal to the manifold's representation size $M_size.",
        )
    end

    if X_size != M_size
        return DomainError(
            X_size,
            "The vector $X can not be tangent at $p on $M, since its size $X_size is not equal to the manifold's representation size $M_size.",
        )
    end

    return nothing
end

@doc raw"""
    check_vector(M::Veronese, p, X; kwargs...)

Check whether `X = [[ν], u]` is a tangent vector at `p = [[λ], x]` on [`Veronese`](@ref) `M`.
The radial component ``\nu`` is unrestricted and the spherical component ``u`` must be tangent
to the unit sphere at ``x``.
"""
function check_vector(M::Veronese{N}, p, X; kwargs...) where {N}
    e = check_size(M, p, X)
    isnothing(e) || return e
    return check_vector(Sphere(N - 1), p[2], X[2]; kwargs...)
end

@doc raw"""
    embed(M::Veronese, p)
    embed!(M::Veronese, q, p)

Embed `p = [[λ], x]` into the full tensor space as

````math
\Phi(\lambda,x)=\lambda x^{\otimes D}.
````

The embedded tensor is represented as a vector of length ``N^D``.
"""
embed(::Veronese, p)

function embed!(::Veronese{N, D}, q, p) where {N, D}
    λ = p[1][1]
    x = p[2]
    if D == 1
        q .= λ .* x
    else
        q .= λ .* kron(ntuple(_ -> x, Val(D))...)
    end
    return q
end

@doc raw"""
    embed(M::Veronese, p, X)
    embed!(M::Veronese, Y, p, X)

Embed the tangent vector `X = [[ν], u]` at `p = [[λ], x]` using the
differential of the Veronese parametrization,

````math
D\Phi_{(\lambda,x)}(\nu,u)
=
\nu x^{\otimes D}
+
\lambda\sum_{j=1}^{D}
x^{\otimes(j-1)}\otimes u\otimes x^{\otimes(D-j)}.
````
"""
embed(::Veronese, p, X)

function embed!(::Veronese{N, D}, Y, p, X) where {N, D}
    λ = p[1][1]
    x = p[2]
    ν = X[1][1]
    u = X[2]

    if D == 1
        Y .= ν .* x .+ λ .* u
        return Y
    end

    Y .= ν .* kron(ntuple(_ -> x, Val(D))...)
    for j in 1:D
        Y .+= λ .* kron(ntuple(i -> i == j ? u : x, Val(D))...)
    end
    return Y
end

@doc raw"""
    get_embedding(M::Veronese{N,D})

Return the full Euclidean tensor space containing [`Veronese`](@ref) `M`,
represented as ``ℝ^{N^D}``.
"""
get_embedding(::Veronese{N, D}) where {N, D} = Euclidean(N^D)

@doc raw"""
    inner(M::Veronese, p, X, Y)

Compute the Riemannian inner product induced by the Euclidean metric of the
full tensor embedding. For `p = [[λ], x]`, `X = [[ν], u]`, and
`Y = [[ξ], v]`, it is

````math
g_p(X,Y)=\nu\xi+D\lambda^2\langle u,v\rangle.
````
"""
function inner(::Veronese{N, D}, p, X, Y) where {N, D}
    return X[1][1] * Y[1][1] + D * p[1][1]^2 * dot(X[2], Y[2])
end

@doc raw"""
    manifold_dimension(M::Veronese{N,D})

Return the dimension ``N`` of [`Veronese`](@ref) `M`.
"""
manifold_dimension(::Veronese{N, D}) where {N, D} = N

@doc raw"""
    Random.rand(M::Veronese; vector_at=nothing, σ=1)

Generate a random point on [`Veronese`](@ref) `M`. If `vector_at` is provided,
generate a random tangent vector there. The optional `σ` scales random tangent
vectors.
"""
function Random.rand(M::Veronese; vector_at = nothing, σ::Real = 1)
    return rand(Random.default_rng(), M; vector_at, σ)
end
function Random.rand(rng::AbstractRNG, M::Veronese{N}; vector_at = nothing, σ::Real = 1) where {N}
    T = isnothing(vector_at) ? Float64 : number_eltype(vector_at)
    pX = [zeros(T, 1), zeros(T, N)]
    return rand!(rng, M, pX; vector_at, σ)
end

function Random.rand!(M::Veronese, pX; kwargs...)
    return rand!(Random.default_rng(), M, pX; kwargs...)
end
function Random.rand!(
        rng::AbstractRNG,
        M::Veronese{N},
        pX;
        vector_at = nothing,
        σ::Real = 1,
    ) where {N}
    if isnothing(vector_at)
        randn!(rng, pX[1])
        while iszero(pX[1][1])
            randn!(rng, pX[1])
        end
        rand!(rng, Sphere(N - 1), pX[2])
    else
        randn!(rng, pX[1])
        pX[1] .*= σ
        rand!(rng, Sphere(N - 1), pX[2]; vector_at = vector_at[2], σ)
    end
    return pX
end

function Base.show(io::IO, ::Veronese{N, D}) where {N, D}
    return print(io, "Veronese($N, $D)")
end
