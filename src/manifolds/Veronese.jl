@doc raw"""
    Veronese{T} <: AbstractManifold{ℝ}

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

The metric is induced by the Euclidean metric on the full tensor space ``(ℝ^N)^{\otimes D}``.
Its geometry is a special case of the Segre--Veronese geometry described in
[JacobssonSwijsenVandervekenVannieuwenhoven:2024](@cite).

# Constructor

    Veronese(n::Int, d::Int; parameter::Symbol=:type)

Generate the Veronese manifold of nonzero symmetric rank-one tensors of order
`d` over ``ℝ^n``. Both `n` and `d` must be positive. The `parameter` keyword
determines whether `n` and `d` are stored in the type (`:type`) or as a field
(`:field`).
"""
struct Veronese{T} <: AbstractManifold{ℝ}
    size::T
end

function Veronese(n::Int, d::Int; parameter::Symbol = :type)
    n > 0 || throw(ArgumentError("The vector-space dimension n must be positive."))
    d > 0 || throw(ArgumentError("The tensor order d must be positive."))
    size = wrap_type_parameter(parameter, (n, d))
    return Veronese{typeof(size)}(size)
end

function ManifoldsBase.allocate_result(M::Veronese, ::typeof(rand))
    n, _ = get_parameter(M.size)
    return [zeros(1), zeros(n)]
end

function ManifoldsBase.allocate_result(M::Veronese, ::typeof(zero_vector), p)
    n, _ = get_parameter(M.size)
    T = number_eltype(p)
    return [zeros(T, 1), zeros(T, n)]
end

function ManifoldsBase.allocate_result_embedding(
        ::Veronese,
        ::typeof(project),
        A,
        p,
    )
    return [similar(p[1]), similar(p[2])]
end

@doc raw"""
    check_point(M::Veronese, p; kwargs...)

Check whether `p = [[λ], x]` represents a point on [`Veronese`](@ref) `M`.
The scale ``\lambda`` must be finite and nonzero, and ``x`` must lie on the
unit sphere.
"""
function check_point(M::Veronese, p; kwargs...)
    e = check_size(M, p)
    isnothing(e) || return e

    λ = p[1][1]
    if !isfinite(λ) || iszero(λ)
        return DomainError(λ, "The scale of a point on $M must be finite and nonzero.")
    end
    n, _ = get_parameter(M.size)
    return check_point(Sphere(n - 1; parameter = get_parameter_type(M)), p[2]; kwargs...)
end

@doc raw"""
    check_size(M::Veronese, p)
    check_size(M::Veronese, p, X)

Check that a point `p` and, optionally, a tangent vector `X` use the nested
representation `[[scalar], vector]` of sizes `(1,)` and `(N,)`.
"""
function check_size(M::Veronese, p)
    p_size = only.(size.(p))
    n, _ = get_parameter(M.size)
    M_size = [1, n]

    if p_size != M_size
        return DomainError(
            p_size,
            "The point $p can not belong to $M, since its size $p_size is not equal to the manifold's representation size $M_size.",
        )
    end

    return nothing
end
function check_size(M::Veronese, p, X)
    p_size = only.(size.(p))
    X_size = only.(size.(X))
    n, _ = get_parameter(M.size)
    M_size = [1, n]

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
function check_vector(M::Veronese, p, X; kwargs...)
    e = check_size(M, p, X)
    isnothing(e) || return e
    n, _ = get_parameter(M.size)
    return check_vector(
        Sphere(n - 1; parameter = get_parameter_type(M)), p[2], X[2]; kwargs...,
    )
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

function embed!(M::Veronese, q, p)
    _, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    if d == 1
        q .= λ .* x
    else
        q .= λ .* kron(ntuple(_ -> x, d)...)
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

function embed!(M::Veronese, Y, p, X)
    _, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    ν = X[1][1]
    u = X[2]

    if d == 1
        Y .= ν .* x .+ λ .* u
        return Y
    end

    Y .= ν .* kron(ntuple(_ -> x, d)...)
    for j in 1:d
        Y .+= λ .* kron(ntuple(i -> i == j ? u : x, d)...)
    end
    return Y
end

@doc raw"""
    project(M::Veronese, p, A)
    project!(M::Veronese, Y, p, A)

Orthogonally project an ambient tensor `A` onto the tangent space of
[`Veronese`](@ref) `M` at `p`. The ambient tensor is represented as a vector of
length ``N^D``, while the result uses the tangent representation `Y = [[ν], u]`.

For every mode ``j``, let ``c_j`` be the contraction of `A` with ``x`` in all
modes except ``j`` and let ``c = \sum_{j=1}^D c_j``. Then

````math
\nu = \langle A, x^{\otimes D}\rangle,
\qquad
u = \frac{c-D\nu x}{D\lambda}.
````
"""
project(::Veronese, p, A)

function project!(M::Veronese, Y, p, A)
    n, d = get_parameter(M.size)
    λ = p[1][1]
    x = p[2]
    A_tensor = reshape(A, ntuple(_ -> n, d))

    T = promote_type(number_eltype(p), number_eltype(A))
    c = zeros(T, n)
    prefix = Vector{T}(undef, d + 1)
    suffix = Vector{T}(undef, d + 1)
    ν = zero(T)

    for I in CartesianIndices(A_tensor)
        prefix[1] = one(T)
        @inbounds for j in 1:d
            prefix[j + 1] = prefix[j] * x[I[j]]
        end

        suffix[d + 1] = one(T)
        @inbounds for j in d:-1:1
            suffix[j] = suffix[j + 1] * x[I[j]]
        end

        a = A_tensor[I]
        ν += a * prefix[d + 1]
        @inbounds for j in 1:d
            c[I[j]] += a * prefix[j] * suffix[j + 1]
        end
    end

    Y[1][1] = ν
    Y[2] .= (c .- d .* ν .* x) ./ (d * λ)
    Y[2] .-= dot(x, Y[2]) .* x
    return Y
end

@doc raw"""
    get_embedding(M::Veronese)

Return the full Euclidean tensor space containing [`Veronese`](@ref) `M`,
represented as ``ℝ^{N^D}``.
"""
function get_embedding(M::Veronese)
    n, d = get_parameter(M.size)
    return Euclidean(n^d; parameter = get_parameter_type(M))
end

get_parameter_type(::Veronese{<:TypeParameter}) = :type
get_parameter_type(::Veronese{Tuple{Int, Int}}) = :field

@doc raw"""
    inner(M::Veronese, p, X, Y)

Compute the Riemannian inner product induced by the Euclidean metric of the
full tensor embedding. For `p = [[λ], x]`, `X = [[ν], u]`, and
`Y = [[ξ], v]`, it is

````math
g_p(X,Y)=\nu\xi+D\lambda^2\langle u,v\rangle.
````
"""
function inner(M::Veronese, p, X, Y)
    _, d = get_parameter(M.size)
    return X[1][1] * Y[1][1] + d * p[1][1]^2 * dot(X[2], Y[2])
end

@doc raw"""
    manifold_dimension(M::Veronese)

Return the dimension ``N`` of [`Veronese`](@ref) `M`.
"""
manifold_dimension(M::Veronese) = get_parameter(M.size)[1]

@doc raw"""
    Random.rand(M::Veronese; vector_at=nothing, σ=1)

Generate a random point on [`Veronese`](@ref) `M`. If `vector_at` is provided,
generate a random tangent vector there. The optional `σ` scales random tangent
vectors.
"""
Random.rand(M::Veronese; vector_at = nothing, σ::Real = 1)

function Random.rand!(
        rng::AbstractRNG,
        M::Veronese,
        pX;
        vector_at = nothing,
        σ::Real = 1,
    )
    n, _ = get_parameter(M.size)
    sphere = Sphere(n - 1; parameter = get_parameter_type(M))
    if isnothing(vector_at)
        randn!(rng, pX[1])
        pX[1][1] += iszero(pX[1][1])
        rand!(rng, sphere, pX[2])
    else
        randn!(rng, pX[1])
        pX[1] .*= σ
        rand!(rng, sphere, pX[2]; vector_at = vector_at[2], σ)
    end
    return pX
end

@doc raw"""
    zero_vector!(M::Veronese, X, p)

Set `X` to the zero tangent vector at `p` on [`Veronese`](@ref).
"""
function zero_vector!(::Veronese, X, p)
    fill!(X[1], zero(eltype(X[1])))
    fill!(X[2], zero(eltype(X[2])))
    return X
end

function Base.show(io::IO, M::Veronese)
    n, d = get_parameter(M.size)
    parameter = get_parameter_type(M) === :field ? "; parameter=:field" : ""
    return print(io, "Veronese($n, $d$parameter)")
end
