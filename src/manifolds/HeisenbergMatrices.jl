@doc raw"""
    HeisenbergMatrices{T} <: AbstractDecoratorManifold{𝔽}

Heisenberg matrices `HeisenbergMatrices(n)` is the manifold of ``(n+2)×(n+2)`` matrices [BinzPods:2008](@cite)

```math
\begin{bmatrix} 1 & \mathbf{a} & c \\
\mathbf{0}_n & I_n & \mathbf{b} \\
0 & \mathbf{0}_n^\mathrm{T} & 1 \end{bmatrix}
```

where ``I_n`` is the ``n×n`` unit matrix, ``\mathbf{a}`` is a row vector of length ``n``,
``\mathbf{b}`` is a column vector of length ``n``, ``\mathbf{0}_n`` is the column zero vector
of length ``n``, and ``c`` is a real number.

It is a submanifold of [`Euclidean`](@ref)`(n+2, n+2)` and the manifold of the
[`HeisenbergGroup`](@ref).

# Constructor

    HeisenbergMatrices(n::Int; parameter::Symbol=:type)

Generate the manifold of ``(n+2)×(n+2)`` Heisenberg matrices.
"""
struct HeisenbergMatrices{T} <: AbstractDecoratorManifold{ℝ}
    size::T
end

function HeisenbergMatrices(n::Int; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n,))
    return HeisenbergMatrices{typeof(size)}(size)
end

function active_traits(f, ::HeisenbergMatrices, args...)
    return merge_traits(IsEmbeddedSubmanifold())
end

function check_point(M::HeisenbergMatrices, p; kwargs...)
    n = get_parameter(M.size)[1]
    if !isone(p[1, 1])
        return DomainError(
            p[1, 1],
            "The matrix $(p) does not lie on $(M), since p[1, 1] is not equal to 1.",
        )
    end
    if !isone(p[n + 2, n + 2])
        return DomainError(
            p[n + 2, n + 2],
            "The matrix $(p) does not lie on $(M), since p[n+2, n+2] is not equal to 1.",
        )
    end
    if !iszero(p[2:(n + 2), 1])
        return DomainError(
            norm(iszero(p[2:(n + 2), 1])),
            "The matrix $(p) does not lie on $(M), since p[2:(n + 2), 1] is not equal to 0.",
        )
    end
    if !iszero(p[n + 2, 1:(n + 1)])
        return DomainError(
            norm(iszero(p[n + 2, 1:(n + 1)])),
            "The matrix $(p) does not lie on $(M), since p[n + 2, 1:(n+1)] is not equal to 0.",
        )
    end
    if !isapprox(I, p[2:(n + 1), 2:(n + 1)])
        return DomainError(
            det(p[2:(n + 1), 2:(n + 1)] - I),
            "The matrix $(p) does not lie on $(M), since p[2:(n+1), 2:(n+1)] is not an identity matrix.",
        )
    end

    return nothing
end

function check_vector(M::HeisenbergMatrices, p, X; kwargs...)
    n = get_parameter(M.size)[1]
    if !iszero(X[1, 1])
        return DomainError(
            X[1, 1],
            "The matrix $(X) does not lie in the tangent space of $(M), since X[1, 1] is not equal to 0.",
        )
    end
    if !iszero(X[n + 2, n + 2])
        return DomainError(
            X[n + 2, n + 2],
            "The matrix $(X) does not lie in the tangent space of $(M), since X[n+2, n+2] is not equal to 0.",
        )
    end
    if !iszero(X[2:(n + 2), 1:(n + 1)])
        return DomainError(
            norm(X[2:(n + 2), 1:(n + 1)]),
            "The matrix $(X) does not lie in the tangent space of $(M), since X[2:(n + 2), 1:(n + 1)] is not a zero matrix.",
        )
    end
    return nothing
end

embed(::HeisenbergMatrices, p) = p
embed(::HeisenbergMatrices, p, X) = X

function get_embedding(::HeisenbergMatrices{TypeParameter{Tuple{n}}}) where {n}
    return Euclidean(n + 2, n + 2)
end
function get_embedding(M::HeisenbergMatrices{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return Euclidean(n + 2, n + 2; parameter=:field)
end

"""
    is_flat(::HeisenbergMatrices)

Return true. [`HeisenbergMatrices`](@ref) is a flat manifold.
"""
is_flat(M::HeisenbergMatrices) = true

"""
    manifold_dimension(M::HeisenbergMatrices)

Return the dimension of [`HeisenbergMatrices`](@ref)`(n)`, which is equal to ``2n+1``.
"""
manifold_dimension(M::HeisenbergMatrices) = 2 * get_parameter(M.size)[1] + 1

function Base.show(io::IO, ::HeisenbergMatrices{TypeParameter{Tuple{n}}}) where {n}
    return print(io, "HeisenbergMatrices($(n))")
end
function Base.show(io::IO, M::HeisenbergMatrices{Tuple{Int}})
    n = get_parameter(M.size)[1]
    return print(io, "HeisenbergMatrices($(n); parameter=:field)")
end

@doc raw"""
    Y = Weingarten(M::HeisenbergMatrices, p, X, V)
    Weingarten!(M::HeisenbergMatrices, Y, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`HeisenbergMatrices`](@ref) `M`
with respect to the tangent vector ``X \in T_p\mathcal M`` and the normal vector
``V \in N_p\mathcal M``.

Since this a flat space by itself, the result is always the zero tangent vector.
"""
Weingarten(::HeisenbergMatrices, p, X, V)

Weingarten!(::HeisenbergMatrices, Y, p, X, V) = fill!(Y, 0)
