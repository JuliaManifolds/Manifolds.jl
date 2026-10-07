@doc raw"""
    CenteredMatrices{𝔽,T} <: AbstractDecoratorManifold{𝔽}

The manifold of ``m×n`` real-valued or complex-valued matrices whose columns sum to zero, i.e.
````math
\bigl\{ p ∈ 𝔽^{m×n}\ \big|\ [1 … 1] * p = [0 … 0] \bigr\},
````
where ``𝔽 ∈ \{ℝ,ℂ\}``.

# Constructor
    CenteredMatrices(m, n[, field=ℝ]; parameter::Symbol=:type)

Generate the manifold of `m`-by-`n` (`field`-valued) matrices whose columns sum to zero.

`parameter`: whether a type parameter should be used to store `m` and `n`. By default size
is stored in type. Value can either be `:field` or `:type`.
"""
struct CenteredMatrices{𝔽, T} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function CenteredMatrices(m::Int, n::Int, field::AbstractNumbers = ℝ; parameter::Symbol = :type)
    size = wrap_type_parameter(parameter, (m, n))
    return CenteredMatrices{field, typeof(size)}(size)
end

@doc raw"""
    check_point(M::CenteredMatrices, p; kwargs...)

Check whether the matrix is a valid point on the
[`CenteredMatrices`](@ref) `M`, i.e. is an `m`-by-`n` matrix whose columns sum to
zero.

The norm of the column sums of `p` has to be at most `atol`, or at most `rtol * sqrt(m) * norm(p)` if this scale is finite.
The relative tolerance `rtol` refers to the size of `p`; its default is the one of `isapprox`.
"""
function check_point(
        M::CenteredMatrices,
        p::T;
        atol::Real = sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
        rtol::Real = sqrt(eps(real(float(number_eltype(T))))),
        kwargs...,
    ) where {T}
    m, n = get_parameter(M.size)
    r = norm(sum(p, dims = 1))
    s = sqrt(m) * norm(p)
    if !(r <= atol || (isfinite(s) && r <= rtol * s))
        return DomainError(
            p,
            string(
                "The point $(p) does not lie on $(M), since its columns do not sum to zero.",
            ),
        )
    end
    return nothing
end

"""
    check_vector(M::CenteredMatrices, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`CenteredMatrices`](@ref) `M`, i.e. that `X` is a matrix of size `(m, n)` whose columns
sum to zero and its values are from the correct [`AbstractNumbers`](@extref ManifoldsBase number-system).
The column sums of `X` have to vanish up to `max(atol, rtol * sqrt(m) * norm(X))`.
The relative tolerance `rtol` refers to the size of `X`; its default is the one of `isapprox`.
"""
function check_vector(
        M::CenteredMatrices,
        p,
        X::T;
        atol::Real = sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
        rtol::Real = sqrt(eps(real(float(number_eltype(T))))),
        kwargs...,
    ) where {T}
    m, n = get_parameter(M.size)
    r = norm(sum(X, dims = 1))
    if !(r <= atol || r <= rtol * sqrt(m) * norm(X))
        return DomainError(
            X,
            "The vector $(X) is not a tangent vector to $(p) on $(M), since its columns do not sum to zero.",
        )
    end
    return nothing
end

embed(::CenteredMatrices, p) = p
embed(::CenteredMatrices, p, X) = X

@doc raw"""
    get_coordinates(M::CenteredMatrices, p, X, ::DefaultOrthonormalBasis{ℝ})

Compute the coordinates of ``HX`` in the [`Euclidean`](@ref) space of ``(m-1)×n`` matrices,
where ``H`` is the ``(m-1)×m`` Helmert submatrix, see [Gentle:2017; Section 8.8.1](@cite).
"""
get_coordinates(::CenteredMatrices, p, X, ::DefaultOrthonormalBasis{ℝ})

function get_coordinates_orthonormal!(M::CenteredMatrices{𝔽}, c, p, X, ::RealNumbers) where {𝔽}
    m, n = get_parameter(M.size)
    H = helmert_submatrix(real(eltype(c)), m)
    E = Euclidean(m - 1, n; field = 𝔽, parameter = :field)
    return get_coordinates_orthonormal!(E, c, H * p, H * X, ℝ)
end

function get_embedding(::CenteredMatrices{𝔽, TypeParameter{Tuple{m, n}}}) where {m, n, 𝔽}
    return Euclidean(m, n; field = 𝔽)
end
function get_embedding(M::CenteredMatrices{𝔽, Tuple{Int, Int}}) where {𝔽}
    m, n = get_parameter(M.size)
    return Euclidean(m, n; field = 𝔽, parameter = :field)
end

function ManifoldsBase.get_embedding_type(::CenteredMatrices)
    return ManifoldsBase.EmbeddedSubmanifoldType()
end

@doc raw"""
    get_vector(M::CenteredMatrices, p, c, ::DefaultOrthonormalBasis{ℝ})

Compute ``X = H^{\mathrm{T}}Z``, where ``Z`` is the ``(m-1)×n`` matrix with the coordinates `c` in the
[`Euclidean`](@ref) space and ``H`` is the ``(m-1)×m`` Helmert submatrix, see [Gentle:2017; Section 8.8.1](@cite).
"""
get_vector(::CenteredMatrices, p, c, ::DefaultOrthonormalBasis{ℝ})

function get_vector_orthonormal!(M::CenteredMatrices{𝔽}, Y, p, c, ::RealNumbers) where {𝔽}
    m, n = get_parameter(M.size)
    H = helmert_submatrix(real(eltype(Y)), m)
    E = Euclidean(m - 1, n; field = 𝔽, parameter = :field)
    Y .= H' * get_vector_orthonormal!(E, similar(Y, m - 1, n), H * p, c, ℝ)
    return Y
end

"""
    is_flat(::CenteredMatrices)

Return true. [`CenteredMatrices`](@ref) is a flat manifold.
"""
is_flat(M::CenteredMatrices) = true

@doc raw"""
    manifold_dimension(M::CenteredMatrices)

Return the manifold dimension of the [`CenteredMatrices`](@ref) `m`-by-`n` matrix `M` over the number system
`𝔽`, i.e.

````math
\dim(\mathcal M) = (m*n - n) \dim_ℝ 𝔽,
````
where ``\dim_ℝ 𝔽`` is the [`real_dimension`](@extref `ManifoldsBase.real_dimension-Tuple{ManifoldsBase.AbstractNumbers}`) of `𝔽`.
"""
function manifold_dimension(M::CenteredMatrices{𝔽}) where {𝔽}
    m, n = get_parameter(M.size)
    return (m * n - n) * real_dimension(𝔽)
end

@doc raw"""
    project(M::CenteredMatrices, p)

Projects `p` from the embedding onto the [`CenteredMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\mathcal M}(p) = p - \begin{bmatrix}
1\\
⋮\\
1
\end{bmatrix} * [c_1 \dots c_n],
````
where ``c_i = \frac{1}{m}\sum_{j=1}^m p_{j,i}`` for ``i = 1, \dots, n``.
"""
project(::CenteredMatrices, ::Any)

project!(::CenteredMatrices, q, p) = copyto!(q, p .- mean(p, dims = 1))

@doc raw"""
    project(M::CenteredMatrices, p, X)

Project the matrix `X` onto the tangent space at `p` on the [`CenteredMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_p(X) = X - \begin{bmatrix}
1\\
⋮\\
1
\end{bmatrix} * [c_1 \dots c_n],
````
where ``c_i = \frac{1}{m}\sum_{j=1}^m x_{j,i}``  for ``i = 1, \dots, n``.
"""
project(::CenteredMatrices, ::Any, ::Any)

project!(::CenteredMatrices, Y, p, X) = (Y .= X .- mean(X, dims = 1))

@doc raw"""
    rand(M::CenteredMatrices; vector_at=nothing, σ::Real=1.0)
    rand!(M::CenteredMatrices, pX; vector_at=nothing, σ::Real=1.0)

Project a matrix of independent normally distributed entries with standard deviation `σ`
onto `M`, which yields a random point as well as a random tangent vector at `vector_at`.
"""
function Random.rand!(
        rng::AbstractRNG, M::CenteredMatrices, pX;
        vector_at = nothing, σ::Real = one(real(eltype(pX)))
    )
    return project!(M, pX, σ .* randn(rng, eltype(pX), representation_size(M)))
end

representation_size(M::CenteredMatrices) = get_parameter(M.size)

function Base.show(io::IO, ::CenteredMatrices{𝔽, TypeParameter{Tuple{m, n}}}) where {m, n, 𝔽}
    return print(io, "CenteredMatrices($(m), $(n), $(𝔽))")
end
function Base.show(io::IO, M::CenteredMatrices{𝔽, Tuple{Int, Int}}) where {𝔽}
    m, n = get_parameter(M.size)
    return print(io, "CenteredMatrices($(m), $(n), $(𝔽); parameter=:field)")
end

@doc raw"""
    Y = Weingarten(M::CenteredMatrices, p, X, V)
    Weingarten!(M::CenteredMatrices, Y, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`CenteredMatrices`](@ref) `M` with respect to the
tangent vector ``X \in T_p\mathcal M`` and the normal vector ``V \in N_p\mathcal M``.

Since this a flat space by itself, the result is always the zero tangent vector.
"""
Weingarten(::CenteredMatrices, p, X, V)

Weingarten!(::CenteredMatrices, Y, p, X, V) = fill!(Y, 0)
