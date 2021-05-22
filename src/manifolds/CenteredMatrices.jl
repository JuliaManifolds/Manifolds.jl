@doc raw"""
    CenteredMatrices{m,n,𝔽} <: AbstractEmbeddedManifold{𝔽,TransparentIsometricEmbedding}

The manifold of $m × n$ real-valued or complex-valued matrices whose columns sum to zero, i.e.
````math
\bigl\{ p ∈ 𝔽^{m × n}\ \big|\ [1 … 1] * p = [0 … 0] \bigr\},
````
where $𝔽 ∈ \{ℝ,ℂ\}$.

# Constructor
    CenteredMatrices(m, n[, field=ℝ])

Generate the manifold of `m`-by-`n` (`field`-valued) matrices whose columns sum to zero.
"""
struct CenteredMatrices{M,N,𝔽} <: AbstractEmbeddedManifold{𝔽,TransparentIsometricEmbedding} end

function CenteredMatrices(m::Int, n::Int, field::AbstractNumbers=ℝ)
    return CenteredMatrices{m,n,field}()
end

@doc raw"""
    check_point(M::CenteredMatrices{m,n,𝔽}, p; kwargs...)

Check whether the matrix is a valid point on the
[`CenteredMatrices`](@ref) `M`, i.e. is an `m`-by-`n` matrix whose columns sum to
zero.

The tolerance for the column sums of `p` can be set using `kwargs...`.
"""
function check_point(M::CenteredMatrices{m,n,𝔽}, p; kwargs...) where {m,n,𝔽}
    mpv = invoke(check_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(sum(p, dims=1), zeros(1, n); kwargs...)
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
    check_vector(M::CenteredMatrices{m,n,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`CenteredMatrices`](@ref) `M`, i.e. that `X` is a matrix of size `(m, n)` whose columns
sum to zero and its values are from the correct [`AbstractNumbers`](@ref).
The tolerance for the column sums of `p` and `X` can be set using `kwargs...`.
"""
function check_vector(M::CenteredMatrices{m,n,𝔽}, p, X; kwargs...) where {m,n,𝔽}
    mpv = invoke(
        check_vector,
        Tuple{supertype(typeof(M)),typeof(p),typeof(X)},
        M,
        p,
        X;
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(sum(X, dims=1), zeros(1, n); kwargs...)
        return DomainError(
            X,
            "The vector $(X) is not a tangent vector to $(p) on $(M), since its columns do not sum to zero.",
        )
    end
    return nothing
end

decorated_manifold(M::CenteredMatrices{m,n,𝔽}) where {m,n,𝔽} = Euclidean(m, n; field=𝔽)

@doc raw"""
    manifold_dimension(M::CenteredMatrices{m,n,𝔽})

Return the manifold dimension of the [`CenteredMatrices`](@ref) `m`-by-`n` matrix `M` over the number system
`𝔽`, i.e.

````math
\dim(\mathcal M) = (m*n - n) \dim_ℝ 𝔽,
````
where $\dim_ℝ 𝔽$ is the [`real_dimension`](@ref) of `𝔽`.
"""
function manifold_dimension(::CenteredMatrices{m,n,𝔽}) where {m,n,𝔽}
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
where $c_i = \frac{1}{m}\sum_{j=1}^m p_{j,i}$ for $i = 1, \dots, n$.
"""
project(::CenteredMatrices, ::Any)

project!(M::CenteredMatrices, q, p) = copyto!(q, p .- mean(p, dims=1))

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
where $c_i = \frac{1}{m}\sum_{j=1}^m x_{j,i}$  for $i = 1, \dots, n$.
"""
project(::CenteredMatrices, ::Any, ::Any)

project!(M::CenteredMatrices, Y, p, X) = (Y .= X .- mean(X, dims=1))

@generated representation_size(::CenteredMatrices{m,n,𝔽}) where {m,n,𝔽} = (m, n)

function Base.show(io::IO, ::CenteredMatrices{m,n,𝔽}) where {m,n,𝔽}
    return print(io, "CenteredMatrices($(m), $(n), $(𝔽))")
end
