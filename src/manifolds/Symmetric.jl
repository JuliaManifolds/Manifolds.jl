@doc raw"""
    SymmetricMatrices{n,𝔽} <: AbstractEmbeddedManifold{TransparentIsometricEmbedding}

The [`Manifold`](@ref) $ \operatorname{Sym}(n)$ consisting of the real- or complex-valued
symmetric matrices of size $n × n$, i.e. the set

````math
\operatorname{Sym}(n) = \bigl\{p  ∈ 𝔽^{n × n} \big| p^{\mathrm{H}} = p \bigr\},
````
where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transpose,
and the field $𝔽 ∈ \{ ℝ, ℂ\}$.

Though it is slightly redundant, usually the matrices are stored as $n × n$ arrays.

Note that in this representation, the complex valued case has to have a real-valued diagonal,
which is also reflected in the [`manifold_dimension`](@ref manifold_dimension(::SymmetricMatrices{N,𝔽}) where {N,𝔽}).

# Constructor

    SymmetricMatrices(n::Int, field::AbstractNumbers=ℝ)

Generate the manifold of $n × n$ symmetric matrices.
"""
struct SymmetricMatrices{n,𝔽} <: AbstractEmbeddedManifold{TransparentIsometricEmbedding} end

function SymmetricMatrices(n::Int, field::AbstractNumbers = ℝ)
    SymmetricMatrices{n,field}()
end

function allocation_promotion_function(
    M::SymmetricMatrices{<:Any,ℂ},
    ::typeof(get_vector),
    args::Tuple,
)
    return complex
end

base_manifold(M::SymmetricMatrices) = M

@doc raw"""
    check_manifold_point(M::SymmetricMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SymmetricMatrices`](@ref) `M`, i.e.
whether `p` is a symmetric matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@ref) `𝔽`.

The tolerance for the symmetry of `p` can be set using `kwargs...`.
"""
function check_manifold_point(M::SymmetricMatrices{n,𝔽}, p; kwargs...) where {n,𝔽}
    mpv = invoke(check_manifold_point, Tuple{supertype(typeof(M)), typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(norm(p - p'), 0.0; kwargs...)
        return DomainError(
            norm(p - p'),
            "The point $(p) does not lie on $M, since it is not symmetric.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::SymmetricMatrices{n,𝔽}, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SymmetricMatrices`](@ref) `M`, i.e. `X` has to be a symmetric matrix of size `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](@ref).
The optional parameter `check_base_point` indicates, whether to call
 [`check_manifold_point`](@ref)  for `p`.
The tolerance for the symmetry of `p` and `X` can be set using `kwargs...`.
"""
function check_tangent_vector(
    M::SymmetricMatrices{n,𝔽},
    p,
    X;
    check_base_point = true,
    kwargs...,
) where {n,𝔽}
    if check_base_point
        mpe = check_manifold_point(M, p; kwargs...)
        mpe === nothing || return mpe
    end
    mpv = invoke(
        check_tangent_vector,
        Tuple{supertype(typeof(M)), typeof(p), typeof(X)},
        M,
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...
    )
    mpv === nothing || return mpv
    if !isapprox(norm(X - X'), 0.0; kwargs...)
        return DomainError(
            norm(X - X'),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not symmetric.",
        )
    end
    return nothing
end

decorated_manifold(M::SymmetricMatrices{N,𝔽}) where {N,𝔽} = Euclidean(N, N; field = 𝔽)

embed!(M::SymmetricMatrices, q, p) = copyto!(q, p)
embed!(M::SymmetricMatrices, Y, p, X) = copyto!(Y, X)

function get_basis(M::SymmetricMatrices, p, B::DiagonalizingOrthonormalBasis)
    Ξ = get_basis(M, p, DefaultOrthonormalBasis()).data
    κ = zeros(real(eltype(p)), manifold_dimension(M))
    return CachedBasis(B, κ, Ξ)
end

function get_coordinates!(
    M::SymmetricMatrices{N,ℝ},
    Y,
    p,
    X,
    B::DefaultOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    @assert size(Y) == (dim,)
    @assert size(X) == (N, N)
    @assert dim == div(N * (N + 1), 2)
    k = 1
    for i = 1:N, j = i:N
        scale = ifelse(i == j, 1, sqrt(2))
        @inbounds Y[k] = X[i, j] * scale
        k += 1
    end
    return Y
end
function get_coordinates!(
    M::SymmetricMatrices{N,ℂ},
    Y,
    p,
    X,
    B::DefaultOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    @assert size(Y) == (dim,)
    @assert size(X) == (N, N)
    @assert dim == N * N
    k = 1
    for i = 1:N, j = i:N
        scale = ifelse(i == j, 1, sqrt(2))
        @inbounds Y[k] = real(X[i, j]) * scale
        k += 1
        if i != j # imag zero on the diagonal
            @inbounds Y[k] = imag(X[i, j]) * scale
            k += 1
        end
    end
    return Y
end

function get_vector!(
    M::SymmetricMatrices{N,ℝ},
    Y,
    p,
    X,
    B::DefaultOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    @assert size(X) == (dim,)
    @assert size(Y) == (N, N)
    k = 1
    for i = 1:N, j = i:N
        scale = ifelse(i == j, 1, 1 / sqrt(2))
        @inbounds Y[i, j] = X[k] * scale
        @inbounds Y[j, i] = X[k] * scale
        k += 1
    end
    return Y
end
function get_vector!(
    M::SymmetricMatrices{N,ℂ},
    Y,
    p,
    X,
    B::DefaultOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    @assert size(X) == (dim,)
    @assert size(Y) == (N, N)
    k = 1
    for i = 1:N, j = i:N
        scale = ifelse(i == j, 1, 1 / sqrt(2))
        @inbounds Y[i, j] = X[k] + (i == j ? 0 : X[k+1]*1im) * scale
        @inbounds Y[j, i] = Y[i, j]
        k += (i == j ? 1 : 2)
    end
    return Y
end
## unify within bases later.

@doc raw"""
    manifold_dimension(M::SymmetricMatrices{n,𝔽})

Return the dimension of the [`SymmetricMatrices`](@ref) matrix `M` over the number system
`𝔽`, i.e.

````math
\dim \operatorname{Sym}(n,ℝ) = \frac{n(n+1)}{2},
````

and

````math
\dim \operatorname{Sym}(n,ℂ) = 2\frac{n(n+1)}{2} - n = n^2,
````

where the last $-n$ is due to the zero imaginary part for Hermitian matrices
"""
function manifold_dimension(::SymmetricMatrices{N,𝔽}) where {N,𝔽}
    return div(N * (N + 1), 2) * real_dimension(𝔽) - (𝔽 === ℂ ? N : 0)
end

@doc raw"""
    project_point(M::SymmetricMatrices, p)

Projects `p` from the embedding onto the [`SymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\operatorname{Sym}(n)}(p) = \frac{1}{2} \bigl( p + p^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transposed.
"""
project_point(::SymmetricMatrices, ::Any...)

project_point!(M::SymmetricMatrices, q, p) = copyto!(q, (p + p') ./ 2)

@doc raw"""
    project_tangent(M::SymmetricMatrices, p, X)

Project the matrix `X` onto the tangent space at `p` on the [`SymmetricMatrices`](@ref) `M`,

````math
\operatorname{proj}_p(X) = \frac{1}{2} \bigl( X + X^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transposed.
"""
project_tangent(::SymmetricMatrices, ::Any...)

project_tangent!(M::SymmetricMatrices, Y, p, X) = (Y .= (X .+ transpose(X)) ./ 2)

function show(io::IO, ::SymmetricMatrices{n,F}) where {n,F}
    print(io, "SymmetricMatrices($(n), $(F))")
end
