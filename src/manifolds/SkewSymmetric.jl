@doc raw"""
    SkewSymmetricMatrices{n,𝔽} <: AbstractEmbeddedManifold{𝔽,TransparentIsometricEmbedding}

The [`Manifold`](@ref) $ \operatorname{SkewSym}(n)$ consisting of the real- or
complex-valued skew-symmetric matrices of size $n × n$, i.e. the set

````math
\operatorname{SkewSym}(n) = \bigl\{p  ∈ 𝔽^{n × n}\ \big|\ p^{\mathrm{H}} = -p \bigr\},
````
where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transpose,
and the field $𝔽 ∈ \{ ℝ, ℂ\}$.

Though it is slightly redundant, usually the matrices are stored as $n × n$ arrays.

Note that in this representation, the (real-valued part of) the diagonal has to be zero,
which is also reflected in the [`manifold_dimension`](@ref manifold_dimension(::SkewSymmetricMatrices{N,𝔽}) where {N,𝔽}).

# Constructor

    SkewSymmetricMatrices(n::Int, field::AbstractNumbers=ℝ)

Generate the manifold of $n × n$ symmetric matrices.
"""
struct SkewSymmetricMatrices{n,𝔽} <:
       AbstractEmbeddedManifold{𝔽,TransparentIsometricEmbedding} end

function SkewSymmetricMatrices(n::Int, field::AbstractNumbers = ℝ)
    SkewSymmetricMatrices{n,field}()
end

function allocation_promotion_function(
    M::SkewSymmetricMatrices{<:Any,ℂ},
    ::typeof(get_vector),
    args::Tuple,
)
    return complex
end

@doc raw"""
    check_manifold_point(M::SkewSymmetricMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SkewSymmetricMatrices`](@ref) `M`, i.e.
whether `p` is a skew-symmetric matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@ref) `𝔽`.

The tolerance for the skew-symmetry of `p` can be set using `kwargs...`.
"""
function check_manifold_point(M::SkewSymmetricMatrices{n,𝔽}, p; kwargs...) where {n,𝔽}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
    mpv === nothing || return mpv
    if !isapprox(norm(p + p'), 0.0; kwargs...)
        return DomainError(
            norm(p + p'),
            "The point $(p) does not lie on $M, since it is not symmetric.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::SkewSymmetricMatrices{n,𝔽}, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SkewSymmetricMatrices`](@ref) `M`, i.e. `X` has to be a skew-symmetric matrix of size `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](@ref).
The optional parameter `check_base_point` indicates, whether to call
 [`check_manifold_point`](@ref)  for `p`.
The tolerance for the skew-symmetry of `p` and `X` can be set using `kwargs...`.
"""
function check_tangent_vector(
    M::SkewSymmetricMatrices{n,𝔽},
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
        Tuple{supertype(typeof(M)),typeof(p),typeof(X)},
        M,
        p,
        X;
        check_base_point = false, # already checked above
        kwargs...,
    )
    mpv === nothing || return mpv
    if !isapprox(norm(X + adjoint(X)), 0.0; kwargs...)
        return DomainError(
            norm(X + adjoint(X)),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not symmetric.",
        )
    end
    return nothing
end

decorated_manifold(M::SkewSymmetricMatrices{N,𝔽}) where {N,𝔽} = Euclidean(N, N; field = 𝔽)

embed!(M::SkewSymmetricMatrices, q, p) = copyto!(q, p)
embed!(M::SkewSymmetricMatrices, Y, p, X) = copyto!(Y, X)

function get_basis(M::SkewSymmetricMatrices, p, B::DiagonalizingOrthonormalBasis)
    Ξ = get_basis(M, p, DefaultOrthonormalBasis()).data
    κ = zeros(real(eltype(p)), manifold_dimension(M))
    return CachedBasis(B, κ, Ξ)
end

function get_coordinates!(
    M::SkewSymmetricMatrices{N,ℝ},
    Y,
    p,
    X,
    B::DefaultOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    @assert size(Y) == (dim,)
    @assert size(X) == (N, N)
    @assert dim == div(N * (N - 1), 2)
    k = 1
    for i = 1:N, j = (i+1):N
        @inbounds Y[k] = X[i, j] * sqrt(2)
        k += 1
    end
    return Y
end
function get_coordinates!(
    M::SkewSymmetricMatrices{N,ℂ},
    Y,
    p,
    X,
    B::DefaultOrthonormalBasis{ℂ},
) where {N}
    dim = manifold_dimension(M)
    @assert size(Y) == (dim,)
    @assert size(X) == (N, N)
    @assert dim == N^2
    k = 1
    for i = 1:N, j = i:N
        if i == j # real-part zero on the diagonal -> just one basis vector per diag entry
            @inbounds Y[k] = imag(X[i, j])
            k += 1
        else
            @inbounds Y[k] = real(X[i, j]) * sqrt(2)
            @inbounds Y[k+1] = imag(X[i, j]) * sqrt(2)
            k += 2
        end
    end
    return Y
end

function get_vector!(
    M::SkewSymmetricMatrices{N,ℝ},
    Y,
    p,
    X,
    B::DefaultOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    @assert size(X) == (dim,)
    @assert size(Y) == (N, N)
    k = 1
    for i = 1:N
        Y[i, i] = convert(eltype(p), 0.0)
    end
    for i = 1:N, j = (i+1):N
        @inbounds Y[i, j] = X[k] / sqrt(2)
        @inbounds Y[j, i] = -X[k] / sqrt(2)
        k += 1
    end
    return Y
end
function get_vector!(
    M::SkewSymmetricMatrices{N,ℂ},
    Y,
    p,
    X,
    B::DefaultOrthonormalBasis{ℂ},
) where {N}
    dim = manifold_dimension(M)
    @assert size(X) == (dim,)
    @assert size(Y) == (N, N)
    k = 1
    for i = 1:N, j = i:N
        if i == j # real zero on the diag
            @inbounds Y[i, j] = X[k] * 1im
            k += 1
        else
            @inbounds Y[i, j] = (X[k] + X[k+1] * 1im) / sqrt(2)
            k += 2
            @inbounds Y[j, i] = -conj(Y[i, j])
        end
    end
    return Y
end

@doc raw"""
    manifold_dimension(M::SkewSymmetricMatrices{n,𝔽})

Return the dimension of the [`SkewSymmetricMatrices`](@ref) matrix `M` over the number system
`𝔽`, i.e.

````math
\begin{aligned}
\dim \mathrm{SkewSym}(n,ℝ) &= \frac{n(n-1)}{2},\\
\dim \mathrm{SkewSym}(n,ℂ) &= 2*\frac{n(n-1)}{2} + n = n^2,
\end{aligned}
````

where the last $n$ is due to an imaginary diagonal that is allowed $\dim_ℝ 𝔽$ is the [`real_dimension`](@ref) of `𝔽`.
"""
function manifold_dimension(::SkewSymmetricMatrices{N,𝔽}) where {N,𝔽}
    return div(N * (N - 1), 2) * real_dimension(𝔽) + (𝔽 === ℂ ? N : 0)
end

@doc raw"""
    project(M::SkewSymmetricMatrices, p)

Projects `p` from the embedding onto the [`SkewSymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\operatorname{SkewSym}(n)}(p) = \frac{1}{2} \bigl( p - p^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transposed.
"""
project(::SkewSymmetricMatrices, ::Any)

project!(M::SkewSymmetricMatrices, q, p) = copyto!(q, (p - p') ./ 2)

@doc raw"""
    project(M::SkewSymmetricMatrices, p, X)

Project the matrix `X` onto the tangent space at `p` on the [`SkewSymmetricMatrices`](@ref) `M`,

````math
\operatorname{proj}_p(X) = \frac{1}{2} \bigl( X - X^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transposed.
"""
project(::SkewSymmetricMatrices, ::Any, ::Any)

project!(M::SkewSymmetricMatrices, Y, p, X) = (Y .= (X .- X') ./ 2)

function show(io::IO, ::SkewSymmetricMatrices{n,F}) where {n,F}
    print(io, "SkewSymmetricMatrices($(n), $(F))")
end
