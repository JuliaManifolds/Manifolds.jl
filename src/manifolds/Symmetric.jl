@doc raw"""
    SymmetricMatrices{n,F} <: Manifold

The [`Manifold`](@ref) $ \operatorname{Sym} (n)$ consisting of the real- or complex-valued
symmetric matrices of size $ n× n$, i.e. the set

````math
\operatorname{Sym}(n) = \bigl\{p  ∈ 𝔽^{n × n} \big| p^{\mathrm{H}} = p \bigr\},
````
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed
and the field $𝔽 ∈ \{ ℝ, ℂ\}$ is set by the
[`AbstractNumbers`](@ref) `F`.

Though it is slighty redundant, usually the matrices are safed as $n × n$ arrays.

# Constructor

    SymmetricMatrices(n::Int, F::AbstractNumbers=ℝ)

Generate the manifold of $n × n$ symmetric metrices.
"""
struct SymmetricMatrices{n,F} <: Manifold end

SymmetricMatrices(n::Int, F::AbstractNumbers = ℝ) = SymmetricMatrices{n,F}()

@doc raw"""
    check_manifold_point(M::SymmetricMatrices{n,F}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SymmetricMatrices`](@ref) `M`, i.e.
whether `p` is a symmetric matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@ref) `F`.

The tolerance for the symmetry of `p` can be set using `kwargs...`.
"""
function check_manifold_point(M::SymmetricMatrices{n,F}, p; kwargs...) where {n,F}
    if (F === ℝ) && !(eltype(p) <: Real)
        return DomainError(
            eltype(p),
            "The matrix $(p) does not lie on $M, since its values are not real.",
        )
    end
    if (F === ℂ) && !(eltype(p) <: Real) && !(eltype(p) <: Complex)
        return DomainError(
            eltype(p),
            "The matrix $(p) does not lie on $M, since its values are not complex.",
        )
    end
    if size(p) != (n, n)
        return DomainError(
            size(p),
            "The point $(p) does not lie on $M since its size ($(size(p))) does not match the representation size ($(representation_size(M))).",
        )
    end
    if !isapprox(norm(p - transpose(p)), 0.0; kwargs...)
        return DomainError(
            norm(p - transpose(p)),
            "The point $(p) does not lie on $M, since it is not symmetric.",
        )
    end
    return nothing
end

"""
    check_tangent_vector(M::SymmetricMatrices{n,F}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SymmetricMatrices`](@ref) `M`, i.e. `X` has to be a symmetric matrix of dimension `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](@ref).

The tolerance for the symmetry of `p` and `X` can be set using `kwargs...`.
"""
function check_tangent_vector(M::SymmetricMatrices{n,F}, p, X; kwargs...) where {n,F}
    t = check_manifold_point(M, p; kwargs...)
    t === nothing || return t
    if (F === ℝ) && !(eltype(X) <: Real)
        return DomainError(
            eltype(X),
            "The matrix $(X) is not a tangent to a point on $M, since its values are not real.",
        )
    end
    if (F === ℂ) && !(eltype(X) <: Real) && !(eltype(X) <: Complex)
        return DomainError(
            eltype(X),
            "The matrix $(X) is not a tangent to a point on $M, since its values are not complex.",
        )
    end
    if size(X) != (n, n)
        return DomainError(
            size(X),
            "The vector $(X) is not a tangent to a point on $(M) since its size ($(size(X))) does not match the representation size ($(representation_size(M))).",
        )
    end
    if !isapprox(norm(X - transpose(X)), 0.0; kwargs...)
        return DomainError(
            norm(X - transpose(X)),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not symmetric.",
        )
    end
    return nothing
end

@doc raw"""
    distance(M::SymmetricMatrices, p, q)

Compute distance using the inherited metric, i.e. taking the Frobenius-norm of the
difference.
"""
distance(M::SymmetricMatrices, p, q) = norm(p - q)

@doc raw"""
    exp(M::SymmetricMatrices, p, X)

Compute the exponential map emanating from `p` in tangent direction `X` on the
[`SymmetricMatrices`](@ref) `M`, which reads

````math
\exp_pX = p + X.
````
"""
exp(::SymmetricMatrices, ::Any...)

exp!(M::SymmetricMatrices, q, p, X) = (q .= p .+ X)

@doc raw"""
    flat(M::SymmetricMatrices, p, X::FVector{TangentSpaceType})

Compute the [`flat`](@ref flat(M::Manifold, p, X::FVector)) isomorphism of the
[`SymmetricMatrices`](@ref) `M` on the manifold point `p` and tangent vector `X`.

Since `M` is already a vector space over $ℝ$, this returns just the vector `X`.
"""
flat(::SymmetricMatrices, ::Any...)

flat!(M::SymmetricMatrices, ξ::CoTFVector, p, X::TFVector) = copyto!(ξ, X)

function get_coordinates(
    M::SymmetricMatrices{N,ℝ},
    p,
    X,
    B::ArbitraryOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    Y = similar(X, dim)
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
function get_coordinates(
    M::SymmetricMatrices{N,ℂ},
    p,
    X,
    B::ArbitraryOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    Y = similar(X, dim)
    @assert size(X) == (N, N)
    @assert dim == N * (N + 1)
    k = 1
    for i = 1:N, j = i:N
        scale = ifelse(i == j, 1, sqrt(2))
        @inbounds Y[k] = real(X[i, j]) * scale
        k += 1
        @inbounds Y[k] = imag(X[i, j]) * scale
        k += 1
    end
    return Y
end

function get_vector(
    M::SymmetricMatrices{N,ℝ},
    p,
    X,
    B::ArbitraryOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    Y = allocate_result(M, get_vector, p)
    @assert size(X) == (div(N * (N + 1), 2),)
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
function get_vector(
    M::SymmetricMatrices{N,ℂ},
    p,
    X,
    B::ArbitraryOrthonormalBasis{ℝ},
) where {N}
    dim = manifold_dimension(M)
    Y = allocate_result(M, get_vector, p, p .* 1im)
    @assert size(X) == (N * (N + 1),)
    @assert size(Y) == (N, N)
    k = 1
    for i = 1:N, j = i:N
        scale = ifelse(i == j, 1, 1 / sqrt(2))
        @inbounds Y[i, j] = Complex(X[k], X[k+1]) * scale
        @inbounds Y[j, i] = Y[i, j]
        k += 2
    end
    return Y
end

@doc raw"""
    inner(M::SymmetricMatrices, p, X, Y)

Compute the inner product of the two tangent vectors `X`, `Y` from the tangent
space at `p` on the [`SymmetricMatrices`](@ref) `M` using the restriction of the
metric from the embedding, i.e.

````math
g_p(X,Y) = \operatorname{tr}(X^{\mathrm{H}}Y),
````

where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
@inline inner(M::SymmetricMatrices, p, X, Y) = dot(X, Y)

@doc raw"""
    log(M::SymmetricMatrices, p, q)
Compute the logarithmic map from `p` to `q` on the [`SymmetricMatrices`](@ref) `M`, which
reads

````math
\log_pq = q-p.
````
"""
log(::SymmetricMatrices, ::Any...)

log!(M::SymmetricMatrices, X, p, q) = (X .= q .- p)

@doc raw"""
manifold_dimension(M::SymmetricMatrices{n,𝔽})

Return the dimension of the [`SymmetricMatrices`](@ref) matrix `M` over the number system
`𝔽`, i.e.

````math
\dim \operatorname{Sym}(n,𝔽) = \frac{n(n+1)}{2} \dim_ℝ 𝔽,
````

where $\dim_ℝ 𝔽$ is the [`real_dimension`](@ref) of `𝔽`.
"""
function manifold_dimension(::SymmetricMatrices{N,𝔽}) where {N,𝔽}
    return div(N * (N + 1), 2) * real_dimension(𝔽)
end

@doc raw"""
    norm(M::SymmetricMatrices, p, X)

Compute the norm of the tangent vector `X` from the tangent space at `p` on the
[`SymmetricMatrices`](@ref) `M`, which is the norm from the embedding, i.e.

````math
\lVert X \rVert_p = \lVert X \rVert_2
````
"""
norm(M::SymmetricMatrices, p, X) = norm(X)

@doc raw"""
    project_point(M::SymmetricMatrices, p)

Projects `p` from the embedding onto the [`SymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\operatorname{Sym}(n)}(p) = \frac{1}{2} \bigl( p + p^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project_point(::SymmetricMatrices, ::Any...)

project_point!(M::SymmetricMatrices, p) = (p .= (p + p') ./ 2)

@doc raw"""
    project_tangent(M::SymmetricMatrices, p, X)

Project the matrix `X` onto the tangent space at `p` on the [`SymmetricMatrices`](@ref) `M`,

````math
\operatorname{proj}_p(X) = \frac{1}{2} \bigl( X + X^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project_tangent(::SymmetricMatrices, ::Any...)

project_tangent!(M::SymmetricMatrices, Y, p, X) = (Y .= (X .+ transpose(X)) ./ 2)

@doc raw"""
    representation_size(M::SymmetricMatrices)

Returns the size points on the [`SymmetricMatrices`](@ref) `M` are represented as, i.e.
for the $n × n$ it's `(n,n)`.
"""
@generated representation_size(::SymmetricMatrices{N}) where {N} = (N, N)

@doc raw"""
    sharp(M::SymmetricMatrices, x, w::FVector{CotangentSpaceType})

Compute the [`sharp`](@ref sharp(M::Manifold, x, w::FVector)) isomorphism of the
[`SymmetricMatrices`](@ref) `M` on the manifold point `x` and cotangent vector `w`.

Since `M` is already a vector space over $ℝ$, this returns just the vector `w`.
"""
sharp(::SymmetricMatrices, ::Any...)

sharp!(M::SymmetricMatrices, X::TFVector, p, ξ::CoTFVector) = copyto!(X, ξ)

function show(io::IO, ::SymmetricMatrices{n,F}) where {n,F}
    print(io, "SymmetricMatrices($(n), $(F))")
end

@doc raw"""
    vector_transport_to(M::SymmetricMatrices, p, X, q, ::ParallelTransport)

Compute the parallel
[`vector_transport_to`](@ref vector_transport_to(M::Manifold, p, X, y, ParallelTransport()))
of `X` from the tangent space at `p` on the [`SymmetricMatrices`](@ref) `M` to `q`.
Since the metric is inherited from the embedding space, this is just the identity, i.e.

````math
\mathcal P_{q←p}(X) = X.
````
"""
vector_transport_to(::SymmetricMatrices, ::Any...)

function vector_transport_to!(M::SymmetricMatrices, Y, p, X, q, ::ParallelTransport)
    return copyto!(Y, X)
end

@doc raw"""
    zero_tangent_vector(M, p)

Return the zero tangent vector for the tangent space at `p` on the
[`SymmetricMatrices`](@ref) `M`, i.e. the zero matrix.
"""
zero_tangent_vector(::SymmetricMatrices, ::Any...)

zero_tangent_vector!(M::SymmetricMatrices, X, p) = fill!(X, 0)
