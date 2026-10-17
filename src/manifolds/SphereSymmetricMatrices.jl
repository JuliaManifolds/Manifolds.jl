@doc raw"""
    SphereSymmetricMatrices{𝔽, T} <: AbstractDecoratorManifold{𝔽}

The [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)  consisting of the ``n×n`` symmetric matrices
of unit Frobenius norm, i.e.
````math
\mathcal{S}_{\text{sym}} :=\bigl\{p  ∈ 𝔽^{n×n}\ \big|\ p^{\mathrm{H}} = p, \lVert p \rVert = 1 \bigr\},
````
where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transpose,
and the field ``𝔽 ∈ \{ ℝ, ℂ\}``.

# Constructor
    SphereSymmetricMatrices(n[, field=ℝ]; parameter::Symbol=:type)

Generate the manifold of `n`-by-`n` symmetric matrices of unit Frobenius norm.
"""
struct SphereSymmetricMatrices{𝔽, T} <: AbstractDecoratorManifold{𝔽}
    size::T
end

function SphereSymmetricMatrices(n::Int, field::AbstractNumbers = ℝ; parameter::Symbol = :type)
    n < 1 && throw(
        DomainError(
            n,
            "The matrix size n of symmetric matrices of unit norm must be positive, but was $(n).",
        ),
    )
    size = wrap_type_parameter(parameter, (n,))
    return SphereSymmetricMatrices{field, typeof(size)}(size)
end

@doc raw"""
    check_point(M::SphereSymmetricMatrices, p; kwargs...)

Check whether the matrix is a valid point on the [`SphereSymmetricMatrices`](@ref) `M`,
i.e. is an `n`-by-`n` symmetric matrix of unit Frobenius norm.

The tolerance for the symmetry of `p` can be set using `kwargs...`.
"""
function check_point(
        M::SphereSymmetricMatrices,
        p::T;
        atol::Real = sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
        kwargs...,
    ) where {T}
    if !isapprox(norm(p - p'), 0; atol = atol, kwargs...)
        return DomainError(
            norm(p - p'),
            "The point $(p) does not lie on $M, since it is not symmetric.",
        )
    end
    return nothing
end

"""
    check_vector(M::SphereSymmetricMatrices, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SphereSymmetricMatrices`](@ref) `M`. That is, `X` has to be a symmetric matrix of size `(n,n)`
that is orthogonal to `p` with respect to the Frobenius inner product.

The tolerance for the symmetry of `p` and `X` can be set using `kwargs...`.
"""
function check_vector(
        M::SphereSymmetricMatrices,
        p,
        X::T;
        atol::Real = sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
        kwargs...,
    ) where {T}
    if !isapprox(norm(X - X'), 0; atol = atol, kwargs...)
        return DomainError(
            norm(X - X'),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not symmetric.",
        )
    end
    return nothing
end

embed(::SphereSymmetricMatrices, p) = p
embed(::SphereSymmetricMatrices, p, X) = X

function get_embedding(::SphereSymmetricMatrices{𝔽, TypeParameter{Tuple{n}}}) where {n, 𝔽}
    return ArraySphere(n, n; field = 𝔽)
end
function get_embedding(M::SphereSymmetricMatrices{𝔽, Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return ArraySphere(n, n; field = 𝔽, parameter = :field)
end

function ManifoldsBase.get_embedding_type(::SphereSymmetricMatrices)
    return ManifoldsBase.EmbeddedSubmanifoldType()
end

"""
    is_flat(::SphereSymmetricMatrices)

Return false. [`SphereSymmetricMatrices`](@ref) is not a flat manifold.
"""
is_flat(M::SphereSymmetricMatrices) = false

@doc raw"""
    manifold_dimension(M::SphereSymmetricMatrices{𝔽})

Return the manifold dimension of the [`SphereSymmetricMatrices`](@ref) `n`-by-`n` symmetric matrix `M` of unit
Frobenius norm over the number system `𝔽`, i.e.

````math
\begin{aligned}
\dim(\mathcal{S}_{\text{sym}})(n,ℝ) &= \frac{n(n+1)}{2} - 1,\\
\dim(\mathcal{S}_{\text{sym}})(n,ℂ) &= 2\frac{n(n+1)}{2} - n -1.
\end{aligned}
````
"""
function manifold_dimension(M::SphereSymmetricMatrices{𝔽}) where {𝔽}
    n = get_parameter(M.size)[1]
    return div(n * (n + 1), 2) * real_dimension(𝔽) - (𝔽 === ℂ ? n : 0) - 1
end

@doc raw"""
    project(M::SphereSymmetricMatrices, p)

Projects `p` from the embedding onto the [`SphereSymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\mathcal{S}_{\text{sym}}}(p) = \frac{s}{\lVert s \rVert},
\qquad\text{where } s = \frac{1}{2} \bigl( p + p^{\mathrm{H}} \bigr),
````
where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transposed,
and ``\lVert ⋅ \rVert`` denotes the Frobenius norm.
"""
project(::SphereSymmetricMatrices, ::Any)

function project!(M::SphereSymmetricMatrices, q, p)
    return project!(get_embedding(M), q, (p + p') ./ 2)
end

@doc raw"""
    project(M::SphereSymmetricMatrices, p, X)

Project the matrix `X` onto the tangent space at `p` on the [`SphereSymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_p(X) = \frac{X + X^{\mathrm{H}}}{2} - ⟨p, \frac{X + X^{\mathrm{H}}}{2}⟩p,
````
where ``⋅^{\mathrm{H}}`` denotes the Hermitian, i.e. complex conjugate transposed.
"""
project(::SphereSymmetricMatrices, ::Any, ::Any)

function project!(M::SphereSymmetricMatrices, Y, p, X)
    return project!(get_embedding(M), Y, p, (X .+ X') ./ 2)
end

function Random.rand!(
        rng::AbstractRNG, M::SphereSymmetricMatrices, pX;
        vector_at = nothing, σ::Real = one(real(eltype(pX)))
    )
    if vector_at === nothing
        project!(M, pX, randn(rng, eltype(pX), representation_size(M)))
    else
        n = σ * randn(rng, eltype(pX), size(pX)) # Gaussian in the embedding
        project!(M, pX, vector_at, n) # project to TpM (keeps Gaussianness)
    end
    return pX
end

function representation_size(M::SphereSymmetricMatrices)
    n = get_parameter(M.size)[1]
    return (n, n)
end

function Base.show(io::IO, ::SphereSymmetricMatrices{𝔽, TypeParameter{Tuple{n}}}) where {n, 𝔽}
    return print(io, "SphereSymmetricMatrices($(n), $(𝔽))")
end
function Base.show(io::IO, M::SphereSymmetricMatrices{𝔽, Tuple{Int}}) where {𝔽}
    n = get_parameter(M.size)[1]
    return print(io, "SphereSymmetricMatrices($(n), $(𝔽); parameter=:field)")
end
