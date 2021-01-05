@doc raw"""
    SphereSymmetricMatrices{n,𝔽} <: AbstractEmbeddedManifold{ℝ,TransparentIsometricEmbedding}

The [`Manifold`](@ref) consisting of the $n × n$ symmetric matrices
of unit Frobenius norm, i.e.
````math
\mathcal{S}_{\text{sym}} :=\bigl\{p  ∈ 𝔽^{n × n}\ \big|\ p^{\mathrm{H}} = p, \lVert p \rVert = 1 \bigr\},
````
where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transpose,
and the field $𝔽 ∈ \{ ℝ, ℂ\}$.

# Constructor
    SphereSymmetricMatrices(n[, field=ℝ])

Generate the manifold of `n`-by-`n` symmetric matrices of unit Frobenius norm.
"""
struct SphereSymmetricMatrices{N,𝔽} <:
       AbstractEmbeddedManifold{𝔽,TransparentIsometricEmbedding} end

function SphereSymmetricMatrices(n::Int, field::AbstractNumbers=ℝ)
    return SphereSymmetricMatrices{n,field}()
end

@doc raw"""
    check_manifold_point(M::SphereSymmetricMatrices{n,𝔽}, p; kwargs...)

Check whether the matrix is a valid point on the [`SphereSymmetricMatrices`](@ref) `M`,
i.e. is an `n`-by-`n` symmetric matrix of unit Frobenius norm.

The tolerance for the symmetry of `p` can be set using `kwargs...`.
"""
function check_manifold_point(M::SphereSymmetricMatrices{n,𝔽}, p; kwargs...) where {n,𝔽}
    mpv =
        invoke(check_manifold_point, Tuple{supertype(typeof(M)),typeof(p)}, M, p; kwargs...)
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
    check_tangent_vector(M::SphereSymmetricMatrices{n,𝔽}, p, X; check_base_point = true, kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SphereSymmetricMatrices`](@ref) `M`, i.e. `X` has to be a symmetric matrix of size `(n,n)`
of unit Frobenius norm.
The optional parameter `check_base_point` indicates, whether to call
 [`check_manifold_point`](@ref)  for `p`.

The tolerance for the symmetry of `p` and `X` can be set using `kwargs...`.
"""
function check_tangent_vector(
    M::SphereSymmetricMatrices{n,𝔽},
    p,
    X;
    check_base_point=true,
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
        check_base_point=false, # already checked above
        kwargs...,
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

function decorated_manifold(M::SphereSymmetricMatrices{n,𝔽}) where {n,𝔽}
    return ArraySphere(n, n; field=𝔽)
end

@doc raw"""
    manifold_dimension(M::SphereSymmetricMatrices{n,𝔽})

Return the manifold dimension of the [`SphereSymmetricMatrices`](@ref) `n`-by-`n` symmetric matrix `M` of unit
Frobenius norm over the number system `𝔽`, i.e.

````math
\begin{aligned}
\dim(\mathcal{S}_{\text{sym}})(n,ℝ) &= \frac{n(n+1)}{2} - 1,\\
\dim(\mathcal{S}_{\text{sym}})(n,ℂ) &= 2\frac{n(n+1)}{2} - n -1.
\end{aligned}
````
"""
function manifold_dimension(::SphereSymmetricMatrices{n,𝔽}) where {n,𝔽}
    return div(n * (n + 1), 2) * real_dimension(𝔽) - (𝔽 === ℂ ? n : 0) - 1
end

@doc raw"""
    project(M::SphereSymmetricMatrices, p)

Projects `p` from the embedding onto the [`SphereSymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\mathcal{S}_{\text{sym}}}(p) = \frac{1}{2} \bigl( p + p^{\mathrm{H}} \bigr),
````
where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transposed.
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
where $\cdot^{\mathrm{H}}$ denotes the Hermitian, i.e. complex conjugate transposed.
"""
project(::SphereSymmetricMatrices, ::Any, ::Any)

function project!(M::SphereSymmetricMatrices, Y, p, X)
    return project!(get_embedding(M), Y, p, (X .+ X') ./ 2)
end

@generated representation_size(::SphereSymmetricMatrices{n,𝔽}) where {n,𝔽} = (n, n)

function Base.show(io::IO, ::SphereSymmetricMatrices{n,𝔽}) where {n,𝔽}
    return print(io, "SphereSymmetricMatrices($(n), $(𝔽))")
end
