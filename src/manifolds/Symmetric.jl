@doc raw"""
    SymmetricMatrices{n,𝔽} <: Manifold

The [`Manifold`](@ref) $ \operatorname{Sym}(n)$ consisting of the real- or complex-valued
symmetric matrices of size $n × n$, i.e. the set

````math
\operatorname{Sym}(n) = \bigl\{p  ∈ 𝔽^{n × n} \big| p^{\mathrm{H}} = p \bigr\},
````
where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transpose,
and the field $𝔽 ∈ \{ ℝ, ℂ\}$.

Though it is slightly redundant, usually the matrices are stored as $n × n$ arrays.

# Constructor

    SymmetricMatrices(n::Int, field::AbstractNumbers=ℝ)

Generate the manifold of $n × n$ symmetric matrices.
"""
struct SymmetricMatrices{n,𝔽} <: AbstractEmbeddedManifold{DefaultIsometricEmbedding}
    manifold::Euclidean{Tuple{n,n},𝔽}
end

function SymmetricMatrices(n::Int, field::AbstractNumbers = ℝ)
    SymmetricMatrices{n,field}(Euclidean(n,n; field=field))
end

base_manifold(M::SymmetricMatrices) = M

@doc raw"""
    check_manifold_point(M::SymmetricMatrices{n,𝔽}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`SymmetricMatrices`](@ref) `M`, i.e.
whether `p` is a symmetric matrix of size `(n,n)` with values from the corresponding
[`AbstractNumbers`](@ref) `𝔽`.

The tolerance for the symmetry of `p` can be set using `kwargs...`.
"""
function check_manifold_point(M::SymmetricMatrices{n,𝔽}, p, ::Val{false}; kwargs...) where {n,𝔽}
    if (𝔽 === ℝ) && !(eltype(p) <: Real)
        return DomainError(
            eltype(p),
            "The matrix $(p) does not lie on $M, since its values are not real.",
        )
    end
    if (𝔽 === ℂ) && !(eltype(p) <: Real) && !(eltype(p) <: Complex)
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
    check_tangent_vector(M::SymmetricMatrices{n,𝔽}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`SymmetricMatrices`](@ref) `M`, i.e. `X` has to be a symmetric matrix of size `(n,n)`
and its values have to be from the correct [`AbstractNumbers`](@ref).

The tolerance for the symmetry of `p` and `X` can be set using `kwargs...`.
"""
function check_tangent_vector(M::SymmetricMatrices{n,𝔽}, p, X, ::Val{false}; kwargs...) where {n,𝔽}
    t = check_manifold_point(M, p; kwargs...)
    t === nothing || return t
    if (𝔽 === ℝ) && !(eltype(X) <: Real)
        return DomainError(
            eltype(X),
            "The matrix $(X) is not a tangent to a point on $M, since its values are not real.",
        )
    end
    if (𝔽 === ℂ) && !(eltype(X) <: Real) && !(eltype(X) <: Complex)
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

embed!(M::SymmetricMatrices,p, q) = (q .= p)

_vec_to_sym(v,n) = Symmetric([ i<=j ? v[div(j*(j-1),2)+i] : 0 for i=1:n, j=1:n ])
function _basis_vec_sym(k,n,unit=true,𝔽::AbstractNumbers=ℝ)
        v = _vec_to_sym([ (i==k ? 1.0 : 0.0) + (𝔽==ℂ ? 0.0*im : 0.0) for i=1:div(n*(n+1),2)],n)
        return (norm(Diagonal(v)) == 0 && unit ? 1/sqrt(2) : 1)*v;
end

function get_basis(M::SymmetricMatrices{n,ℝ}, p, B::ArbitraryOrthonormalBasis{ℝ}, ::Val{false}) where {n}
    vecs = [_basis_vec_sym(k,n,true,ℝ) for k=1:manifold_dimension(M)]
    return PrecomputedOrthonormalBasis(vecs)
end
function get_basis(M::SymmetricMatrices{n,ℂ}, p, B::ArbitraryOrthonormalBasis{ℂ}, ::Val{false}) where {n}
    vecs = [_basis_vec_sym(k,n,true,ℂ) for k=1:manifold_dimension(M)]
    return PrecomputedOrthonormalBasis([vec; im*vecs])
end
function get_basis(M::SymmetricMatrices, p, B::DiagonalizingOrthonormalBasis, ::Val{false}) where {n}
    vecs = get_basis(M, p, ArbitraryOrthonormalBasis()).vectors
    kappas = zeros(real(eltype(p)), manifold_dimension(M))
    return PrecomputedDiagonalizingOrthonormalBasis(vecs, kappas)
end

function get_basis(M::SymmetricMatrices{n,T}, p, B::AbstractOrthonormalBasis{T}, ::Val{false}) where {n,T}
    return PrecomputedProductOrthonormalBasis(
        [_basis_sym(k,manifold_dimension(M),true,T) for k=1:manifold_dimension(M)]
    )
end

function get_coordinates(
     M::SymmetricMatrices{N,T},
     p,
     X,
     B::ArbitraryOrthonormalBasis{T},
     ::Val{false}
) where {N,T}
    V = get_basis(M,p,B)
    return [inner(M,p,X,Y) for Y in V.vectors]
end

function get_coordinates(
     M::SymmetricMatrices{N,T},
     p,
     X,
     B::AbstractPrecomputedOrthonormalBasis{T},
     ::Val{false}
) where {N,T}
    return [inner(M,p,X,Y) for Y in B.vectors]
end

function get_vector(
    M::SymmetricMatrices{N,T},
    p,
    X,
    B::ArbitraryOrthonormalBasis{T},
    ::Val{false}
) where {N,T}
    V = get_basis(M,p,B)
    dim = manifold_dimension(M)
    @assert size(X) == (dim,)
    return sum(X.*V.vectors)
end
function get_vector(
    M::SymmetricMatrices{N,T},
    p,
    X,
    B::AbstractPrecomputedOrthonormalBasis{T},
    ::Val{false}
) where {N,T}
    dim = manifold_dimension(M)
    @assert size(X) == (dim,)
    return sum(X.*B.vectors)
end
@doc raw"""
manifold_dimension(M::SymmetricMatrices{n,𝔽})

Return the dimension of the [`SymmetricMatrices`](@ref) matrix `M` over the number system
`𝔽`, i.e.

````math
\dim \operatorname{Sym}(n,𝔽) = \frac{n(n+1)}{2} \dim_ℝ 𝔽,
````

where $\dim_ℝ 𝔽$ is the [`real_dimension`](@ref) of `𝔽`.
"""
function manifold_dimension(::SymmetricMatrices{N,𝔽}, ::Val{false}) where {N,𝔽}
    return div(N * (N + 1), 2) * real_dimension(𝔽)
end

@doc raw"""
    project_point(M::SymmetricMatrices, p)

Projects `p` from the embedding onto the [`SymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\operatorname{Sym}(n)}(p) = \frac{1}{2} \bigl( p + p^{\mathrm{H}} \bigr),
````

where $\cdot^{\mathrm{H}}$ denotes the hermitian, i.e. complex conjugate transposed.
"""
project_point(::SymmetricMatrices, ::Any...)

project_point!(M::SymmetricMatrices, p, ::Val{false}) = (p .= (p + p') ./ 2)

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

function show(io::IO, ::SymmetricMatrices{n,F}) where {n,F}
    print(io, "SymmetricMatrices($(n), $(F))")
end