@doc raw"""
    SphereSymmetricMatrices{n,ℝ} <: AbstractEmbeddedManifold{ℝ,DefaultIsometricEmbeddingType}

The [`Manifold`](@ref) consisting of the $n × n$ real symmetric matrices 
of unit Frobenius norm, i.e. 
````math
\bigl\{p  ∈ ℝ^{n × n}\ \big|\ p^T = p, \lVert p \rVert = 1 \bigr\}.
````

# Constructor
    SphereSymmetricMatrices(n[, field=ℝ])

Generate the manifold of `n`-by-`n` real-valued symmetric matrices of unit Frobenius norm.
"""
struct SphereSymmetricMatrices{N,ℝ} <:
       AbstractEmbeddedManifold{ℝ,DefaultIsometricEmbeddingType} end

function SphereSymmetricMatrices(n::Int, field::AbstractNumbers = ℝ)
    return SphereSymmetricMatrices{n,field}()
end

@doc raw"""
    check_manifold_point(M::SphereSymmetricMatrices{n,ℝ}, p; kwargs...) 

Check whether the matrix is a valid point on the
[`SphereSymmetricMatrices`](@ref) `M`, i.e. is an `n`-by-`n` symmetric matrix of unit 
Frobenius norm.

The tolerance for the symmetry of `p` can be set using `kwargs...`.
"""
function check_manifold_point(M::SphereSymmetricMatrices{n,ℝ}, p; kwargs...) where {n,ℝ}
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
[`SphereSymmetricMatrices`](@ref) `M`, i.e. `X` has to be a real-valued symmetric matrix of size `(n,n)`
of unit Frobenius norm.
The optional parameter `check_base_point` indicates, whether to call
 [`check_manifold_point`](@ref)  for `p`.

The tolerance for the symmetry of `p` and `X` can be set using `kwargs...`.
"""
function check_tangent_vector(
    M::SphereSymmetricMatrices{n,𝔽},
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
    if !isapprox(norm(X - X'), 0.0; kwargs...)
        return DomainError(
            norm(X - X'),
            "The vector $(X) is not a tangent vector to $(p) on $(M), since it is not symmetric.",
        )
    end
    return nothing
end

function decorated_manifold(M::SphereSymmetricMatrices{n,ℝ}) where {n,ℝ}
    return ArraySphere{Tuple([n, n]),ℝ}
end

embed!(M::SphereSymmetricMatrices, q, p) = copyto!(q, p)
embed!(M::SphereSymmetricMatrices, Y, p, X) = copyto!(Y, X)

@doc raw"""
    manifold_dimension(M::SphereSymmetricMatrices{n,ℝ})

Return the manifold dimension of the [`SphereSymmetricMatrices`](@ref) `n`-by-`n` real-valued symmetric matrix `M` of unit
Frobenius norm, i.e.

````math
\dim(\mathcal M) = \frac{n*(n + 1)}{2} - 1.
````
"""
function manifold_dimension(::SphereSymmetricMatrices{n,ℝ}) where {n,ℝ}
    return n * (n + 1) / 2 - 1
end

@doc raw"""
    project(M::SphereSymmetricMatrices, p)

Projects `p` from the embedding onto the [`SphereSymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_{\mathcal M}(p) = \frac{p}{\lVert p \rVert}.
````
"""
project(::SphereSymmetricMatrices, ::Any)

function project!(M::SphereSymmetricMatrices, q, p)
    return project!(get_embedding(M), q, (p + transpose(p)) / 2)
end

@doc raw"""
    project(M::SphereSymmetricMatrices, p, X)

Project the matrix `X` onto the tangent space at `p` on the [`SphereSymmetricMatrices`](@ref) `M`, i.e.

````math
\operatorname{proj}_p(X) = \frac{X + X^T}{2} - ⟨p, \frac{X + X^T}{2}⟩p.
````
"""
project(::SphereSymmetricMatrices, ::Any, ::Any)

function project!(M::SphereSymmetricMatrices, Y, p, X)
    return project!(get_embedding(M), Y, p, (X + transpose(X)) / 2)
end

@generated representation_size(::SphereSymmetricMatrices{n,ℝ}) where {n,ℝ} = (n, n)

function Base.show(io::IO, ::SphereSymmetricMatrices{n,ℝ}) where {n,ℝ}
    return print(io, "SphereSymmetricMatrices($(n), $(ℝ))")
end
