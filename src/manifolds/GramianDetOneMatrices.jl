#TODO: Maybe discuss name and use “UnitVolumeMatrices”?
@doc raw"""
    GramianDetOneMatrices{T} <: AbstractDecoratorManifold{ℝ}

The [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)
consisting of the real- or complex-valued matrices of Gramian determinant one,
that is the set

```math
\bigl\{p  ∈ ℝ^{n×k}\ \big|\ \det(p'*p) = 1 \bigr\},
`````.

The tangent space at any point `p` is the set of matrices `X` satisfying `tr(p(p'*p)^{-1}X)=0

# Constructor

    GramianDetOneMatrices(n, k; parameter = :type)

Generate the manifold of ``n×k`` matrices of gramian determinant one.
"""
struct GramianDetOneMatrices{T} <: AbstractManifold{ℝ}
    size::T
end
function GramianDetOneMatrices(n::Int, k::Int; parameter::Symbol = :type)
    size = ManifoldsBase.wrap_type_parameter(parameter, (n, k))
    return GramianDetOneMatrices{typeof(size)}(size)
end

@doc raw"""
    check_point(M::GramianDetOneMatrices{n}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`GramianDetOneMatrices`](@ref) `M`, i.e.
whether `p'*p` has a determinant of ``1``.

The check is performed with `isapprox` and all keyword arguments are passed to this
"""
function check_point(M::GramianDetOneMatrices, p; kwargs...)
    if !isapprox(det(p' * p), 1; kwargs...)
        return DomainError(
            det(p' * p),
            "The point $(p) does not lie on $(M), since its determinant is $(det(p' * p)) and not 1.",
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M::GramianDetOneMatrices{n,k}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`GramianDetOneMatrices`](@ref) `M`, which are all matrices `X`of size ``n×k``
with ``\mathrm{tr}((p'*p)^{-1} p' X)=0``.
"""
function check_vector(M::GramianDetOneMatrices, p, X; kwargs...)
    if !isapprox(tr(inv(p' * p) * p' * X), 0; kwargs...)
        return DomainError(
            tr(inv(p' * p) * p' * X),
            "The tangent vector $(X) does not lie in the Tangent space at $(p) of $(M), since $(tr(inv(p' * p) * p' * X)) is not zero.",
        )
    end
    return nothing
end

"""
    default_retraction_method(M::GramianDetOneMatrices)

Return the default retraction method for the [`GramianDetOneMatrices`](@ref) manifold `M`,
which is the [`ProjectionRetraction`](@extref `ManifoldsBase.ProjectionRetraction`)
"""
default_retraction_method(::GramianDetOneMatrices) = ProjectionRetraction()

embed(::GramianDetOneMatrices, p) = p
embed(::GramianDetOneMatrices, p, X) = X

"""
    get_embedding(M::GramianDetOneMatrices{(n,k)})

Return the embedding manifold of the [`GramianDetOneMatrices`](@ref) `M`,
which is the space of matrices ``ℝ^{n×k}``, see [`Euclidean`](@ref)`(n, k)`.
"""
get_embedding(::GramianDetOneMatrices)

function get_embedding(::GramianDetOneMatrices{ManifoldsBase.TypeParameter{Tuple{n, k}}}) where {n, k}
    return Euclidean(n, k)
end
function get_embedding(M::GramianDetOneMatrices{Tuple{Int, Int}})
    n = get_parameter(M.size)[1]
    k = get_parameter(M.size)[2]
    return Euclidean(n, k)
end

function ManifoldsBase.get_embedding_type(::GramianDetOneMatrices)
    return ManifoldsBase.EmbeddedSubmanifoldType()
end

# TODO: Check whether necessary, the submanifold should pass this on I think?
# otherwise switch to embedding_type `IsometricallyEmbeddedManifoldType`, cf. `Sphere`,
# which is the way to specify we inherthe the inner product from the embedding manifold
function inner(G::GramianDetOneMatrices, p, X, Y)
    return inner(get_embedding(G), p, X, Y)
end

"""
    manifold_dimension(M::GramianDetOneMatrices{(n, k)})

Return the dimension of the [`GramianDetOneMatrices`](@ref) manifold `M`, which is
given by ``n*k - 1`` for ``n×k`` matrices.
"""
function manifold_dimension(M::GramianDetOneMatrices)
    return manifold_dimension(get_embedding(M)) - 1
end

@doc raw"""
    project(G::GramianDetOneMatrices, p)
    project!(G::GramianDetOneMatrices, q, p)

    Project point onto the manifold by diving by an appropriate power (´´1/(2k)´´)of its gramian determinant
"""
project(::GramianDetOneMatrices, p)

function project!(M::GramianDetOneMatrices, q, p)
    k = get_parameter(M.size)[2]
    grdetp = det(p' * p)
    isapprox(grdetp, 1) && return copyto!(q, p)
    q .= p ./ (grdetp^(1 / 2 / k))
    return q
end

@doc raw"""
    project(G::GramianDetOneMatrices, p, X)
    project!(G::GramianDetOneMatrices, Y, p, X)

Orthogonally project ``X ∈ ℝ^{n×k}`` onto the tangent space of ``p`` to the
[`GramianDetOneMatrices`](@ref).
TODO Dokumentiere Formel

"""
project(::GramianDetOneMatrices, p, X)

function project!(M::GramianDetOneMatrices, Y, p, X)
    A = inv(p' * p)
    Y .= p * A
    alpha = dot(X, Y) / sum(A) # inner(get_embedding(G),p,X,Y)/ inner(get_embedding(G),p,Y,Y)
    Y .*= -alpha
    Y .+= X
    return Y
end

"""
    representation_size(M::GramianDetOneMatrices)

Return the size of points on the [`GramianDetOneMatrices`](@ref) manifold `M`.
"""
function representation_size(M::GramianDetOneMatrices{Tuple{Int, Int}})
    return (get_parameter(M.size)[1], get_parameter(M.size)[2])
end
function representation_size(::GramianDetOneMatrices{ManifoldsBase.TypeParameter{Tuple{n, k}}}) where {n, k}
    return (n, k)
end

# TODO: Necessary? With project (see above) this should be available automatically?
function retract_project!(M::GramianDetOneMatrices, q, p, X)
    q .= p + X
    qq = copy(M, q)
    return project!(M, q, qq)
end

#distance(::GramianDetOneMatrices, p, q, r::Real = 2) = norm(p - reshape(q,size(p)), r)

function get_vectors(
        M::GramianDetOneMatrices, p, ::DefaultOrthonormalBasis; kwargs...,
    )
    n, k = get_parameter(M.size)
    pp = copy(M, p)
    A = pp' * pp
    q = pp / A
    pperp = nullspace([reshape(q, n * k, 1) zeros(n * k, n * k - 1)]')
    V = [reshape(pperp[:, i], n, k) for i in 1:(n * k - 1)]
    return V
end

function get_vector_orthonormal!(M::GramianDetOneMatrices, Y, p, c, N::RealNumbers)
    V = get_vectors(M, p, DefaultOrthonormalBasis())
    fill!(Y, 0.0)
    length(c) < length(V) && error(
        "Coordinate vector too short. Expected $(length(V)), but only got $(length(c)) entries.",
    )
    @inbounds for i in 1:length(V)
        Y .+= c[i] .* V[i]
    end
    return Y
end

# TODO: gescheit implementieren – und dokumentieren...aktuell gibt dies auch noch Dinge zurück, die nicht GramianDetOne sind
function Random.rand!(
        rng::AbstractRNG,
        M::GramianDetOneMatrices,
        pX;
        vector_at = nothing,
        kwargs...,
    )
    #TODO: Also distignuish point p and tangent vector X (vector_at is set)
    rand!(rng, get_embedding(M), pX; kwargs...)
    return pX
end

function Base.show(io::IO, ::GramianDetOneMatrices{ManifoldsBase.TypeParameter{Tuple{n, k}}}) where {n, k}
    return print(io, "GramianDetOneMatrices($(n), $(k))")
end
function Base.show(io::IO, M::GramianDetOneMatrices{Tuple{Int, Int}})
    n = get_parameter(M.size)[1]
    k = get_parameter(M.size)[2]
    return print(io, "GramianDetOneMatrices($(n), $(k); parameter=:field)")
end

zero_vector!(::GramianDetOneMatrices, X, p) = fill!(X, 0.0)
