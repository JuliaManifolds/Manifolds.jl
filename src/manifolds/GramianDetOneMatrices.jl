@doc raw"""
    GramianDetOneMatrices{T} <: AbstractDecoratorManifold{ℝ}

The [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)
consisting of the real- or complex-valued matrices of gramian determinant one,
that is the set

```math
\bigl\{p  ∈ ℝ^{n×k}\ \big|\ \det(p'*p) = 1 \bigr\},
`````.

The tangent space at any point `p` is the set of matrices `X` satisfying `tr(p(p'*p)^{-1}X)=0

# Constructor

    GramianDetOneMatrices(n::Int)

Generate the manifold of ``n×k`` matrices of gramian determinant one.
"""
struct GramianDetOneMatrices{T} <: AbstractManifold{ℝ}
    size::T
end

function GramianDetOneMatrices(n::Int, k::Int; parameter::Symbol = :type)
    size = ManifoldsBase.wrap_type_parameter(parameter, (n,k))
    return GramianDetOneMatrices{typeof(size)}(size)
end

function representation_size(M::GramianDetOneMatrices{Tuple{Int, Int}})
    return (get_parameter(M.size)[1],get_parameter(M.size)[2])
end

function representation_size(::GramianDetOneMatrices{ManifoldsBase.TypeParameter{Tuple{n, k}}}) where {n, k}
    return (n,k)
end


@doc raw"""
    check_point(M::GramianDetOneMatrices{n}, p; kwargs...)

Check whether `p` is a valid manifold point on the [`DeterminantOneMatrices`](@ref) `M`, i.e.
whether `p'*p` has a determinant of ``1``.

The check is performed with `isapprox` and all keyword arguments are passed to this
"""
function check_point(M::GramianDetOneMatrices, p; kwargs...)
    if !isapprox(det(p'*p), 1; kwargs...)
        return DomainError(
            det(p'*p),
            "The point $(p) does not lie on $(M), since its determinant is $(det(p'*p)) and not 1.",
        )
    end
    return nothing
end

"""
    check_vector(M::GramianDetOneMatrices{n,k}, p, X; kwargs... )

Check whether `X` is a tangent vector to manifold point `p` on the
[`DeterminantOneMatrices`](@ref) `M`, which are all matrices `X`of size ``n×k``
with `trace(inv(p'*p)*p'*X)=0` .
"""
function check_vector(M::GramianDetOneMatrices, p, X; kwargs...)
    if !isapprox(tr(inv(p'*p)*p'*X), 0; kwargs...)
        return DomainError(
            tr(inv(p'*p)*p'*X),
            "The tangent vector $(X) does not lie in the Tangent space at $(p) of $(M), since $(tr(inv(p'*p)*p'*X)) is not zero.",
        )
    end
    return nothing
end

	
embed(::GramianDetOneMatrices, p) = p
embed(::GramianDetOneMatrices, p, X) = X

#Todo: Doku
function get_embedding(::GramianDetOneMatrices{ManifoldsBase.TypeParameter{Tuple{n, k}}}) where {n, k}
    return Euclidean(n, k)
end

#Todo: Doku
function get_embedding(M::GramianDetOneMatrices{Tuple{Int, Int}})
    n = get_parameter(M.size)[1]
    k = get_parameter(M.size)[2]
    return Euclidean(n, k)
end

function ManifoldsBase.get_embedding_type(::GramianDetOneMatrices)
    return ManifoldsBase.EmbeddedSubmanifoldType()
 end
	
function manifold_dimension(M::GramianDetOneMatrices{Tuple{Int, Int}})
    return get_parameter(M.size)[1]*get_parameter(M.size)[2]-1
end

#Todo: Doku
function manifold_dimension(::GramianDetOneMatrices{ManifoldsBase.TypeParameter{Tuple{n, k}}}) where {n, k}
    return n*k-1
end

@doc raw"""
    project(G::GramianDetOneMatrices, p)
    project!(G::GramianDetOneMatrices, q, p)

    Project point onto the manifold by diving by an appropriate power (´´1/(2k)´´)of its gramian determinant
"""
project(::GramianDetOneMatrices, p)

function project!(M::GramianDetOneMatrices, q, p)
    k = get_parameter(M.size)[2]
    grdetp = det(p'*p)
    isapprox(grdetp, 1) && return copyto!(q, p)
    q .= p./(grdetp^(1/2/k))
    return q
end

#Todo: Doku
function retract_project!(M::GramianDetOneMatrices, q, p, X)
    q .= p+X
    qq = copy(M,q)
    return project!(M,q,qq)
end
# Todo: Doku
default_retraction_method(::GramianDetOneMatrices) = ProjectionRetraction()

@doc raw"""
    project(G::GramianDetOneMatrices, p, X)
    project!(G::GramianDetOneMatrices, Y, p, X)

Orthogonally project ``X ∈ ℝ^{n×k}`` onto the tangent space of ``p`` to the
[`DeterminantOneMatrices`](@ref).
TODO Dokumentiere Formel

"""
project(::GramianDetOneMatrices, p, X)


function project!(G::GramianDetOneMatrices, Y, p, X)

    n, k = get_parameter(M.size)
    A = inv(p'*p)
	Y.=p*A
 	alpha = dot(X,Y)/sum(A)# inner(get_embedding(G),p,X,Y)/ inner(get_embedding(G),p,Y,Y)
	Y .*= -alpha
	Y .+= X
	return Y
end

function inner(G::GramianDetOneMatrices,p,X,Y)
    return inner(get_embedding(G),p,X,Y)
end

#distance(::GramianDetOneMatrices, p, q, r::Real = 2) = norm(p - reshape(q,size(p)), r)

function get_vectors(
        M::GramianDetOneMatrices,
        p,
        B::DefaultOrthonormalBasis{ℝ, TangentSpaceType};
        kwargs...,
    )
    n, k = get_parameter(M.size)
    pp = copy(M,p)
    A = pp'*pp
    q = pp/A
    pperp = nullspace([reshape(q,n*k,1) zeros(n*k, n*k-1)]')
    V=[reshape(pperp[:,i],n,k) for i in 1:n*k-1]
    return V
end

zero_vector!(::GramianDetOneMatrices, X, p) = fill!(X, 0.0)


function get_vector_orthonormal!(M::GramianDetOneMatrices, Y, p, c, N::RealNumbers)
    V = get_vectors(M, p, DefaultOrthonormalBasis())
    fill!(Y,0.0)
    length(c) < length(V) && error(
        "Coordinate vector too short. Expected $(length(V)), but only got $(length(c)) entries.",
    )
    @inbounds for i in 1:length(V)
        Y .+= c[i] .* V[i]
    end
    return Y
end

# TODO: brauchen wir das?
function Random.rand!(M::GramianDetOneMatrices, pX; kwargs...)
    return rand!(Random.default_rng(), M, pX; kwargs...)
end
# TODO: gescheit implementieren:
function Random.rand!(
        rng::AbstractRNG,
        M::GramianDetOneMatrices,
        pX;
        vector_at = nothing,
        kwargs...,
    )
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
