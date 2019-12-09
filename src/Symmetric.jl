@doc doc"""
    SymmetricMatrices{N,T} <: Manifold

The manifold $ Sym (N)$ consisting of the `T`-valued, symmetric matrices of sice $ n\times n$, i.e. the set 

$\text{Sym}(n) = \bigl\{A \in T^{n\times n} \big| A^{\mathrm{T}} = A \bigr\}$

Being slightly unefficient, the matrices are safed as $n\times n$ arrays despite the redundance of the lower triangle.


# Constructor

    SymmetricMatrices(n,T = Real)

"""
struct SymmetricMatrices{N,T} <: Manifold end
SymmetricMatrices(N::Int,T::Type = Real) = SymmetricMatrices{N,T}()

@traitimpl HasMetric{Symmetric,EuclideanMetric}

function representation_size(::SymmetricMatrices{N,T}) where {N,T}
    return (N,N)
end

@doc doc"""
    manifold_dimension(M::SymmetricMatrices{n,T})

Return the dimension of the manifold $M=\text{Sym}(n)$, i.e. $\frac{n(n+1)}{2}$.
"""
manifold_dimension(M::SymmetricMatrices{N,T}) where {N,T} = div(N*(N+1), 2)

project_point!(M::SymmetricMatrices, x) = (x += transpose(x))

project_tangent!(M::SymmetricMatrices, w, x, v) = (x -= transpose(x))

@doc doc"""
    inner(M::SymmetricMatrices, x, w, v)

compute the inner product of the two tangent vectors `w,v` from the tangent
plane at `x` on the manifold $M=\text{Sym}(n)$ using the restriction of the
metric from the embedding, i.e. $ (v,w)_x = v^\mathrm{T}w $.
"""
inner(M::SymmetricMatrices, x, w, v) = dot(w, v)

norm(M::SymmetricMatrices, x, v) = norm(v)

"""
    distance(M::SymmetricMatrices, x, y)

by using the inherited metric, i.e. taking the Frobenius-norm of the difference.

"""

distance(M::SymmetricMatrices, x, y) = norm(x-y)

function exp!(M::SymmetricMatrices, y, x, v)
    y = x + v
    return y
end

function log!(M::SymmetricMatrices, v, x, y)
    v = y-x
    return v
end

function zero_tangent_vector!(M::SymmetricMatrices, v, x)
    fill!(v, 0)
    return v
end


@doc doc"""
    vector_transport_to!(M::SymmetricMatrices, vto, x, v, y)
compute the vector transport of the tangential vector v from the point $x\in M=\text{Sym}(n)$
to the point $y\in M$. The result is stored in vto.
Since the metric is inherited from the embedding space, this is just the identity.
"""
function vector_transport_to!(M::SymmetricMatrices, vto, x, v, y)
    vto=v
    return vto
end

function flat!(M::SymmetricMatrices, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

function sharp!(M::SymmetricMatrices, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    copyto!(v.data, w.data)
    return v
end


@doc doc"""
    is_manifold_point(M::SymmetricMatrices,x)

checks, whether `x` is a valid point in the symmetric matrices $\text{Sym}(n)$, i.e. is a symmetric matrix
of size `(n,n)` with values of type `T`.
"""
function is_manifold_point(M::SymmetricMatrices{N,T},x; kwargs...) where {N,T}
    if (T != eltype(x))
        throw(DomainError(eltype(x),"The matrix $(x) does not lie on $M, since its values are not of type $T."))
    end
    if size(x) != (N,N)
        throw(DomainError(size(x),"The point $(x) does not lie on $M, since its size is not ($N,$N)."))
    end
    if x != transpose(x)
        throw(DomainError(norm(x-transpose(x)), "The point $(x) does not lie on $M, since it is not symmetric."))
    end
    return true
end

"""
    is_tangent_vector(M,x,v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`SymmetricMatrices`](@ref) matrices `M`, i.e.
after [`is_manifold_point`](@ref)`(S,x)`, `v` has to be a symmetric matrix of dimension `(n,n)`.
"""
function is_tangent_vector(M::SymmetricMatrices{N,T},x,v; kwargs...) where {N,T}
    is_manifold_point(M,x)
    if (T != eltype(v))
        throw(DomainError(eltype(v),"The matrix $(v) is not a tangent to a point on $M, since its values are not of type $T."))
    end
    if size(v) != (N,N)
        throw(DomainError(size(v),
            "The vector $(v) is not a tangent to a point on $(M) since its size does not match ($(N),$(N))."))
    end
    if v != transpose(v)
        throw(DomainError(norm(v-transpose(v)),
            "The vector $(v) is not a tangent vector to $(x) on $(M), since it is not symmetric."
        ))
    end
    return true
end

