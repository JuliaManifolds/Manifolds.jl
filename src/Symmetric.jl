@doc doc"""
    Symmetric{N} <: Manifold

The manifold $ Sym (N)$ consisting of the real-valued, symmetric matrices of sice $ n\times n$, i.e. the set 

$\text{Sym}(n) = \bigl\{A \in \mathbb{R}^{n\times n} \big| A^{\mathrm{T}} = A \bigr\}$

Being slightly unefficient, the matrices are safed as $n\times n$ arrays despite the redundance of the lower triangle.


# Constructor

    Symmetric(n)

"""
struct Symmetric{N} <: Manifold end
Symmetric(n::Int) = Symmetric{n}()

@traitimpl HasMetric{Symmetric,EuclideanMetric}

function representation_size(::Symmetric{N}) where N
    return (N,N)
end

@doc doc"""
    manifold_dimension(M::Symmetric)

Return the dimension of the manifold $M=\text{Sym}(n)$, i.e. $\frac{n(n+1)}{2}$.
"""
manifold_dimension(M::Symmetric{N}) where {N} = N*(N+1)/2

project_point!(M::Symmetric, x) = (x += transpose(x))

project_tangent!(M::Symmetric, w, x, v) = (x -= transpose(x))

@doc doc"""
    inner(M::Symmetric, x, w, v)

compute the inner product of the two tangent vectors `w,v` from the tangent
plane at `x` on the manifold $M=\text{Sym}(n)$ using the restriction of the
metric from the embedding, i.e. $ (v,w)_x = v^\mathrm{T}w $.
"""
inner(M::Symmetric, x, w, v) = dot(w, v)

norm(M::Symmetric, x, v) = norm(v)

"""
    distance(M::Symmetric, x, y)

by using the inherited metric, i.e. taking the Frobenius-norm of the difference.

"""

distance(M::Symmetric, x, y) = norm(x-y)

function exp!(M::Symmetric, y, x, v)rans
    y = x + v
    return y
end

function log!(M::Symmetric, v, x, y)
    v = y-x
    return v
end

function zero_tangent_vector!(M::Symmetric, v, x)
    fill!(v, 0)
    return v
end


@doc doc"""
    vector_transport_to!(M::Symmetric, vto, x, v, y)
compute the vector transport of the tangential vector v from the point $x\in M=\text{Sym}(n)$
to the point $y\in M$. The result is stored in vto.
Since the metric is inherited from the embedding space, it is just the identity.
"""
function vector_transport_to!(M::Symmetric, vto, x, v, y)
    vto=v
    return vto
end

function flat!(M::Symmetric, v::FVector{CotangentSpaceType}, x, w::FVector{TangentSpaceType})
    copyto!(v.data, w.data)
    return v
end

function sharp!(M::Symmetric, v::FVector{TangentSpaceType}, x, w::FVector{CotangentSpaceType})
    copyto!(v.data, w.data)
    return v
end


@doc doc"""
    is_manifold_point(M::Symmetric,x)

checks, whether `x` is a valid point in the symmetric matrices $\text{Sym}(n)$, i.e. is a symmetric matrix
of size `(n,n)`.
"""
function is_manifold_point(M::Symmetric{N},x; kwargs...) where {N}
    if size(x) != (N,N)
        throw(DomainError(size(x),"The point $(x) does not lie on $M, since its size is not ($N,$N)."))
    end
    if x != transpose(x)
        throw(DomainError(norm(x), "The point $(x) does not lie on $M, since it is not symmetric."))
    end
    return true
end

"""
    is_tangent_vector(M,x,v; kwargs... )

checks whether `v` is a tangent vector to `x` on the [`Symmetric`](@ref) matrices `M`, i.e.
after [`is_manifold_point`](@ref)`(S,x)`, `v` has to be a symmetric matrix of dimension `(n,n)`.
"""
function is_tangent_vector(M::Symmetric{N},x,v; kwargs...) where N
    is_manifold_point(M,x)
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

