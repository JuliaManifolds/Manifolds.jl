@doc raw"""
    Spectrahedron{T} <: AbstractDecoratorManifold{ℝ}

The Spectrahedron manifold, also known as the set of correlation matrices (symmetric
positive semidefinite matrices) of rank ``k`` with unit trace.

````math
\begin{aligned}
\mathcal S(n,k) =
\bigl\{p ∈ ℝ^{n×n}\ \big|\ &a^\mathrm{T}pa \geq 0 \text{ for all } a ∈ ℝ^{n},\\
&\operatorname{tr}(p) = \sum_{i=1}^n p_{ii} = 1,\\
&\text{and } p = qq^{\mathrm{T}} \text{ for } q \in  ℝ^{n×k}
\text{ with } \operatorname{rank}(p) = \operatorname{rank}(q) = k
\bigr\}.
\end{aligned}
````

This manifold is working solely on the matrices ``q``. Note that this ``q`` is not unique,
indeed for any orthogonal matrix ``A`` we have ``(qA)(qA)^{\mathrm{T}} = qq^{\mathrm{T}} = p``,
so the manifold implemented here is the quotient manifold. The unit trace translates to
unit frobenius norm of ``q``.


The tangent space at ``p``, denoted ``T_p\mathcal E(n,k)``, is also represented by matrices
``Y\in ℝ^{n×k}`` and reads as

````math
T_p\mathcal S(n,k) = \bigl\{
X ∈ ℝ^{n×n}\,|\,X = qY^{\mathrm{T}} + Yq^{\mathrm{T}}
\text{ with } \operatorname{tr}(X) = \sum_{i=1}^{n}X_{ii} = 0
\bigr\}
````
endowed with the [`Euclidean`](@ref) metric from the embedding, i.e. from the ``ℝ^{n×k}``


This manifold was for example
investigated in [JourneeBachAbsilSepulchre:2010](@cite).

# Constructor

    Spectrahedron(n::Int, k::Int; parameter::Symbol=:type)

generates the manifold ``\mathcal S(n,k) \subset ℝ^{n×n}``.
"""
struct Spectrahedron{T} <: AbstractDecoratorManifold{ℝ}
    size::T
end

function Spectrahedron(n::Int, k::Int; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n, k))
    return Spectrahedron{typeof(size)}(size)
end

active_traits(f, ::Spectrahedron, args...) = merge_traits(IsIsometricEmbeddedManifold())

@doc raw"""
    check_point(M::Spectrahedron, q; kwargs...)

checks, whether `q` is a valid representation of a point ``p=qq^{\mathrm{T}}`` on the
[`Spectrahedron`](@ref) `M`, i.e. is a matrix
of size `(N,K)`, such that ``p`` is symmetric positive semidefinite and has unit trace,
i.e. ``q`` has to have unit frobenius norm.
Since by construction ``p`` is symmetric, this is not explicitly checked.
Since ``p`` is by construction positive semidefinite, this is not checked.
The tolerances for positive semidefiniteness and unit trace can be set using the `kwargs...`.
"""
function check_point(M::Spectrahedron, q; kwargs...)
    fro_n = norm(q)
    if !isapprox(fro_n, 1.0; kwargs...)
        return DomainError(
            fro_n,
            "The point $(q) does not represent a point p=qq^T on $(M) since q has not Frobenius norm 1 (and hence p not unit trace).",
        )
    end
    return nothing
end

@doc raw"""
    check_vector(M::Spectrahedron, q, Y; kwargs...)

Check whether ``X = qY^{\mathrm{T}} + Yq^{\mathrm{T}}`` is a tangent vector to
``p=qq^{\mathrm{T}}`` on the [`Spectrahedron`](@ref) `M`,
i.e. atfer [`check_point`](@ref) of `q`, `Y` has to be of same dimension as `q`
and a ``X`` has to be a symmetric matrix with trace.
The tolerance for the base point check and zero diagonal can be set using the `kwargs...`.
Note that symmetry of ``X`` holds by construction and is not explicitly checked.
"""
function check_vector(
    M::Spectrahedron,
    q,
    Y::T;
    atol::Real=sqrt(prod(representation_size(M))) * eps(real(float(number_eltype(T)))),
    kwargs...,
) where {T}
    X = q * Y' + Y * q'
    n = tr(X)
    if !isapprox(n, 0; atol=atol, kwargs...)
        return DomainError(
            n,
            "The vector $(X) is not a tangent to a point on $(M) (represented py $(q) and $(Y), since its trace is nonzero.",
        )
    end
    return nothing
end

function get_embedding(::Spectrahedron{TypeParameter{Tuple{n,k}}}) where {n,k}
    return Euclidean(n, k)
end
function get_embedding(M::Spectrahedron{Tuple{Int,Int}})
    n, k = get_parameter(M.size)
    return Euclidean(n, k; parameter=:field)
end

"""
    is_flat(::Spectrahedron)

Return false. [`Spectrahedron`](@ref) is not a flat manifold.
"""
is_flat(M::Spectrahedron) = false

@doc raw"""
    manifold_dimension(M::Spectrahedron)

returns the dimension of
[`Spectrahedron`](@ref) `M```=\mathcal S(n,k), n,k ∈ ℕ``, i.e.
````math
\dim \mathcal S(n,k) = nk - 1 - \frac{k(k-1)}{2}.
````
"""
function manifold_dimension(M::Spectrahedron)
    N, K = get_parameter(M.size)
    return N * K - 1 - div(K * (K - 1), 2)
end

"""
    project(M::Spectrahedron, q)

project `q` onto the manifold [`Spectrahedron`](@ref) `M`, by normalizing w.r.t. the
Frobenius norm
"""
project(::Spectrahedron, ::Any)

project!(::Spectrahedron, r, q) = copyto!(r, q ./ norm(q))

"""
    project(M::Spectrahedron, q, Y)

Project `Y` onto the tangent space at `q`, i.e. row-wise onto the Spectrahedron manifold.
"""
project(::Spectrahedron, ::Any...)

function project!(::Spectrahedron, Z, q, Y)
    Y2 = Y - sum(q .* Y) * q
    Z .= Y2 - q * lyap(q' * q, -(q' * Y2 - Y2' * q))
    return Z
end

@doc raw"""
    retract(M::Spectrahedron, q, Y, ::ProjectionRetraction)

compute a projection based retraction by projecting ``q+Y`` back onto the manifold.
"""
retract(::Spectrahedron, ::Any, ::Any, ::ProjectionRetraction)

function ManifoldsBase.retract_project!(M::Spectrahedron, r, q, Y)
    return project!(M, r, q .+ Y)
end
function ManifoldsBase.retract_project_fused!(M::Spectrahedron, r, q, Y, t::Number)
    return project!(M, r, q .+ t .* Y)
end

@doc raw"""
    representation_size(M::Spectrahedron)

Return the size of an array representing an element on the
[`Spectrahedron`](@ref) manifold `M`, i.e. ``n×k``, the size of such factor of ``p=qq^{\mathrm{T}}``
on ``\mathcal M = \mathcal S(n,k)``.
"""
representation_size(M::Spectrahedron) = get_parameter(M.size)

function Base.show(io::IO, M::Spectrahedron{TypeParameter{Tuple{n,k}}}) where {n,k}
    return print(io, "Spectrahedron($n, $k)")
end
function Base.show(io::IO, M::Spectrahedron{Tuple{Int,Int}})
    n, k = get_parameter(M.size)
    return print(io, "Spectrahedron($n, $k; parameter=:field)")
end

"""
    vector_transport_to(M::Spectrahedron, p, X, q)

transport the tangent vector `X` at `p` to `q` by projecting it onto the tangent space
at `q`.
"""
vector_transport_to(::Spectrahedron, ::Any, ::Any, ::Any, ::ProjectionTransport)

function vector_transport_to_project!(M::Spectrahedron, Y, p, X, q)
    project!(M, Y, q, X)
    return Y
end

@doc raw"""
    zero_vector(M::Spectrahedron,p)

returns the zero tangent vector in the tangent space of the symmetric positive
definite matrix `p` on the [`Spectrahedron`](@ref) manifold `M`.
"""
zero_vector(::Spectrahedron, ::Any...)

zero_vector!(::Spectrahedron, X, ::Any) = fill!(X, 0)
