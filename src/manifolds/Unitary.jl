
@doc raw"""
    const UnitaryMatrices{n,𝔽} = AbstarctUnitaryMatrices{n,𝔽,AbsoluteDeterminantOneMatrices}

The manifold ``U(n,𝔽)`` of ``n×n`` complex matrices (when 𝔽=ℂ) or quaternionic matrices
(when 𝔽=ℍ) such that

``p^{\mathrm{H}}p = \mathrm{I}_n,``

where ``\mathrm{I}_n`` is the ``n×n`` identity matrix.
Such matrices `p` have a property that ``\lVert \det(p) \rVert = 1``.

The tangent spaces are given by

```math
    T_pU(n) \coloneqq \bigl\{
    X \big| pY \text{ where } Y \text{ is skew symmetric, i. e. } Y = -Y^{\mathrm{H}}
    \bigr\}
```

But note that tangent vectors are represented in the Lie algebra, i.e. just using ``Y`` in
the representation above.
If you prefer the representation as `X` you can use the [`Stiefel`](@ref)`(n, n, ℂ)` manifold.

# Constructor

    UnitaryMatrices(n, 𝔽::AbstractNumbers=ℂ)

see also [`OrthogonalMatrices`](@ref) for the real valued case.
"""
const UnitaryMatrices{T,𝔽} = GeneralUnitaryMatrices{T,𝔽,AbsoluteDeterminantOneMatrices}

function UnitaryMatrices(n::Int, 𝔽::AbstractNumbers=ℂ; parameter::Symbol=:type)
    size = wrap_type_parameter(parameter, (n,))
    return UnitaryMatrices{typeof(size),𝔽}(size)
end

check_size(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p::Number) = nothing
check_size(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p, X::Number) = nothing

embed(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p::Number) = SMatrix{1,1}(p)

embed(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p, X::Number) = SMatrix{1,1}(X)

function exp(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p, X::Number)
    return p * exp(X)
end
function exp(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p, X::Number, t::Number)
    return p * exp(t * X)
end

function get_vector_orthonormal(
    ::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ},
    p::Quaternions.Quaternion,
    c,
    ::QuaternionNumbers,
)
    i = firstindex(c)
    return Quaternions.quat(0, c[i], c[i + 1], c[i + 2])
end

injectivity_radius(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}) = π

function _isapprox(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, x, y; kwargs...)
    return isapprox(x[], y[]; kwargs...)
end
function _isapprox(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p, X, Y; kwargs...)
    return isapprox(X[], Y[]; kwargs...)
end

function log(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p::Number, q::Number)
    return log(conj(p) * q)
end

@doc raw"""
    manifold_dimension(M::UnitaryMatrices{n,ℂ}) where {n}

Return the dimension of the manifold unitary matrices.
```math
\dim_{\mathrm{U}(n)} = n^2.
```
"""
function manifold_dimension(M::UnitaryMatrices{<:Any,ℂ})
    n = get_parameter(M.size)[1]
    return n^2
end
@doc raw"""
    manifold_dimension(M::UnitaryMatrices{<:Any,ℍ})

Return the dimension of the manifold unitary matrices.
```math
\dim_{\mathrm{U}(n, ℍ)} = n(2n+1).
```
"""
function manifold_dimension(M::UnitaryMatrices{<:Any,ℍ})
    n = get_parameter(M.size)[1]
    return n * (2n + 1)
end

number_of_coordinates(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, ::AbstractBasis{ℍ}) = 3

project(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p) = sign(p)

project(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p, X) = (X - conj(X)) / 2

function Random.rand(M::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}; vector_at=nothing)
    if vector_at === nothing
        return sign(rand(Quaternions.QuaternionF64))
    else
        project(M, vector_at, rand(Quaternions.QuaternionF64))
    end
end
function Random.rand(
    rng::AbstractRNG,
    M::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ};
    vector_at=nothing,
)
    if vector_at === nothing
        return sign(rand(rng, Quaternions.QuaternionF64))
    else
        project(M, vector_at, rand(rng, Quaternions.QuaternionF64))
    end
end

@doc raw"""
    rand(::Unitary; vector_at=nothing, σ::Real=1.0)

Generate a random point on the [`UnitaryMatrices`](@ref) manifold,
if `vector_at` is nothing, by computing the QR decomposition of
a ``n×x`` matrix.

Generate a tangent vector at `vector_at` by projecting a normally
distributed matrix onto the tangent space.
"""
rand(::UnitaryMatrices; σ::Real=1.0)

function Random.rand!(
    rng::AbstractRNG,
    M::UnitaryMatrices,
    pX;
    vector_at=nothing,
    σ::Real=one(real(eltype(pX))),
)
    n = get_parameter(M.size)[1]
    if vector_at === nothing
        A = σ * randn(rng, eltype(pX), n, n)
        pX .= Matrix(qr(A).Q)
    else
        Z = σ * randn(rng, eltype(pX), size(pX))
        project!(M, pX, vector_at, Z)
    end
    return pX
end

function Base.show(io::IO, ::UnitaryMatrices{TypeParameter{Tuple{n}},ℂ}) where {n}
    return print(io, "UnitaryMatrices($(n))")
end
function Base.show(io::IO, M::UnitaryMatrices{Tuple{Int},ℂ})
    n = get_parameter(M.size)[1]
    return print(io, "UnitaryMatrices($n; parameter=:field)")
end
function Base.show(io::IO, ::UnitaryMatrices{TypeParameter{Tuple{n}},ℍ}) where {n}
    return print(io, "UnitaryMatrices($(n), ℍ)")
end
function Base.show(io::IO, M::UnitaryMatrices{Tuple{Int},ℍ})
    n = get_parameter(M.size)[1]
    return print(io, "UnitaryMatrices($n, ℍ; parameter=:field)")
end

@doc raw"""
    riemannian_Hessian(M::UnitaryMatrices, p, G, H, X)

The Riemannian Hessian can be computed by adopting Eq. (5.6) [Nguyen:2023](@cite),
so very similar to the complex Stiefel manifold.
The only difference is, that here the tangent vectors are stored
in the Lie algebra, i.e. the update direction is actually ``pX`` instead of just ``X`` (in Stiefel).
and that means the inverse has to be appliead to the (Euclidean) Hessian
to map it into the Lie algebra.
"""
riemannian_Hessian(M::UnitaryMatrices, p, G, H, X)
function riemannian_Hessian!(M::UnitaryMatrices, Y, p, G, H, X)
    symmetrize!(Y, G' * p)
    project!(M, Y, p, p' * H - X * Y)
    return Y
end

@doc raw"""
    Weingarten(M::UnitaryMatrices, p, X, V)

Compute the Weingarten map ``\mathcal W_p`` at `p` on the [`Stiefel`](@ref) `M` with respect to the
tangent vector ``X \in T_p\mathcal M`` and the normal vector ``V \in N_p\mathcal M``.

The formula is due to [AbsilMahonyTrumpf:2013](@cite) given by

```math
\mathcal W_p(X,V) = -\frac{1}{2}p\bigl(V^{\mathrm{H}}X - X^\mathrm{H}V\bigr)
```
"""
Weingarten(::UnitaryMatrices, p, X, V)

function Weingarten!(::UnitaryMatrices, Y, p, X, V)
    Y .= V' * X
    Y .= -p * 1 / 2 * (Y - Y')
    return Y
end

zero_vector(::UnitaryMatrices{TypeParameter{Tuple{1}},ℍ}, p) = zero(p)
