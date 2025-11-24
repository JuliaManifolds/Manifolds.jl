"""
    AbstractAtlas{𝔽}

An abstract class for atlases with charts that have values in the vector space `𝔽ⁿ`
for some value of `n`. `𝔽` is a number system determined by an [`AbstractNumbers`](@extref ManifoldsBase number-system)
object.
"""
abstract type AbstractAtlas{𝔽} end

@doc raw"""
    RetractionAtlas{
        𝔽,
        TRetr<:AbstractRetractionMethod,
        TInvRetr<:AbstractInverseRetractionMethod,
        TBasis<:AbstractBasis,
    } <: AbstractAtlas{𝔽}

An atlas indexed by points on a manifold, ``\mathcal M = I`` and parameters (local coordinates)
are given in ``T_p\mathcal M``.
This means that a chart ``φ_p = \mathrm{cord}\circ\mathrm{retr}_p^{-1}`` is only locally
defined (around ``p``), where ``\mathrm{cord}`` is the decomposition of the tangent vector
into coordinates with respect to the given basis of the tangent space, cf. [`get_coordinates`](@ref).
The parametrization is given by ``φ_p^{-1}=\mathrm{retr}_p\circ\mathrm{vec}``,
where ``\mathrm{vec}`` turns the basis coordinates into a tangent vector, cf. [`get_vector`](@ref).

In short: The coordinates with respect to a basis are used together with a retraction as a parametrization.

# See also

[`AbstractAtlas`](@ref), [`AbstractInverseRetractionMethod`](@extref `ManifoldsBase.AbstractInverseRetractionMethod`),
[`AbstractRetractionMethod`](@extref `ManifoldsBase.AbstractRetractionMethod`), [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`)
"""
struct RetractionAtlas{
        𝔽,
        TRetr <: AbstractRetractionMethod,
        TInvRetr <: AbstractInverseRetractionMethod,
        TBasis <: AbstractBasis{<:Any, TangentSpaceType},
    } <: AbstractAtlas{𝔽}
    retr::TRetr
    invretr::TInvRetr
    basis::TBasis
end

function RetractionAtlas(
        retr::AbstractRetractionMethod,
        invretr::AbstractInverseRetractionMethod,
    )
    basis = DefaultOrthonormalBasis()
    return RetractionAtlas{ℝ, typeof(retr), typeof(invretr), typeof(basis)}(
        retr,
        invretr,
        basis,
    )
end
RetractionAtlas() = RetractionAtlas(ExponentialRetraction(), LogarithmicInverseRetraction())

"""
    affine_connection(M::AbstractManifold, A::AbstractAtlas, i, a, Xc, Yc)

Calculate affine connection on manifold `M` at point with parameters `a` in chart `i` of
[`AbstractAtlas`](@ref) `A` of vectors with coefficients `Xc` and `Yc` in induced basis.
"""
function affine_connection(M::AbstractManifold, A, i, a, Xc, Yc)
    Zc = allocate(Xc)
    return affine_connection!(M, Zc, A, i, a, Xc, Yc)
end

"""
    affine_connection!(M::AbstractManifold, Zc, A::AbstractAtlas, i, a, Xc, Yc)

Calculate affine connection on manifold `M` at point with parameters `a` in chart `i` of an
an [`AbstractAtlas`](@ref) `A` of vectors with coefficients `Zc` and `Yc` in induced basis and save the result
in `Zc`.
"""
affine_connection!(M::AbstractManifold, Zc, A::AbstractAtlas, i, a, Xc, Yc)

"""
    get_default_atlas(::AbstractManifold)

Determine the default real-valued atlas for the given manifold.
"""
function get_default_atlas(::AbstractManifold)
    return RetractionAtlas()
end

@doc raw"""
    get_parameters(M::AbstractManifold, A::AbstractAtlas, i, p)

Calculate parameters (local coordinates) of point `p` on manifold `M` in chart from an [`AbstractAtlas`](@ref)
`A` at index `i`.
This function is hence an implementation of the chart ``φ_i(p), i\in I``.
The parameters are in the number system determined by `A`.
If the point ``p\notin U_i`` is not in the domain of the chart, this method should throw an error.

# See also

[`get_point`](@ref), [`get_chart_index`](@ref)

"""
get_parameters(::AbstractManifold, ::AbstractAtlas, ::Any, ::Any)

function get_parameters(M::AbstractManifold, A::AbstractAtlas, i, p)
    a = allocate_result(M, get_parameters, p)
    get_parameters!(M, a, A, i, p)
    return a
end

function allocate_result(M::AbstractManifold, f::typeof(get_parameters), p)
    T = allocate_result_type(M, f, (p,))
    return allocate(p, T, manifold_dimension(M))
end
# disambiguation
@invoke_maker 1 AbstractManifold allocate_result(
    M::AbstractDecoratorManifold,
    f::typeof(get_parameters),
    p,
)

"""
    check_chart_switch(M::AbstractManifold, A::AbstractAtlas, i, a)

Determine whether chart should be switched when an operation in chart `i` from an [`AbstractAtlas`](@ref) `A`
reaches parameters `a` in that chart.

By default `false` is returned.
"""
check_chart_switch(M::AbstractManifold, A::AbstractAtlas, i, a) = false

function get_parameters!(M::AbstractManifold, a, A::RetractionAtlas, i, p)
    return get_coordinates!(M, a, i, inverse_retract(M, i, p, A.invretr), A.basis)
end

function get_parameters(M::AbstractManifold, A::RetractionAtlas, i, p)
    return get_coordinates(M, i, inverse_retract(M, i, p, A.invretr), A.basis)
end

@doc raw"""
    get_point(M::AbstractManifold, A::AbstractAtlas, i, a)

Calculate point at parameters (local coordinates) `a` on manifold `M` in chart from
an [`AbstractAtlas`](@ref) `A` at index `i`.
This function is hence an implementation of the inverse ``φ_i^{-1}(a), i\in I`` of a chart, also called a parametrization.

# See also

[`get_parameters`](@ref), [`get_chart_index`](@ref)
"""
get_point(::AbstractManifold, ::AbstractAtlas, ::Any, ::Any)

function get_point(M::AbstractManifold, A::AbstractAtlas, i, a)
    p = allocate_result(M, get_point, a)
    get_point!(M, p, A, i, a)
    return p
end

function allocate_result(M::AbstractManifold, f::typeof(get_point), a)
    T = allocate_result_type(M, f, (a,))
    return allocate(a, T, representation_size(M)...)
end
# disambiguation
@invoke_maker 1 AbstractManifold allocate_result(
    M::AbstractDecoratorManifold,
    f::typeof(get_point),
    a,
)

function get_point(M::AbstractManifold, A::RetractionAtlas, i, a)
    return retract(M, i, get_vector(M, i, a, A.basis), A.retr)
end

function get_point!(M::AbstractManifold, p, A::RetractionAtlas, i, a)
    return retract!(M, p, i, get_vector(M, i, a, A.basis), A.retr)
end

"""
    get_chart_index(M::AbstractManifold, A::AbstractAtlas, p)

Select a chart from an [`AbstractAtlas`](@ref) `A` for manifold `M` that is suitable for
representing the neighborhood of point `p`. This selection should be deterministic, although
different charts may be selected for arbitrarily close but distinct points.

# See also

[`get_default_atlas`](@ref)
"""
get_chart_index(::AbstractManifold, ::AbstractAtlas, ::Any)

get_chart_index(::AbstractManifold, ::RetractionAtlas, p) = p

"""
    get_chart_index(M::AbstractManifold, A::AbstractAtlas, i, a)

Select a chart from an [`AbstractAtlas`](@ref) `A` for manifold `M` that is suitable for
representing the neighborhood of point with parametrization `a` in chart `i`. This selection
should be deterministic, although different charts may be selected for arbitrarily close but
distinct points.

# See also

[`get_default_atlas`](@ref)
"""
get_chart_index(::AbstractManifold, ::AbstractAtlas, ::Any, ::Any)

"""
    inner(M::AbstractManifold, A::AbstractAtlas, i, a, Xc, Yc)

Calculate inner product on manifold `M` at point with parameters `a` in chart `i` of an
atlas `A` of vectors with coefficients `Xc` and `Yc` in induced basis.
"""
inner(M::AbstractManifold, A::AbstractAtlas, i, a, Xc, Yc)

"""
    local_metric(M::AbstractManifold, A::AbstractAtlas{ℝ}, i, a)

Compute the local metric tensor on the manifold `M` at the point with parameters `a` in chart `i` 
of an [`AbstractAtlas`](@ref) `A`. The local metric tensor is represented as a matrix, where 
each entry corresponds to the inner product of basis vectors in the tangent space at the given point.

# Arguments

- `M::AbstractManifold`: The manifold on which the metric is computed.
- `A::AbstractAtlas{ℝ}`: The atlas defining the charts and coordinate systems on the manifold.
- `i`: The index of the chart in the atlas.
- `a`: The parameters (local coordinates) of the point in the chart.

# Returns

A matrix representing the local metric tensor at the given point.

# See also

[`local_metric!`](@ref), [`inner`](@ref)
"""
function local_metric(M::AbstractManifold, A::AbstractAtlas{ℝ}, i, a)
    n = length(a)
    g = zeros(eltype(a), n, n)
    return local_metric!(M, g, A, i, a)
end
function local_metric!(M::AbstractManifold, g::AbstractMatrix{T}, A::AbstractAtlas{ℝ}, i, a) where {T}
    n = length(a)

    e_p = zeros(T, n)
    e_q = zeros(T, n)
    for p in 1:n, q in 1:n
        e_p[p] = 1
        e_q[q] = 1
        g[p, q] = inner(M, A, i, a, e_p, e_q)
        e_p[p] = 0
        e_q[q] = 0
    end
    return g
end

"""
    inverse_local_metric(M::AbstractManifold, A::AbstractAtlas{ℝ}, i, a)

Compute the inverse of the local metric tensor on the manifold `M` at the point with parameters `a` 
in chart `i` of an [`AbstractAtlas`](@ref) `A`. The inverse local metric tensor is represented as a matrix, 
where each entry corresponds to the inverse of the inner product of basis vectors in the tangent space 
at the given point.

# Arguments

- `M::AbstractManifold`: The manifold on which the metric is computed.
- `A::AbstractAtlas{ℝ}`: The atlas defining the charts and coordinate systems on the manifold.
- `i`: The index of the chart in the atlas.
- `a`: The parameters (local coordinates) of the point in the chart.

# Returns

A matrix representing the inverse of the local metric tensor at the given point.

# See also

[`local_metric`](@ref), [`inner`](@ref)
"""
function inverse_local_metric(M::AbstractManifold, A::AbstractAtlas{ℝ}, i, a)
    n = length(a)
    ginv = zeros(eltype(a), n, n)
    return inverse_local_metric!(M, ginv, A, i, a)
end
function inverse_local_metric!(M::AbstractManifold, ginv::AbstractMatrix{T}, A::AbstractAtlas{ℝ}, i, a) where {T}
    # inverse metric
    ginv .= inv(local_metric(M, A, i, a))
    return ginv
end

"""
    norm(M::AbstractManifold, A::AbstractAtlas, i, a, Xc)

Calculate norm on manifold `M` at point with parameters `a` in chart `i` of an
[`AbstractAtlas`](@ref) `A` of vector with coefficients `Xc` in induced basis.
"""
norm(M::AbstractManifold, A::AbstractAtlas, i, a, Xc) = sqrt(inner(M, A, i, a, Xc, Xc))

@doc raw"""
    transition_map(M::AbstractManifold, A_from::AbstractAtlas, i_from, A_to::AbstractAtlas, i_to, a)
    transition_map(M::AbstractManifold, A::AbstractAtlas, i_from, i_to, a)

Given coordinates `a` in chart `(A_from, i_from)` of a point on manifold `M`, returns
coordinates of that point in chart `(A_to, i_to)`. If `A_from` and `A_to` are equal, `A_to`
can be omitted.

Mathematically this function is the transition map or change of charts, but it
might even be between two atlases ``A_{\text{from}} = \{(U_i,φ_i)\}_{i\in I} `` and ``A_{\text{to}} = \{(V_j,\psi_j)\}_{j\in J}``,
and hence ``I, J`` are their index sets.
We have ``i_{\text{from}}\in I``, ``i_{\text{to}}\in J``.

This method then computes
```math
\bigl(\psi_{i_{\text{to}}}\circ φ_{i_{\text{from}}}^{-1}\bigr)(a)
```

Note that, similarly to [`get_parameters`](@ref), this method should fail the same way if ``V_{i_{\text{to}}}\cap U_{i_{\text{from}}}=\emptyset``.

# See also

[`AbstractAtlas`](@ref), [`get_parameters`](@ref), [`get_point`](@ref)
"""
function transition_map(
        M::AbstractManifold,
        A_from::AbstractAtlas,
        i_from,
        A_to::AbstractAtlas,
        i_to,
        a,
    )
    return get_parameters(M, A_to, i_to, get_point(M, A_from, i_from, a))
end

function transition_map(M::AbstractManifold, A::AbstractAtlas, i_from, i_to, a)
    return transition_map(M, A, i_from, A, i_to, a)
end

function transition_map!(
        M::AbstractManifold,
        y,
        A_from::AbstractAtlas,
        i_from,
        A_to::AbstractAtlas,
        i_to,
        a,
    )
    return get_parameters!(M, y, A_to, i_to, get_point(M, A_from, i_from, a))
end

function transition_map!(M::AbstractManifold, y, A::AbstractAtlas, i_from, i_to, a)
    return transition_map!(M, y, A, i_from, A, i_to, a)
end

"""
    transition_map_diff(M::AbstractManifold, A::AbstractAtlas, i_from, a, c, i_to)

Compute differential of transition map from chart `i_from` to chart `i_to` from an
[`AbstractAtlas`](@ref) `A` on manifold `M` at point with parameters `a` on tangent vector
with coordinates `c` in the induced basis.
"""
function transition_map_diff(M::AbstractManifold, A::AbstractAtlas, i_from, a, c, i_to)
    old_B = induced_basis(M, A, i_from)
    new_B = induced_basis(M, A, i_to)
    p_final = get_point(M, A, i_from, a)
    return change_basis(M, p_final, c, old_B, new_B)
end

"""
    transition_map_diff!(M::AbstractManifold, c_out, A::AbstractAtlas, i_from, a, c, i_to)

Compute [`transition_map_diff`](@ref) on given arguments and save the result in `c_out`.
"""
function transition_map_diff!(
        M::AbstractManifold,
        c_out,
        A::AbstractAtlas,
        i_from,
        a,
        c_in,
        i_to,
    )
    old_B = induced_basis(M, A, i_from)
    new_B = induced_basis(M, A, i_to)
    p_final = get_point(M, A, i_from, a)
    return change_basis!(M, c_out, p_final, c_in, old_B, new_B)
end

"""
    induced_basis(M::AbstractManifold, A::AbstractAtlas, i, p, VST::VectorSpaceType)

Basis of vector space of type `VST` at point `p` from manifold `M` induced by
chart (`A`, `i`).

# See also

[`VectorSpaceType`](@extref `ManifoldsBase.VectorSpaceType`), [`AbstractAtlas`](@ref)
"""
induced_basis(M::AbstractManifold, A::AbstractAtlas, i, VST::VectorSpaceType)

function induced_basis(
        ::AbstractManifold,
        A::RetractionAtlas{
            <:AbstractRetractionMethod,
            <:AbstractInverseRetractionMethod,
            <:DefaultOrthonormalBasis,
        },
        i,
        p,
        ::TangentSpaceType,
    )
    return A.basis
end
function induced_basis(
        M::AbstractManifold,
        A::RetractionAtlas{
            <:AbstractRetractionMethod,
            <:AbstractInverseRetractionMethod,
            <:DefaultOrthonormalBasis,
        },
        i,
        p,
        ::CotangentSpaceType,
    )
    return dual_basis(M, p, A.basis)
end

@doc raw"""
    InducedBasis(vs::VectorSpaceType, A::AbstractAtlas, i)

The basis induced by chart with index `i` from an [`AbstractAtlas`](@ref) `A` of vector
space of type `vs`.

For the `vs` a [`TangentSpace`](@extref `ManifoldsBase.TangentSpace`)
this works as  follows:

Let ``n`` denote the dimension of the manifold ``\mathcal M``.

Let the parameter ``a=φ_i(p) ∈ \mathbb R^n`` and ``j∈\{1,…,n\}``.
We can look at the ``j``th parameter curve ``b_j(t) = a + te_j``, where ``e_j`` denotes the ``j``th unit vector.
Using the parametrisation we obtain a curve ``c_j(t) = φ_i^{-1}(b_j(t))`` which fulfills ``c(0) = p``.

Now taking the derivative(s) with respect to ``t`` (and evaluate at ``t=0``),
we obtain a tangent vector for each ``j`` corresponding to an equivalence class of curves (having the same derivative) as

```math
X_j = [c_j] = \frac{\mathrm{d}}{\mathrm{d}t} c_i(t) \Bigl|_{t=0}
```

and the set ``\{X_1,\ldots,X_n\}`` is the chart-induced basis of ``T_p\mathcal M``.

# See also

[`VectorSpaceType`](@extref `ManifoldsBase.VectorSpaceType`), [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`)
"""
struct InducedBasis{𝔽, VST <: VectorSpaceType, TA <: AbstractAtlas, TI} <: AbstractBasis{𝔽, VST}
    vs::VST
    A::TA
    i::TI
end

"""
    induced_basis(::AbstractManifold, A::AbstractAtlas, i, VST::VectorSpaceType = TangentSpaceType())

Get the basis induced by chart with index `i` from an [`AbstractAtlas`](@ref) `A` of vector
space of type `vs`. Returns an object of type [`InducedBasis`](@ref).

# See also

[`VectorSpaceType`](@extref `ManifoldsBase.VectorSpaceType`), [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`)
"""
function induced_basis(
        ::AbstractManifold{𝔽},
        A::AbstractAtlas,
        i,
        VST::VectorSpaceType = TangentSpaceType(),
    ) where {𝔽}
    return InducedBasis{𝔽, typeof(VST), typeof(A), typeof(i)}(VST, A, i)
end

"""
    inverse_chart_injectivity_radius(M::AbstractManifold, A::AbstractAtlas, i)

Injectivity radius of `get_point` for chart `i` from an [`AbstractAtlas`](@ref) `A` of a manifold `M`.
"""
inverse_chart_injectivity_radius(M::AbstractManifold, A::AbstractAtlas, i)

function dual_basis(
        M::AbstractManifold{𝔽},
        ::Any,
        B::InducedBasis{𝔽, TangentSpaceType},
    ) where {𝔽}
    return induced_basis(M, B.A, B.i, CotangentSpaceType())
end
function dual_basis(
        M::AbstractManifold{𝔽},
        ::Any,
        B::InducedBasis{𝔽, CotangentSpaceType},
    ) where {𝔽}
    return induced_basis(M, B.A, B.i, TangentSpaceType())
end

function ManifoldsBase._get_coordinates(M::AbstractManifold, p, X, B::InducedBasis)
    return get_coordinates_induced_basis(M, p, X, B)
end
function get_coordinates_induced_basis(M::AbstractManifold, p, X, B::InducedBasis)
    Y = allocate_result(M, get_coordinates, p, X, B)
    return get_coordinates_induced_basis!(M, Y, p, X, B)
end

function ManifoldsBase._get_coordinates!(M::AbstractManifold, Y, p, X, B::InducedBasis)
    return get_coordinates_induced_basis!(M, Y, p, X, B)
end
function get_coordinates_induced_basis! end

function ManifoldsBase._get_vector(M::AbstractManifold, p, c, B::InducedBasis)
    return get_vector_induced_basis(M, p, c, B)
end
function get_vector_induced_basis(M::AbstractManifold, p, c, B::InducedBasis)
    Y = allocate_result(M, get_vector, p, c)
    return get_vector!(M, Y, p, c, B)
end

function ManifoldsBase._get_vector!(M::AbstractManifold, Y, p, c, B::InducedBasis)
    return get_vector_induced_basis!(M, Y, p, c, B)
end
function get_vector_induced_basis! end

"""
    local_metric(M::AbstractManifold, p, B::InducedBasis)

Compute the local metric tensor for vectors expressed in terms of coordinates
in basis `B` on manifold `M`. The point `p` is not checked.
"""
local_metric(::AbstractManifold, ::Any, ::InducedBasis)

"""
    levi_civita_affine_connection!(M::AbstractManifold, Zc, A::AbstractAtlas, i, a, Xc, Yc; backend::AbstractADType = AutoForwardDiff())

Compute the Levi-Civita affine connection on the manifold `M` at a point with parameters `a`
in chart `i` of an  [`AbstractAtlas`](@ref) `A`. The connection is calculated for vectors
with coefficients `Xc` and `Yc` in the induced basis,  and the result is stored in `Zc`.

The Levi-Civita connection is computed using the metric tensor (`inner` called in a chart)
of the manifold, ensuring that the connection is torsion-free and compatible with the metric.
The computation involves the Christoffel symbols of the second kind, which are derived from
the metric tensor and its derivatives using automatic differentiation. Note that this
computation is relatively slow. Where performance matters, it should be replaced with
custom-derived formulas.

# Arguments

- `M::AbstractManifold`: The manifold on which the Levi-Civita connection is computed.
- `Zc`: The output vector where the result of the connection is stored.
- `A::AbstractAtlas`: The atlas defining the charts and coordinate systems on the manifold.
- `i`: The index of the chart in the atlas.
- `a`: The parameters (local coordinates) of the point in the chart.
- `Xc`: The coefficients of the first vector in the induced basis.
- `Yc`: The coefficients of the second vector in the induced basis.
- `backend::AbstractADType`: The automatic differentiation backend used for computing derivatives (default: `AutoForwardDiff()`).

# Returns

The result is stored in `Zc`, which represents the Levi-Civita connection in the induced basis.

# Notes

- The computation involves the inverse of the local metric tensor, which is used to raise indices.
- The directional derivatives of the metric tensor are computed using the specified automatic differentiation backend.
- The function assumes that the input vectors `Xc` and `Yc` are expressed in the induced basis of the chart.

# See also

[`affine_connection!`](@ref), [`local_metric`](@ref), [`inner`](@ref)
"""
function levi_civita_affine_connection!(
        M::AbstractManifold, Zc, A::AbstractAtlas, i, a, Xc, Yc;
        backend::AbstractADType = AutoForwardDiff()
    )
    # number of coordinates
    n = length(a)

    ginv = inverse_local_metric(M, A, i, a)
    T = eltype(ginv)

    # helper: directional derivative at a in direction dir of the scalar function
    # f_dir(V1, V2) = d/dt|0 inner(M, A, i, a + t*dir, V1, V2)
    function directional_derivative_scalar(dir, V1, V2)
        f(t) = real(inner(M, A, i, a .+ (t .* dir), V1, V2))
        return DI.derivative(f, backend, 0.0)
    end

    # compute S_k = 1/2 ( X[g(Y, e_k)] + Y[g(X, e_k)] - e_k[g(X,Y)] )
    S = zeros(T, n)
    e_k = zeros(T, n)
    for k in 1:n
        e_k[k] = 1
        term1 = directional_derivative_scalar(Xc, Yc, e_k)
        term2 = directional_derivative_scalar(Yc, Xc, e_k)
        term3 = directional_derivative_scalar(e_k, Xc, Yc)
        S[k] = (term1 + term2 - term3) / 2
        e_k[k] = 0
    end

    # raise index: (∇_X Y)^l = g^{l k} S_k
    Zc .= ginv * S

    return Zc
end

"""
    get_coordinates_induced_basis_generic!(M::AbstractManifold, c, p, X, B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas}; backend::AbstractADType = AutoForwardDiff())

Compute the coordinates of a tangent vector `X` at a point `p` on the manifold `M` in the induced basis `B` 
and store the result in `c`. This function uses automatic differentiation to compute the coordinates.

# Arguments

- `M::AbstractManifold`: The manifold on which the computation is performed.
- `c`: The output array where the coordinates of the tangent vector will be stored.
- `p`: The point on the manifold where the tangent vector `X` is located.
- `X`: The tangent vector at `p` whose coordinates are to be computed.
- `B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas}`: The induced basis in which the coordinates are expressed.
- `backend::AbstractADType`: The automatic differentiation backend used for computing derivatives (default: `AutoForwardDiff()`).

# Returns

The result is stored in `c`, which contains the coordinates of the tangent vector `X` in the induced basis `B`.

# Notes

- This function computes the coordinates by differentiating the chart map at the given point `p` in the direction of `X`.
- The computation relies on automatic differentiation.

# See also

[`get_coordinates`](@ref), [`InducedBasis`](@ref), [`AbstractAtlas`](@ref)
"""
function get_coordinates_induced_basis_generic!(
        M::AbstractManifold,
        c,
        p,
        X,
        B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas};
        backend::AbstractADType = AutoForwardDiff(),
    )
    DI.derivative!(t -> get_parameters(M, B.A, B.i, p + t * X), c, backend, zero(eltype(c)))
    return c
end

"""
    get_vector_induced_basis_generic!(M::AbstractManifold, Y, p, Xc, B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas}; backend::AbstractADType = AutoForwardDiff())

Compute the tangent vector `Y` at a point `p` on the manifold `M` corresponding to the coordinates `Xc` 
in the induced basis `B` and store the result in `Y`. This function uses automatic differentiation 
to compute the tangent vector.

# Arguments

- `M::AbstractManifold`: The manifold on which the computation is performed.
- `Y`: The output tangent vector at `p` corresponding to the coordinates `Xc` in the induced basis.
- `p`: The point on the manifold where the tangent vector is located.
- `Xc`: The coordinates of the tangent vector in the induced basis `B`.
- `B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas}`: The induced basis in which the coordinates `Xc` are expressed.
- `backend::AbstractADType`: The automatic differentiation backend used for computing derivatives (default: `AutoForwardDiff()`).

# Returns

The result is stored in `Y`, which represents the tangent vector at `p` corresponding to the coordinates `Xc` 
in the induced basis `B`.

# Notes

- This function computes the tangent vector by differentiating the chart map at the given point `p` 
  in the direction of the coordinates `Xc`.
- The computation relies on automatic differentiation.

# See also

[`get_coordinates_induced_basis_generic!`](@ref), [`InducedBasis`](@ref), [`AbstractAtlas`](@ref)
"""
function get_vector_induced_basis_generic!(
        M::AbstractManifold,
        Y,
        p,
        Xc,
        B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas};
        backend::AbstractADType = AutoForwardDiff(),
    )
    p_i = get_parameters(M, B.A, B.i, p)
    DI.derivative!(t -> get_point(M, B.A, B.i, p_i + t * Xc), Y, backend, zero(eltype(p_i)))
    return Y
end
