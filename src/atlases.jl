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
    Zc = similar(Xc, Base.promote_type(eltype(Xc), eltype(Yc), eltype(a)))
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
    christoffel_symbols_second(M::AbstractManifold, A::AbstractAtlas, i, a)

Compute values of the Christoffel symbol of the second kind in chart `i` of atlas `A`
at point with parameters `a`.
"""
function christoffel_symbols_second(M::AbstractManifold, A::AbstractAtlas, i, a; kwargs...)
    n = length(a)
    Γ = zeros(eltype(a), n, n, n)
    return christoffel_symbols_second!(M, Γ, A, i, a; kwargs...)
end

function christoffel_symbols_second!(
        M::AbstractManifold, Γ, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()
    )
    # number of coordinates
    n = length(a)

    ginv = inverse_local_metric(M, A, i, a)
    T = eltype(ginv)

    # Precompute all directional derivatives ∂_i g_jl
    e = [zeros(T, n) for _ in 1:n]  # Basis vectors
    for k in 1:n
        e[k][k] = 1
    end

    dg = zeros(T, n, n, n)  # Tensor to store ∂_i g_jl
    for i in 1:n, j in 1:n, l in 1:n
        dg[i, j, l] = DI.derivative(
            t -> inner(M, A, i, a .+ t .* e[i], e[j], e[l]),
            backend,
            0.0,
        )
    end

    # Compute Christoffel symbols Γ^k_ij = 1/2 g^kl ( ∂_i g_jl + ∂_j g_il - ∂_l g_ij )
    for i in 1:n, j in 1:n, k in 1:n
        Γ[k, i, j] = sum(
            ginv[k, l] * (dg[i, j, l] + dg[j, i, l] - dg[l, i, j]) for l in 1:n
        ) / 2
    end
    return Γ
end

"""
    christoffel_symbols_first(M::AbstractManifold, A::AbstractAtlas, i, a; backend::AbstractADType = AutoForwardDiff())

Compute the Christoffel symbols of the first kind ``Γ_{i j k}`` in chart `i` of
[`AbstractAtlas`] `A` at coordinates `a`.

The symbols are obtained by lowering the first index of the second-kind Christoffel
symbols:

````math
    Γ_{i j k} = g_{k l} Γ^l_{i j}
````

# Arguments

- `M::AbstractManifold` : manifold
- `A::AbstractAtlas`   : atlas providing charts / induced basis
- `i`                  : chart index in `A`
- `a`                  : coordinates of the point in chart `i` (length `n`)

# Keyword arguments

- `backend::AbstractADType` : automatic-differentiation backend (default `AutoForwardDiff()`).
                              It is passed to [`christoffel_symbols_second`](@ref) in the
                              default implementation to compute the Christoffel symbol of
                              the second kind.

Returns an ``n×n×n`` array with ordering ``(i, j, k)``.
"""
function christoffel_symbols_first(
        M::AbstractManifold, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()
    )
    n = length(a)
    T = eltype(a)
    Γ = zeros(T, n, n, n)
    return christoffel_symbols_first!(M, Γ, A, i, a; backend = backend)
end

function christoffel_symbols_first!(
        M::AbstractManifold, Γ, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()
    )
    # Compute second-kind symbols and lower the first index: Γ_{i j k} = g_{k l} Γ^l_{i j}
    Γ2 = christoffel_symbols_second(M, A, i, a; backend = backend)
    g = local_metric(M, A, i, a)
    n = length(a)
    T = eltype(Γ2)
    fill!(Γ, zero(T))
    for ii in 1:n, j in 1:n, k in 1:n
        s = zero(T)
        for l in 1:n
            s += g[k, l] * Γ2[l, ii, j]
        end
        Γ[ii, j, k] = s
    end
    return Γ
end

@doc raw"""
    det_local_metric(M::AbstractManifold, A::AbstractAtlas, i, a)

Return the determinant of local matrix representation of the metric tensor at the point
with parametrization `a` in chart `i` of [`AbstractAtlas`](@ref) `A`.

See also [`local_metric`](@ref)
"""
function det_local_metric(M::AbstractManifold, A::AbstractAtlas, i, a)
    return det(local_metric(M, A, i, a))
end

"""
    einstein_tensor(M::AbstractManifold, A::AbstractAtlas, i, a; backend::AbstractADType = AutoForwardDiff())

Compute the Einstein tensor of the manifold `M` at the point specified by coordinates `a`
in chart `i` of atlas `A`.

The Einstein tensor is defined as

````math
    G_{ij} = Ric_{ij} - 1/2 g_{ij} R
````

where `Ric` is the Ricci tensor, `g` the local metric and `R` the scalar curvature.
They are computed using, respectively, [`ricci_tensor`](@ref), [`local_metric`](@ref) and
[`ricci_curvature`](@ref).
"""
function einstein_tensor(
        M::AbstractManifold, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()
    )
    n = length(a)
    T = eltype(a)
    G = zeros(T, n, n)
    return einstein_tensor!(M, G, A, i, a; backend = backend)
end

function einstein_tensor!(
        M::AbstractManifold, G::AbstractMatrix, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()
    )
    # compute Ricci tensor and scalar curvature
    ricci_tensor!(M, G, A, i, a; backend = backend)
    R = ricci_curvature(M, A, i, a; backend = backend)

    g = local_metric(M, A, i, a)
    G .-= g .* (R / 2)

    return G
end

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
each entry corresponds to the inner product of basis vectors in the tangent space at the point
with given parameters.

In contrast, `local_metric(M::AbstractManifold, p, ::InducedBasis)` requires passing a point
instead of its parameters in a chart.

# Arguments

- `M::AbstractManifold`: The manifold on which the metric is computed.
- `A::AbstractAtlas{ℝ}`: The atlas defining the charts and coordinate systems on the manifold.
- `i`: The index of the chart in the atlas.
- `a`: The parameters of the point in the chart.

# Returns

A matrix representing the local metric tensor at the point with given parameters.

# See also

[`inverse_local_metric`](@ref), [`inner`](@ref)
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

@doc raw"""
    log_local_metric_density(M::AbstractManifold, A::AbstractAtlas, i, a)

Return the natural logarithm of the metric density ``ρ`` of `M` at the point with
parametrization `a` in chart `i` of [`AbstractAtlas`](@ref) `A`, which is given by
````math
ρ = \log \sqrt{\lvert \det g_{ij} \rvert}
````
for the metric tensor expressed in the same chart.

# See also

[`local_metric`](@ref), [`det_local_metric`](@ref)
"""
function log_local_metric_density(M::AbstractManifold, A::AbstractAtlas, i, a)
    return log(abs(det_local_metric(M, A, i, a))) / 2
end

"""
    inverse_local_metric(M::AbstractManifold, A::AbstractAtlas{ℝ}, i, a)

Compute the inverse of the local metric tensor on the manifold `M` at the point with parameters `a` 
in chart `i` of an [`AbstractAtlas`](@ref) `A`. The inverse local metric tensor is represented as a matrix, 
where each entry corresponds to the inverse of the inner product of basis vectors in the tangent space 
at the point with given parameters.

In contrast, `inverse_local_metric(M::AbstractManifold, p, ::InducedBasis)` requires passing
a point instead of its parameters in a chart.

# Arguments

- `M::AbstractManifold`: The manifold on which the metric is computed.
- `A::AbstractAtlas{ℝ}`: The atlas defining the charts and coordinate systems on the manifold.
- `i`: The index of the chart in the atlas.
- `a`: The parameters of the point in the chart.

# Returns

A matrix representing the inverse of the local metric tensor at the point with given parameters.

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
    induced_basis(M::AbstractManifold, A::AbstractAtlas, i, VST::VectorSpaceType)

Basis of vector space of type `VST` at point `p` from manifold `M` induced by
chart (`A`, `i`).

# See also

[`VectorSpaceType`](@extref `ManifoldsBase.VectorSpaceType`), [`AbstractAtlas`](@ref)
"""
induced_basis(M::AbstractManifold, A::AbstractAtlas, i, VST::VectorSpaceType)

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

function ManifoldsBase._get_coordinates(M::AbstractManifold, p, X, B::InducedBasis; kwargs...)
    return get_coordinates_induced_basis(M, p, X, B; kwargs...)
end
function get_coordinates_induced_basis(M::AbstractManifold, p, X, B::InducedBasis; kwargs...)
    Y = allocate_result(M, get_coordinates, p, X, B)
    return get_coordinates_induced_basis!(M, Y, p, X, B; kwargs...)
end

function ManifoldsBase._get_coordinates!(M::AbstractManifold, Y, p, X, B::InducedBasis; kwargs...)
    return get_coordinates_induced_basis!(M, Y, p, X, B; kwargs...)
end
function get_coordinates_induced_basis! end

function ManifoldsBase._get_vector(M::AbstractManifold, p, c, B::InducedBasis; kwargs...)
    return get_vector_induced_basis(M, p, c, B; kwargs...)
end
function get_vector_induced_basis(M::AbstractManifold, p, c, B::InducedBasis; kwargs...)
    Y = allocate_result(M, get_vector, p, c)
    return get_vector!(M, Y, p, c, B; kwargs...)
end

function ManifoldsBase._get_vector!(M::AbstractManifold, Y, p, c, B::InducedBasis; kwargs...)
    return get_vector_induced_basis!(M, Y, p, c, B; kwargs...)
end
function get_vector_induced_basis! end

"""
    kretschmann_scalar(M::AbstractManifold, A::AbstractAtlas, i, a; backend::AbstractADType = AutoForwardDiff())

Compute the Kretschmann scalar ``K = R_{abcd} R^{abcd}`` at the point given by coordinates `a`
in chart `i` of atlas `A` on manifold `M`.

This implementation uses the Riemann tensor in the form ``R^u_{ijk}`` (returned by `riemann_tensor`)
and the inverse local metric `g^{ij}` (returned by `inverse_local_metric`) to form the full
contraction:

````math
    K = g^{u v} g^{i p} g^{j q} g^{k r} R^u_{i j k} R^v_{p q r}
````

# Arguments

- `M::AbstractManifold` : manifold
- `A::AbstractAtlas`   : atlas providing charts / induced basis
- `i`                  : chart index in `A`
- `a`                  : coordinates of the point in chart `i` (length `n`)
- `backend::AbstractADType` : automatic-differentiation backend (default `AutoForwardDiff()`)

# Returns

Scalar (same element type as `a`) equal to the Kretschmann scalar at the point
"""
function kretschmann_scalar(
        M::AbstractManifold, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff(),
    )
    n = length(a)
    T = eltype(a)
    R = riemann_tensor(M, A, i, a; backend = backend)   # R[u, ii, j, k] == R^u_{ijk}
    ginv = inverse_local_metric(M, A, i, a)             # g^{ij}

    K = zero(T)
    for u in 1:n, ii in 1:n, j in 1:n, k in 1:n
        Ruijk = R[u, ii, j, k]
        if iszero(Ruijk)
            continue
        end
        for v in 1:n, p in 1:n, q in 1:n, r in 1:n
            K += ginv[u, v] * ginv[ii, p] * ginv[j, q] * ginv[k, r] * Ruijk * R[v, p, q, r]
        end
    end
    return K
end

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
with coefficients `Xc` and `Yc` in the induced basis, and the result is stored in `Zc`.

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

# Keyword arguments

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
    get_coordinates_induced_basis!(M::AbstractManifold, c, p, X, B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas};
        backend::AbstractADType = AutoForwardDiff())

Compute the coordinates of a tangent vector `X` at a point `p` on the manifold `M` in the induced basis `B` 
and store the result in `c`. This function uses automatic differentiation to compute the coordinates.

# Arguments

- `M::AbstractManifold`: The manifold on which the computation is performed.
- `c`: The output array where the coordinates of the tangent vector will be stored.
- `p`: The point on the manifold where the tangent vector `X` is located.
- `X`: The tangent vector at `p` whose coordinates are to be computed.
- `B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas}`: The induced basis in which the coordinates are expressed.

# Keyword arguments

- `backend::AbstractADType`: The automatic differentiation backend used for computing derivatives (default: `AutoForwardDiff()`).

# Returns

The result is stored in `c`, which contains the coordinates of the tangent vector `X` in the induced basis `B`.

# Notes

- This function computes the coordinates by differentiating the chart map at the given point `p` in the direction of `X`.
- The computation relies on automatic differentiation.

# See also

[`InducedBasis`](@ref), [`AbstractAtlas`](@ref)
"""
function get_coordinates_induced_basis!(
        M::AbstractManifold,
        c,
        p,
        X,
        B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas};
        backend::AbstractADType = AutoForwardDiff(),
    )
    DI.derivative!(t -> get_parameters(M, B.A, B.i, p .+ t .* X), c, backend, zero(eltype(c)))
    return c
end

"""
    get_vector_induced_basis!(M::AbstractManifold, Y, p, Xc, B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas};
        backend::AbstractADType = AutoForwardDiff())

Compute the tangent vector `Y` at a point `p` on the manifold `M` corresponding to the coordinates `Xc` 
in the induced basis `B` and store the result in `Y`. This function uses automatic differentiation 
to compute the tangent vector.

# Arguments

- `M::AbstractManifold`: The manifold on which the computation is performed.
- `Y`: The output tangent vector at `p` corresponding to the coordinates `Xc` in the induced basis.
- `p`: The point on the manifold where the tangent vector is located.
- `Xc`: The coordinates of the tangent vector in the induced basis `B`.
- `B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas}`: The induced basis in which the coordinates `Xc` are expressed.

# Keyword arguments

- `backend::AbstractADType`: The automatic differentiation backend used for computing derivatives (default: `AutoForwardDiff()`).

# Returns

The result is stored in `Y`, which represents the tangent vector at `p` corresponding to the coordinates `Xc` 
in the induced basis `B`.

# Notes

- This function computes the tangent vector by differentiating the chart map at the given point `p` 
  in the direction of the coordinates `Xc`.
- The computation relies on automatic differentiation.

# See also

[`InducedBasis`](@ref), [`AbstractAtlas`](@ref)
"""
function get_vector_induced_basis!(
        M::AbstractManifold,
        Y,
        p,
        Xc,
        B::InducedBasis{ℝ, TangentSpaceType, <:AbstractAtlas};
        backend::AbstractADType = AutoForwardDiff(),
    )
    p_i = get_parameters(M, B.A, B.i, p)
    DI.derivative!(t -> get_point(M, B.A, B.i, p_i .+ t .* Xc), Y, backend, zero(eltype(p_i)))
    return Y
end

"""
    ricci_curvature(M::AbstractManifold, A::AbstractAtlas, i, a; backend=AutoForwardDiff())

Compute the scalar Ricci curvature (Ricci scalar) of the manifold `M` at the point
given by coordinates `a` in chart `i` of atlas `A`.

The scalar curvature is the trace of the Ricci tensor with respect to the inverse
local metric:
````math
    R = g^{ij} R_{ij}
````math

# Arguments

- `M::AbstractManifold` : manifold
- `A::AbstractAtlas`   : atlas providing charts / induced basis
- `i`                  : chart index in `A`
- `a`                  : coordinates of the point in chart `i` (length `n`)
- `backend::AbstractADType` : automatic-differentiation backend (default `AutoForwardDiff()`)

# Returns

- scalar (same element type as `a`) equal to the Ricci scalar at the point
"""
function ricci_curvature(
        M::AbstractManifold, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff(),
    )
    Ginv = inverse_local_metric(M, A, i, a)
    Ric = ricci_tensor(M, A, i, a; backend = backend)
    S = sum(Ginv .* Ric)
    return S
end

"""
    ricci_tensor(M::AbstractManifold, A::AbstractAtlas, i, a; backend::AbstractADType=AutoForwardDiff())

Compute the Ricci tensor of the manifold `M` at the point specified by coordinates `a`
in chart `i` of atlas `A`.

The Ricci tensor is the contraction of the Riemann tensor:

````math
    Ric_{p q} = R^u_{p u q}
````

# Arguments

- `M::AbstractManifold` : manifold
- `A::AbstractAtlas`   : atlas providing charts / induced basis
- `i`                  : chart index in `A`
- `a`                  : coordinates of the point in chart `i` (length `n`)
- `backend::AbstractADType` : automatic-differentiation backend (default `AutoForwardDiff()`)

# Returns

- `n×n` matrix with components `Ric[p, q]` (same element type as `a`)

# See also

- [`riemann_tensor`](@ref)
"""
function ricci_tensor(
        M::AbstractManifold, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()
    )
    n = length(a)
    T = eltype(a)
    Ric = zeros(T, n, n)
    return ricci_tensor!(M, Ric, A, i, a; backend = backend)
end

function ricci_tensor!(
        M::AbstractManifold, Ric, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()
    )
    # compute full Riemann tensor and contract: Ric_{ij} = R^u_{i u j}
    R = riemann_tensor(M, A, i, a; backend = backend)
    n = length(a)
    fill!(Ric, zero(eltype(R)))
    for p in 1:n, q in 1:n
        s = zero(eltype(R))
        for u in 1:n
            s += R[u, p, u, q]
        end
        Ric[p, q] = s
    end
    return Ric
end

"""
    riemann_tensor(M::AbstractManifold, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()

Compute the Riemann curvature tensor of manifold `M` at the point given by parameters `a` in
chart `i` of atlas `A`.

Returns a 4-dimensional array `R` of size (n,n,n,n) with components `R[u,i,j,k] = R^u_{ijk}`,
where the first index is the contravariant (upper) index and the remaining three are covariant
(lower) indices. The components satisfy, for coordinate vector fields e_i:

````math
    R^u_{ijk} e_u = (∇_{e_i} ∇_{e_j} - ∇_{e_j} ∇_{e_i} - ∇_{[e_i,e_j]}) e_k
````

# Arguments

- `M::AbstractManifold`: manifold
- `A::AbstractAtlas`: atlas used for coordinates/induced basis
- `i`: chart index in `A`
- `a`: coordinates of the point in chart `i` (length `n`)
- `backend::AbstractADType` : automatic-differentiation backend (default `AutoForwardDiff()`)

# Notes

- The default implementation computes connection coefficients via `affine_connection`
  and their directional derivatives (using automatic differentiation), so it can be
  expensive. Manifold-specific overrides yielding closed-form curvature are recommended
  for performance-critical code.
- Use `riemann_tensor!(M, Wc, A, i, a, Xc, Yc, Zc)` to compute the action `R(X,Y)Z` on
  coordinate vectors `Xc`, `Yc`, `Zc` without constructing the full 4-tensor.

# See also

[`affine_connection`](@ref)
"""
function riemann_tensor(
        M::AbstractManifold, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()
    )
    n = length(a)
    T = eltype(a)
    R = zeros(T, n, n, n, n) # R[u, i, j, k]
    return riemann_tensor!(M, R, A, i, a; backend = backend)
end

function riemann_tensor!(
        M::AbstractManifold, R, A::AbstractAtlas, i, a;
        backend::AbstractADType = AutoForwardDiff()
    )
    n = length(a)
    T = eltype(a)

    # basis vectors for directional derivatives
    e = [zeros(T, n) for _ in 1:n]
    for k in 1:n
        e[k][k] = 1
    end

    # Compute connection coefficients Γ^u_{ii j} by applying affine_connection to basis vectors
    Γ = zeros(T, n, n, n) # Γ[u, i, j]
    for ii in 1:n, j in 1:n
        affine_connection!(M, view(Γ, :, ii, j), A, i, a, e[ii], e[j])
    end

    # directional derivatives ∂_p Γ^u_{ij}
    dΓ = zeros(T, n, n, n, n) # dΓ[p, u, ii, j]
    for p in 1:n, ii in 1:n, j in 1:n
        DI.derivative!(
            t -> affine_connection(M, A, i, a .+ t .* e[p], e[ii], e[j]),
            view(dΓ, p, :, ii, j),
            backend,
            0.0,
        )
    end

    # Compute R^u_{ii j k} = ∂_i Γ^u_{j k} - ∂_j Γ^u_{ii k}
    #                     + Σ_m ( Γ^m_{j k} Γ^u_{ii m} - Γ^m_{ii k} Γ^u_{j m} )

    for u in 1:n, ii in 1:n, j in 1:n, k in 1:n
        R[u, ii, j, k] = dΓ[ii, u, j, k] - dΓ[j, u, ii, k]
        for m in 1:n
            R[u, ii, j, k] += Γ[m, j, k] * Γ[u, ii, m] - Γ[m, ii, k] * Γ[u, j, m]
        end
    end
    return R
end

"""
    riemann_tensor(M::AbstractManifold, A::AbstractAtlas, i, a, Xc, Yc, Zc;
                   backend::AbstractADType = AutoForwardDiff())

Compute the action of the Riemann curvature tensor `R` on tangent vectors with coordinates
`Xc`, `Yc` and `Zc` at the point specified by parameters `a` in chart `i` of atlas `A` on
manifold `M`.

This function returns the vector `W (in induced-chart coordinates) given by
``(R(X, Y) Z)``, i.e. the result of applying the curvature operator to `Zc`.

# Arguments
- `M::AbstractManifold` : manifold
- `A::AbstractAtlas`   : atlas providing charts / induced basis
- `i`                  : chart index in `A`
- `a`                  : coordinates of the point in chart `i` (length `n`)
- `Xc, Yc, Zc`         : coordinates of tangent vectors X, Y, Z in the chart-induced basis

# Keyword arguments
- `backend::AbstractADType = AutoForwardDiff()` : AD backend used when numerical derivatives are required

# Returns
- A vector (same shape/type as `Xc`) containing coordinates of ``(R(X, Y) Z)`` in the induced basis.

# Notes
- The default implementation builds the full 4-index Riemann tensor using the
  chart affine connection and its derivatives, then contracts with `Xc`, `Yc`, `Zc`.
  This is convenient but can be expensive; prefer manifold-specific overrides
  that compute R(X,Y)Z directly for performance-critical code.
- Inputs are expected to be expressed in the chart-induced basis associated with `A` and `i`.
- The function uses automatic differentiation (via `backend`) when computing
  directional derivatives of connection coefficients.

# See also
- `riemann_tensor(M, A, i, a)` which returns the full 4-tensor
- []`affine_connection`](@ref) used to obtain connection coefficients (see [`levi_civita_affine_connection!`](@ref) for a generic implementation)
"""
function riemann_tensor(
        M::AbstractManifold, A::AbstractAtlas, i, a, Xc, Yc, Zc;
        backend::AbstractADType = AutoForwardDiff()
    )
    Wc = similar(Xc)
    return riemann_tensor!(M, Wc, A, i, a, Xc, Yc, Zc; backend = backend)
end

function riemann_tensor!(
        M::AbstractManifold, Wc, A::AbstractAtlas, i, a, Xc, Yc, Zc;
        backend::AbstractADType = AutoForwardDiff()
    )
    # number of coordinates
    n = length(a)

    # Get the tensor
    R = riemann_tensor(M, A, i, a; backend = backend)

    # Apply to vectors: (R(X,Y)Z)^u = Σ_{i,j,k} R^u_{i j k} X^i Y^j Z^k
    fill!(Wc, 0)
    for u in 1:n
        for i in 1:n, j in 1:n, k in 1:n
            Wc[u] += R[u, i, j, k] * Xc[i] * Yc[j] * Zc[k]
        end
    end

    return Wc
end

for mf in [
        christoffel_symbols_first,
        christoffel_symbols_second,
        det_local_metric,
        einstein_tensor,
        flat!,
        gaussian_curvature,
        inverse_local_metric,
        local_metric,
        log_local_metric_density,
        mean,
        mean!,
        median,
        median!,
        ricci_curvature,
        ricci_tensor,
        riemann_tensor,
        riemannian_gradient,
        riemannian_gradient!,
        riemannian_Hessian,
        riemannian_Hessian!,
        sharp!,
    ]
    @eval is_metric_function(::typeof($mf)) = true
end
