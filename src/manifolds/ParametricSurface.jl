@doc raw"""
    ParametricSurface{TMP<:AbstractManifold{ℝ},TE<:Euclidean,TF,TIF,TDF} <: AbstractDecoratorManifold{ℝ}

Surface in ℝⁿ described by a parametric function `f` defined on a parameter space `M_param`.
The embedding ℝⁿ is described by a Euclidean manifold `M_embed`. The metric is the
restriction of the Euclidean metric to the surface.

The functions `f!`, `inverse_f!`, and `jacobian_f!` should be defined in-place. They map
parameters `p` to embedded points `q`, embedded points to parameters, and parameters to the
Jacobian of `f`, respectively.

# Constructor

    ParametricSurface(M_param, M_embed, f!, inverse_f!, jacobian_f!)
"""
struct ParametricSurface{
        TMP <: AbstractManifold{ℝ},
        TE <: Euclidean,
        TF,
        TIF,
        TDF,
    } <: AbstractDecoratorManifold{ℝ}
    M_param::TMP
    M_embed::TE
    f!::TF
    inverse_f!::TIF
    jacobian_f!::TDF
end

"""
    decorated_manifold(M::ParametricSurface)

Return the parameter-space manifold of `M`.
"""
decorated_manifold(M::ParametricSurface) = M.M_param

"""
    get_embedding(M::ParametricSurface)

Return the Euclidean embedding manifold of `M`.
"""
get_embedding(M::ParametricSurface) = M.M_embed

"""
    ManifoldsBase.get_embedding_type(::ParametricSurface)

Declare `ParametricSurface` as directly and isometrically embedded in its Euclidean embedding.
"""
ManifoldsBase.get_embedding_type(::ParametricSurface) =
    ManifoldsBase.IsometricallyEmbeddedManifoldType(ManifoldsBase.DirectEmbedding())

"""
    manifold_dimension(M::ParametricSurface)

Return the dimension of the parameter space of `M`.
"""
manifold_dimension(M::ParametricSurface) = manifold_dimension(M.M_param)

"""
    representation_size(M::ParametricSurface)

Return the representation size of the Euclidean embedding of `M`.
"""
representation_size(M::ParametricSurface) = representation_size(M.M_embed)

"""
    check_point(M::ParametricSurface, q; kwargs...)

Check whether the embedded point `q` lies on `M` by applying the inverse parametrization and
reconstructing it with `f!`.
"""
function check_point(M::ParametricSurface, q; kwargs...)
    p = ManifoldsBase.allocate_on(M.M_param)
    M.inverse_f!(p, q)
    q_reconstructed = similar(q)
    M.f!(q_reconstructed, p)
    if !isapprox(q, q_reconstructed; kwargs...)
        return DomainError(q, "The point $(q) does not lie on the $(M).")
    end
    return nothing
end

"""
    check_vector(M::ParametricSurface, p, X; atol::Real=sqrt(eps(float(number_eltype(p)))), kwargs...)

Check whether `X` is tangent to the parametric surface `M` at embedded point `p`.
"""
function check_vector(
        M::ParametricSurface, p, X;
        atol::Real = sqrt(eps(float(number_eltype(p)))), kwargs...,
    )
    parameters = ManifoldsBase.allocate_on(M.M_param)
    M.inverse_f!(parameters, p)
    J = Matrix{promote_type(eltype(p), eltype(X))}(
        undef,
        manifold_dimension(M.M_embed),
        manifold_dimension(M.M_param),
    )
    M.jacobian_f!(J, parameters)
    residual = X - J * (J \ X)
    if !isapprox(norm(residual), 0; atol = atol, kwargs...)
        return DomainError(residual, "The vector $(X) is not tangent to $(p) from $(M).")
    end
    return nothing
end

"""
    project!(M::ParametricSurface, q, p)

Project `p` onto `M` by applying the inverse parametrization followed by `f!` and store the
result in `q`.
"""
function project!(M::ParametricSurface, q, p)
    parameters = ManifoldsBase.allocate_on(M.M_param)
    M.inverse_f!(parameters, p)
    M.f!(q, parameters)
    return q
end

"""
    project!(M::ParametricSurface, Y, p, X)

Project the ambient vector `X` orthogonally onto the tangent space of `M` at `p` and store it
in `Y`.
"""
function project!(M::ParametricSurface, Y, p, X)
    parameters = ManifoldsBase.allocate_on(M.M_param)
    M.inverse_f!(parameters, p)
    J = Matrix{promote_type(eltype(p), eltype(X))}(
        undef,
        manifold_dimension(M.M_embed),
        manifold_dimension(M.M_param),
    )
    M.jacobian_f!(J, parameters)
    Y .= J * (J \ X)
    return Y
end

"""
    ParametricSurfaceAtlas(A)

Atlas for a [`ParametricSurface`](@ref) obtained by forwarding the parameter-space atlas `A`.
"""
struct ParametricSurfaceAtlas{TA <: AbstractAtlas{ℝ}} <: AbstractAtlas{ℝ}
    A::TA
end

"""
    get_default_atlas(M::ParametricSurface)

Return the [`ParametricSurfaceAtlas`](@ref) forwarding the default atlas of the parameter
space of `M`.
"""
get_default_atlas(M::ParametricSurface) = ParametricSurfaceAtlas(get_default_atlas(M.M_param))

"""
    affine_connection!(M::ParametricSurface, Zc, A::ParametricSurfaceAtlas, i, a, Xc, Yc)

Store the Levi-Civita affine connection in forwarded chart coordinates in `Zc`.
"""
function affine_connection!(M::ParametricSurface, Zc, A::ParametricSurfaceAtlas, i, a, Xc, Yc)
    return levi_civita_affine_connection!(M, Zc, A, i, a, Xc, Yc)
end

"""
    check_chart_switch(M::ParametricSurface, A::ParametricSurfaceAtlas, i, a; kwargs...)

Delegate the chart-switch condition to the parameter-space atlas wrapped by `A`.
"""
check_chart_switch(M::ParametricSurface, A::ParametricSurfaceAtlas, i, a; kwargs...) =
    check_chart_switch(M.M_param, A.A, i, a; kwargs...)

"""
    gaussian_curvature(M::ParametricSurface, p; kwargs...)

Return the Gaussian curvature of `M` at embedded point `p`, computed in its forwarded atlas.
"""
function gaussian_curvature(M::ParametricSurface, p; kwargs...)
    A = get_default_atlas(M)
    i = get_chart_index(M, A, p)
    a = get_parameters(M, A, i, p)
    return ricci_curvature(M, A, i, a; kwargs...) / 2
end

"""
    inner(M::ParametricSurface, A::ParametricSurfaceAtlas, i, a, Xc, Yc)

Return the pullback Euclidean inner product of chart-coordinate vectors `Xc` and `Yc`.
"""
function inner(M::ParametricSurface, A::ParametricSurfaceAtlas, i, a, Xc, Yc)
    parameters = get_point(M.M_param, A.A, i, a)
    J = Matrix{promote_type(eltype(parameters), eltype(Xc), eltype(Yc))}(
        undef,
        manifold_dimension(M.M_embed),
        manifold_dimension(M.M_param),
    )
    M.jacobian_f!(J, parameters)
    return dot(J * Xc, J * Yc)
end

"""
    get_chart_index(M::ParametricSurface, A::ParametricSurfaceAtlas, p)

Return the chart index in `A` containing embedded point `p`.
"""
function get_chart_index(M::ParametricSurface, A::ParametricSurfaceAtlas, p)
    parameters = ManifoldsBase.allocate_on(M.M_param)
    M.inverse_f!(parameters, p)
    return get_chart_index(M.M_param, A.A, parameters)
end

"""
    get_chart_index(M::ParametricSurface, A::ParametricSurfaceAtlas, i, a)

Return the chart index in `A` containing the point represented by local coordinates `a` in
chart `i`.
"""
function get_chart_index(M::ParametricSurface, A::ParametricSurfaceAtlas, i, a)
    parameters = get_point(M.M_param, A.A, i, a)
    return get_chart_index(M.M_param, A.A, parameters)
end

"""
    get_parameters!(M::ParametricSurface, a, A::ParametricSurfaceAtlas, i, p)

Store in `a` the local parameters of embedded point `p` in chart `i` of
[`ParametricSurfaceAtlas`](@ref) `A`.
"""
function get_parameters!(M::ParametricSurface, a, A::ParametricSurfaceAtlas, i, p)
    parameters = ManifoldsBase.allocate_on(M.M_param)
    M.inverse_f!(parameters, p)
    return get_parameters!(M.M_param, a, A.A, i, parameters)
end

"""
    get_point!(M::ParametricSurface, p, A::ParametricSurfaceAtlas, i, a)

Store in `p` the embedded point represented by local parameters `a` in chart `i` of
[`ParametricSurfaceAtlas`](@ref) `A`.
"""
function get_point!(M::ParametricSurface, p, A::ParametricSurfaceAtlas, i, a)
    parameters = ManifoldsBase.allocate_on(M.M_param)
    get_point!(M.M_param, parameters, A.A, i, a)
    M.f!(p, parameters)
    return p
end

"""
    get_coordinates_induced_basis!(
        M::ParametricSurface,
        cX,
        p,
        X,
        B::InducedBasis{ℝ, TangentSpaceType, <:ParametricSurfaceAtlas}
    )

Store in `cX` the coordinates of embedded tangent vector `X` at `p` in the basis induced by
forwarded atlas `B`.
"""
function get_coordinates_induced_basis!(
        M::ParametricSurface, cX, p, X,
        B::InducedBasis{ℝ, TangentSpaceType, <:ParametricSurfaceAtlas},
    )
    parameters = ManifoldsBase.allocate_on(M.M_param)
    M.inverse_f!(parameters, p)
    J = Matrix{promote_type(eltype(p), eltype(X))}(
        undef,
        manifold_dimension(M.M_embed),
        manifold_dimension(M.M_param),
    )
    M.jacobian_f!(J, parameters)
    B_param = induced_basis(M.M_param, B.A.A, B.i)
    return get_coordinates!(M.M_param, cX, parameters, J \ X, B_param)
end

"""
    get_vector_induced_basis!(
        M::ParametricSurface,
        X,
        p,
        cX,
        B::InducedBasis{ℝ, TangentSpaceType, <:ParametricSurfaceAtlas},
    )

Store in `X` the embedded tangent vector at `p` represented by coordinates `cX` in the basis
induced by forwarded atlas `B`.
"""
function get_vector_induced_basis!(
        M::ParametricSurface,
        X,
        p,
        cX,
        B::InducedBasis{ℝ, TangentSpaceType, <:ParametricSurfaceAtlas},
    )
    parameters = ManifoldsBase.allocate_on(M.M_param)
    M.inverse_f!(parameters, p)
    B_param = induced_basis(M.M_param, B.A.A, B.i)
    X_param = get_vector(M.M_param, parameters, cX, B_param)
    J = Matrix{promote_type(eltype(p), eltype(X_param))}(
        undef,
        manifold_dimension(M.M_embed),
        manifold_dimension(M.M_param),
    )
    M.jacobian_f!(J, parameters)
    X .= J * X_param
    return X
end
