function allocate_coordinates(::ProductManifold, p, T, n::Int)
    return allocate(submanifold_component(p, 1), T, n)
end

function adjoint_Jacobi_field!(M::ProductManifold, Y, p, q, t, X, β::Tβ) where {Tβ}
    map(
        adjoint_Jacobi_field!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, q),
        ntuple(_ -> t, length(M.manifolds)),
        submanifold_components(M, X),
        ntuple(_ -> β, length(M.manifolds)),
    )
    return Y
end

metric(::ProductManifold) = ProductMetric()

@doc raw"""
    flat(M::ProductManifold, p, X::FVector{TangentSpaceType})

use the musical isomorphism to transform the tangent vector `X` from the tangent space at
`p` on the [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`)
`M` to a cotangent vector. This can be done elementwise for every entry of `X` (with respect
to the corresponding entry in `p`) separately.
"""
flat(::ProductManifold, ::Any...)

function jacobi_field!(M::ProductManifold, Y, p, q, t, X, β::Tβ) where {Tβ}
    map(
        jacobi_field!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, q),
        ntuple(_ -> t, length(M.manifolds)),
        submanifold_components(M, X),
        ntuple(_ -> β, length(M.manifolds)),
    )
    return Y
end

"""
    manifold_volume(M::ProductManifold)

Return the volume of [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`)
`M`, i.e. product of volumes of the manifolds `M` is constructed from.
"""
manifold_volume(M::ProductManifold) = mapreduce(manifold_volume, *, M.manifolds)

function riemannian_gradient!(M::ProductManifold, Y, p, X)
    map(
        riemannian_gradient!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return Y
end

@doc raw"""
    Y = riemannian_Hessian(M::ProductManifold, p, G, H, X)
    riemannian_Hessian!(M::ProductManifold, Y, p, G, H, X)

Compute the Riemannian Hessian ``\operatorname{Hess} f(p)[X]`` given the
Euclidean gradient ``∇ f(\tilde p)`` in `G` and the Euclidean Hessian ``∇^2 f(\tilde p)[\tilde X]`` in `H`,
where ``\tilde p, \tilde X`` are the representations of ``p,X`` in the embedding,.

On a product manifold, this decouples and can be computed elementwise.
"""
riemannian_Hessian(M::ProductManifold, p, G, H, X)

function riemannian_Hessian!(M::ProductManifold, Y, p, G, H, X)
    map(
        riemannian_Hessian!,
        M.manifolds,
        submanifold_components(M, Y),
        submanifold_components(M, p),
        submanifold_components(M, G),
        submanifold_components(M, H),
        submanifold_components(M, X),
    )
    return Y
end

@doc raw"""
    sharp(M::ProductManifold, p, ξ::FVector{CotangentSpaceType})

Use the musical isomorphism to transform the cotangent vector `ξ` from the tangent space at
`p` on the [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`)
`M` to a tangent vector. This can be done elementwise for every entry of `ξ` (and `p`)
separately
"""
sharp(::ProductManifold, ::Any...)

@doc raw"""
    volume_density(M::ProductManifold, p, X)

Return volume density on the [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`)
`M`, i.e. product of constituent volume densities.
"""
function volume_density(M::ProductManifold, p, X)
    dens = map(
        volume_density,
        M.manifolds,
        submanifold_components(M, p),
        submanifold_components(M, X),
    )
    return prod(dens)
end

"""
    ProductAtlas(atlases::NTuple{N, AbstractAtlas{𝔽}}) where {N, 𝔽}

Atlas on a [`ProductManifold`](@extref `ManifoldsBase.ProductManifold`) obtained by
taking the product of the factor atlases in `atlases`. Chart indices are tuples of the
corresponding factor chart indices and coordinates are concatenated factor coordinates.
"""
struct ProductAtlas{𝔽, TA <: NTuple{N, AbstractAtlas{𝔽}} where {N}} <: AbstractAtlas{𝔽}
    atlases::TA
end
ProductAtlas(atlases::AbstractAtlas{𝔽}...) where {𝔽} = ProductAtlas{𝔽, typeof(atlases)}(atlases)

"""
    get_default_atlas(M::ProductManifold)

Return the product of the default atlases of the factor manifolds of `M`.
"""
get_default_atlas(M::ProductManifold) = ProductAtlas(map(get_default_atlas, M.manifolds))

get_chart_index(M::ProductManifold, A::ProductAtlas, p) =
    map(get_chart_index, M.manifolds, A.atlases, submanifold_components(M, p))

function get_chart_index(M::ProductManifold, A::ProductAtlas, i::Tuple, a)
    offset = 0
    return map(M.manifolds, A.atlases, i) do Mj, Aj, ij
        dim = manifold_dimension(Mj)
        result = get_chart_index(Mj, Aj, ij, view(a, (offset + 1):(offset + dim)))
        offset += dim
        return result
    end
end

function get_parameters!(M::ProductManifold, a, A::ProductAtlas, i::Tuple, p)
    offset = 0
    map(M.manifolds, A.atlases, i, submanifold_components(M, p)) do Mj, Aj, ij, pj
        dim = manifold_dimension(Mj)
        get_parameters!(Mj, view(a, (offset + 1):(offset + dim)), Aj, ij, pj)
        offset += dim
    end
    return a
end

function get_point!(M::ProductManifold, p, A::ProductAtlas, i::Tuple, a)
    offset = 0
    map(M.manifolds, submanifold_components(M, p), A.atlases, i) do Mj, pj, Aj, ij
        dim = manifold_dimension(Mj)
        get_point!(Mj, pj, Aj, ij, view(a, (offset + 1):(offset + dim)))
        offset += dim
    end
    return p
end

function get_coordinates_induced_basis!(
        M::ProductManifold,
        c,
        p,
        X,
        B::InducedBasis{ℝ, TangentSpaceType, <:ProductAtlas},
    )
    offset = 0
    map(M.manifolds, submanifold_components(M, p), submanifold_components(M, X), B.A.atlases, B.i) do Mj, pj, Xj, Aj, ij
        dim = manifold_dimension(Mj)
        get_coordinates_induced_basis!(
            Mj,
            view(c, (offset + 1):(offset + dim)),
            pj,
            Xj,
            induced_basis(Mj, Aj, ij),
        )
        offset += dim
    end
    return c
end

function get_coordinates!(M::ProductManifold, c, p, X, B::InducedBasis{ℝ, TangentSpaceType, <:ProductAtlas})
    return get_coordinates_induced_basis!(M, c, p, X, B)
end


function get_vector_induced_basis!(
        M::ProductManifold,
        Y,
        p,
        c,
        B::InducedBasis{ℝ, TangentSpaceType, <:ProductAtlas},
    )
    offset = 0
    map(M.manifolds, submanifold_components(M, Y), submanifold_components(M, p), B.A.atlases, B.i) do Mj, Yj, pj, Aj, ij
        dim = manifold_dimension(Mj)
        get_vector_induced_basis!(
            Mj,
            Yj,
            pj,
            view(c, (offset + 1):(offset + dim)),
            induced_basis(Mj, Aj, ij),
        )
        offset += dim
    end
    return Y
end

function get_vector!(M::ProductManifold, X, p, Xⁱ, B::InducedBasis{ℝ, TangentSpaceType, <:ProductAtlas})
    return get_vector_induced_basis!(M, X, p, Xⁱ, B)
end
