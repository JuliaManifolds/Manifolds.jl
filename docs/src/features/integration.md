# Integration

This part of documentation covers integration of scalar functions defined on manifolds.
The basic concepts are derived from geometric measure theory.
In principle, there are many ways in which a manifold can be equipped with a measure that can be later used to define an integral.
One of the most popular ways is based on pushing the Lebesgue measure on a tangent space through the exponential map.
Any other suitable atlas could be used, not just the one defined by normal coordinates, though each one requires different volume density corrections due to the Jacobian determinant of the pushforward.
`Manifolds.jl` provides the function [`volume_density`](@ref) that calculates that quantity.

While many sources define volume density as a function of two points, `Manifolds.jl` decided to use the more general point-tangent vector formulation. The two-points variant can be implemented as `volume_density_two_points(M, p, q) = volume_density(M, p, log(M, p, q))`.

The simplest way to of integrating a function on a compact manifold is through a Monte Carlo integrator.
A simple variant can be implemented as follows:

```@example 1
using Manifolds, LinearAlgebra, Distributions, SpecialFunctions
function simple_mc_integrate(M::AbstractManifold, f; N::Int = 1000)
    V = manifold_volume(M)
    sum = 0.0
    q = rand(M)
    for i in 1:N
        sum += f(M, q)
        rand!(M, q)
    end
    return V * sum/N
end
```

[`manifold_volume`](@ref) and [`volume_density`](@ref) are closely related to each other, though very few sources explore this connection, and some even claiming a certain level of arbitrariness in defining `manifold_volume`.
Volume is sometimes considered arbitrary because Riemannian metrics on some spaces like the manifold of rotations are defined with arbitrary constants.
However, once a constant is picked (and it must be picked before any useful computation can be performed), all geometric operations must follow in a consistent way: inner products, exponential and logarithmic maps, volume densities, etc.
`Manifolds.jl` consistently picks such constants and provides a unified framework, though it sometimes results in picking a different constant than what is the most popular in some sub-communities.

On Lie groups the situation is more complicated. Invariance under left or right group action is a desired property that leads one to consider Haar measures.
It is, however, unclear what are the practical benefits of considering Haar measures over the Lebesgue measure of the underlying manifold, which often turns out to be invariant anyway.

Last point before we turn to consider an example is the matter of integration through charts.
This is an approach not currently supported by `Manifolds.jl`.
One has to define a suitable set of disjoint charts covering the entire manifold and use a method for multivariate Euclidean integration.
Note that ranges of parameters have to be adjusted for each manifold and scaling based on the metric needs to be applied.
See [^BoyaSudarshanTilma2003] for some considerations on symmetric spaces.

## Distributions

We will now try to verify that volume density correction correctly changes probability density of a (wrapped) normal distribution.
`pdf_tangent_space` represents probability density of a normally distributed random variable in the tangent space.
`pdf_manifold` refers to the probability density of the distribution from the tangent space wrapped using exponential map on the manifold.
[`volume_density`](@ref) function calculates the necessary correction.

Note that our simplified `pdf_manifold` assumes that the probability mass of `pdf_tangent_space` outside of (local) injectivity radius is negligible. If it was not, a summation over vectors in the tangent plane pointing at `q` would have to be performed.

```@example 1
function pdf_tangent_space(M::AbstractManifold, p)
    return pdf(MvNormal(zeros(manifold_dimension(M)), 0.2*I), p)
end

function pdf_manifold(M::AbstractManifold, q)
    p = [1.0, 0.0, 0.0]
    X = log(M, p, q)
    Xc = get_coordinates(M, p, X, DefaultOrthonormalBasis())
    vd = abs(volume_density(M, p, X))
    if vd > eps()
        return pdf_tangent_space(M, Xc) / vd
    else
        return 0.0
    end
end

println(simple_mc_integrate(Sphere(2), pdf_manifold; N=1000000))
```

We can also make a Pelletier's isotropic kernel density estimator:

```@example 1
struct PelletierKDE{TM<:AbstractManifold,TPts<:AbstractVector}
    M::TM
    bandwidth::Float64
    pts::TPts
end

(kde::PelletierKDE)(::AbstractManifold, p) = kde(p)
function (kde::PelletierKDE)(p)
    n = length(kde.pts)
    d = manifold_dimension(kde.M)
    sum_kde = 0.0
    function epanechnikov_kernel(x)
        if x < 1
            return gamma(2+d/2) * (1-x^2)/(π^(d/2))
        else
            return 0.0
        end
    end
    for i in 1:n
        X = log(kde.M, p, kde.pts[i])
        Xn = norm(kde.M, p, X)
        sum_kde += epanechnikov_kernel(Xn / kde.bandwidth) / volume_density(kde.M, p, X)
    end
    sum_kde /= n * kde.bandwidth^d
    return sum_kde
end

M = Sphere(2)
pts = rand(M, 8)
kde = PelletierKDE(M, 0.7, pts)
println(simple_mc_integrate(Sphere(2), kde; N=1000000))
println(kde(rand(M)))
```

The radially symmetric multivariate Epanechnikov kernel is described in [^LangrenéWarin2019].

## Documentation

```@docs
manifold_volume(::AbstractManifold)
volume_density(::AbstractManifold, ::Any, ::Any)
```

## References

[^Tornier2020]:
    > S. Tornier, “Haar Measures.” arXiv, Jun. 19, 2020.
    > doi: 10.48550/arXiv.2006.10956.

[^LangrenéWarin2019]:
    > N. Langrené and X. Warin, “Fast and Stable Multivariate Kernel Density Estimation by
    > Fast Sum Updating,” Journal of Computational and Graphical Statistics, vol. 28, no. 3,
    > pp. 596–608, Jul. 2019,
    > doi: 10.1080/10618600.2018.1549052.
