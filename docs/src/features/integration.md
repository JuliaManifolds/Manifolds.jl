# Integration

A simple Monte Carlo integrator on a compact manifold can be realized as follows:

```@example 1
using Manifolds, LinearAlgebra, Distributions
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

We will now try to verify that volume density correction correctly changes probability density of a (wrapped) normal distribution

```@example 1
function pdf_basic(M::AbstractManifold, p)
    return pdf(MvNormal(zeros(manifold_dimension(M)), 0.2*I), p)
end

function pdf_corrected(M::AbstractManifold, q)
    p = [1.0, 0.0, 0.0]
    X = log(M, p, q)
    Xc = get_coordinates(M, p, X, DefaultOrthonormalBasis())
    vd = abs(volume_density(M, p, X))
    if vd > eps()
        return pdf_basic(M, Xc) / vd
    else
        return 0.0
    end
end

simple_mc_integrate(Sphere(2), pdf_corrected; N=1000000)
```

We can also make a Pelletier's isotropic kernel density estimator:

```@example 1
struct PelletierKDE{TM<:AbstractManifold,TPts<:AbstractVector}
    M::TM
    bandwidth::Float64
    pts::TPts
end

(kde::PelletierKDE)(M, p) = kde(p)
function (kde::PelletierKDE)(p)
    n = length(kde.pts)
    sum_kde = 0.0
    function epanechnikov_kernel(x)
        if x < 1
            return 3*(1-x^2)/4
        else
            return 0.0
        end
    end
    for i in 1:n
        X = log(kde.M, p, kde.pts[i])
        Xn = norm(kde.M, p, X)
        sum_kde += epanechnikov_kernel(Xn / kde.bandwidth) / volume_density(kde.M, p, X)
    end
    sum_kde /= n * kde.bandwidth^manifold_dimension(kde.M)
    return sum_kde
end

M = Sphere(2)
pts = rand(M, 1)
kde = PelletierKDE(M, 0.2, pts)
println(kde(rand(M)))
```

## Documentation

```@docs
manifold_volume(::AbstractManifold)
volume_density(::AbstractManifold, ::Any, ::Any)
```
