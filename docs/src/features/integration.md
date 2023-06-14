# Integration

A simple Monte Carlo integrator on a manifold can be realized as follows:
```@example 1
using Manifolds, LinearAlgebra, Distributions
function simple_mc_integrate(M::AbstractManifold, f; N::Int = 1000)
    V = Manifolds.manifold_volume(M)
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
function pdf_basic(p)
    return pdf(MvNormal([0.0, 0.0], 0.2*I), p)
end

function pdf_corrected(M::AbstractManifold, q)
    p = [1.0, 0.0, 0.0]
    X = log(M, p, q)
    return pdf_basic(get_coordinates(M, p, X, DefaultOrthonormalBasis())) / Manifolds.volume_density(M, p, X) 
end

simple_mc_integrate(Sphere(2), pdf_corrected; N=1000000)
```
