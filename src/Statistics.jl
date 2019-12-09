using Random: shuffle!
@doc doc"""
    mean(M,x; weights=1/n*ones(n), stop_iter=100, kwargs... )

computes the Riemannian center of mass also known as Karcher mean of the vector
`x` of `n` points on the [`Manifold`](@ref) `M`. This function
uses the gradient descent scheme. Optionally one can provide
weights $w_i$ for the weighted Riemannian center of mass.
The general formula to compute the minimizer reads
````math
\argmin_{y\in\mathcal M} \frac{1}{2}\sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}^2(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemannian [`distance`](@ref).

Finally `stop_iter` denotes the maximal number of iterations to perform and
the `kwargs...` are passed to [`isapprox`](@ref) to stop, when the inimal change
between two iterates is small. For more stopping criteria
check the [`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

The algorithm is further described in 
> Afsari, B; Tron, R.; Vidal, R.: On the Convergence of Gradient
> Descent for Finding the Riemannian Center of Mass,
> SIAM Journal on Control and Optimization (2013), 51(3), pp. 2230–2260, 
> doi: [10.1137/12086282X](https://doi.org/10.1137/12086282X),
> arxiv: [1201.0925](https://arxiv.org/abs/1201.0925">1201.0925)
"""
function mean(M::Manifold, y, x::AbstractVector}; kwargs...)
    y = x[1]
    return mean!(M, y, x; kwargs...)
end
function mean!(M::Manifold, y, x::AbstractVector};
            weights= ones(length(x)) / length(x),
            stop_iter=100,
            kwargs...
        ) where {T}
    iter = 0
    yold = y
    v = fill(zero_tangent_vector(M,x),5)
    for i=1:stop_iter
        v = weights.*log!.*(Ref(M), v, Ref(yold), x)
        yold, y = y, exp(M, yold, sum( v ) / 2 )
        isapprox(M,y,yold; kwargs...) && break
    end
    return y
end

@doc doc"""
    median(M,x; weights=1/n*ones(n), stop_tol=10^-10, stop_iter=10000, use_rand = false )

computes the Riemannian median of the vector `x`  of `n` points on the
[`Manifold`](@ref) `M`. This function is nonsmooth (i.e nondifferentiable) and
uses a cyclic proximal point scheme. Optionally one can provide
weights $w_i$ for the weighted Riemannian median. The general formula to compute
the minimizer reads
````math
\argmin_{y\in\mathcal M}\sum_{i=1}^n w_i\mathrm{d}_{\mathcal M}(y,x_i),
````
where $\mathrm{d}_{\mathcal M}$ denotes the Riemannian [`distance`](@ref).

The cyclic proximal point can be run with a random cyclic order by setting
`use_rand` to `true`

Finally `stop_iter` denotes the maximal number of iterations to perform and
the `kwargs...` are passed to [`isapprox`](@ref) to stop, when the inimal change
between two iterates is small. For more stopping criteria
check the [`Manopt.jl`](https://manoptjl.org) package and use a solver therefrom.

The algorithm is further described in Algorithm 4.3 and 4.4 in 
> Bačák, M: Computing Medians and Means in Hadamard Spaces.
> SIAM Journal on Optimization (2014), 24(3), pp. 1542–1566,
> doi: [10.1137/140953393](https://doi.org/10.1137/140953393),
> arxiv: [1210.2145](https://arxiv.org/abs/1210.2145)
"""
function median(M::Manifold, x::AbstractVector; kwargs...)
    y = x[1]
    return median!(M,y,x; kwargs...)
end
function median!(M::Manifold, y, x::AbstractVector;
            weights= ones(length(x)) / length(x),
            stop_iter=10000,
            use_random = false,
            kwargs...
        ) where {T}
    n = length(x)
    yold = y
    order = collect(1:n)
    iter = 0
    for i=1:stop_iter
        λ = 1/(iter+1)
        yold = y
        use_random && shuffle!(order)
        for i=1:n
            t = min( λ * weights[order[i]] / distance(M,y,x[order[i]]) , 1 )
            exp!( M, y, yold, t*log(M, y, x[order[i]]) )
        end
        isapprox(M,y,yold; kwargs...) && break
    end
    return y
end