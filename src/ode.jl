@doc doc"""
    solve_exp_ode(M::MetricManifold,
                  x,
                  v,
                  tspan;
                  solver=AutoVern9(Rodas5()),
                  kwargs...)

Approximate the exponential map on the manifold over the provided timespan
assuming the Levi-Civita connection by solving the ordinary differential
equation

$\frac{d^2}{dt^2} x^k + \Gamma^k_{ij} \frac{d}{dt} x_i \frac{d}{dt} x_j = 0,$

where $\Gamma^k_{ij}$ are the Christoffel symbols of the second kind, and
the Einstein summation convention is assumed. The arguments `tspan` and
`solver` follow the `OrdinaryDiffEq` conventions. `kwargs...` specify keyword
arguments that will be passed to `OrdinaryDiffEq.solve`.

Currently, the numerical integration is only accurate when using a single
coordinate chart that covers the entire manifold. This excludes coordinates
in an embedded space.
"""
function solve_exp_ode(M::MetricManifold,
                       x,
                       v,
                       tspan;
                       solver=AutoVern9(Rodas5()),
                       kwargs...)
    n = length(x)
    iv = SVector{n}(1:n)
    ix = SVector{n}(n+1:2n)
    u0 = similar(x, 2n)
    u0[iv] .= v
    u0[ix] .= x

    function exp_problem(u, p, t)
        M = p[1]
        dx = u[iv]
        x = u[ix]
        ddx = similar(u, Size(n))
        du = similar(u)
        Γ = christoffel_symbols_second(M, x)
        @einsum ddx[k] = -Γ[k,i,j] * dx[i] * dx[j]
        du[iv] .= ddx
        du[ix] .= dx
        return Base.convert(typeof(u), du)
    end

    p = (M,)
    prob = ODEProblem(exp_problem, u0, tspan, p)
    sol = solve(prob, solver; kwargs...)
    return sol
end

@traitfn function exp(M::MMT,
                      x,
                      v,
                      T::AbstractVector) where {MT<:Manifold,
                                                GT<:Metric,
                                                MMT<:MetricManifold{MT,GT};
                                                !HasMetric{MT,GT}}
    sol = solve_exp_ode(M, x, v, extrema(T); dense=false, saveat=T)
    n = length(x)
    return map(i -> sol.u[i][n+1:end], 1:length(T))
end

@traitfn function exp!(M::MMT, y, x, v) where {MT<:Manifold,
                                               GT<:Metric,
                                               MMT<:MetricManifold{MT,GT};
                                               !HasMetric{MT,GT}}
    tspan = (0.0, 1.0)
    sol = solve_exp_ode(M, x, v, tspan; dense=false, saveat=[1.0])
    n = length(x)
    y .= sol.u[1][n+1:end]
    return y
end
