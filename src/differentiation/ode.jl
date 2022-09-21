function solve_exp_ode(
    M::AbstractManifold,
    p,
    X;
    basis::AbstractBasis=DefaultOrthonormalBasis(),
    solver=AutoVern9(Rodas5()),
    backend=default_differential_backend(),
    kwargs...,
)
    n = length(p)
    iv = SVector{n}(1:n)
    ix = SVector{n}((n + 1):(2n))
    u0 = allocate(p, 2n)
    u0[iv] .= X
    u0[ix] .= p

    function exp_problem(u, params, t)
        M = params[1]
        dx = u[iv]
        p = u[ix]
        ddx = allocate(u, Size(n))
        du = allocate(u)
        Γ = christoffel_symbols_second(M, p, basis; backend=backend)
        @einsum ddx[k] = -Γ[k, i, j] * dx[i] * dx[j]
        du[iv] .= ddx
        du[ix] .= dx
        return Base.convert(typeof(u), du)
    end

    params = (M,)
    prob = ODEProblem(exp_problem, u0, (0.0, 1.0), params)
    sol = solve(prob, solver; kwargs...)
    q = sol.u[1][(n + 1):(2 * n)]
    return q
end
# also define exp! for metric manifold anew in this case
function exp!(
    ::TraitList{IsMetricManifold},
    M::AbstractDecoratorManifold,
    q,
    p,
    X;
    kwargs...,
)
    copyto!(M, q, solve_exp_ode(M, p, X; kwargs...))
    return q
end

function chart_exp_problem(u, params, t)
    M = params[1]
    B = params[2]
    p = u.x[1]
    dx = u.x[2]
    du = affine_connection(M, p, dx, dx, B)
    return ArrayPartition(dx, du)
end

function solve_chart_exp_ode(
    M::AbstractManifold,
    p,
    X,
    A::AbstractAtlas,
    i;
    solver=AutoVern9(Rodas5()),
    kwargs...,
)
    u0 = ArrayPartition(p, X)
    B = induced_basis(M, A, i, TangentSpaceType())
    params = (M, B)
    prob = ODEProblem(chart_exp_problem, u0, (0.0, 1.0), params)
    sol = solve(prob, solver; kwargs...)
    q = sol.u[1].x[1]
    return q
end

# also define exp! for metric manifold anew in this case
function exp!(
    ::TraitList{IsMetricManifold},
    M::AbstractDecoratorManifold,
    q,
    p,
    X;
    kwargs...,
)
    copyto!(M, q, solve_exp_ode(M, p, X; kwargs...))
    return q
end
