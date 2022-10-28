using LinearAlgebra

# A simple implementation of LBFGS based on minFunc.m
# https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html

@doc raw"""
    _lbfgs_calc!(z, g, S, Y, rho, lbfgs_start, lbfgs_end, Hdiag, al_buff)
    LBFGS Search Direction
    This function returns the (L-BFGS) approximate inverse Hessian,
    multiplied by the negative gradient
    Use saved data Y S YS
    To avoid repeated allocation of memory use fix Y, S, YS but
    cycling the indices
    data is saved between lbfgs_start and lbfgs_end, inclusively both ends
    Hdiag is the diagonal Hessian approximation. al_buff is a buffer for the alpha coefficients
"""
@inline function _lbfgs_calc!(z, g, S, Y, rho, lbfgs_start, lbfgs_end, Hdiag, al_buff)
    # Set up indexing
    nVars, maxCorrections = size(S)
    if lbfgs_start == 1
        ind = Array(1:lbfgs_end)
        nCor = lbfgs_end - lbfgs_start + 1
    else
        ind = vcat(Array(lbfgs_start:maxCorrections),
                   Array(1:(lbfgs_end)))
        nCor = maxCorrections
    end
    @views begin
        al = al_buff[1:nCor]
    end
    al .= 0
    # we use q and z as different valriables in the algorithm
    # description but here we use one name to save storage    
    z .= -g
    nid = size(ind)[1]
    for j in 1:nid
        i = ind[nid-j+1]
        al[i] = dot(S[:, i], z)*rho[i]
        # z .-= al[i]*Y[:, i]
        mul!(z, UniformScaling(-al[i]), Y[:, i], 1, 1)
    end
    # Multiply by Initial Hessian.
    z .= Hdiag .* z    
    for i in ind
        be_i = dot(Y[:, i], z)*rho[i]
        # z .+= S[:, i]*(al[i]-be_i)
        mul!(z, UniformScaling(al[i]-be_i), S[:, i], 1, 1)        
    end
    return z
end
        
@doc raw"""
    _save!(y, s, S, Y, rho, lbfgs_start, lbfgs_end, Hdiag, precondx)
    append data to S, Y, rho.
     To avoid repeated allocation of memory use fixed storage
     but keep tract of indices, and recycle if the storage is full.
    Data to save:
    S: change in x (steps) (also called q_i)
    Y: Change in gradient
    YS: contraction of Y and S
"""        
@inline function _save!(y, s, S, Y, rho, lbfgs_start, lbfgs_end, Hdiag, precondx)
    ys = dot(y, s)
    skipped = 0
    corrections = size(S)[2]
    if ys > 1e-10
        if lbfgs_end < corrections
            lbfgs_end = lbfgs_end+1
            if lbfgs_start != 1
                if lbfgs_start == corrections
                    lbfgs_start = 1
                else
                    lbfgs_start = lbfgs_start+1
                end
            end
        else
            lbfgs_start = min(2, corrections)
            lbfgs_end = 1
        end

        S[:, lbfgs_end] .= s
        Y[:, lbfgs_end] .= y
        rho[lbfgs_end] = 1.0/ys

        # Update scale of initial Hessian approximation
        # Hdiag .= ys/sum(y.*pcondx.*y)*pcondx
        if isnothing(precondx)
            Hdiag .=  ys/dot(y,  y)
        else
            mul!(Hdiag,  ys/dot3(y, precondx, y), precondx)
        end
    else
        skipped = 1
    end
    return lbfgs_start, lbfgs_end, skipped
end

@doc raw"""
"""
@inline function _cubic_interpolate(x1, f1, g1, x2, f2, g2; bounds=nothing)
    # Compute bounds of interpolation area
    if !isnothing(bounds)
        xmin_bound, xmax_bound = bounds
    else
        xmin_bound, xmax_bound = x1 <= x2 ? (x1, x2) : (x2, x1)
    end
    # Code for most common case: cubic interpolation of 2 points
    #   w/ function and derivative values for both
    # Solution in this case (where x2 is the farthest point):
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1*d1 - g1 * g2
    if d2_square >= 0
        d2 = sqrt(d2_square)
        if x1 <= x2
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        end
        return min(max(min_pos, xmin_bound), xmax_bound)
    else
        return (xmin_bound + xmax_bound) / 2.
    end
end

@inline function  multd!(x_buff, x, t, d)
    mul!(x_buff, t, d)
    x_buff .+= x
end    

@doc raw""" Strong Wolf condition
"""    
@inline function _strong_wolfe(fun_obj, x, t, d, f, g, gx_buff, gtd,
                               c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25)
    d_norm = max_abs(d)
    # g = copy(g)
    # evaluate objective and gradient using initial step
    p = size(x)[1]
    @views begin
        g_new = gx_buff[1:p]
        g_prev = gx_buff[p+1:2*p]
        x_buff = gx_buff[2*p+1:3*p]
    end
    multd!(x_buff, x, t, d)
    
    # f_new = 0.
    f_new = fun_obj(0., g_new, x_buff)
    ls_func_evals = 1
    gtd_new = dot(g_new, d)

    # bracket an interval containing a point satisfying the Wolfe criteria
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = false
    ls_iter = 0
    bracket_gtd = Array([NaN, NaN])
    idx_g = 0
    idx_g_new = 1
    idx_g_prev = 2    
    while ls_iter < max_ls
        # check conditions
        if f_new > (f + c1 * t * gtd) || ((ls_iter > 1) && (f_new >= f_prev))
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [idx_g_prev, idx_g_new]
            bracket_gtd .= [gtd_prev, gtd_new]
            break
        end
        
        if abs(gtd_new) <= -c2 * gtd
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [idx_g_new]
            done = true
            break
        end
        if gtd_new >= 0
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [idx_g_prev, idx_g_new]
            bracket_gtd .= [gtd_prev, gtd_new]
            break
        end
        # interpolate
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev,
            f_prev,
            gtd_prev,
            t,
            f_new,
            gtd_new,
            bounds=(min_step, max_step))        

        # next step
        t_prev = tmp
        f_prev = f_new
        g_prev .= g_new
        gtd_prev = gtd_new
        multd!(x_buff, x, t, d)
        f_new = fun_obj(f_new, g_new, x_buff)
        ls_func_evals += 1
        gtd_new = dot(g_new, d)
        ls_iter += 1
    end
    # reached max number of iterations?
    if ls_iter == max_ls
        bracket = [0.0, t]
        bracket_f = [f, f_new]
        bracket_g = [idx_g, idx_g_new]
    end    
    # zoom phase: we now have a point satisfying the criteria, or
    # a bracket around it. We refine the bracket until we find the
    # exact point satisfying the criteria
    insuf_progress = false
    # find high and low points in bracket
    low_pos, high_pos = bracket_f[1] <= bracket_f[end] ? (1, 2) : (2, 1)    
    
    while !done && (ls_iter < max_ls)
        # line-search bracket is so small
        if abs(bracket[2] - bracket[1]) * d_norm < tolerance_change
            break
        end
        
        # compute new trial value
        t = _cubic_interpolate(bracket[1], bracket_f[1], bracket_gtd[1],
                               bracket[2], bracket_f[2], bracket_gtd[2])

        # test that we are making sufficient progress
        # in case `t` is so close to boundary, we mark that we are making
        # insufficient progress, and if
        #   + we have made insufficient progress in the last step, or
        #   + `t` is at one of the boundary,
        # we will move `t` to a position which is `0.1 * len(bracket)`
        # away from the nearest boundary point.
        bklo, bkhi =  bracket[1] <= bracket[2] ? (bracket[1], bracket[2]) : (bracket[2], bracket[1])
        eps = 0.1 * (bkhi - bklo)
        if min(bkhi - t, t - bklo) < eps
            # interpolation close to boundary
            if insuf_progress || (t >= bkhi) || (t <= bklo)
                # evaluate at 0.1 away from boundary
                if abs(t - bkhi) < abs(t - bklo)
                    t = bkhi - eps
                else
                    t = bklo + eps
                end
                insuf_progress = false
            else
                insuf_progress = true
            end      
        else
            insuf_progress = false
        end        
        # Evaluate new point
        multd!(x_buff, x, t, d)
        f_new = fun_obj(f_new, g_new, x_buff)
        ls_func_evals += 1
        gtd_new = dot(g_new, d)
        ls_iter += 1
        if (f_new > (f + c1 * t * gtd)) || (f_new >= bracket_f[low_pos])
            # Armijo condition not satisfied or not lower than lowest point
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = idx_g_new
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos =  bracket_f[1] <= bracket_f[2] ? (1, 2) : (2, 1)
        else
            if abs(gtd_new) <= -c2 * gtd
                # Wolfe conditions satisfied
                done = true
            elseif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0
                # old high becomes new low
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]
            end
            # new point becomes new low
            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = idx_g_new            
            bracket_gtd[low_pos] = gtd_new
            
        end
    end
    # return stuff
    t = bracket[low_pos]
    f_new = bracket_f[low_pos]
    function set_g!(g, idx)
        if idx == 1
            g .= g_new
        elseif idx == 2
            g .= g_prev
        end
    end    
    set_g!(g,  bracket_g[low_pos])
    return f_new, t, ls_func_evals
end                

@doc raw"""minimize
    minimize(fun_obj, x0, options)
"""
function minimize(
    fun_obj, x0;
    max_fun_evals=120,
    max_itr=100,
    grad_tol=1e-15,
    func_tol=1e-15,
    corrections=10,
    c1=1e-4, c2=0.9, max_ls=25,
    precond=nothing)
    
    # Initialize
    p = size(x0)[1]
    d = zeros(p)
    x = copy(x0)
    t = 1.0
    if isnothing(precond)
        precondx = nothing
    else
        precondx = precond(x)
    end
    
    funEvalMultiplier = 1.
    
    # Evaluate Initial Point
    g = similar(x)
    gx_buff = Array{eltype(x), 1}(undef, 3*p)

    # f = 0.0
    f = fun_obj(0., g, x)
    funEvals = 1
    g_old = similar(g)

    # Compute optimality of initial point
    optCond = max_abs(g)
    
    # Exit if initial point is optimal
    if optCond <= grad_tol
        exitflag = 1
        msg = "Optimality Condition below grad_tol"
        output = Dict("iterations" => 0,
                      "funcCount" => 1,
                      "firstorderopt" => max_abs(g),
                      "message" => msg)
        return x, f, exitflag, output
    end
    d = -g # Initially use steepest descent direction
    Tp = Base.eltype(x0)

    S = zeros(p, corrections)
    Y = zeros(p, corrections)
    rho = zeros(corrections)
    lbfgs_start = 0
    lbfgs_end = 0
    Hdiag = ones(p)
    al_buff = zeros(corrections)
    # Perform up to a maximum of 'max_itr' descent steps:
    i = 1
    while i <= max_itr
        # COMPUTE DESCENT DIRECTION 
        # Update the direction and step sizes
        if i > 1
            lbfgs_start, lbfgs_end, skipped = _save!(
                g - g_old, t*d, S, Y, rho,
                lbfgs_start, lbfgs_end, Hdiag, precondx)
            _lbfgs_calc!(d, g, S, Y, rho, lbfgs_start, lbfgs_end, Hdiag, al_buff)
            # display(string(i)*" "*string(lbfgs_start)*" "*string(lbfgs_end)*string(Hdiag[1:5]))
        end
        g_old .= g

        if !_is_legal(d)
            display("Step direction is illegal!\n")
            output = Dict("iterations" => i,
                          "funcCount" => funEvals*funEvalMultiplier,
                          "firstorderopt" => NaN,
                          "message" => "Step direction is illegal!\n")

            return x, f, -1, output
        end
        
        # COMPUTE STEP LENGTH
        # Directional Derivative
        gtd = dot(g, d)
        
        # Check that progress can be made along direction
        if gtd > -func_tol
            exitflag = 2
            msg = "Directional Derivative below func_tol"
            break
        end

        # Select Initial Guess
        if i == 1
            t = min(1.0, 1.0/sum_abs(g))
        else
            t = 1.
        end
        f_old = f
        gtd_old = gtd
        
        # function obj_func(f, grad, x, t, d)
        #    return funObj(f, grad, x + t*d)
        # end
        f, t, LSfunEvals = _strong_wolfe(
            fun_obj, x, t, d, f, g, gx_buff, gtd, c1, c2, func_tol, max_ls)
        funEvals = funEvals + LSfunEvals
        x .+= t*d
        # Check Optimality Condition
        if optCond <= grad_tol
            exitflag = 1
            msg = "Optimality Condition below grad_tol"
            break
        end
        # Check for lack of progress

        if max_abs(t*d) <= func_tol
            exitflag = 2
            msg = "Step Size below func_tol"
            break
        end
        if abs(f-f_old) < func_tol
            exitflag = 2
            msg = "Function Value changing by less than func_tol"
            break
        end
        # Check for going over iteration/evaluation limit
        if funEvals*funEvalMultiplier >= max_fun_evals
            exitflag = 0
            msg = "Reached Maximum Number of Function Evaluations"
            break
        end
        
        if i == max_itr
            exitflag = 0
            msg = "Reached Maximum Number of Iterations"
            break
        end
        i += 1
    end
    output = Dict("iterations" => i,
                  "funcCount" => funEvals*funEvalMultiplier,
                  "firstorderopt" => max_abs(g),
                  "message" => msg)
    return x, f, exitflag, output
end        

# internal functions. Self explanatory - for vector functions
@inline function _is_legal(v)
    return isreal(v) && sum(isnan.(v))==0 && sum(isinf.(v)) == 0;
end    

@inline function dot3(a, b, c)
    s = 0
    for i in 1:size(a)[1]
        s += a[i]*b[i]*c[i]
    end
    return s
end

@inline function sum_abs(a)
    s = 0
    for i in 1:size(a)[1]
        s += abs(a[i])
    end
    return s
end

@inline function max_abs(a)
    s = -Inf
    for i in 1:size(a)[1]
        if s < abs(a[i])
            s = abs(a[i])
        end
    end
    return s
end    
