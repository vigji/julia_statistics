# Gaussian distribution:
gauss(x, μ, σ) = (1/√(2π)σ)*exp(-(x-μ)^2/(2σ^2))

# Bimodal distribution:
bimodal(x, μ1, σ1, μ2, σ2) = gauss(x, μ1, σ1) + gauss(x, μ2, σ2)


function gauss_step_on_dist(curr_x::Float64, funct::Function; step_dev=0.1, xmin=0., xmax=1.)
    # 1. Generate step from normal distribution:
    m = randn() * step_dev
    
    if xmin <= curr_x + m < xmax
    # 2. Compare distribution values to get p:
        p_move = min(funct(curr_x + m) / funct(curr_x), 1)

        # 3. Accept move with prob. p_move:
        curr_x = curr_x + m * (rand() < p_move)
    end
    return curr_x
end


function continuous_metropolis(funct, n_steps; xmin=0., xmax=1., step_dev=0.1)
    log = Array{Float64}(undef, n_steps)
    
    # Seed the chain at a random point of the distribution:
    log[1] = rand() * (xmax - xmin) + xmin
    
    for i in 2:n_steps
        log[i] = gauss_step_on_dist(log[i-1], funct)
    end
    
    return log
end
