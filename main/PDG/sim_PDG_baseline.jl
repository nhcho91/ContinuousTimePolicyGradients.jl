# main function definition
include("main_PDG.jl")

## execute optimisation and simulation
for t_f_val in Float32.(38:1:44)
    @time (result, policy_NN, fwd_ensemble_sol, loss_history) = main(1000, 100, 1f-1; k_r_f_val = 1f6, k_v_f_val = 1f5, t_f_val = t_f_val)

    # re-execute optimisation and simulation
    p_NN_prev = result.u
    @time (result, policy_NN, fwd_ensemble_sol, loss_history) = main(0, 0, 1f-1; k_r_f_val = 0.0f0, k_v_f_val = 0.0f0, p_NN_0 = p_NN_prev)

    # save results
    dir_string = "t_f_$(round(Int,t_f_val))"
    mkdir(dir_string)
    cd(dir_string)

    jldsave("sim_dat.jld2"; result, fwd_ensemble_sol, loss_history, main)

    # plot simulation results
    # (result, fwd_ensemble_sol, loss_history) = load(".jld2", "result", "fwd_ensemble_sol", "loss_history")

    x_names = ["\$X\$" "\$Y\$" "\$Z\$" "\$V_X\$" "\$V_Y\$" "\$V_Z\$" "\$m\$"]
    vars_x = 1:7 # [1,2,3, (1,2), (2,3)]
    u_names = ["\$T\$" "\$\\sigma_T\$" "\$\\eta_T\$"]
    vars_u = 1:3
    y_names = ["\$X\$" "\$Y\$" "\$Z\$" "\$V_X\$" "\$V_Y\$" "\$V_Z\$" "\$m\$" "\$h\$" "\$V\$" "\$e_{r}\$" "\$e_{v}\$" "\$x_{E}\$" "\$y_{N}\$" "\$z_{U}\$"]
    vars_y = 10:14
    y_NN_names = ["\$T\$" "\$\\sigma_T\$" "\$\\eta_T\$"]
    vars_y_NN = 1:3

    (f_x, f_u, f_y, f_y_NN, f_L) = view_result([], fwd_ensemble_sol, loss_history; x_names = x_names, vars_x = vars_x, u_names = u_names, vars_u = vars_u, y_names = y_names, vars_y = vars_y, y_NN_names = y_NN_names, vars_y_NN = vars_y_NN, linealpha = 0.6)

    # 3D plot
    y = hcat(fwd_ensemble_sol[1].y...)'

    @show (h_f, V_f, e_r_f, e_v_f, m_f) = y[end,7:11];

    f_EN = plot(y[:,12], y[:,13], aspect_ratio = :equal, label="Lander", xlabel= "\$x_{E}\$", ylabel ="\$y_{N}\$")
    plot!(f_EN, zeros(2), zeros(2), label="Goal", marker=:circle)
    display(f_EN)
    savefig(f_EN, "f_EN.pdf")

    f_NU = plot(y[:,13], y[:,14], aspect_ratio = :equal, label="Lander", xlabel= "\$y_{N}\$", ylabel ="\$z_{U}\$")
    plot!(f_NU, zeros(2), zeros(2), label="Goal", marker=:circle)
    display(f_NU)
    savefig(f_NU, "f_NU.pdf")

    f_3D = plot(y[:,12], y[:,13], y[:,14], aspect_ratio = :equal, label="Lander", xlabel= "\$x_{E}\$", ylabel ="\$y_{N}\$", zlabel="\$z_{U}\$")
    plot!(f_3D, zeros(2), zeros(2), zeros(2), label="Goal", marker=:circle)
    display(f_3D)
    savefig(f_3D, "f_3D.pdf")

    # Learning curve
    # using Plots, JLD2
    # loss_base       = load("DS_base.jld2", "loss_history")

    # f_J = plot([[loss_base], [loss_unscaled], [loss_discrete]], label = ["base" "unscaled" "discrete"], xlabel = "iteration", ylabel = "cost \$J\$", yaxis = :log10)
    # savefig(f_J, "f_J.pdf")

    cd("..")
end


## All case
t_f_list = 38:1:44
y_f_list = zeros(length(t_f_list), 3)
f_EN = plot()
f_NU = plot()
f_3D = plot()
f_u1 = plot()
f_u2 = plot()
f_u3 = plot()

for i in 1:length(t_f_list)
    @show t_f_val = t_f_list[i];
    dir_string = "t_f_$(t_f_val)"
    label_string = "\$t_{f} = $(t_f_val)\$"

    (result, fwd_ensemble_sol, loss_history) = load(joinpath("baseline","3rd",dir_string,"sim_dat.jld2"), "result", "fwd_ensemble_sol", "loss_history")
    t = fwd_ensemble_sol[1].sol.t
    y = hcat(fwd_ensemble_sol[1].y...)'
    u = hcat(fwd_ensemble_sol[1].u...)'
    
    @show (e_r_f, e_v_f, m_f) = y_f_list[i,:] = y[end,[10,11,7]];

    plot!(f_EN, y[:,12], y[:,13], aspect_ratio = :equal, label=label_string, xlabel= "\$x_{E}\$ [m]", ylabel ="\$y_{N}\$ [m]")    
    
    plot!(f_NU, y[:,13], y[:,14], aspect_ratio = :equal, label=label_string, xlabel= "\$y_{N}\$ [m]", ylabel ="\$z_{U}\$ [m]", legend=:bottomleft)

    plot!(f_3D, y[:,12], y[:,13], y[:,14], aspect_ratio = :equal, label=label_string, xlabel= "\$x_{E}\$ [m]", ylabel ="\$y_{N}\$ [m]", zlabel="\$z_{U}\$ [m]") 

    plot!(f_u1, t, u[:,1], label=label_string, xlabel = "\$t\$ [s]", ylabel = "\$\\delta_{T}\$")
    plot!(f_u2, t, rad2deg.(u[:,2]), label=:false, xlabel = "\$t\$ [s]", ylabel = "\$\\sigma_{T}\$ [deg]")
    plot!(f_u3, t, rad2deg.(u[:,3]), label=:false, xlabel = "\$t\$ [s]", ylabel = "\$\\eta_{T}\$ [deg]")
end
plot!(f_EN, zeros(2), zeros(2), label="Goal", marker=:circle)
plot!(f_NU, zeros(2), zeros(2), label="Goal", marker=:circle)
plot!(f_3D, zeros(2), zeros(2), zeros(2), label="Goal", marker=:circle, camera = (60,20))
f_u = plot(f_u1, f_u2, f_u3, layout = (3,1))

f_y_f = scatter(t_f_list, y_f_list, layout=(3,1), label=:false, xlabel = "\$t_{f}\$ [s]", ylabel = ["\$e_{r_{f}} = \\left\\| \\mathbf{r}_{f} - \\mathbf{r}_{f_{d}} \\right\\|\$ [m]" "\$e_{v_{f}} = \\left\\| \\mathbf{v}_{f} - \\mathbf{v}_{f_{d}} \\right\\|\$ [m/s]" "\$m_{f}\$ [kg]"], size = (600,600))

display(f_EN)
display(f_NU)
display(f_3D)
display(f_u)
display(f_y_f)

savefig(f_EN, "f_EN_summary.pdf")
savefig(f_NU, "f_NU_summary.pdf")
savefig(f_3D, "f_3D_summary.pdf")
savefig(f_u, "f_u_summary.pdf")
savefig(f_y_f, "f_final_output_summary.pdf")