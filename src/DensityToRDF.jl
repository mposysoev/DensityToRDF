module DensityToRDF

using DelimitedFiles
using Flux
using BSON: @save, @load
using Plots

export validate_data, prepare_training_data, create_model, train_model, evaluate_model,
       save_model, save_losses, plot_loss_values, plot_evaluation_results, plot_rdf_model,
       load_reference_data, plot_results

function plot_rdf_model(rdf_data, concentration)
    p = plot(
        xlabel = "Distance (Points)",
        ylabel = "g(r)",
        title = "Radial Distribution Functions",
        legend = :topright,
        linewidth = 2
    )

    plot!(p, rdf_data, label = "$concentration%")

    display(p)
end

function plot_results(test_concentrations, reference_data, model)
    for (i, c) in enumerate(test_concentrations)
        xs = reference_data[i][:, 1]
        y_true = reference_data[i][:, 2]
        y_pred = model([Float32(c)])[:]  # Ensure c is wrapped in an array and output is a vector

        p = plot(
            xlabel = "Distance (Å)",
            ylabel = "g(r)",
            title = "Radial Distribution Function for $c%",
            legend = :topright,
            size = (800, 600)  # Increase plot size for better visibility
        )

        plot!(p, xs, y_pred,
            linewidth = 3,
            label = "NN Prediction",
            color = :blue
        )
        plot!(p, xs, y_true,
            linewidth = 2,
            label = "Reference",
            color = :red,
            linestyle = :dash
        )

        mse = round(Flux.mse(y_pred, y_true), digits = 5)
        annotate!(p, xs[end] / 2, maximum(y_true), text("MSE = $mse", :red, 10))

        # Add a subplot for the difference between prediction and reference
        diff = y_pred .- y_true
        p2 = plot(xs, diff,
            xlabel = "Distance (Å)",
            ylabel = "Difference",
            title = "Prediction - Reference",
            legend = false,
            color = :green,
            linewidth = 2,
            size = (800, 300)
        )
        hline!(p2, [0], color = :black, linestyle = :dash)

        # Combine the two plots
        final_plot = plot(p, p2, layout = (2, 1), size = (800, 900))

        display(final_plot)
        # savefig(final_plot, "rdf_comparison_$c%.png")  # Save each plot as an image
    end
end

function plot_loss_values(losses, log = true)
    epochs = 1:length(losses)

    p = plot(epochs, losses,
        xlabel = "Epoch",
        ylabel = "Loss",
        title = "Training Loss over Epochs",
        line = :solid,
        legend = false,
        linewidth = 2
    )

    if log
        yaxis!(p, :log)
    end

    display(p)
end

function plot_evaluation_results(evaluation_results)
    concentrations = [result.input for result in evaluation_results]
    mse_values = [result.mse for result in evaluation_results]

    p = plot(concentrations, mse_values,
        xlabel = "Concentration",
        ylabel = "Mean Squared Error (MSE)",
        title = "MSE vs Concentration",
        marker = :circle,
        line = :solid,
        legend = false
    )

    display(p)
end

function read_rdf_data(filename::AbstractString)
    data = readdlm(filename, Float32, comments = true)
    return data
end

function load_reference_data(rdf_paths::Vector{String})
    return [read_rdf_data(path) for path in rdf_paths]
end

function prepare_training_data(concentrations, rdf_paths)
    X = [[Float32(c)] for c in concentrations]
    Y = [read_rdf_data(path)[:, 2] for path in rdf_paths]
    return collect(zip(X, Y))
end

function create_model(output_neurons::Int, activation = identity)
    Chain(Dense(1, output_neurons, activation))
end

save_model(model::Flux.Chain, filename::AbstractString) = @save filename model

function load_model(filename::AbstractString)
    model = nothing
    @load filename model
    return model
end

save_losses(losses, filename::AbstractString) = writedlm(filename, losses)

function validate_data(concentrations, rdf_paths)
    length(concentrations) == length(rdf_paths) ||
        error("Mismatch in number of concentrations and RDF paths.")
    all(isfile, rdf_paths) || error("Some RDF files are missing.")
    all(c -> isa(c, Number), concentrations) || error("All concentrations must be numbers.")
end

function evaluate_model(model, test_data)
    predictions = [model(x) for (x, _) in test_data]
    true_values = [y for (_, y) in test_data]
    input_values = [first(x) for (x, _) in test_data]
    mse_values = [Flux.mse(y_pred, y_true)
                  for (y_pred, y_true) in zip(predictions, true_values)]
    return [(input = x, mse = mse) for (x, mse) in zip(input_values, mse_values)]
end

function train_model(model::Flux.Chain, data, params)
    println("Training for $(params.epochs) with learning rate: $(params.learning_rate)")
    opt_state = Flux.setup(Flux.Adam(params.learning_rate), model)
    losses = zeros(Float64, params.epochs)

    for epoch in 1:(params.epochs)
        mean_loss = 0.0
        for (x, y) in data
            loss, grads = Flux.withgradient(model) do m
                model_output = m(x)
                Flux.mse(model_output, y)
            end
            Flux.update!(opt_state, model, grads[1])
            mean_loss += loss
        end
        losses[epoch] = mean_loss / length(data)
    end

    return model, losses
end

end # module DensityToRDF
