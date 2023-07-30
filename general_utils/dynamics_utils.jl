module DynamicsUtils
using Statistics
export run_estimate_diffusion_coefficient, run_estimate_diffusion_coefficient_time_rescaling, run_estimate_diffusion_coefficient_lamperti

function run_estimate_diffusion_coefficient(q_chunk, dt, kT; segment_length)
    num_segments = div(length(q_chunk), segment_length)

    # Divide the trajectory into segments
    segments = [q_chunk[(i-1)*segment_length+1:i*segment_length] for i in 1:num_segments]

    # Compute mean squared displacement (MSD) for each segment
    msd_values = [mean((seg[end] .- seg).^2) for seg in segments]
    time_points = [segment_length * dt * i for i in 1:num_segments]

    # Compute spatial coordinates for each segment
    spatial_coordinates = [mean(seg) for seg in segments]

    # Fit MSD to obtain local diffusion coefficients
    local_diffusion_coefficients = msd_values ./ (2 * kT * time_points)

    return spatial_coordinates, local_diffusion_coefficients
end

function run_estimate_diffusion_coefficient_lamperti(q_chunk, dt, kT; segment_length, x_of_y)
    # Transform the trajector back to the original coordinate system
    q_chunk = x_of_y.(q_chunk)

    num_segments = div(length(q_chunk), segment_length)

    # Divide the trajectory into segments
    segments = [q_chunk[(i-1)*segment_length+1:i*segment_length] for i in 1:num_segments]

    # Compute mean squared displacement (MSD) for each segment
    msd_values = [mean((seg[end] .- seg).^2) for seg in segments]
    time_points = [segment_length * dt * i for i in 1:num_segments]

    # Compute spatial coordinates for each segment
    spatial_coordinates = [mean(seg) for seg in segments]

    # Fit MSD to obtain local diffusion coefficients
    local_diffusion_coefficients = msd_values ./ (2 * kT * time_points)

    return spatial_coordinates, local_diffusion_coefficients
end

function run_estimate_diffusion_coefficient_time_rescaling(q_chunk, dt, kT, segment_length)
    num_segments = div(length(q_chunk), segment_length)

    # Divide the trajectory into segments
    segments = [q_chunk[(i-1)*segment_length+1:i*segment_length] for i in 1:num_segments]

    # Compute mean squared displacement (MSD) and time points for each segment
    msd_values = []
    time_points = []
    for (i, seg) in enumerate(segments)
        segment_msd = mean((seg[end] .- seg).^2)
        push!(msd_values, segment_msd)
        segment_time_points = cumsum(dt[(i-1)*segment_length+1:i*segment_length])
        push!(time_points, segment_time_points)
    end

    # Flatten the time points and compute spatial coordinates for each segment
    flat_time_points = vcat(time_points...)
    spatial_coordinates = [mean(seg) for seg in segments]

    # Fit MSD to obtain local diffusion coefficients
    local_diffusion_coefficients = msd_values ./ (2 * kT * flat_time_points)

    return spatial_coordinates, local_diffusion_coefficients
end

end # module DynamicsUtils