module AutocorrelationUtils
using Statistics, FFTW
export autocorrelation_direct, autocorrelation_fft


function autocorrelation_direct(x::Vector{T}) where T
    n = length(x)
    x̄ = mean(x)
    centered_data = x .- x̄

    acf = Vector{T}(undef, n)

    for k in 0:n-1
        acf[k+1] = sum(centered_data[i] * centered_data[i+k+1] for i in 1:n-k) / (n-k)
    end

    return acf
end


function autocorrelation_fft(x::Vector{T}) where T
    n = length(x)
    x̄ = mean(x)
    centered_data = x .- x̄

    # Compute the FFT
    fft_result = fft(centered_data)

    # Calculate the squared magnitude of the FFT
    squared_magnitude = abs2.(fft_result)

    # Perform the inverse FFT to get the autocorrelation function
    acf = real(ifft(squared_magnitude)) / n

    return acf
end


end