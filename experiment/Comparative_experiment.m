clear;
clc;
close all;

% Input your simulation signal
% The simulation signal includes the original signal + noise signal

%% EEMD combined with improved wavelet threshold denoising plot
original_f = original_f(:);
noise_f = noise_f(:);
f = f(:);
c = 1.7;
y_eemd_wavelet = voice_gaijin_processSignal(f, 0.38, 0.001, c);

figure;
subplot(2, 1, 1);
plot(n, f);
title('Simulated Noise Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

subplot(2, 1, 2);
plot(n, y_eemd_wavelet);
title('EEMD with Improved Wavelet Threshold Denoising');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

%% EEMD denoising plot
y_eemd = voice_EEMD(f, 0.20);

figure;
subplot(2, 1, 1);
plot(n, f);
title('Simulated Noise Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

subplot(2, 1, 2);
plot(n, y_eemd);
title('EEMD Denoising');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

%% Wavelet threshold denoising plot
y_wave = wavelet_threshold_denoising(f, 0.01, 'db4', 6);

figure;
subplot(2, 1, 1);
plot(n, f);
title('Simulated Noise Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

subplot(2, 1, 2);
plot(n, y_wave);
title('Wavelet Threshold Denoising');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

%% Improved wavelet threshold denoising plot
y_wave_gaijin = improved_wavelet_threshold_denoising(f, 0.01, 1.7, 'db4', 6);

figure;
subplot(2, 1, 1);
plot(n, f);
title('Simulated Noise Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

subplot(2, 1, 2);
plot(n, y_wave_gaijin);
title('Improved Wavelet Threshold Denoising');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

%% CEEMDAN denoising plot
[y_wave_ceemdan, retained_imfs, y_its] = ceemdan(f, 0.1, 100, 1000);

figure;
subplot(2, 1, 1);
plot(n, f);
title('Simulated Noise Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

subplot(2, 1, 2);
plot(n, y_wave_ceemdan);
title('Denoised Signal Using CEEMDAN');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-1.2, 1.2]);  % Set y-axis range

%% Plot waveform overlay
figure;
plot(n, original_f, 'k', 'LineWidth', 2);  % Original signal in black
hold on;
plot(n, f, 'Color', [0, 0.5, 1], 'LineWidth', 1.2);  % Noisy signal in light blue
plot(n, y_eemd_wavelet, 'Color', [0.4, 0.7, 0.2], 'LineWidth', 1.2);  % EEMD + wavelet denoised signal in light green
plot(n, y_eemd, 'Color', [1, 0.5, 0], 'LineWidth', 0.7);  % EEMD denoised signal in orange
plot(n, y_wave, 'Color', [0.2, 0.6, 0.6], 'LineWidth', 0.7);  % Wavelet denoised signal in dark cyan
plot(n, y_wave_gaijin, 'Color', [0.9, 0.7, 0], 'LineWidth', 0.7);  % Improved wavelet denoised signal in light yellow
legend('Original Signal', 'Noisy Signal', 'EEMD + Wavelet Denoised Signal', 'EEMD Denoised Signal', 'Wavelet Denoised Signal', 'Improved Wavelet Denoised Signal');
title('Comparison of Denoised Signals');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

%% Plot waveform scatter
figure;
subplot(4, 1, 1);
plot(n, original_f, 'k', 'LineWidth', 1.5);  % Original signal in black, line width 1.5
hold on;
plot(n, y_eemd_wavelet, 'Color', [1, 0.5, 0.5], 'LineWidth', 1.2);  % EEMD + wavelet denoised signal in red
legend('Original Signal', 'EEMD + Wavelet Denoised Signal');
title('Comparison of Denoised Signals');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-0.5, 0.5]);  % Set y-axis range
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);

subplot(4, 1, 2);
plot(n, original_f, 'k', 'LineWidth', 1.5);  % Original signal in black, line width 1.5
hold on;
plot(n, y_eemd, 'Color', [1, 0.5, 0.5], 'LineWidth', 1.2);  % EEMD denoised signal in red
legend('Original Signal', 'EEMD Denoised Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-0.5, 0.5]);  % Set y-axis range
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);

subplot(4, 1, 3);
plot(n, original_f, 'k', 'LineWidth', 1.5);  % Original signal in black, line width 1.5
hold on;
plot(n, y_wave, 'Color', [1, 0.5, 0.5], 'LineWidth', 1.2);  % Wavelet denoised signal in red
legend('Original Signal', 'Wavelet Denoised Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-0.5, 0.5]);  % Set y-axis range
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);

subplot(4, 1, 4);
plot(n, original_f, 'k', 'LineWidth', 1.5);  % Original signal in black, line width 1.5
hold on;
plot(n, y_wave_gaijin, 'Color', [1, 0.5, 0.5], 'LineWidth', 1.2);  % Improved wavelet denoised signal in red
legend('Original Signal', 'Improved Wavelet Denoised Signal');
xlabel('Time (seconds)');
ylabel('Amplitude');
ylim([-0.5, 0.5]);  % Set y-axis range
grid on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);

%% Output SNR and RMSE values
snr_eemd_wavelet = calculate_snr(original_f, y_eemd_wavelet);
rmse_eemd_wavelet = calculate_rmse(original_f, y_eemd_wavelet);

snr_eemd = calculate_snr(original_f, y_eemd);
rmse_eemd = calculate_rmse(original_f, y_eemd);

snr_wave = calculate_snr(original_f, y_wave);
rmse_wave = calculate_rmse(original_f, y_wave);

snr_wave_gaijin = calculate_snr(original_f, y_wave_gaijin);
rmse_wave_gaijin = calculate_rmse(original_f, y_wave_gaijin);

y_wave_ceemdan_1 = y_wave_ceemdan.';
snr_wave_ceemdan = calculate_snr(original_f, y_wave_ceemdan_1);
rmse_wave_ceemdan = calculate_rmse(original_f, y_wave_ceemdan_1);

fprintf('EEMD + Improved Wavelet Denoising: SNR = %.2f dB, RMSE = %.5f\n', snr_eemd_wavelet, rmse_eemd_wavelet);
fprintf('EEMD Denoising: SNR = %.2f dB, RMSE = %.5f\n', snr_eemd, rmse_eemd);
fprintf('Wavelet Threshold Denoising: SNR = %.2f dB, RMSE = %.5f\n', snr_wave, rmse_wave);
fprintf('Improved Wavelet Threshold Denoising: SNR = %.2f dB, RMSE = %.5f\n', snr_wave_gaijin, rmse_wave_gaijin);
fprintf('CEEMDAN Denoising: SNR = %.2f dB, RMSE = %.5f\n', snr_wave_ceemdan, rmse_wave_ceemdan);

%% EEMD + Improved Wavelet Threshold Denoising

function thresholded_coefficients = applyThreshold(coefficients, beta, c)
    thresholded_coefficients = sign(coefficients) .* (abs(coefficients) - beta .* (1 - exp(-log(c+1) ./ (abs(coefficients) - beta))));
    thresholded_coefficients(abs(coefficients) <= beta) = 0;
end

function voice_yout = voice_gaijin_processSignal(voice_yin, voice_threshold, voice_thresholdFactor, c)
    tic;
    Nstd = 0.2 * std(voice_yin);
    NE = 30;
    imf = eemd(voice_yin, Nstd, NE);
    [~, n] = size(imf);

    if n >= 2
        imf = imf(:, 2:min(n, 11));
    else
        error('Not enough IMF components for denoising');
    end
    
    % Plot IMF components
    num_imf = size(imf, 2);
    figure;
    for i = 1:num_imf
        subplot(num_imf, 1, i);
        plot(imf(:, i));
        title(['IMF', num2str(i)]);
        xlabel('Sample Points');
        ylabel('Amplitude');
    end

    r = zeros(1, size(imf, 2));
    
    for i = 1:size(imf, 2)
        r(i) = sum((voice_yin - mean(voice_yin)) .* (imf(:, i) - mean(imf(:, i)))) / sqrt(sum((voice_yin - mean(voice_yin)).^2) * sum((imf(:, i) - mean(imf(:, i))).^2));
        % Display correlation coefficients
        disp(['IMF ', num2str(i), ' Correlation coefficient: ', num2str(r(1, i))]);
    end
    
    voice_selectedIMF = imf(:, r > voice_threshold);
    
    if isempty(voice_selectedIMF)
        [~, maxIndex] = max(r);
        voice_selectedIMF = imf(:, maxIndex);
    end
    
    wname = 'db4';
    level = 6;
    denoisedIMF = zeros(size(voice_selectedIMF));
    
    for i = 1:size(voice_selectedIMF, 2)
        [thr, ~] = ddencmp('den', 'wv', voice_selectedIMF(:, i));
        thresholded_coefficients = applyThreshold(voice_selectedIMF(:, i), thr * voice_thresholdFactor, c);
        denoisedIMF(:, i) = waverec(thresholded_coefficients, [length(voice_yin) level], wname);
    end
    
    voice_yout = sum(denoisedIMF, 2);
    t = toc;
    fprintf('EEMD + Improved Wavelet Denoising completed in %.4f seconds.\n', t);
end

%% EEMD signal denoising method
function voice_yout_EEMD = voice_EEMD(voice_yin, voice_threshold)
    tic;
    Nstd = 0.2 * std(voice_yin);
    NE = 30;
    imf = eemd(voice_yin, Nstd, NE);
    [~, n] = size(imf);

    if n >= 2
        imf = imf(:, 2:min(n, 11));
    else
        error('Not enough IMF components for denoising');
    end
    
    r = zeros(1, size(imf, 2));
    
    for i = 1:size(imf, 2)
        r(i) = sum((voice_yin - mean(voice_yin)) .* (imf(:, i) - mean(imf(:, i)))) / sqrt(sum((voice_yin - mean(voice_yin)).^2) * sum((imf(:, i) - mean(imf(:, i))).^2));
        % Display correlation coefficients
        disp(['IMF ', num2str(i), ' Correlation coefficient: ', num2str(r(1, i))]);
    end
    
    voice_selectedIMF = imf(:, r > voice_threshold);
    
    if isempty(voice_selectedIMF)
        [~, maxIndex] = max(r);
        voice_selectedIMF = imf(:, maxIndex);
    end

    voice_yout_EEMD = sum(voice_selectedIMF,2);
    t = toc;
    fprintf('EEMD Denoising completed in %.4f seconds.\n', t);
end

%% Wavelet threshold denoising
function denoised_signal = wavelet_threshold_denoising(signal, threshold_factor, wavelet_name, num_levels)
    tic;
    [C, L] = wavedec(signal, num_levels, wavelet_name);

    threshold = threshold_factor * median(abs(C)) / 0.6745;

    C_den = soft_threshold(C, threshold);

    denoised_signal = waverec(C_den, L, wavelet_name);
    t = toc;
    fprintf('Wavelet Threshold Denoising completed in %.4f seconds.\n', t);
end

function thresholded_coeffs = soft_threshold(coeffs, threshold)
    thresholded_coeffs = sign(coeffs) .* max(abs(coeffs) - threshold, 0);
end

%% Improved wavelet threshold denoising
function denoised_signal_gaijin = improved_wavelet_threshold_denoising(signal, threshold_factor, c, wavelet_name, num_levels)
    tic;
    [C, L] = wavedec(signal, num_levels, wavelet_name);

    threshold = threshold_factor * sqrt(2 * log(numel(signal)));

    C_den = applyThreshold(C, threshold, c);

    denoised_signal_gaijin = waverec(C_den, L, wavelet_name);
    t = toc;
    fprintf('Improved Wavelet Threshold Denoising completed in %.4f seconds.\n', t);
end

%% CEEMDAN denoising method
function [denoised_signal, retained_imfs, its] = ceemdan(x, Nstd, NR, MaxIter)
    tic;  % Start timing the function execution
    % Initialize the signal and scaling
    x = x(:)';  % Ensure x is a row vector
    desvio_x = std(x);
    x = x / desvio_x;  % Normalize the input signal

    % Initialization
    modes = zeros(size(x));
    aux = zeros(size(x));
    acum = zeros(size(x));
    iter = zeros(NR, round(log2(length(x))+5));
    white_noise = cell(1, NR);
    modes_white_noise = cell(1, NR);

    % Create white noise realizations and calculate their EMD modes
    for i = 1:NR
        white_noise{i} = randn(size(x));  % Creates the noise realizations
        modes_white_noise{i} = emd(white_noise{i});  % Calculates the modes of white Gaussian noise
    end

    % Calculate the first mode
    for i = 1:NR
        temp = x + Nstd * white_noise{i};
        [temp, o, it] = emd(temp, 'MaxModes', 1, 'MaxIterations', MaxIter);
        temp = temp(1,:);
        aux = aux + temp / NR;
        iter(i,1) = it;
    end
    modes = aux;  % Saves the first mode
    k = 1;
    aux = zeros(size(x));
    acum = sum(modes, 1);

    % Calculate the rest of the modes
    while nnz(diff(sign(diff(x-acum)))) > 2
        for i = 1:NR
            tamanio = size(modes_white_noise{i});
            if tamanio(1) >= k + 1
                noise = modes_white_noise{i}(k,:);
                noise = noise / std(noise);
                noise = Nstd * noise;
                try
                    [temp, o, it] = emd(x - acum + std(noise) * noise, 'MaxModes', 1, 'MaxIterations', MaxIter);
                    temp = temp(1,:);
                catch
                    it = 0;
                    temp = x - acum;
                end
            else
                [temp, o, it] = emd(x - acum, 'MaxModes', 1, 'MaxIterations', MaxIter);
                temp = temp(1,:);
            end
            aux = aux + temp / NR;
            iter(i, k+1) = it;
        end
        modes = [modes; aux];
        aux = zeros(size(x));
        acum = sum(modes, 1);
        k = k + 1;
    end
    modes = [modes; (x - acum)];
    modes = modes * desvio_x;  % Rescale modes to the original amplitude
    [a, b] = size(modes);
    iter = iter(:, 1:a);
    
    % Calculate correlation coefficients and reconstruct the signal
    denoised_signal = zeros(size(x));
    retained_imfs = [];
    correlation_threshold = 0.2;
    x_normalized = (x - mean(x)) / std(x);
    for i = 1:size(modes, 1)
        mode_normalized = (modes(i, :) - mean(modes(i, :))) / std(modes(i, :));
        corr_coeff = abs(corr(x_normalized', mode_normalized'));
        if corr_coeff > correlation_threshold
            denoised_signal = denoised_signal + modes(i, :);
            retained_imfs = [retained_imfs, i];
        end
    end

    % Stop timing and output the duration
    elapsed_time = toc;
    fprintf('CEEMDAN Denoising completed in: %.4f seconds.\n', elapsed_time);

    its = iter;  % Return the iteration counts
end

%% Calculate SNR and RMSE values
function snr_value = calculate_snr(original, denoised)
    % Calculate signal-to-noise ratio (SNR)
    signal_power = sum(original.^2);
    noise_power = sum((original - denoised).^2);
    snr_value = 10 * log10(signal_power / noise_power);
end

function rmse_value = calculate_rmse(original, denoised)
    % Calculate root mean square error (RMSE)
    error = original - denoised;
    rmse_value = sqrt(mean(error.^2));
end
