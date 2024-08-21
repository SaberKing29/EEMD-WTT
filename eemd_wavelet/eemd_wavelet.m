data = readmatrix('./processed_voice.csv', 'Range', [20002, 1, 40002, 3]);
sample = data(:, 1);
time = data(:, 2);
amplitude = data(:, 3);

x=amplitude;
y1 = voice_processSignal(x, 0.2, 0.9);
c = 1.3;
y2 = voice_gaijin_processSignal(x, 0.2, 0.9,c);

figure;
subplot(3, 1, 1);
plot(time, x);
title('Original Sound Signal');
xlabel('Sample');
ylabel('Amplitude');

subplot(3, 1, 2);
plot(time, y1);
title('EEMD + Wavelet Threshold Denoising');
xlabel('Sample');
ylabel('Amplitude');

subplot(3, 1, 3);
plot(time, y2);
title('EEMD + Improved Wavelet Threshold Denoising');
xlabel('Sample');
ylabel('Amplitude');


function voice_yout = voice_processSignal(voice_yin, voice_threshold,voice_thresholdFactor)

    Nstd = 0.2 * std(voice_yin);
    NE = 30;
    imf = eemd(voice_yin, Nstd, NE);
    imf = imf(:, 2:11);
    n = 10;
    r = zeros(1, n);
    for i = 1:n
       
        r(1, i) = sum((voice_yin - mean(voice_yin)) .* (imf(:, i) - mean(imf(:, i))), 1) ./ sqrt(sum(((voice_yin - mean(voice_yin)).^2), 1) .* sum(((imf(:, i) - mean(imf(:, i))).^2), 1));

    end
 
    voice_selectedIMF = imf(:, r > voice_threshold);
 
    if isempty(voice_selectedIMF)
        [~, maxIndex] = max(r);
        voice_selectedIMF = imf(:, maxIndex);
    end

    wname = 'db4'; 
    level = 1; 
    denoisedIMF = zeros(size(voice_selectedIMF));
    for i = 1:size(voice_selectedIMF, 2)
        [thr, sorh] = ddencmp('den', 'wv', voice_selectedIMF(:, i)); 
        denoisedIMF(:, i) = wdencmp('gbl', voice_selectedIMF(:, i), wname, level, thr*voice_thresholdFactor, sorh,true); 
    end
    voice_yout = sum(denoisedIMF, 2);

end

function thresholded_coefficients = applyThreshold(coefficients, beta, c)
    thresholded_coefficients = sign(coefficients) .* (abs(coefficients) - beta .* (1 - exp(-log(c+1) ./ (abs(coefficients) - beta))));
    thresholded_coefficients(abs(coefficients) <= beta) = 0;
end

function voice_yout = voice_gaijin_processSignal(voice_yin, voice_threshold, voice_thresholdFactor,c)
    Nstd = 0.2 * std(voice_yin);
    NE = 30;
    imf = eemd(voice_yin, Nstd, NE);
    imf = imf(:, 2:11);
    n = size(imf, 2);
    r = zeros(1, n);
  for i = 1:n

    r(i) = sum((voice_yin - mean(voice_yin)) .* (imf(:, i) - mean(imf(:, i)))) / sqrt(sum((voice_yin - mean(voice_yin)).^2) * sum((imf(:, i) - mean(imf(:, i))).^2));
  end
    voice_selectedIMF = imf(:, r > voice_threshold);
    if isempty(voice_selectedIMF)
        [~, maxIndex] = max(r);
        voice_selectedIMF = imf(:, maxIndex);
    end
    wname = 'db4';
    level = 1;
    denoisedIMF = zeros(size(voice_selectedIMF));
    for i = 1:size(voice_selectedIMF, 2)
        [thr, ~] = ddencmp('den', 'wv', voice_selectedIMF(:, i));
        thresholded_coefficients = applyThreshold(voice_selectedIMF(:, i), thr * voice_thresholdFactor, c);
        denoisedIMF(:, i) = waverec(thresholded_coefficients, [length(voice_yin) level], wname);
    end
    
    voice_yout = sum(denoisedIMF, 2);
end

