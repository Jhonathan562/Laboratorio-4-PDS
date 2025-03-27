% ======= CONFIGURACIÓN =======
device = 'Dev2';     % Nombre de tu DAQ (ajusta si es diferente)
channel = 'ai0';     % Canal de entrada (por ejemplo, ai0)
sampleRate = 1000;   % Frecuencia de muestreo (Hz)
duration = 60;       % Duración total (segundos)
outputFile = 'emg_signal.csv';  % Nombre del archivo a guardar

% ======= CREAR SESIÓN =======
d = daq("ni");  % Crear sesión para DAQ NI
addinput(d, device, channel, "Voltage");  % Agregar canal de entrada
d.Rate = sampleRate;

% ======= VARIABLES =======
timeVec = [];  % Vector de tiempo
signalVec = [];  % Vector de señal

% ======= CONFIGURAR GRÁFICA =======
figure('Name', 'Señal en Tiempo Real', 'NumberTitle', 'off');
h = plot(NaN, NaN);
xlabel('Tiempo (s)');
ylabel('Voltaje (V)');
title('Señal EMG en Tiempo Real');
xlim([0, duration]);
ylim([0, 4]);  % Ajusta el rango de voltaje si es necesario
grid on;

% ======= ADQUISICIÓN Y GUARDADO =======
disp('Iniciando adquisición...');
startTime = datetime('now');

while seconds(datetime('now') - startTime) < duration
    % Leer una muestra
    [data, timestamp] = read(d, "OutputFormat", "Matrix");
    
    % Guardar datos en vectores
    t = seconds(datetime('now') - startTime);
    timeVec = [timeVec; t];
    signalVec = [signalVec; data];
    
    % Actualizar gráfica
    set(h, 'XData', timeVec, 'YData', signalVec);
    drawnow;
end

% ======= GUARDAR LOS DATOS =======
disp('Adquisición finalizada. Guardando archivo...');
T = table(timeVec, signalVec, 'VariableNames', {'Tiempo (s)', 'Voltaje (V)'});
writetable(T, outputFile);
disp(['Datos guardados en: ', outputFile]);

% ======= CERRAR SESIÓN =======
clear d;

% LAB4 JHONATHAN Y JOSE