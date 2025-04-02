## LABORATORIO #3 PDS
### Señales electromiograficas EMG

Este informe tiene como objetivo aplicar el filtrado de señales continuas para procesar una señal electromigráfica
y detectar la fatiga muscular a través del análisis espectral de la misma.

Para entender el codigo necesitamos tener en cuenta unos conceptos importantes. 
### Contracción muscular
La contracción muscular se inicia cuando un potencial de acción, generado en la motoneurona alfa, viaja por el axón hasta la placa motora, liberando acetilcolina en la hendidura sináptica. La acetilcolina se une a receptores nicotínicos en la membrana de la fibra muscular, despolarizándola y propagando un nuevo potencial de acción a lo largo del sarcomera y los túbulos T. Esto activa los receptores de rianodina en el retículo sarcoplásmico, liberando iones de calcio (Ca⁺) al sarcoplasma. El Ca⁺ se une a la troponina en los filamentos de actina, desplazando la tropomiosina y exponiendo sitios de unión para las cabezas de miosina. Mediante el ciclo de los puentes cruzados (hidrólisis de ATP), las cabezas de miosina "caminan" sobre la actina, acortando el sarcómero y generando fuerza contráctil. La *señal EMG* capta la suma de estos potenciales eléctricos extracelulares desde múltiples unidades motoras, cuya amplitud y frecuencia dependen de la tasa de disparo de las motoneuronas.  

### Que es la electromiografia (EMG)
La electromiografía (EMG) es una técnica utilizada para evaluar y registrar la actividad eléctrica producida por los músculos esqueléticos.La EMG implica la detección de señales eléctricas generadas por las fibras musculares durante la contracción. Estas señales se transmiten a través del sistema nervioso y pueden ser medidas utilizando electrodos de superficie colocados sobre la piel o electrodos intramusculares insertados en el músculo. Esta información es importante  para diagnosticar trastornos neuromusculares, evaluar la fatiga muscular y guiar estrategias de rehabilitación.

### Toma de datos para una señal EMG
 En la adquisición de señales de electromiografía (EMG)requiere un sistema de adquisición con electrodos superficiales o intramusculares en nuestro caso se usaron electrodos de superficie, segun los buscado en articulos y paginas que hablan sobre recolección de datos dice que deberia ser  configurando a una frecuencia de muestreo ≥1000 Hz para captar la actividad muscular que es tipica  entre los (20-500 Hz). Ademas de esto nos afirman que es importante  aplicar filtros notch (50/60 Hz) para eliminar ruido de línea y filtros pasa-bandas para aislar la señal biológica. La señal original se rectifica y normaliza para su análisis en dominios de *tiempo* (amplitud, RMS) y *frecuencia* (FFT, espectro de potencia).  

Para implementación de la toma de datos EMG en Python o MATLAB, se recomienda usar librerías como *SciPy*.

### Que es un sistema de adquisición de datos DAQ

El *DAQ NI USB-6001/6002/6003* de National Instruments es un dispositivo de adquisición de datos (DAQ) compacto y de bajo costo, diseñado para aplicaciones de medición y control básicas, incluyendo la captura de señales biomédicas como *electromiografía (EMG). Estos dispositivos tienen entradas analógicas diferenciales (8/16 bits, ±10V, frecuencias de muestreo de hasta 10–48 kS/s dependiendo del modelo) con filtros anti-aliasing integrados para registrar señales EMG en el rango de 20–500 Hz. Tienen una conexión vía *USB* que permite una integración sencilla con software como LabVIEW, MATLAB o Python (usando librerías como nidaqmx), donde se pueden configurar parámetros de muestreo, aplicar filtros digitales como lo son, notch 50/60 Hz, pasa-bandas y extraer características en tiempo real. Además, incluyen entradas y salidas digitales . Aunque no es un equipo médico certificado, su relación costo-eficiencia lo hace popular en prototipos de investigación y proyectos de biomecánica, donde se requiere capturar la actividad muscular para procesamiento posterior.

### Implementación de filtros para una señal
Las señales EMG a menudo están contaminadas con ruido y artefactos, lo que nos hace  aplicar diversas técnicas de procesamiento de señales para un análisis preciso como los son los filtros.

El *filtro pasa altas* (típicamente con frecuencia de corte entre 10-20 Hz) elimina componentes de baja frecuencia como los artefactos de movimiento, deriva de la línea base y potenciales lentos asociados al desplazamiento de los electrodos, que pueden enmascarar la verdadera actividad muscular. Por otro lado, el *filtro pasa bajas* (con corte entre 400-500 Hz) atenúa el ruido de alta frecuencia generado por interferencias electromagnéticas, ruido térmico de los componentes electrónicos y artefactos de conmutación, preservando las componentes espectrales relevantes de la señal EMG que generalmente se concentran entre 20-400 Hz. Esta combinación de filtrado (que técnicamente forma un filtro pasa banda cuando se aplican secuencialmente) mejora significativamente la relación señal-ruido, permitiendo una mejor caracterización de parámetros como la amplitud RMS, la densidad espectral de potencia o los patrones de activación muscular.

![alt text](<../Captura de pantalla 2025-04-02 055130.png>)
 *Imagen 1. FIR filter software.*

### Aventanamiento
l *aventanamiento* (o windowing) es una técnica usada en PDS para reducir las discontinuidades en los bordes de un segmento finito de señal al aplicar la *Transformada de Fourier Discreta (DFT). Como las señales reales son de duración limitada, truncarlas abruptamente introduce *fugas espectrales (artefactos de alta frecuencia). Las *ventanas* suavizan los bordes de la señal, atenuando estos efectos.  

#### *Ventanas de Hamming y Hann(ing)*  
- *Ventana de Hamming:* Definida como 
   
      w(n) = 0.54 - 0.46 cos(2pi n)/N-1

ES ideal para aplicaciones donde se prioriza la atenuación de fugas espectrales (ej: análisis de frecuencias cercanas).  

- *Ventana de Hann(ing):* Dada por 

      w(n) = 0.5(1-cos(2pi n/N-1))
  
 Se utiliza 
  cuando se necesita mejor resolución en frecuencia (ej: identificación de tonos musicales). Ambas ventanas son *no paramétricas* y se usan en análisis espectral (FFT).

  ## Procedimiento de la practica
  - Inicialmente se colocaron electrodos al brazo de una persona y se le pidió que realizara unas contracciones hasta llegar a la fatiga muscular.

- Posteriormente se capturó la señal en tiempo real por medio de mathlab, por medio del sistema DAQ.

- Se realizo el codigo en Python que en principio tiene como objetivo registrar la señal EMG en tiempo real durante todo el proceso. Para ello se utilizó.

       import pandas as pd
       import numpy as np
       import matplotlib.pyplot as plt
       from scipy.signal import butter, filtfilt
       file_path = "emg_signal.csv" 
       df = pd.read_csv(file_path)
     
      tiempo = df.iloc[:, 0]  # Primera columna (Tiempo)
      voltaje = df.iloc[:, 1]  # Segunda columna (Voltaje)

Lee los datos de un archivo CSV que contiene la señal EMG.

Una vez cargados los archivos de CSV pasamos a  calcular los intervalos de tiempo entre muestras (tiempo.diff()), convertir los intervalos a frecuencia (1/intervalo), toma el promedio (fs_mean) para obtener la frecuencia de muestreo en Hz.  
        
    fs_estimates = 1 / tiempo.diff().dropna().unique()
    fs_mean = fs_estimates.mean()
- Siguiente a esto se genera un  gráfico de la señal EMG en el dominio del tiempo.  

        plt.figure(figsize=(10, 4))
        plt.plot(tiempo, voltaje, label="Señal EMG", color="b")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Voltaje (V)")
        plt.title("Señal EMG Original")
        plt.legend()
        plt.grid(True)
        plt.show()

- Diseñado para tener una respuesta plana en la banda de paso. Con la función **filtfilt** para el filtrado en ambas direcciones para evitar retrasos de fase. 


      def butterworth_filter(data, cutoff, fs, filter_type, order=4):
      nyquist = 0.5 * fs  # Frecuencia de Nyquist
      normal_cutoff = cutoff / nyquist
      b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
      return filtfilt(b, a, data)

      # Aplicar filtro pasa altas (10 Hz)
      filtered_high = butterworth_filter(voltaje, 20, fs_mean, 'high')

      # Aplicar filtro pasa bajas (60 Hz)
      filtered_signal = butterworth_filter(filtered_high, 20, fs_mean, 'low')

- En esta parte podemos observar la grafica ya filtrada

      plt.figure(figsize=(10, 4))
      plt.plot(tiempo, voltaje, label="Señal Original", alpha=0.5, color="gray")
      plt.plot(tiempo, filtered_signal, label="Señal Filtrada", color="blue")
      plt.xlabel("Tiempo (s)")
      plt.ylabel("Voltaje (V)")
      plt.title("Señal EMG antes y después del filtrado")
      plt.legend()
      plt.grid(True)
      plt.show()

- Lo siguiente se hace para  reducir las *fugas espectrales*, a demás la *ventana de Hamming* suaviza los bordes de cada segmento, para despues hacer una grafica de las ventanas.  
 
        window_size = 1  # 1 segundo por ventana
        samples_per_window = int(window_size * fs_mean)  # Convertir a muestras

        # Aplicar aventanamiento
        num_windows = len(filtered_signal) // samples_per_window
        windows = [filtered_signal[i * samples_per_window:(i + 1) * samples_per_window] for i in range(num_windows)]

        # Aplicar ventana de Hamming
        windowed_signals = [w * np.hamming(len(w)) for w in windows]
        
        # Graficar algunas ventanas
        plt.figure(figsize=(10, 4))
        for i in range(min(5, len(windowed_signals))):
        plt.plot(windowed_signals[i], label=f'Ventana {i+1}')
        plt.xlabel("Muestras")
        plt.ylabel("Voltaje (V)")
        plt.title("Señales EMG con ventana de Hamming")
        plt.legend()
        plt.grid(True)
        plt.show()

- Transformada de Fourier y analisis espectral 
        
        # Aplicar Transformada de Fourier (FFT) a cada ventana
        fft_results = [np.fft.fft(w) for w in windowed_signals]
        frequencies = np.fft.fftfreq(samples_per_window, d=1/fs_mean)

        # Tomar solo la mitad del espectro (parte positiva)
        half_spectrum = samples_per_window // 2
        frequencies = frequencies[:half_spectrum]
        fft_magnitudes = [np.abs(fft[:half_spectrum]) for fft in fft_results]

        # Graficar el espectro de frecuencia de algunas ventanas
        plt.figure(figsize=(10, 4))
        for i in range(min(5, len(fft_magnitudes))):
        plt.plot(frequencies, fft_magnitudes[i], label=f'Ventana {i+1}')
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Magnitud")
        plt.title("Espectro de Frecuencia de la Señal EMG")
        plt.legend()
        plt.grid(True)
         plt.show()

 Aqui buscamos el *espectro de frecuencias* de la señal EMG (rango típico útil: 20–500 Hz), los picos en frecuencias específicas pueden indicar actividad muscular o patologías.  





