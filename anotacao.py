
"""
LIBRIOSA
[0, 2048[
[512, 2560[
...
[i * hop_length, i * hop_length + n_fft[



COMO ENCONTRAR UM INDICE DE UM FRAME DENTRO DO WAVEFORM DE UM AUDIO
WAVEFORM: ANTES DA TRANSFORMADA
FRAME: DEPOIS

CROSSING RATE DE CADA FRAME

np.mean(librosa.feature.zero_crossing_rate(signal)[0])
np.array([librosa.feature.zero_crossing_rate(signal[i * hop_length : i * hop_length + n_fft]) for i in range(0, len(signal)/hop_length)])
[librosa.feature.zero_crossing_rate(signal[i * hop_length : i * hop_length + n_fft]) for i in range(0, len(signal)/hop_length)]

eh jeito de aumentar qnt de dados -> eficiencia (dataaugmenting)
"""