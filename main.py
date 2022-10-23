from datetime import datetime
import librosa
import librosa.display
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
import re
from scipy.signal import butter, filtfilt
import warnings

import sys
sys.path.append('./python-som/')

def extract_data_from_filename(fpath: str) -> list:
    #Criação de um padrão através de ReGex
    #DÚVIDA: NÃO DARIA PARA DEFINIR NO INICO DO CÓDIGO? EH MT PARECIDO UM COM O  OUTRO
    #Padrão da ER: [TUDO][GDIGITO]_[G6DIGITOS]-[GCARACTER]-[G2DIGITOS][END OF STRING]
    #groups()[0]:[GDIGITO]    --> FOLD
    #groups()[1]:[G6DIGITOS]  --> CLID_ID
    #groups()[2]:[GCARACTER]  --> TAKE
    #groups()[3]:[G2DIGITOS]  --> TARGET
    #POR QUE NÃO FUNCIONA? pattern: str = (r"^.*(\d)-(\d{1,6})-(\[A-Z])-(\d{1,2}).*$")
    pattern: str = (r"^.*(\d)-(\d{1,6})-(\w)-(\d{1,2}).*$")

    #Matches: lista de objetos matches que representam nomes de arquivos que respeitam o padrão
    #Objeto match: a string mas dividida segundo algum padrão
    #re.fullmatch(): retorna um objeto match se toda a string respeitar o padrão. Se não, retorna None
      #Parâmetro pattern: ER definindo o padrão
      #Parâmetro fpath: string a ser testada    
    match: re.Match = re.fullmatch(pattern, fpath)

    #Constrói as primeiras quatro colunas (metadados) do áudio
    #matches[i].groups()[j]: acessa o grupo j do objeto matches[i]
    data =  [ fpath,              #NOME DO ARQUIVO  
              match.groups()[0], #FOLD
              match.groups()[3], #TARGET
              match.groups()[1], #SRC_FILE OU CLIP_ID
              match.groups()[2]  #TAKE
            ]

    return data

def extract_feature_means(audio_file_path: str, verbose: bool = True) -> pd.DataFrame:
  #Define alguns valores importantes para extração
  if verbose:
      print("File:", audio_file_path)
  #DÚVIDA: QUEM E PORQUE FOR DETERMINADO O 20, 2048 E 512
  #DÚVIDA: ESCREVER AQUI O QUE É CADA UM
  number_of_mfcc = 20 #pensar em diminuir ou aumentar (testar)
  n_fft = 2048  # FFT window size
  hop_length = 512  # number audio of frames between STFT columns

  #Extrai os metadados deste áudio com a função criada anteriormente
  if verbose:
      print("0.Extracting info from filename...")
  filename, fold, target, src_file, take = extract_data_from_filename(audio_file_path)

  #Carrega o áudio com librosa
  if verbose:
      print("1.Importing file with librosa...")
  try:
      y, sr = librosa.load(audio_file_path)
  except Exception as e:
      print(e)
      return None

  #librosa.effects.trim(): Corta o silêncio inicial e final de um sinal de áudio
    #Retorna signal: série temporal do sinal cortado
    #Retorna o intervalo não silencioso do sinal
  signal, _ = librosa.effects.trim(y)

  #Armazena em d_audio a magnitude da frequencia do sinal após aplicar o STFT
  #lr.stft(x): Short-time Fourier transform (STFT) que representa o sinal no domínico do tempo e frequencia
    #DÚVIDA: PORQUE O ABS RETORNA A MAGNITUDE E O ANGLE A FASE? NÃO ENTENDI
    #np.abs(D[..., f, t]): função magnitude da frequencia *******
    #np.angle(D[..., f, t]): função fase da frequencia
  if verbose:
      print("2.Fourier transform...")
  #Short-time Fourier transform (STFT)
  d_audio = np.abs(librosa.stft(signal, n_fft = n_fft, hop_length = hop_length))

  #Armazena em db_audio, após transformar a magnitude da frequencia, a unidade comparativa dB do sinal
  #lr.amplitude_to_db(): converte a amplitude do espectrogram para dB
    #np.abs(lr.stft(x)): amplitude
    #ref = np.max: DÚVIDA NÃO ENTENDI ISSO AQUIIIIIII
      #OBS:np.max: pega o maior valor de um eixo dado
  if verbose:
      print("3.Spectrogram...")
  db_audio = librosa.amplitude_to_db(d_audio, ref=np.max)

  #Armazena em s_db_audio, após extrair o espectograma em escala mel, a unidade comparativa dB do sinal
  #librosa.feature.melspectrogram(): computa o espectograma em escala mel do sinal
    #Parâmetro signal: serie temporal do audio
    #Parâmtro sr: taxa de amostragem
  if verbose:
      print("4.Mel spectograms...")
  s_audio = librosa.feature.melspectrogram(signal, sr=sr)
  s_db_audio = librosa.amplitude_to_db(s_audio, ref=np.max)

  #Captura o elemento harmômico e percusivo do sinal
  #librosa.effects.hpss(): decompõem uma serie temporal de audio componentes harmônicos e percursivos
    #Retorno y_harm: serie temporal do elemento harmonico
    #Retorno y_perc: serie temporal do elemento percursivo
  if verbose:
      print("6.Harmonics and perceptrual...")
  y_harm, y_perc = librosa.effects.hpss(signal)

  #Captura as features relacionadas a spectral centroid
  #librosa.feature.spectral_centroid(): computa o spectral centroid
    #Parâmetro signal: serie temporal do audio
    #Parâmtro sr: taxa de amostragem
    #Retorna: centroidnp.ndarray [shape=(…, 1, t)]
  #ibrosa.feature.delta(): computa delta features (estimativa local de derivadas dos dados de entrada)
    #Parâmetro order: a ordem da derivada (primeira, segunda, etc.)
  #DÚVIDA: POR QUE AS DERIVADAS SÃO IMPORTANTES MESMO?
  if verbose:
      print("7.Spectral centroid...")
  # Calculate the Spectral Centroids
  spectral_centroids = librosa.feature.spectral_centroid(signal, sr=sr)[0]
  spectral_centroids_delta = librosa.feature.delta(spectral_centroids)
  spectral_centroids_accelerate = librosa.feature.delta(spectral_centroids, order=2)

  #Captura o cromagrama
  #librosa.feature.chroma_stft(): computa o cromagrama da onda
    #Parâmetro signal: audio time series
    #Parâmetro sr: sample rate
    #Parâmetro hop_length: FFT window size
    #Retorna: um np.ndarray() com a "energia normalizada" para cada croma em cada frama
    #DÚVIDA: não entendi muito bem o que seria energia normalizada
  if verbose:
      print("8.Chroma features...")
  chromagram = librosa.feature.chroma_stft(signal,
                                            sr=sr,
                                            hop_length=hop_length)
  #DÚVIDA: QUE FEATURE É ESSA?
  #librosa.beat.beat_track(): detecta o BPM (MAS O QUE É ISSO?)
    #Retorna: float relativo ao tempo global estimado (batidas por minuto)
    #Retorna: np.ndarr0ay relativo a posição estimada dos eventos de batida (em geral, unidade dado em frames)
  if verbose:
      print("9.Tempo BPM...")
  tempo_y, _ = librosa.beat.beat_track(signal, sr=sr)

  #Captura Spectral Rollof, Spectral Flux e Spectral Bandwidth
  if verbose:
      print("10.Spectral rolloff...")
  #-->Spectral RollOff Vector
  #librosa.feature.spectral_rolloff():
    #Retorna: np.ndarray relativo a frequencia roll-off para cada frame
  #???? ler
  #DÚVIDA: POR QUE PEGAMOS APENAS O FRAME 0?
  spectral_rolloff = librosa.feature.spectral_rolloff(signal, sr=sr)[0]
  #-->Spectral Flux
  #DÚVIDA: O QUE É
  """
  relaciondo ao inicio de um som
  """
  #librosa.onset.onset_strength(): compute o spectral flux "onset strength envelope"
    #Retorna: vetor contendo onset stregth "envelopado"
  onset_env = librosa.onset.onset_strength(y=signal, sr=sr)
  #-->Spectral Bandwidth
  #librosa.feature.spectral_bandwidth(): computa a p-ésima ordem do spectal bandwith
    #Returno: np.ndarray relativo a frequencia bandwidth para cada frame
  spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(signal, sr=sr)[0]
  spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(signal,
                                                            sr=sr,
                                                            p=3)[0]
  spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(signal,
                                                            sr=sr,
                                                            p=4)[0]

    #columns = ['filename', 'fold', 'target', 'src_file', 'take']
  #Formatação das colunas e linhas do DataFrame
  audio_features = {
      "filename":
      filename,
      "fold":
      fold,
      "target":
      target,
      "src_file":
      src_file,
      "take":
      take,
      #librosa.feature.zero_crossing_rate(): computa a taxa de "zero-crossing" de uma serie temporal de audio
        #Returna: np.ndarray relativo a fração de "zero-crossing" em cada frame
      "zero_crossing_rate":
      np.mean(librosa.feature.zero_crossing_rate(signal)[0]),
      "zero_crossings":
      #librosa.zero_crossings(): encontra "zero-crossing" do sinal 
        #Parâmetro pad = False: não considera signal[0] como sendo válido
        #Retorna: np.ndarray indicando "zero-crossing" do sinal ao longo do eixo
      np.sum(librosa.zero_crossings(signal, pad=False)),
      "spectrogram":
      np.mean(db_audio[0]),
      "mel_spectrogram":
      np.mean(s_db_audio[0]),
      "harmonics":
      np.mean(y_harm),
      "perceptual_shock_wave":
      np.mean(y_perc),
      "spectral_centroids":
      np.mean(spectral_centroids),
      "spectral_centroids_delta":
      np.mean(spectral_centroids_delta),
      "spectral_centroids_accelerate":
      np.mean(spectral_centroids_accelerate),
      "chroma1":
      np.mean(chromagram[0]),
      "chroma2":
      np.mean(chromagram[1]),
      "chroma3":
      np.mean(chromagram[2]),
      "chroma4":
      np.mean(chromagram[3]),
      "chroma5":
      np.mean(chromagram[4]),
      "chroma6":
      np.mean(chromagram[5]),
      "chroma7":
      np.mean(chromagram[6]),
      "chroma8":
      np.mean(chromagram[7]),
      "chroma9":
      np.mean(chromagram[8]),
      "chroma10":
      np.mean(chromagram[9]),
      "chroma11":
      np.mean(chromagram[10]),
      "chroma12":
      np.mean(chromagram[11]),
      "tempo_bpm":
      tempo_y,
      "spectral_rolloff":
      np.mean(spectral_rolloff),
      "spectral_flux":
      np.mean(onset_env),
      "spectral_bandwidth_2":
      np.mean(spectral_bandwidth_2),
      "spectral_bandwidth_3":
      np.mean(spectral_bandwidth_3),
      "spectral_bandwidth_4":
      np.mean(spectral_bandwidth_4),
  }

  #Extraí o DataFrame contendo o MFCC do áudio e suas derivadas primeira e segunda
  mfcc_df = extract_mfcc_feature_means(audio_file_path,
                                        signal,
                                        sample_rate=sr,
                                        number_of_mfcc=number_of_mfcc)

  #Converte o dicionário em DataFrame
  df = pd.DataFrame.from_records(data=[audio_features])

  #pd.merge(): fundi objetos DataFrames/Series com memso estilo de database
    #Parâmetro df e mfcc_df: os dois DataFrames que queremos juntas
    #Parâmetro on: a coluna que será usada como orientação para juntar
  df = pd.merge(df, mfcc_df, on='filename')

  if verbose:
      print("DONE:", audio_file_path)

  return df

  # librosa.feature.mfcc(signal)[0, 0]

def extract_mfcc_feature_means(audio_file_name: str, 
                               signal: np.ndarray,
                               sample_rate: int,
                               number_of_mfcc: int) -> pd.DataFrame:
    
    #Computa MFCC de uma serie temporal de áudio, bem como a feature delta de ordem 1 e 2 relacionado a ela
    #librosa.feature.mfcc(): computa MFCC de uma série de áudio
      #Retorna: np.ndarray relativo à sequencia MFCC
      #DÚVIDA: O QUE CADA ELEMENTO DA SEQUÊNCIA QUER DIZER?
    mfcc_alt = librosa.feature.mfcc(y=signal,
                                    sr=sample_rate,
                                    n_mfcc=number_of_mfcc)
    delta = librosa.feature.delta(mfcc_alt)
    accelerate = librosa.feature.delta(mfcc_alt, order=2)

    #Cria o dicionário inicial para formação do DataFrame
    mfcc_features = {
        "filename": audio_file_name,
    }

    #DÚVIDA: DA ONDE SAIU A IDEIA DE CONSIDERAR A DERIVADA PRIMERIA E SEGUNDA NA ANÁLISE
    #Adiciona cada esquema e elemento do MFCC e seus deltas ao dicionário mfcc_features
    for i in range(0, number_of_mfcc):
        # dict.update({'key3': 'geeks'})

        #Adiciona mfcc[i] coefficient ao dicionário
        #string.join(): função embutida nas variáveis tipo string que 
          #junta elementos de uma lista separando-os pela string
        key_name = "".join(['mfcc', str(i)])
        mfcc_value = np.mean(mfcc_alt[i])
        #dictionary.update({key_name: value}): adiciona mais um par ao dicionário
        mfcc_features.update({key_name: mfcc_value})

        #Adiciona mfcc[i] delta coefficient ao dicionário
        key_name = "".join(['mfcc_delta_', str(i)])
        mfcc_value = np.mean(delta[i])
        mfcc_features.update({key_name: mfcc_value})
        
        #Adiciona mfcc[i] acclerate coefficient ao dicionário
        key_name = "".join(['mfcc_accelerate_', str(i)])
        mfcc_value = np.mean(accelerate[i])
        mfcc_features.update({key_name: mfcc_value})

    #pandas.DataFrame.from_records(data): converte estrutura ou registro ndarray em DataFrame
    df = pd.DataFrame.from_records(data=[mfcc_features])

    return df
    
EXTENSION: str = '.wav'
COMMON_PATH: str = '/home/tarcidio/analysis-ESC50/'

NUM_ITERATION = '1'
DATA_NAME = 'audio' + NUM_ITERATION
DATA_DIR: str = os.path.join(COMMON_PATH, DATA_NAME)
CSV_PATH = os.path.join(COMMON_PATH, 'csv_feature')
CSV_NAME = 'feauture' + NUM_ITERATION + '.csv'

fpaths: list = sorted([
    os.path.join(root, file) 
      for root, _, files in os.walk(DATA_DIR)
        for file in files if file.endswith(EXTENSION)
])

pattern: str = (r"^.*(\d)-(\d{1,6})-([A-Z])-(\d{1,2}).*$")

#Matches: lista de objetos matches que representam nomes de arquivos que respeitam o padrão
matches: list = [re.fullmatch(pattern, fpath) for fpath in fpaths]

if len(matches) != len(fpaths):
    raise ValueError("check fpath patterns")

#Constrói as primeiras quatro colunas (metadados) do áudio
rows = np.array([ [ fpaths[i],              #NOME DO ARQUIVO  
                    matches[i].groups()[0], #FOLD
                    matches[i].groups()[3], #TARGET
                    matches[i].groups()[1], #SRC_FILE OU CLIP_ID
                    matches[i].groups()[2]  #TAKE
] for i in range(len(matches))])

#Define o nome dos cinco primeiros atributos que do DataFrame
columns = ['filename', 'fold', 'target', 'src_file', 'take']
print('Number of identified audio files:', len(fpaths))

#Cria um DataFrame cujo atributos são 'datetime', 'fpath', 'ala' e 'grupo'
audios = pd.DataFrame(data=rows, columns=columns)

print('Processing', len(audios), 'audios...')

#Vetor cujos elementos serão, cada um, a linha do DataFrame final contendo os metadados e as features
result = []

#Variável que determina em quantos processos serão feitas as extrações
#DÚVIDA: POR QUE FOI ESCOLHIDO A QUANTIDADE 32?
n_processes = 4

#Cria lista com os nomes dos arquivos de áudios
iterable = list(audios['filename'])
#print(iterable)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    with mp.Pool(processes=n_processes) as pool:
        """result recebe, de vários processamentos, elementos que contem os metadados
e as features de cada arquivo de áudio. A variável result será usada como
argumento para concatenar todas as informações e gerar o DataFrame"""
        result = pool.map(extract_feature_means, #Função a ser aplicada a cada nome de arquivo
                          iterable=iterable,     #Lista com o nome dos arquivos
                          chunksize=len(iterable) // n_processes
                                                 #Distribuição para cada processo
                          ) 
        pool.close()
        pool.join()


print("Done processing audios. Concatenating and writing to output file...")
for idx, i in enumerate(result):
    if i is None:
        del result[idx]
#pandas.concat(): concatena objetos pandas
audios_means = pd.concat(result)
audios_means.set_index(['filename'], drop = True, inplace = True, verify_integrity=False)
output_path = os.path.join(CSV_PATH, CSV_NAME)
audios_means.to_csv(output_path, index=False)
print("End")