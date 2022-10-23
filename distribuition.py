#Importando biblioteca para controlar as pastas
import os
import shutil

#Definindo a pasta origem de saida dos audios
ORIGINAL_DIR : str = r'audios/origin'
#Definindo a pasta raiz comum das pastas que serão criadas
COMMON_DIR : str = r'audios'
#Definin nome padrão para as pastas que serão criadas
PATTERN_NAME : str = r'audio'

#Quantidade de arquivos por pasta
NUM_FILES : int = 660

#Contador para indicar quantos arquivos já foram movidos
cont : int = 0
#Contador para indicar o número de pastas que serão criadas
num_dir : int = 1
#Nome do novo diretório que será criado
new_name_dir : str = PATTERN_NAME + str(num_dir)
#Concatena nome da pasta e o caminhho
new_dir : str = os.path.join(COMMON_DIR, new_name_dir)
#Cria pasta
os.mkdir(new_dir)

#Para cada arquivo da pasta que estão os audios, faça:
for file in os.listdir(ORIGINAL_DIR):
    #Formata o caminho do arquivo
    path : str = os.path.join(ORIGINAL_DIR, file)
    if cont < NUM_FILES:
        #Atualiza o contador
        cont = cont + 1
        #Move o arquivo
        shutil.move(path, new_dir)
    else:
        #Reseta o contador
        cont = 0
        #Atualiza o nome do novo diretório
        num_dir = num_dir + 1
        new_name_dir = PATTERN_NAME + str(num_dir)
        new_dir = os.path.join(COMMON_DIR, new_name_dir)
        #Cria a pasta
        os.mkdir(new_dir)
        #Move o arquivo
        shutil.move(path, new_dir)