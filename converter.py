import neo
import scipy.io
import numpy as np


def convert_plx_to_mat(plx_file, mat_file):
    """
    Converte um arquivo .plx (Plexon) para .mat (MATLAB), otimizando o tamanho do arquivo.

    :param plx_file: Caminho para o arquivo .plx de entrada.
    :param mat_file: Caminho para o arquivo .mat de saída.
    """
    # Carregar o arquivo .plx usando neo
    reader = neo.io.PlexonIO(filename=plx_file)
    block = reader.read_block()

    # Criar um dicionário para armazenar os dados extraídos
    data_dict = {}

    # Salvar os sinais analógicos sem conversão de tipo
    analog_signals = block.segments[0].analogsignals
    for i, signal in enumerate(analog_signals):
        num_channels = signal.shape[1] if signal.ndim > 1 else 1
        for ch in range(num_channels):
            channel_name = f'AD{ch + 2:02d}'  # Ajuste para nomeação correta
            data_dict[channel_name] = np.array(signal[:, ch] if num_channels > 1 else signal)
            data_dict[f'{channel_name}_ts'] = np.array([[float(signal.times[0])]])
            data_dict[f'{channel_name}_ind'] = np.array([[1]], dtype=np.int32)  # Índice como inteiro reduz tamanho
            data_dict[f'{channel_name}_ts_step'] = np.array([[float(signal.sampling_period)]])

    # Salvar os spikes (trens de spikes) armazenados de forma otimizada
    spike_trains = block.segments[0].spiketrains
    for i, spikes in enumerate(spike_trains):
        unit_name = f'sig{str(i + 1).zfill(3)}i'  # Nome igual ao do NeuroExplorer
        data_dict[unit_name] = np.array(spikes).reshape(-1, 1)  # Ajuste para formato (N, 1)

    # Salvar os dados no arquivo .mat
    scipy.io.savemat(mat_file, data_dict, do_compression=True)
    print(f'Conversão concluída: {mat_file}')


# Exemplo de uso
plx_file = "/caminho/para/o/arquivo.plx"
mat_file = "/caminho/para/o/arquivo.mat"

convert_plx_to_mat(plx_file, mat_file)
