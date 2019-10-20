from tensorflow_functions import matrix_dot, cosine_knn
from embedding import load_embedding
from utils import batch
import urllib.request
import argparse
import datetime
import tarfile
import os

global device
batch_size = 1024


CPUcolor = '\033[94m'
GPUcolor = '\033[92m'
STORAGEcolor = '\033[31m'
RESETcolor = '\033[0m'


def benchmark():
    global device

    get_files()
    print("Running Benchmark..")
    time = datetime.datetime.now()

    emb = load_embedding('RWSGwn.emb', length_normalize=False, delete_duplicates=True)
    time = print_time('Loading embedding from Disk to RAM step', time)

    emb.length_normalize()
    time = print_time('Embedding length normalization step (' + CPUcolor + 'CPU' + RESETcolor + ')', time)

    vocab_to_search = emb.words
    for i in range(100):
        for word in vocab_to_search:
            v = emb.word_to_vector(word)
    time = print_time('Searching for vocabulary step (' + CPUcolor + 'CPU' + RESETcolor + ')', time)

    m = emb.vectors
    M = emb.vectors

    for i_batch, mb in enumerate(batch(m, batch_size)):
        _ = matrix_dot(mb, M)

    time = print_time('Matrix dot product step step ' + (
        '(' + CPUcolor + 'CPU' + RESETcolor + ')' if device == 'CPU' else '(' + GPUcolor + 'GPU' + RESETcolor + ')'),
                      time)

    for i_batch, mb in enumerate(batch(m, batch_size)):
        _ = cosine_knn(mb, M, 10)

    time = print_time('Searching for nearest neighbors step ' + (
        '(' + CPUcolor + 'CPU' + RESETcolor + ')' if device == 'CPU' else '(' + GPUcolor + 'GPU' + RESETcolor + ')'),
                      time)

    emb.export('temp.emb')
    time = print_time('Exporting embedding from RAM to Disk step', time)

    os.remove("temp.emb")
    print()
    print("Benchmark is over.")


def get_files():
    print("Downloading required files...")
    if not os.path.isfile('./RWSGwn.emb'):
        urllib.request.urlretrieve('http://ixa2.si.ehu.es/ukb/embeddings/RWSGwn.emb.tar.gz', 'RWSGwn.emb.tar.gz')
        tar = tarfile.open('RWSGwn.emb.tar.gz', "r:gz")
        tar.extractall()
        tar.close()
        os.remove("RWSGwn.emb.tar.gz")


def print_time(text, time):
    total_time = (datetime.datetime.now() - time).total_seconds()
    print('>>> ' + text + ': \t' + str(total_time) + ' seconds')
    with open('benchmark.log', 'a+') as file:
        print('>>> ' + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' <<< ' + text + ': \t' + str(total_time) + ' seconds.', file=file)

    return datetime.datetime.now()


def test_tensorflow_gpu():
    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        print('No GPU found, check that tensorflow and cuda (or amd ROCm if you are using an AMD GPU) are correctly installed')
        return False
    else:
        print('Found GPU at: {}'.format(device_name))
        return True


if __name__ == '__main__':
    global device
    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', '--use_GPU', action='store_true')
    args = parser.parse_args()

    if not args.use_GPU:
        device = 'CPU'
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print('Benchmark will use the CPU in all the steps')

    else:
        if not test_tensorflow_gpu:
            quit()
        device = 'GPU'
        print('Benchmark will use GPU for matrix operations')

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    benchmark()
