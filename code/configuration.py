data_dir = "../data/"

max_input_seq_length = 25
max_output_seq_length = 25
embeddings_dim = 128
#dropout_val = 0.2
lstm_cell_size=128
do_vocab_pruning = True
max_vocab_size = 12000

use_reverse_encoder=True
share_encoder_decoder_embeddings=True
use_pointer=False
use_pretrained_embeddings = True
pretrained_embeddings_path = "../data/noConstraints_128.p"
pretrained_embeddings_are_trainable = True  # Not supported yet. 

display_step=1
sample_step=2
save_step = 4
batch_size = 32
