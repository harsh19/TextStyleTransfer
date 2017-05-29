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
use_pointer=True
use_pretrained_embeddings = True
pretrained_embeddings_path = "../data/noConstraints_128.p"
pretrained_embeddings_are_trainable = True  
use_additional_info_from_pretrained_embeddings = True # if some word is not in training data set but is there in pretrained embeddings: mark True to add such words also. Otherwise mark False
use_sentinel_loss = True
lambd = 2.0

display_step=1
sample_step=2
save_step = 1
batch_size = 32
