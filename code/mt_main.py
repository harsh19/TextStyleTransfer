from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np
import csv
import configuration as config
from sklearn.preprocessing import LabelEncoder
import mt_model as models
import pickle
import utilities as datasets
import utilities
import mt_solver as solver
import sys

'''
# usage:
python main.py train <num_of_iters> <model_name>
OR
python main.py inference <saved_model_path>
'''

class PreProcessing:

	def __init__(self):
		self.unknown_word = "UNK".lower()
		self.sent_start = "SENTSTART".lower()
		self.sent_end = "SENTEND".lower()
		self.pad_word = "PADWORD".lower()

		word_to_idx = {}
		word_to_idx_ctr = 0 
		idx_to_word = {}
		word_to_idx[self.pad_word] = word_to_idx_ctr # 0 is for padword
		idx_to_word[word_to_idx_ctr]=self.pad_word
		word_to_idx_ctr+=1
		word_to_idx[self.sent_start] = word_to_idx_ctr
		idx_to_word[word_to_idx_ctr]=self.sent_start
		word_to_idx_ctr+=1
		word_to_idx[self.sent_end] = word_to_idx_ctr
		idx_to_word[word_to_idx_ctr]=self.sent_end		
		word_to_idx_ctr+=1
		word_to_idx[self.unknown_word] = word_to_idx_ctr
		idx_to_word[word_to_idx_ctr]=self.unknown_word		
		word_to_idx_ctr+=1

		#strore needed data structres as class variables
		self.word_index = word_to_idx
		self.index_word = idx_to_word
		self.word_to_idx_ctr = word_to_idx_ctr

	def pad_sequences_my(sequences, maxlen, padding='post', truncating='post'):
		ret=[]
		for sequence in sequences:
			if len(sequence)>=maxlen:
				sequence=sequence[:maxlen]
			else:
				if padding=='post':
					sequence = sequence + [0]*(maxlen - len(sequence))
				else:
					sequence = [0]*(maxlen - len(sequence)) + sequence
			ret.append(sequence)
		return np.array(ret)

	def preprocess(self, text_rows):
		return [row.strip().split(' ') for row in text_rows]

	def loadData(self, split, update_vocab=True):

		print "======================================================= loadData: split = ",split
		inp_src = config.data_dir + split + ".modern"
		out_src = config.data_dir + split + ".original"
		inp_data = open(inp_src,"r").readlines()
		out_data = open(out_src,"r").readlines()
		
		inputs = self.preprocess(inp_data)
		outputs = self.preprocess(out_data)
		
		word_to_idx = self.word_index
		idx_to_word = self.index_word
		word_to_idx_ctr = self.word_to_idx_ctr

		if update_vocab:
			# generate vocab
			texts = inputs
			for text in texts:
				for token in text:
					if token not in word_to_idx:
						word_to_idx[token] = word_to_idx_ctr
						idx_to_word[word_to_idx_ctr]=token
						word_to_idx_ctr+=1
			texts = outputs
			for text in texts:
				for token in text:
					if token not in word_to_idx:
						word_to_idx[token] = word_to_idx_ctr
						idx_to_word[word_to_idx_ctr]=token
						word_to_idx_ctr+=1
		#print "Ignoring MAX_VOCAB_SIZE "
		#print "Found vocab size = ",word_to_idx_ctr-1 # -1 due to padword

		# generate sequences
		sequences_input = [ [word_to_idx[token] for token in text] for text in inputs ]
		sequences_input = [ [word_to_idx[self.sent_start]]+text+[word_to_idx[self.sent_end]] for text in sequences_input ]
		sequences_output = [ [word_to_idx[token] for token in text] for text in outputs ]
		sequences_output = [ [word_to_idx[self.sent_start]]+text+[word_to_idx[self.sent_end]] for text in sequences_output ]

		# pad sequences
		#sequences_input, sequences_output = padAsPerBuckets(sequences_input, sequences_output)
		sequences_input = pad_sequences(sequences_input, maxlen=config.max_input_seq_length, padding='pre', truncating='post')
		sequences_output = pad_sequences(sequences_output, maxlen=config.max_output_seq_length, padding='post', truncating='post')
		
		self.word_index = word_to_idx
		self.index_word = idx_to_word
		self.vocab_size = len(word_to_idx)
		self.word_to_idx_ctr = word_to_idx_ctr

		print "Printing few sample sequences... "
		print sequences_input[0],":", self.fromIdxSeqToVocabSeq(sequences_input[0]), "---", sequences_output[0], ":", self.fromIdxSeqToVocabSeq(sequences_output[0])
		print sequences_input[113], sequences_output[113]

		print "================================="
		return sequences_input, sequences_output

	def fromIdxSeqToVocabSeq(self, seq):
		return [self.index_word[x] for x in seq]

	def prepareMTData(self, sequences, seed=123, do_shuffle=False):
		inputs, outputs = sequences

		decoder_inputs = np.array( [ sequence[:-1] for sequence in outputs ] )
		#decoder_outputs = np.array( [ np.expand_dims(sequence[1:],-1) for sequence in outputs ] )
		decoder_outputs = np.array( [ sequence[1:] for sequence in outputs ] )
		encoder_inputs = np.array(inputs)

		if do_shuffle:
			#shuffling
			indices = np.arange(encoder_inputs.shape[0])
			np.random.seed(seed)
			np.random.shuffle(indices)

		return encoder_inputs, decoder_inputs, decoder_outputs
		

def main():
	
	# params
	params = {}
	params['embeddings_dim'] =  config.embeddings_dim
	params['lstm_cell_size'] = config.lstm_cell_size
	params['max_input_seq_length'] = config.max_input_seq_length
	params['max_output_seq_length'] = config.max_output_seq_length-1 #inputs are all but last element, outputs are al but first element
	params['batch_size'] = config.batch_size
	params['pretrained_embeddings']=False
	print "params = ", params
	buckets = {  0:{'max_input_seq_length':params['max_input_seq_length'], 'max_output_seq_length':params['max_output_seq_length']} }
	print "buckets = ",buckets

	# preprocessing
	print "------------------------------------------------------------------------"
	preprocessing = PreProcessing()
	splits =["train","valid","test"]
	data_seq = {split:preprocessing.loadData(split=split) for split in splits}			
	data = { split:preprocessing.prepareMTData(cur_data) for split,cur_data in data_seq.items()  }
	for split,split_data in data.items():
		print "Split: ",split
		inp,dinp,dout = split_data
		print inp.shape, dinp.shape, dout.shape
		'''print inp[0]
		print dinp[0]
		print dout[0]
		print ""
		'''
	print "------------------------------------------------------------------------"
	print "------------------------------------------------------------------------"
	print ""
	#return

	params['vocab_size'] = preprocessing.vocab_size
	
	# train
	train = data['train']
	lim=params['batch_size'] * ( len(train[0])/params['batch_size'] )
	if lim!=-1:
		train_encoder_inputs, train_decoder_inputs, train_decoder_outputs = train
		train_encoder_inputs = train_encoder_inputs[:lim]
		train_decoder_inputs = train_decoder_inputs[:lim]
		train_decoder_outputs = train_decoder_outputs[:lim]
		train = train_encoder_inputs, train_decoder_inputs, train_decoder_outputs
	if params['pretrained_embeddings']:
		pretrained_embeddings = getPretrainedEmbeddings(src="pretrained_embeddings.txt")
		word_to_idx = preprocessing.word_index
		encoder_embedding_matrix = np.random.rand( params['vocab_size'], params['embeddings_dim'] )
		decoder_embedding_matrix = np.random.rand( params['vocab_size'], params['embeddings_dim'] )
		for char,idx in word_to_idx.items():
			if char in pretrained_embeddings:
				encoder_embedding_matrix[idx]=pretrained_embeddings[char]
				decoder_embedding_matrix[idx]=pretrained_embeddings[char]
			else:
				print "No pretrained embedding for ",char
		params['encoder_embeddings_matrix'] = encoder_embedding_matrix 
		params['decoder_embeddings_matrix'] = decoder_embedding_matrix 

	mode=sys.argv[1]
	print "mode = ",mode
	if mode=='train':
		training_iters = int(sys.argv[2])
		model_name = sys.argv[3]
		params['training_iters'] = training_iters
		params['use_reverse_encoder'] = config.use_reverse_encoder
		params['model_name'] = model_name
		train_buckets = {}
		for bucket,_ in enumerate(buckets):
			train_buckets[bucket] = train

		rnn_model = solver.Solver(params,buckets)
		_ = rnn_model.getModel(params, mode='train',reuse=False, buckets=buckets)
		rnn_model.trainModel(config=params, train_feed_dict=train_buckets, val_feed_dct=None, reverse_vocab=preprocessing.index_word, do_init=True)
	
	else:
		saved_model_path = sys.argv[2]
		val_encoder_inputs, val_decoder_inputs, val_decoder_outputs = val
		print "val_encoder_inputs = ",val_encoder_inputs

		if len(val_decoder_outputs.shape)==3:
			val_decoder_outputs=np.reshape(val_decoder_outputs, (val_decoder_outputs.shape[0], val_decoder_outputs.shape[1]))

		params['saved_model_path'] = saved_model_path
		rnn_model = solver.Solver(params, buckets=None, mode='inference')
		_ = rnn_model.getModel(params, mode='inference', reuse=False, buckets=None)
		print "----Running inference-----"
		#rnn_model.runInference(params, val_encoder_inputs[:params['batch_size']], val_decoder_outputs[:params['batch_size']], preprocessing.index_word)
		rnn_model.solveAll(params, val_encoder_inputs, val_decoder_outputs, preprocessing.index_word)

if __name__ == "__main__":
	main()
