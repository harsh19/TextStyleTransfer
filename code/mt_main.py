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
OR
python main.py debug
'''

class PreProcessing:

	def __init__(self):
		self.unknown_word = "UNK".lower()
		self.sent_start = "SENTSTART".lower()
		self.sent_end = "SENTEND".lower()
		self.pad_word = "PADWORD".lower()

		self.word_counters, self.word_to_idx, self.word_to_idx_ctr, self.idx_to_word = self.initVocabItems()

	def initVocabItems(self):

		word_counters = {}
		word_to_idx = {}
		word_to_idx_ctr = 0 
		idx_to_word = {}

		word_to_idx[self.pad_word] = word_to_idx_ctr # 0 is for padword
		idx_to_word[word_to_idx_ctr]=self.pad_word
		word_counters[self.pad_word] = 1
		word_to_idx_ctr+=1
		
		word_to_idx[self.sent_start] = word_to_idx_ctr
		word_counters[self.sent_start] = 1
		idx_to_word[word_to_idx_ctr]=self.sent_start
		word_to_idx_ctr+=1

		word_to_idx[self.sent_end] = word_to_idx_ctr
		word_counters[self.sent_end] = 1
		idx_to_word[word_to_idx_ctr]=self.sent_end		
		word_to_idx_ctr+=1
		
		word_counters[self.unknown_word] = 1
		word_to_idx[self.unknown_word] = word_to_idx_ctr
		idx_to_word[word_to_idx_ctr]=self.unknown_word		
		word_to_idx_ctr+=1

		return word_counters, word_to_idx, word_to_idx_ctr, idx_to_word

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
		return [row.strip().lower().split(' ') for row in text_rows]

	def loadVocab(self, split):

		print "======================================================= loadData: split = ",split
		inp_src = config.data_dir + split + ".original" + ".nltktok" #".modern"
		out_src = config.data_dir + split + ".modern" + ".nltktok" #".original"
		inp_data = open(inp_src,"r").readlines()
		out_data = open(out_src,"r").readlines()
		
		inputs = self.preprocess(inp_data)
		outputs = self.preprocess(out_data)
		
		word_to_idx = self.word_to_idx
		idx_to_word = self.idx_to_word
		word_to_idx_ctr = self.word_to_idx_ctr
		word_counters = self.word_counters

		texts = inputs
		for text in texts:
			for token in text:
				if token not in word_to_idx:
					word_to_idx[token] = word_to_idx_ctr
					idx_to_word[word_to_idx_ctr]=token
					word_to_idx_ctr+=1
					word_counters[token]=0
				word_counters[token]+=1
		texts = outputs
		for text in texts:
			for token in text:
				if token not in word_to_idx:
					word_to_idx[token] = word_to_idx_ctr
					idx_to_word[word_to_idx_ctr]=token
					word_to_idx_ctr+=1
					word_counters[token]=0
				word_counters[token]+=1

		self.word_to_idx = word_to_idx
		self.idx_to_word = idx_to_word
		self.vocab_size = len(word_to_idx)
		self.word_to_idx_ctr = word_to_idx_ctr
		self.word_counters = word_counters

	def pruneVocab(self, max_vocab_size):
		word_to_idx = self.word_to_idx
		idx_to_word = self.idx_to_word
		word_to_idx_ctr = self.word_to_idx_ctr
		word_counters = self.word_counters

		self.word_counters, self.word_to_idx, self.word_to_idx_ctr, self.idx_to_word = self.initVocabItems()

		print "vocab size before pruning = ", len(word_to_idx)
		top_items = sorted( word_counters.items(), key=lambda x:-x[1] )[:max_vocab_size]
		for token_count in top_items:
			token=token_count[0]
			self.word_to_idx[token] = self.word_to_idx_ctr
			self.idx_to_word[self.word_to_idx_ctr] = token
			self.word_to_idx_ctr+=1
		self.vocab_size = len(self.word_to_idx)
		print "vocab size after pruning = ", self.vocab_size


	def loadData(self, split):

		print "======================================================= loadData: split = ",split
		inp_src = config.data_dir + split + ".original" + ".nltktok" #".modern"
		out_src = config.data_dir + split + ".modern" + ".nltktok" #".original"
		inp_data = open(inp_src,"r").readlines()
		out_data = open(out_src,"r").readlines()
		
		inputs = self.preprocess(inp_data)
		outputs = self.preprocess(out_data)
		
		word_to_idx = self.word_to_idx
		idx_to_word = self.idx_to_word
		word_to_idx_ctr = self.word_to_idx_ctr

		# generate sequences
		sequences_input = [] 		
		sequences_output = [] 

		texts = inputs
		for text in texts:
			tmp = [word_to_idx[self.sent_start]]
			for token in text:
				if token not in word_to_idx:
					tmp.append(word_to_idx[self.unknown_word])
				else:
					tmp.append(word_to_idx[token])
			tmp.append(word_to_idx[self.sent_end])
			sequences_input.append(tmp)

		texts = outputs
		for text in texts:
			tmp = [word_to_idx[self.sent_start]]
			for token in text:
				if token not in word_to_idx:
					tmp.append(word_to_idx[self.unknown_word])
				else:
					tmp.append(word_to_idx[token])
			tmp.append(word_to_idx[self.sent_end])
			sequences_output.append(tmp)

		# pad sequences
		# sequences_input, sequences_output = padAsPerBuckets(sequences_input, sequences_output)
		sequences_input = pad_sequences(sequences_input, maxlen=config.max_input_seq_length, padding='pre', truncating='post')
		sequences_output = pad_sequences(sequences_output, maxlen=config.max_output_seq_length, padding='post', truncating='post')

		print "Printing few sample sequences... "
		print sequences_input[0],":", self.fromIdxSeqToVocabSeq(sequences_input[0]), "---", sequences_output[0], ":", self.fromIdxSeqToVocabSeq(sequences_output[0])
		print sequences_input[113], sequences_output[113]
		print "================================="

		return sequences_input, sequences_output

	def fromIdxSeqToVocabSeq(self, seq):
		return [self.idx_to_word[x] for x in seq]

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
	params['pretrained_embeddings'] = config.use_pretrained_embeddings
	params['share_encoder_decoder_embeddings'] = config.share_encoder_decoder_embeddings
	params['use_pointer'] = config.use_pointer
	params['pretrained_embeddings_path'] = config.pretrained_embeddings_path
	params['pretrained_embeddings_are_trainable'] = config.pretrained_embeddings_are_trainable
	params['max_vocab_size'] = config.max_vocab_size
	params['do_vocab_pruning'] = config.do_vocab_pruning

	print "params = ", params
	buckets = {  0:{'max_input_seq_length':params['max_input_seq_length'], 'max_output_seq_length':params['max_output_seq_length']} }
	print "buckets = ",buckets

	# preprocessing
	print "------------------------------------------------------------------------"
	preprocessing = PreProcessing()
	splits =["train","valid","test"]
	#for split in splits: preprocessing.loadVocab(split)
	preprocessing.loadVocab('train')
	if params['do_vocab_pruning']:
		preprocessing.pruneVocab(max_vocab_size=params['max_vocab_size'])
	data_seq = {split:preprocessing.loadData(split=split) for split in splits}			
	data = { split:preprocessing.prepareMTData(cur_data) for split,cur_data in data_seq.items()  }
	for split,split_data in data.items():
		print "Split: ",split
		inp,dinp,dout = split_data
		print inp.shape, dinp.shape, dout.shape
	print "------------------------------------------------------------------------"
	print "------------------------------------------------------------------------"
	print ""
	#return

	params['vocab_size'] = preprocessing.vocab_size
	
	# train
	mode=sys.argv[1]
	print "mode = ",mode

	train = data['train']
	val = data['valid']
	test = data['test']
	if mode=="debug":
		lim = 64
	else:
		lim=params['batch_size'] * ( len(train[0])/params['batch_size'] )
	if lim!=-1:
		train_encoder_inputs, train_decoder_inputs, train_decoder_outputs = train
		train_encoder_inputs = train_encoder_inputs[:lim]
		train_decoder_inputs = train_decoder_inputs[:lim]
		train_decoder_outputs = train_decoder_outputs[:lim]
		train = train_encoder_inputs, train_decoder_inputs, train_decoder_outputs
	if params['pretrained_embeddings']:
		pretrained_embeddings = pickle.load(open(params['pretrained_embeddings_path'],"r"))
		word_to_idx = preprocessing.word_to_idx
		encoder_embedding_matrix = np.random.rand( params['vocab_size'], params['embeddings_dim'] )
		decoder_embedding_matrix = np.random.rand( params['vocab_size'], params['embeddings_dim'] )
		not_found_count = 0
		for token,idx in word_to_idx.items():
			if token in pretrained_embeddings:
				encoder_embedding_matrix[idx]=pretrained_embeddings[token]
				decoder_embedding_matrix[idx]=pretrained_embeddings[token]
			else:
				if not_found_count<10:
					print "No pretrained embedding for (only first 10 such cases will be printed. other prints are suppressed) ",token
				not_found_count+=1
		params['encoder_embeddings_matrix'] = encoder_embedding_matrix 
		params['decoder_embeddings_matrix'] = decoder_embedding_matrix 


	if mode=='train' or mode=="debug":
		if mode=="train":
			training_iters = int(sys.argv[2])
			model_name = sys.argv[3]
		else:
			training_iters = 5
			model_name = "test"
		params['training_iters'] = training_iters
		params['use_reverse_encoder'] = config.use_reverse_encoder
		params['model_name'] = model_name
		train_buckets = {}
		for bucket,_ in enumerate(buckets):
			train_buckets[bucket] = train

		rnn_model = solver.Solver(params,buckets)
		_ = rnn_model.getModel(params, mode='train',reuse=False, buckets=buckets)
		rnn_model.trainModel(config=params, train_feed_dict=train_buckets, val_feed_dct=val, reverse_vocab=preprocessing.idx_to_word, do_init=True)
	
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
		#rnn_model.runInference(params, val_encoder_inputs[:params['batch_size']], val_decoder_outputs[:params['batch_size']], preprocessing.idx_to_word)
		rnn_model.solveAll(params, val_encoder_inputs, val_decoder_outputs, preprocessing.idx_to_word)

if __name__ == "__main__":
	main()
