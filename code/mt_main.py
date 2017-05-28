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
from prepro import PreProcessing

'''
# usage:
python main.py train <num_of_iters> <model_name>
OR
python main.py inference <saved_model_path>
OR
python main.py debug
OR 
python main.py preprocessing
'''

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
	params['use_additional_info_from_pretrained_embeddings'] = config.use_additional_info_from_pretrained_embeddings
	params['max_vocab_size'] = config.max_vocab_size
	params['do_vocab_pruning'] = config.do_vocab_pruning
	params['use_reverse_encoder'] = config.use_reverse_encoder

	print "params = ", params
	buckets = {  0:{'max_input_seq_length':params['max_input_seq_length'], 'max_output_seq_length':params['max_output_seq_length']} }
	print "buckets = ",buckets
	
	# train
	mode=sys.argv[1]
	print "mode = ",mode

	if mode=="preprocessing":
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
		pickle.dump(data,open("./tmp/data.obj","w"))
		pickle.dump(preprocessing, open("./tmp/preprocessing.obj","w") )
		return
	else:
		data = pickle.load(open("./tmp/data.obj","r") )
		preprocessing = pickle.load(open("./tmp/preprocessing.obj","r") )

	params['vocab_size'] = preprocessing.vocab_size

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
		print "not found count = ", not_found_count 
		params['encoder_embeddings_matrix'] = encoder_embedding_matrix 
		params['decoder_embeddings_matrix'] = decoder_embedding_matrix 

		if params['use_additional_info_from_pretrained_embeddings']:
			additional_count=0
			tmp=[]
			for token in pretrained_embeddings:
				if token not in preprocessing.word_to_idx:
					preprocessing.word_to_idx[token] = preprocessing.word_to_idx_ctr
					preprocessing.idx_to_word[preprocessing.word_to_idx_ctr] = token
					preprocessing.word_to_idx_ctr+=1
					tmp.append(pretrained_embeddings[token])
					additional_count+=1
			print "additional_count = ",additional_count
			params['vocab_size'] = preprocessing.word_to_idx_ctr
			tmp = np.array(tmp)
			encoder_embedding_matrix = np.vstack([encoder_embedding_matrix,tmp])
			decoder_embedding_matrix = np.vstack([decoder_embedding_matrix,tmp])
			print "decoder_embedding_matrix.shape ",decoder_embedding_matrix.shape
			print "New vocab size = ",params['vocab_size']


	if mode=='train' or mode=="debug":
		if mode=="train":
			training_iters = int(sys.argv[2])
			model_name = sys.argv[3]
		else:
			training_iters = 5
			model_name = "test"
		params['training_iters'] = training_iters
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
		decoder_outputs_inference, decoder_ground_truth_outputs = rnn_model.solveAll(params, val_encoder_inputs, val_decoder_outputs, preprocessing.idx_to_word)
			   
		validOutFile=open(saved_model_path+".valid.output","w")
		for outputLine,groundLine in zip(decoder_outputs_inference,decoder_ground_truth_outputs):
			print outputLine
			outputLine=preprocessing.fromIdxSeqToVocabSeq(outputLine)
			if "sentend" in outputLine:
				outputLine=outputLine[:outputLine.index("sentend")]
			print outputLine
			print preprocessing.fromIdxSeqToVocabSeq(groundLine)
			outputLine=" ".join(outputLine)+"\n"
			validOutFile.write(outputLine)
		validOutFile.close()
		
		import os
		#os.system("./multi-bleu.perl -lc ../data/valid.original.nltktok < "+saved_model_path+".valid.output")
		BLEUOutput=os.popen("./multi-bleu.perl -lc ../data/valid.original.nltktok < "+saved_model_path+".valid.output").read()
		BLEUOutputFile=open(saved_model_path+".valid.BLEU","w")
		BLEUOutputFile.write(BLEUOutput)
		BLEUOutputFile.close()


if __name__ == "__main__":
	main()
