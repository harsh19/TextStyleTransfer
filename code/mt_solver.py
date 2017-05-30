import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import Merge
from keras.utils import np_utils
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Input, Embedding, LSTM, Dense, merge, SimpleRNN, TimeDistributed
import tensorflow as tf
from tensorflow.contrib import rnn
from utilities import OutputSentence, TopN
import utilities
import utilities as utils
from utilities import *
import configuration
import math

# pointer
if configuration.use_pointer:
	import mt_model_pointer as model
else:
	import mt_model as model

	
class Solver:

	def __init__(self, params, buckets=None, mode='training'):
		if mode=='training':
			self.model_obj = model.RNNModel(buckets, mode=mode, params=params)
		else:
			self.model_obj = model.RNNModel(buckets_dict=None, mode=mode, params=params)

	def getModel(self, config, buckets, mode='train', reuse=False ):

		self.buckets = buckets 
		self.preds = []
		self.encoder_outputs_list = []
		self.cost_list = []
		self.sentinel_cost_list = []
		self.optimizer_list = []

		optimizer_typ =  "adam" #"sgd" #"adam"
		if "optimizer_typ" in config:
			optimizer_typ = config['optimizer_typ']
		self.optimizer_typ = optimizer_typ
		learning_rate= 0.001 #0.001
		print "optimizer_typ, learning_rate= ", optimizer_typ, learning_rate

		if mode=='train':
			#########################
			print "==================================================="
			for bucket_num, bucket_dct in self.buckets.items():
				config['max_input_seq_length'] = bucket_dct['max_input_seq_length']
				config['max_output_seq_length'] = bucket_dct['max_output_seq_length']
				print "------------------------------------------------------------------------------------------------------------------------------------------- "
				encoder_outputs = self.model_obj.getEncoderModel(config, mode='training', reuse= reuse, bucket_num=bucket_num )
				pred = self.model_obj.getDecoderModel(config, encoder_outputs, is_training=True, mode='training', reuse=reuse, bucket_num=bucket_num)
				self.preds.append(pred)
				self.encoder_outputs_list.append(encoder_outputs)
				self.cost_list.append( self.model_obj.cost )
				if config['use_pointer']:
					self.sentinel_cost_list.append( self.model_obj.sentinel_loss )
				cost = self.model_obj.cost 
				if self.optimizer_typ=="sgd":
					optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
					train_op = optimizer
				else: # adam
					optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
					grads = tf.gradients(cost, tf.trainable_variables())
					grads_and_vars = list(zip(grads, tf.trainable_variables()))
					train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars)
				self.optimizer_list.append(train_op)
				reuse=True

			self.encoder_outputs = encoder_outputs = self.model_obj.getEncoderModel(config, mode='inference', reuse=True )
			decoder_outputs_inference, encoder_outputs = self.model_obj.getDecoderModel(config, encoder_outputs, is_training=False, mode='inference', reuse=True)	
			self.decoder_outputs_inference = decoder_outputs_inference
			
			self.token_lookup_sequences_placeholder_list  = self.model_obj.token_lookup_sequences_placeholder_list
			self.token_lookup_sequences_decoder_placeholder_list = self.model_obj.token_lookup_sequences_decoder_placeholder_list
			self.token_output_sequences_decoder_placeholder_list = self.model_obj.token_output_sequences_decoder_placeholder_list
			self.mask_list = self.model_obj.masker_list
			if config['use_pointer']:
				self.token_output_sequences_decoder_inpmatch_placeholder_list = self.model_obj.token_output_sequences_decoder_inpmatch_placeholder_list
		else:
			encoder_outputs = self.model_obj.getEncoderModel(config, mode='inference', reuse=reuse)
			self.decoder_outputs_inference, self.encoder_outputs = self.model_obj.getDecoderModel(config, encoder_outputs, is_training=False, 	mode='inference', reuse=False)

			if configuration.use_pointer:
				self.beamSearchInit(config)

	def trainModel(self, config, train_feed_dict, val_feed_dct, reverse_vocab, do_init=True):
		
		# Initializing the variables
		if do_init:
			init = tf.global_variables_initializer()
			sess = tf.Session()
			sess.run(init)
			self.sess= sess

		saver = tf.train.Saver()

		print("============== \n Printing all trainainble variables")
		for v in tf.trainable_variables():
			print(v)
		print("==================")
		model_name = config['model_name']


		for bucket_num,bucket in enumerate(self.buckets):
			encoder_inputs, decoder_inputs, decoder_outputs, decoder_outputs_inpmatch = train_feed_dict[bucket_num]
			#cost = self.model_obj.cost

			# if y is passed as (N, seq_length, 1): change it to (N,seq_length)
			if len(decoder_outputs.shape)==3:
				decoder_outputs=np.reshape(decoder_outputs, (decoder_outputs.shape[0], decoder_outputs.shape[1]))

			#create temporary feed dictionary
			token_lookup_sequences_placeholder = self.token_lookup_sequences_placeholder_list[bucket_num]
			token_output_sequences_decoder_placeholder = self.token_output_sequences_decoder_placeholder_list[bucket_num]
			token_lookup_sequences_decoder_placeholder = self.token_lookup_sequences_decoder_placeholder_list[bucket_num]
			if config['use_pointer']:
				token_output_sequences_decoder_inpmatch_placeholder = self.token_output_sequences_decoder_inpmatch_placeholder_list[bucket_num]
			else:
				token_output_sequences_decoder_inpmatch_placeholder = None
			feed_dct={token_lookup_sequences_placeholder:encoder_inputs, token_output_sequences_decoder_placeholder:decoder_outputs, token_lookup_sequences_decoder_placeholder:decoder_inputs} 
			if config['use_pointer']:
				feed_dct[token_output_sequences_decoder_inpmatch_placeholder] = decoder_outputs_inpmatch

			#print "token_lookup_sequences_placeholder,  = ",token_lookup_sequences_placeholder, "\n token_output_sequences_decoder_placeholder = ",token_output_sequences_decoder_placeholder,"token_lookup_sequences_decoder_placeholder=",token_lookup_sequences_decoder_placeholder
			#print "\n encoder_inputs = ",encoder_inputs.shape, "\ndecoder_outputs =  ",decoder_outputs.shape, "\n decoder_inputs =  ", decoder_inputs.shape

			pred = self.preds[bucket_num]
			masker = self.mask_list[bucket_num]
			cost = self.cost_list[bucket_num]
			if config['use_pointer']:
				sentinel_cost = self.sentinel_cost_list[bucket_num]
			else:
				sentinel_cost = None

			# Gradient descent
			#learning_rate=0.1
			batch_size=config['batch_size']
			train_op = self.optimizer_list[bucket_num]
			#optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

			sess = self.sess

			training_iters=config['training_iters']
			display_step=configuration.display_step
			sample_step=configuration.sample_step
			save_step = configuration.save_step
			n = feed_dct[token_lookup_sequences_placeholder].shape[0]
			# Launch the graph
			step = 1
			#preds = np.array( sess.run(self.pred, feed_dict= feed_dct) )
			#print preds
			#with tf.Session() as sess:
			while step <= training_iters:
				#num_of_batches =  n/batch_size #(n+batch_size-1)/batch_size
				num_of_batches =  (n+batch_size-1)/batch_size
				for j in range(num_of_batches):
					#print "j= ",j
					feed_dict_cur = {}
					for k,v in feed_dct.items():
						feed_dict_cur[k] = v[j*batch_size:min(n,(j+1)*batch_size)]
						#print feed_dict_cur[k].shape
					cur_out = feed_dict_cur[token_output_sequences_decoder_placeholder]
					x,y = np.nonzero(cur_out)
					mask = np.zeros(cur_out.shape, dtype=np.float)
					mask[x,y]=1
					feed_dict_cur[masker]=mask
					sess.run(train_op, feed_dict=feed_dict_cur )

				if step % display_step == 0:
					#loss = sess.run(cost, feed_dict= feed_dict_cur)
					encoder_input_sequences, decoder_input_sequences, decoder_output_sequences, decoder_outputs_matching_inputs = val_feed_dct
					loss = self.getLoss( config, encoder_input_sequences, decoder_input_sequences, decoder_output_sequences, token_lookup_sequences_placeholder, token_lookup_sequences_decoder_placeholder, token_output_sequences_decoder_placeholder, masker, token_output_sequences_decoder_inpmatch_placeholder, decoder_outputs_matching_inputs, cost, sentinel_cost, sess)
					print "step ",step," : loss = ",loss
					bleu = self.getBleuOnVal( config, reverse_vocab, val_feed_dct, sess, model_name)
					print "step ",step," : bleu = ",bleu
				if step % sample_step == 0:
  					self.runInference( config, encoder_inputs[:batch_size], decoder_outputs[:batch_size], reverse_vocab, sess )
				if step%save_step==0:
					save_path = saver.save(sess, "./tmp/" + model_name + str(step) + ".ckpt")
	  				print "Model saved in file: ",save_path
				step += 1

		self.saver = saver

	###################################################################################

	def runInference(self, config, encoder_inputs, decoder_ground_truth_outputs, reverse_vocab, sess=None, print_all=True): # sampling
		if print_all:
			print " INFERENCE STEP ...... ============================================================"
		if sess==None:
	  		sess = tf.Session()
	  		saver = tf.train.Saver()
			saved_model_path = config['saved_model_path']
	  		saver.restore(sess,  saved_model_path ) #"./tmp/model39.ckpt")
		typ = "greedy" #config['inference_type']
		model_obj = self.model_obj
		feed_dct={model_obj.token_lookup_sequences_placeholder_inference:encoder_inputs}
		batch_size = config['batch_size'] #x_test.shape[0]
		if typ=="greedy":
			decoder_outputs_inference, encoder_outputs = np.array( sess.run([self.decoder_outputs_inference, self.encoder_outputs], feed_dict= feed_dct) ) # timesteps, N
			encoder_outputs = np.array(encoder_outputs)
			decoder_outputs_inference = np.transpose(decoder_outputs_inference) # (N,timesteps)
			if print_all:
				for i,row in enumerate(decoder_outputs_inference):
					ret=""
					for val in row:
					        if val==2: break # sentend. TO DO: load this value from config
						ret+=( " " + reverse_vocab[val] )
					#print "decoder_ground_truth_outputs[i] = ",decoder_ground_truth_outputs[i]
					print "GT: ", [ reverse_vocab[j] for j in decoder_ground_truth_outputs[i]]
					print "prediction: ",ret
					print "row= ",row
					print "matches: ", [ r==x for r,x in zip(row,decoder_ground_truth_outputs[i]) ]
					print ""
					if i>20:
						break
			return decoder_outputs_inference
		elif typ=="beam":
			pass

	def beamSearchInit(self, params):
		#pass
		self.encoder_outputs_beam, self.initial_state_c, self.initial_state_h = self.model_obj.getBeamSearchVars(t=0,params=params)
		self.prev_state_c, self.prev_state_h, self.encoder_input_sequence_beam, self.inputs_beam, self.cur_outputs_beam, self.state_c, self.state_h = self.model_obj.getBeamSearchVars(t=1,params=params)


	def beamSearch(self, config, encoder_inputs, decoder_ground_truth_outputs, reverse_vocab, sess=None, print_all=True, beam_size=8, max_caption_length=30):

		# encoder_inputs: (4, encoder input_size)
		inp_batch_size = 4
		start = 1
		end = 2
		length_normalization_factor = 0.1

		if sess==None:
	  		sess = tf.Session()
	  		saver = tf.train.Saver()
			saved_model_path = config['saved_model_path']
	  		saver.restore(sess,  saved_model_path ) #"./tmp/model39.ckpt")

		model_obj = self.model_obj
		batch_size = config['batch_size'] #x_test.shape[0]

		# Get encoder outputs
		# encoder_inputs_batch =  np.repeat(encoder_inputs,batch_size,axis=0)
		encoder_inputs_batch = encoder_inputs
		for j in range(batch_size - inp_batch_size): 
			encoder_inputs_batch = np.vstack( [encoder_inputs_batch, encoder_inputs_batch[0]] )
		feed_dct={model_obj.token_lookup_sequences_placeholder_inference: encoder_inputs_batch }
		encoder_outputs = np.array( sess.run(self.encoder_outputs, feed_dict= feed_dct) ) # timesteps, N, lstm_cell_size
		encoder_outputs = np.transpose(encoder_outputs, [1,0,2]) # (N,timesteps,lstm_cell_size)
		print encoder_outputs.shape
		##encoder_outputs_cur = encoder_outputs[0] # timesteps
		print "------>>>>>>>>>>>>>>>>>>>>>>>>"
		
		# Get initial state
		feed_dct = {self.encoder_outputs_beam:encoder_outputs}
		initial_state_c, initial_state_h = sess.run( [self.initial_state_c, self.initial_state_h], feed_dict=feed_dct ) # N, lstm_cell_size
		
		# initial beam
		partial_captions_all = []
		complete_captions_all = []
		for i in range(inp_batch_size):
			initial_state_ci, initial_state_hi = initial_state_c[i], initial_state_h[i]
			initial_beam = OutputSentence(
				sentence=[start], # to do: replace with start symbol index
				state=(initial_state_ci, initial_state_hi),
				logprob=0.0,
				score=0.0,
				metadata=[""])
			partial_captions_all.append( TopN(beam_size) )
			partial_captions_all[i].push(initial_beam)
			complete_captions_all[i].append( TopN(beam_size) )

		# Run beam search.
		for _ in range(max_caption_length - 1):	
			partial_captions_list = []
			input_feed = []
			state_feed_c = []
			state_feed_h = []
			mapper = []
			for i in range(inp_batch_size):
				partial_captions = partial_captions_all[i]
				tmp = partial_captions.extract()
				partial_captions_list.extend(tmp)
				for j in range(len(tmp)): mapper.append(i)
				partial_captions.reset()
				input_feed.extend( [c.sentence[-1] for c in partial_captions_list] )
				state_feed_c.extend( [c.state[0] for c in partial_captions_list] )
				state_feed_h.extend( [c.state[1] for c in partial_captions_list] )

			input_feed = np.reshape(np.array(input_feed), [-1,1])
			state_feed_c = np.array(state_feed_c)
			state_feed_h = np.array(state_feed_h)

			#self.prev_state_c, self.prev_state_h, self.encoder_input_sequence_beam, self.inputs_beam, self.cur_outputs_beam, self.state_c, self.state_h
			feed_dct_tmp = { self.prev_state_c:state_feed_c, self.prev_state_h:state_feed_h, self.encoder_input_sequence_beam:encoder_inputs_batch, self.inputs_beam:input_feed, self.encoder_outputs_beam:encoder_outputs}
			feed_dct = {}
			lim = len(input_feed)
			for k,v in feed_dct_tmp.items():
				if len(v)<batch_size:
					gap = batch_size - len(v)
					for j in range(gap):
						v = np.vstack( (v,v[0]) )
				#print k, len(v)
				feed_dct[k]=v
			cur_outputs_beam, state_c, state_h = sess.run( [self.cur_outputs_beam, self.state_c, self.state_h], feed_dict=feed_dct )
			softmax = cur_outputs_beam

			for i, partial_caption in enumerate(partial_captions_list):
				cur_mapper = mapper[i]
				word_probabilities = softmax[i]
				state = state_c[i], state_h[i]
				# For this partial caption, get the beam_size most probable next words.
				words_and_probs = list(enumerate(word_probabilities))
				words_and_probs.sort(key=lambda x: -x[1])
				words_and_probs = words_and_probs[0:beam_size]
				# Each next word gives a new partial caption.
				for w, p in words_and_probs:
					if p < 1e-12:
						continue	# Avoid log(0).
					sentence = partial_caption.sentence + [w]
					logprob = partial_caption.logprob + math.log(p)
					score = logprob
					if w == end:
						if length_normalization_factor > 0:
							score /= len(sentence)**length_normalization_factor
						beam = OutputSentence(sentence, state, logprob, score, [""])
						complete_captions_all[cur_mapper].push(beam)
					else:
						beam = OutputSentence(sentence, state, logprob, score, [""])
						partial_captions_all[cur_mapper].push(beam)
		
			if sum( [partial_captions_all[i].size() for i in range(inp_batch_size)] ) ==0:
				# We have run out of partial candidates; happens when beam_size = 1.
				break

		# If we have no complete captions then fall back to the partial captions.
		# But never output a mixture of complete and partial captions because a
		# partial caption could have a higher score than all the complete captions.
		for i in range(inp_batch_size):
			if not complete_captions_all[i].size():
				complete_captions_all[i] = partial_captions_all[i]

		ret = []
		for i in range(inp_batch_size):
			tmp = complete_captions_all[i].extract(sort=True)
			tmp = np.array( [ c.sentence for c in tmp ] )
			ret.append(tmp)
		return ret
		#return np.array( tmp[0].sentence )
		

	###################################################################################

	def solveAll(self, config, encoder_inputs, decoder_ground_truth_outputs, reverse_vocab, sess=None, print_progress=True, inference_type="beam"): # sampling
		print " SolveAll ...... ============================================================"
		
		if inference_type=="greedy":
			batch_size = config['batch_size']
		else:# beam
			batch_size = 4
		num_batches = ( len(encoder_inputs) + batch_size - 1)/ batch_size 
		print "num_batches = ",num_batches
		print "batch_size = ", batch_size 
		print "len(encoder_inputs) = ",len(encoder_inputs)
		decoder_outputs_inference = []
		for i in range(num_batches):
			if print_progress:
				print "i= ",i
			encoder_inputs_cur = encoder_inputs[i*batch_size:(i+1)*batch_size]
			decoder_gt_outputs_cur = decoder_ground_truth_outputs[i*batch_size:(i+1)*batch_size]
			lim = len(encoder_inputs_cur)
			if len(encoder_inputs_cur)<batch_size:
				gap = batch_size - len(encoder_inputs_cur)
				for j in range(gap):
					encoder_inputs_cur = np.vstack( (encoder_inputs_cur,encoder_inputs[0]) )
					decoder_gt_outputs_cur = np.vstack( (decoder_gt_outputs_cur,decoder_ground_truth_outputs[0]) )
					#decoder_gt_outputs_cur.extend(decoder_ground_truth_outputs[0]*gap)
			if inference_type=="greedy":
				decoder_outputs_inference_cur = self.runInference(config, encoder_inputs_cur, decoder_gt_outputs_cur, reverse_vocab, sess=sess, print_all=False)
				decoder_outputs_inference.extend( decoder_outputs_inference_cur[:lim] )
			else:
				decoder_outputs_inference_cur = self.beamSearch(config, encoder_inputs_cur, decoder_gt_outputs_cur, reverse_vocab, sess=sess, print_all=False)
				decoder_outputs_inference.extend( decoder_outputs_inference_cur )
			break
		print "len(encoder_inputs) = ",len(encoder_inputs)
		print "len(decoder_outputs_inference) = ",len(decoder_outputs_inference)
		print decoder_outputs_inference[0], decoder_ground_truth_outputs[0]

		### debug
		print ' '.join( [reverse_vocab[i] for i in decoder_outputs_inference[0]] )
		print ""
		print ' '.join( [reverse_vocab[i] for i in decoder_outputs_inference[1]] )

		return decoder_outputs_inference, decoder_ground_truth_outputs

	###################################################################################

	def getLoss(self, config, encoder_input_sequences, decoder_input_sequences, decoder_output_sequences, enc_inp_placeholder, dec_in_placeholder, dec_out_placeholder, mask_placeholder, token_output_sequences_decoder_inpmatch_placeholder, decoder_outputs_matching_inputs, loss_variable, sentinel_loss_variable, sess): # Probabilities
		print " getLoss ...... ============================================================"
		batch_size = config['batch_size']
		num_batches = ( len(encoder_input_sequences) + batch_size - 1)/ batch_size 
		loss = []
		sentinel_loss = []
		all_vals=[]
		for i in range(num_batches):
			#print "i= ",i
			cur_input_sequences = encoder_input_sequences[i*batch_size:(i+1)*batch_size]
			cur_decoder_input_sequences = decoder_input_sequences[i*batch_size:(i+1)*batch_size]
			cur_decoder_output_sequences = decoder_output_sequences[i*batch_size:(i+1)*batch_size]
			cur_decoder_outputs_matching_inputs = decoder_outputs_matching_inputs[i*batch_size:(i+1)*batch_size]
			lim = len(cur_input_sequences)
			if len(cur_input_sequences)<batch_size:
				gap = batch_size - len(cur_input_sequences)
				for j in range(gap):
					cur_decoder_output_sequences = np.vstack( (cur_decoder_output_sequences, cur_decoder_output_sequences[0]) )
					cur_decoder_input_sequences = np.vstack( (cur_decoder_input_sequences, cur_decoder_input_sequences[0]) )
					cur_input_sequences = np.vstack( (cur_input_sequences, cur_input_sequences[0]) )
					cur_decoder_outputs_matching_inputs = np.concatenate( (cur_decoder_outputs_matching_inputs, np.array([cur_decoder_outputs_matching_inputs[0]]) ) )
			feed_dct = {enc_inp_placeholder:cur_input_sequences, dec_out_placeholder:cur_decoder_output_sequences, dec_in_placeholder:cur_decoder_input_sequences}
			if config['use_pointer']:
				feed_dct[token_output_sequences_decoder_inpmatch_placeholder] = cur_decoder_outputs_matching_inputs
			mask = np.zeros(cur_decoder_output_sequences.shape, dtype=np.float)
			x,y = np.nonzero(cur_decoder_output_sequences)
			mask[x,y]=1
			feed_dct[mask_placeholder]=mask
			cur_loss = sess.run(loss_variable, feed_dct)
			#vals = sess.run(self.model_obj.vals, feed_dct)
			#all_vals.append(vals)
			loss.append( cur_loss )
			## cur_loss = sess.run(sentinel_loss_variable, feed_dct)
			## sentinel_loss.append( cur_loss )
			#break
		loss = np.array(loss)
		## sentinel_loss = np.array(sentinel_loss)
		## print "s3ntinel loss = ", np.mean(sentinel_loss)
		
		'''vals = all_vals[0]
		for val in vals: # val-> at a time step
			cur_sentinel_attention_loss,sentinel_weight = val[0], val[1]
			print "cur_sentinel_attention_loss = ",cur_sentinel_attention_loss
			print "sentinel_weight = ",sentinel_weight
			print ""
			break
		'''
		
		return np.mean(loss)

	def getBleuOnVal(self, params, reverse_vocab, val_feed, sess, model_name):
		val_encoder_inputs, val_decoder_inputs, val_decoder_outputs, val_decoder_outputs_matching_inputs = val_feed
		decoder_outputs_inference, decoder_ground_truth_outputs = self.solveAll(params, val_encoder_inputs, val_decoder_outputs, reverse_vocab, sess=sess, print_progress=False)        			   
		validOutFile_name = "./tmp/tmp_" + model_name +".valid.output"
		original_data_path = "../data/valid.original.nltktok"
		BLEUOutputFile_path = "./tmp/tmp_" + model_name + ".valid.BLEU"
		utilities.getBlue(validOutFile_name, original_data_path, BLEUOutputFile_path, decoder_outputs_inference, decoder_ground_truth_outputs, params['preprocessing'])
		return open(BLEUOutputFile_path,"r").read()

########################################################################################




