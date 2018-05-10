import cPickle as cp
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score
import os, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, KFold
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
import keras
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

glo_seed = 0
np.random.seed(glo_seed)
rng = np.random.RandomState(seed=glo_seed)
tf.set_random_seed(glo_seed)

learning_rate, number_of_hidden_unit_lone, number_of_hidden_unit_ltwo = 0.005,100,50
d_type = 1

def plot_bar_x(label):
    # this is for plotting purpose
    index = np.arange(len(label))
    plt.bar(index, label)
    plt.xlabel('Features')
    plt.ylabel('Corrolation')
    plt.xticks(index, label, fontsize=7, rotation=30)
    plt.title('Corrolation Coefficient')
    plt.show()

def generate_random_hyperparams(lr_min, lr_max, w_min, w_max, ldim_min, ldim_max):
	'''generate random learning rate, neural network layer size and dimension of latent variable'''
	# random search through log space for learning rate
	random_learning_rate = 10**np.random.uniform(lr_min, lr_max)

	W = np.random.uniform(w_min, w_max)
	ldim = np.random.uniform(ldim_min, ldim_max)
	return int(W), int(ldim), random_learning_rate

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
		self.acc.append(logs.get('acc'))

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


class conv_nn ():
	def __initilize(self, n_dimensions, n_classes, ID="conv_train"):
		self.id = ID
		self.learning_rate, self.number_of_hidden_unit_lone, self.number_of_hidden_unit_ltwo = learning_rate, number_of_hidden_unit_lone, number_of_hidden_unit_ltwo
		self.batch_size = 100
		self.epochs = 30
		self.n_classes = n_classes
		self.n_dimensions = n_dimensions
		
		self.model = Sequential()
		self.model.add(Conv1D(self.number_of_hidden_unit_lone, kernel_size=5, strides=1,
		                 activation='relu',
		                 input_shape=[self.n_dimensions,1]))
		self.model.add(MaxPooling1D(pool_size=2, strides=1))
		self.model.add(Dropout(0.25))
		self.model.add(Conv1D(self.number_of_hidden_unit_ltwo,3, activation='relu'))
		self.model.add(MaxPooling1D(pool_size=2))
		self.model.add(Dropout(0.25))
		self.model.add(Flatten())
		self.model.add(Dense(100, activation='relu'))
		self.model.add(Dense(n_classes, activation='softmax'))

		self.check_cb = keras.callbacks.ModelCheckpoint('checkpoints/' + self.id + '.hdf5',
                                             monitor='val_loss',
                                             verbose=0, save_best_only=True, mode='min')
		self.earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
		self.history = LossHistory()
		# self.optimizer = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)
		self.optimizer = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

		self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['binary_accuracy'])
		# self.model.compile(loss=keras.losses.categorical_crossentropy,
  #             optimizer=keras.optimizers.SGD(lr=0.01),
  #             metrics=['accuracy'])


	def fit(self,train_x,train_y, test_x, test_y):
		self.__initilize(train_x.shape[1],train_y.shape[1], "conv_200")
		# history = AccuracyHistory()
		train_x = np.expand_dims(train_x, axis=-1)
		test_x = np.expand_dims(test_x, axis=-1)
		test_y = np.eye(2)[test_y]
		self.model.fit(train_x, train_y,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=0,
          validation_data=(test_x, test_y),
          callbacks=[self.check_cb, self.earlystop_cb])

	def predict(self, x_test):
		x_test = np.expand_dims(x_test, axis=-1)
		y_pred =  self.model.predict(x_test, batch_size=self.batch_size, verbose=0, steps=None)
		return y_pred[:,-1],  np.argmax(y_pred,-1)


class fw_nn ():
	def __initilize(self, n_dimensions, n_classes, ID="fw_train"):
		self.id = ID
		self.learning_rate, self.number_of_hidden_unit_lone, self.number_of_hidden_unit_ltwo = learning_rate, number_of_hidden_unit_lone, number_of_hidden_unit_ltwo
		self.batch_size = 100
		self.epochs = 30
		self.n_classes = n_classes
		self.n_dimensions = n_dimensions
		
		self.model = Sequential()
		self.model.add(Dense(self.number_of_hidden_unit_lone, activation='relu',input_shape=[self.n_dimensions,]))
		self.model.add(Dense(self.number_of_hidden_unit_ltwo, activation='relu'))
		self.model.add(Dense(n_classes, activation='softmax'))

		self.check_cb = keras.callbacks.ModelCheckpoint('checkpoints/' + self.id + '.hdf5',
                                             monitor='val_loss',
                                             verbose=0, save_best_only=True, mode='min')
		self.earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
		self.history = LossHistory()
		# self.optimizer = keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)
		self.optimizer = keras.optimizers.Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

		self.model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['binary_accuracy'])
		# self.model.compile(loss=keras.losses.categorical_crossentropy,
  #             optimizer=keras.optimizers.SGD(lr=0.01),
  #             metrics=['accuracy'])


	def fit(self,train_x,train_y, test_x, test_y):
		self.__initilize(train_x.shape[1],train_y.shape[1], "fw_200")
		# history = AccuracyHistory()
		# train_x = np.expand_dims(train_x, axis=-1)
		# test_x = np.expand_dims(test_x, axis=-1)
		test_y = np.eye(2)[test_y]
		self.model.fit(train_x, train_y,
          batch_size=self.batch_size,
          epochs=self.epochs,
          verbose=0,
          validation_data=(test_x, test_y),
          callbacks=[self.check_cb, self.earlystop_cb])

	def predict(self, x_test):
		y_pred =  self.model.predict(x_test, batch_size=self.batch_size, verbose=0, steps=None)
		return y_pred[:,-1],  np.argmax(y_pred,-1)

class simple_NN ():

	def __initilize(self, n_dimensions, n_classes, ID="nn_train"):

		self.id = ID
		self.learning_rate, self.number_of_hidden_unit_lone, self.number_of_hidden_unit_ltwo = learning_rate, number_of_hidden_unit_lone, number_of_hidden_unit_ltwo
		self.batch_size = 100
		self.n_classes = n_classes
		self.n_dimensions = n_dimensions
		self.X = tf.placeholder(tf.float32,[None,n_dimensions])
		self.Y = tf.placeholder(tf.float32,[None,n_classes])
		self.training_epochs = 15
		self.weight_lone = self.__weight_variable([n_dimensions,self.number_of_hidden_unit_lone])
		self.bias_lone = self.__bias_variable([self.number_of_hidden_unit_lone])
		self.weight_ltwo = self.__weight_variable([self.number_of_hidden_unit_lone,self.number_of_hidden_unit_ltwo])
		self.bias_ltwo = self.__bias_variable([self.number_of_hidden_unit_ltwo])


		self.weight_lout = self.__weight_variable([self.number_of_hidden_unit_ltwo,n_classes])
		self.bias_lout = self.__bias_variable([n_classes])

		self.l1 = self.__fully_connected_relu(self.X,self.weight_lone,self.bias_lone, name = "layer1")
		self.l2 = self.__fully_connected_relu(self.l1,self.weight_ltwo,self.bias_ltwo, name = "layer2")
		self.logit_ = self.__fully_connected_logit(self.l2,self.weight_lout,self.bias_lout,True, name = "output")

		if d_type == 1:
			self.y_ = self.__fully_connected_softmax(self.l2,self.weight_lout,self.bias_lout,True, name = "output")
			self.loss_function = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits( logits=self.logit_, labels= self.Y)) 
		else:
			self.y_ = self.__fully_connected_sigmoid(self.l2,self.weight_lout,self.bias_lout,True, name = "output")
			self.loss_function = tf.reduce_sum( tf.nn.sigmoid_cross_entropy_with_logits( logits=self.logit_, labels= self.Y)) 

		
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss_function)

	def __weight_variable(self, shape, name = "output"):
		initial = tf.truncated_normal(shape, stddev = 1/np.sqrt(self.n_dimensions), name=name)
		return tf.Variable(initial,name=name)

	def __bias_variable(self, shape,name = "output"):
		initial = tf.constant(0.0, shape = shape, name = name)
		return tf.Variable(initial)

	def __fully_connected_sigmoid(self, x,weight,bias,return_logits = False, name = "output"):
		return tf.nn.sigmoid(tf.add(tf.matmul(x,weight),bias), name = name)

	def __fully_connected_softmax(self, x,weight,bias,return_logits = False, name = "output"):
		return tf.nn.softmax(tf.add(tf.matmul(x,weight),bias), name = name)

	def __fully_connected_logit(self, x,weight,bias,return_logits = False, name = "output"):
		return tf.add(tf.matmul(x,weight),bias)

	def __fully_connected_relu(self,x,weight,bias,name):
		return tf.nn.relu(tf.add(tf.matmul(x,weight),bias),name = name)


	def fit(self,train_x,train_y):

		''' Train a simple neural network'''
		
		total_batches = train_x.shape[0]
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		tf.reset_default_graph()
		self.__initilize(train_x.shape[1],train_y.shape[1], "nn_200")
		saver = tf.train.Saver()
		with tf.Session(config = config) as session:
			session.run(tf.global_variables_initializer())
			session.run(tf.local_variables_initializer())
			# print"--------- Epoch -----------"
			for epoch in range(self.training_epochs):
				cost_history = np.empty((0),dtype=float)
				for b in range(total_batches):	
					offset = (b * self.batch_size) % (train_y.shape[0] - self.batch_size)
					batch_x = train_x[offset:(offset + self.batch_size), :]
					batch_y = train_y[offset:(offset + self.batch_size), :]
					_, c = session.run([self.optimizer,self.loss_function],feed_dict = {self.X: batch_x, self.Y: batch_y})
					cost_history = np.append(cost_history,c)
					# print(c)

				# print("-------------------\n\n")
			try:
				os.mkdir( self.id )
			except:
				pass
			save_path = saver.save( session, '{0}/model'.format( self.id ) )
			# print( "Model saved in file: %s" % save_path )
		session.close()
				# print epoch, 
	

	def predict(self, test_x):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		saver = tf.train.import_meta_graph( "{0}/model.meta".format( self.id ) )
		
		with tf.Session(config = config) as session:
			saver.restore( session, tf.train.latest_checkpoint( '{0}'.format( self.id ) ) )
			y_pred = session.run(self.y_,feed_dict = { self.X: test_x })
			# print "\n\nROC AUC Score: ",roc_auc_score(test_y,y_pred)
			
			# y_pred = np.argmax(y_pred,-1)
			return y_pred[:,-1],  np.argmax(y_pred,-1)

def main():
	acronyms = []
	candidate_list = []
	all_candidates = []
	labels = []
	with open('Files/medstract_bioc_gold', 'rb') as infile:
		dataset = cp.load(infile)
		for x in dataset: 
			all_candidates.extend(x[3:])
		b_set = set(tuple(x) for x in all_candidates)
		all_candidates = [list(x) for x in b_set]

	all_candidates = np.array(all_candidates)
	X = np.array(all_candidates[:,:-1])#[:,[0,  1,  2, 4,6,7, 9,10, 14, 15]]
	Y = np.array(all_candidates[:,-1], dtype=np.int32)
	cor = np.corrcoef(X,Y, rowvar=False)[-1,:-1]
	# plot_bar_x(cor)
	# exit()
	#X = Normalizer().fit_transform(X)
	#X = MinMaxScaler().fit_transform(X)
	X = X[:,np.arange(cor.shape[0])[np.where(abs(cor) >=0.15 )]]

	print(Y.sum())
	print(Y.shape)
	#exit()

	n_split = 5
	nn_search = True
	sss = StratifiedKFold(n_splits=n_split, random_state=0)
	# sss = KFold(n_splits=n_split, random_state=0)

	classifiers = {"LG": LogisticRegression(), "GNB": GaussianNB(), "LSVM": LinearSVC(), "DT": DecisionTreeClassifier()}#, "Conv": conv_nn(), "FW": fw_nn()}# "Conv": conv_nn()}#, "Simple_NN": simple_NN ()}
	cv_split = [[train_index, test_index] for train_index, test_index in sss.split(X, Y)]
	fpr_arr, tpr_arr, roc_auc, pre_arr, rec_arr = dict(), dict(), dict(), dict(), dict()
	best_fpr, best_tpr, best_auc, best_auc_std, best_pr, best_rec, best_ag = dict(), dict(), dict(), dict(), dict(), dict(), dict()

	for classifier in classifiers:

		if classifier == "FW" or classifier == "Conv":
			if nn_search == True:
				learning_rate, number_of_hidden_unit_lone, number_of_hidden_unit_ltwo = generate_random_hyperparams(-3, -2, 150, 200, 50, 100)
				NN_search_iter = 10
			else:
				learning_rate, number_of_hidden_unit_lone, number_of_hidden_unit_ltwo = (94, 50, 0.0014782065507749993)
				NN_search_iter = 1
		else:
			learning_rate, number_of_hidden_unit_lone, number_of_hidden_unit_ltwo = 0,0,0
			NN_search_iter = 1

		start_t = time.time()
		print("Evaluating {} Algorithm ===>".format(classifier))
		best = [-1]*5
		best_config = []
		should_merge = True


		for i in range(NN_search_iter):
			
			eval_scores = np.zeros(5)
			iter = 0
			tprs = []
			recs = []
			aucs = []
			ag = 0
			mean_fpr = np.linspace(0, 1, 100)
			mean_pr = np.linspace(0, 1, 100)

			
			for train_index, test_index in cv_split:

				if should_merge:
					#print(len(test_index))
					#print(len(train_index))
					train_index = np.array(train_index)
					zero_ind = train_index[np.where(Y[train_index] == 0)]
					merge_ind = np.random.choice(zero_ind, size=int(0 * len(zero_ind)), replace=False)
					test_index =  np.hstack((test_index, merge_ind))
					train_index = np.setdiff1d(train_index, merge_ind)
					#print(len(test_index))
					#print(len(train_index))
					#exit()

				X_train, X_test = X[train_index], X[test_index]
				y_train, y_test = Y[train_index], Y[test_index]
				ids = np.arange(len(X_train))
				np.random.shuffle(ids)

				# shuffle
				X_train = X_train[ids]
				y_train = y_train[ids]
				# print(y_train.sum())
				# print(y_train.shape)
				if classifier == "FW" or classifier == "Conv":
					y_train = np.eye(2)[y_train]

				if classifier <> "FW" and classifier <> "Conv":
					classifiers[classifier].fit(X_train, y_train)
				else:
					classifiers[classifier].fit(X_train, y_train, X_test, y_test)
				predY = classifiers[classifier].predict(X_test)
				
				if classifier == "LSVM":
					probs = classifiers[classifier].decision_function(X_test)
				elif classifier <> "FW" and classifier <> "Conv":
					probs = classifiers[classifier].predict_proba(X_test)[:,-1]
				else:
					probs , predY= classifiers[classifier].predict(X_test)
					# predY = np.argmax(probs,-1)



				eval_scores += [precision_score(y_test, predY),recall_score(y_test, predY),f1_score(y_test,predY),accuracy_score(y_test,predY),roc_auc_score(y_test, probs)]
				fpr, tpr, _ = roc_curve(y_test, probs)
				pre, rec, _ = precision_recall_curve(y_test, probs)
				ag += ( average_precision_score (y_test, probs))


				tprs.append(interp(mean_fpr, fpr, tpr))
				# tprs[-1][0] = 0.0
				roc_auc = auc(fpr, tpr)
				aucs.append(roc_auc)

				recs.append(interp(mean_pr, pre, rec))
				# recs[0][-1] = 0.0
				iter += 1



			eval_scores /= n_split
			mean_tpr = np.mean(tprs, axis=0)
			mean_tpr[-1] = 1.0

			mean_recs = np.mean(recs, axis=0)
			ag /= n_split
			# mean_pr[-1] = 1.0

			mean_auc = auc(mean_fpr, mean_tpr)
			std_auc = np.std(aucs)
			# fpr_arr[classifier]  /= n_split
			# tpr_arr[classifier] /= n_split
			# roc_auc[classifier] /= n_split

			if best[2] < eval_scores[2]:
				best = eval_scores
				best_config = (learning_rate, number_of_hidden_unit_lone, number_of_hidden_unit_ltwo)
				best_fpr[classifier], best_tpr[classifier], best_auc[classifier], best_auc_std[classifier], best_pr[classifier], best_rec[classifier], best_ag[classifier] =  mean_fpr, mean_tpr, mean_auc, std_auc, mean_pr, mean_recs, ag
				

		print("Finished evaluating {} Algorithm".format(classifier))
		print("<====== Results ======>")
		print("Precicion = {} , Recall = {}, F1-Score = {}, Accuracy = {}, AUC = {}".format(best[0], best[1], best[2], best[3], best[4]))
		print("<==== Best Configuration ====>")
		print(best_config)

	plt.figure()
	lw = 2
	for classifier in classifiers:
		plt.plot(best_fpr[classifier], best_tpr[classifier],
		lw=lw, label='{0:s} Mean ROC AUC = {1:0.2f} $\pm$ {2:0.2f}'.format(classifier, best_auc[classifier], best_auc_std[classifier]))
	plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()

	


	# colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue'])

	colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red'])

	plt.figure()
	f_scores = np.linspace(0.2, 0.8, num=4)
	lines = []
	labels = []
	for f_score in f_scores:
	    x = np.linspace(0.01, 1)
	    y = f_score * x / (2 * x - f_score)
	    l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
	    plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

	lines.append(l)
	labels.append('iso-f1 curves')
	for classifier, color in zip(classifiers.keys(), colors):
	    l, = plt.plot(best_rec[classifier], best_pr[classifier], color=color, lw=2)
	    lines.append(l)
	    labels.append('Precision-recall for {0}, (area = {1:0.2f})'
	                  ''.format(classifier, best_ag[classifier]))

	fig = plt.gcf()
	fig.subplots_adjust(bottom=0.25)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.title('Precision-Recall curve of the various algorithms')
	plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))


	plt.show()

	cor = np.corrcoef(X,Y, rowvar=False)
	plot_bar_x(cor[-1,:-1])

	exit()

main()