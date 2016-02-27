#	encode: utf8

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F
from chainer.functions.loss import mean_squared_error

class AE(chainer.Chain):
	def __init__(self, dims, train=True, noise=0.0, activation=None):
		super(AE, self).__init__(
			enc=L.Linear(dims[0], dims[1]),
			dec=L.Linear(dims[1], dims[0])
		)

		self.train = train
		self.noise = noise
		self.activation = activation

	def __call__(self, x, t=None):
		if t is not None:
			#	Noise x
			t = chainer.Variable(
				np.random.normal(x.data, self.noise).astype(np.float32), 
				volatile=x.volatile)

		#	Encode and Decode
		self.y = F.relu(self.enc(F.dropout(x, train=self.train)))
		if t is None:
			self.t = self.dec(F.dropout(self.y, train=self.train))
		else:
			self.t = self.activation(self.dec(F.dropout(self.y, train=self.train)))

		self.loss = F.mean_squared_error(t, self.t)

		return self.loss

class MLAE(chainer.Chain):
	"""
		Multi-layer Auto-Encoder
	"""
	def __init__(self, dims, train=True, noise=0.0, activation=None):
		super(MLAE, self).__init__(
			enc1=L.Linear(dims[0], dims[1]),
			enc2=L.Linear(dims[1], dims[2]),
			dec1=L.Linear(dims[2], dims[3]),
			dec2=L.Linear(dims[3], dims[0])
		)

		self.train = train
		self.noise = noise
		self.activation = activation

	def encode(self, x, train=True):
		h1 = F.relu(self.enc1(F.dropout(x, train=train)))
		y = F.relu(self.enc2(F.dropout(h1, train=train)))

		return y

	def decode(self, y, train=True):
		h1 = F.relu(self.dec1(F.dropout(y, train=train)))
		if self.activation is None:
			t = self.dec2(F.dropout(h1, train=train))
		else:
			t = self.activation(self.dec2(F.dropout(h1, train=train)))

		return t

	def __call__(self, x, t = None):
		if t is not None:
			#	Noise x
			t = chainer.Variable(
				np.random.normal(x.data, self.noise).astype(np.float32), 
				volatile=x.volatile)

		#	Encode and Decode
		self.y = self.encode(x, self.train)
		self.t = self.decode(self.y, self.train)

		self.loss = F.mean_squared_error(t, self.t)

		return self.loss

class RAE(chainer.Chain):
	"""
		Recurrsive Auto-Encoder
	"""
	def __init__(self, dims, train=True, noise=0.0, activation=None):
		super(RAE, self).__init__(
			enc=L.LSTM(dims[0], dims[1]),
			dec=L.Linear(dims[1], dims[0])
		)

		self.train = train
		self.noise = noise
		self.activation = activation

	def __call__(self, x, t=None):
		if t is not None:
			#	Noise x
			t = chainer.Variable(
				np.random.normal(x.data, self.noise).astype(np.float32), 
				volatile=x.volatile)

		self.y = self.enc(F.dropout(x, train=self.train))

		if self.activation is None:
			self.t = self.dec(F.dropout(self.y, train=self.train))
		else:
			self.t = self.activation(self.dec(F.dropout(self.y, train=self.train)))

		self.loss = F.mean_squared_error(t, self.t)

		return self.loss

class RMLAE(chainer.Chain):
	"""
		Recurrsive Multi-layer Auto-Encoder
	"""
	def __init__(self, dims, train=True, noise=0.0, activation=None):
		super(RMLAE, self).__init__(
			enc1=L.LSTM(dims[0], dims[1]),
			enc2=L.LSTM(dims[1], dims[2]),
			dec1=L.Linear(dims[2], dims[3]),
			dec2=L.Linear(dims[3], dims[0])
		)

		self.train = train
		self.noise = noise
		self.activation = activation

	def reset_state(self):
		self.enc1.reset_state()
		self.enc2.reset_state()

	def encode(self, x, train=True):
		h1 = self.enc1(F.dropout(x, train=train))
		y = self.enc2(F.dropout(h1, train=train))

		return y

	def decode(self, y, train=True):
		h1 = F.relu(self.dec1(F.dropout(y, train=train)))

		if self.activation is None:
			t = self.dec2(F.dropout(h1, train=train))
		else:
			t = self.activation(self.dec2(F.dropout(h1, train=train)))

		return t

	def __call__(self, x, t=None):
		if t is not None:
			#	Noise x
			t = chainer.Variable(
				np.random.normal(x.data, self.noise).astype(np.float32), 
				volatile=x.volatile)

		#	Encode and Decode
		self.y = self.encode(x, self.train)
		self.t = self.decode(self.y, self.train)

		self.loss = F.mean_squared_error(t, self.t)

		return self.loss

		