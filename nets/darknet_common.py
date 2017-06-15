import numpy as np
import pickle
import os

# cfg
def parser(model):
	"""
	Read the .cfg file to extract layers into `layers`
	as well as model-specific parameters into `meta`
	"""
	def _parse(l, i = 1):
		return l.split('=')[i].strip()

	with open(model, 'rb') as f:
		lines = f.readlines()

	lines = [line.decode() for line in lines]	
	
	meta = dict(); layers = list() # will contains layers' info
	h, w, c = [int()] * 3; layer = dict()
	for line in lines:
		line = line.strip()
		line = line.split('#')[0]
		if '[' in line:
			if layer != dict(): 
				if layer['type'] == '[net]': 
					h = layer['height']
					w = layer['width']
					c = layer['channels']
					meta['net'] = layer
				else:
					if layer['type'] == '[crop]':
						h = layer['crop_height']
						w = layer['crop_width']
					layers += [layer]				
			layer = {'type': line}
		else:
			try:
				i = float(_parse(line))
				if i == int(i): i = int(i)
				layer[line.split('=')[0].strip()] = i
			except:
				try:
					key = _parse(line, 0)
					val = _parse(line, 1)
					layer[key] = val
				except:
					'banana ninja yadayada'

	meta.update(layer) # last layer contains meta info
	if 'anchors' in meta:
		splits = meta['anchors'].split(',')
		anchors = [float(x.strip()) for x in splits]
		meta['anchors'] = anchors
	meta['model'] = model # path to cfg, not model name
	meta['inp_size'] = [h, w, c]
	return layers, meta

def cfg_yielder(model, binary):
	"""
	yielding each layer information to initialize `layer`
	"""
	layers, meta = parser(model); yield meta;
	h, w, c = meta['inp_size']; l = w * h * c

	# Start yielding
	flat = False # flag for 1st dense layer
	conv = '.conv.' in model
	for i, d in enumerate(layers):
		#-----------------------------------------------------
		if d['type'] == '[crop]':
			yield ['crop', i]
		#-----------------------------------------------------
		elif d['type'] == '[local]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			activation = d.get('activation', 'logistic')
			w_ = (w - 1 - (1 - pad) * (size - 1)) // stride + 1
			h_ = (h - 1 - (1 - pad) * (size - 1)) // stride + 1
			yield ['local', i, size, c, n, stride, 
					pad, w_, h_, activation]
			if activation != 'linear': yield [activation, i]
			w, h, c = w_, h_, n
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[convolutional]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size // 2
			activation = d.get('activation', 'logistic')
			batch_norm = d.get('batch_normalize', 0) or conv
			yield ['convolutional', i, size, c, n, 
				   stride, padding, batch_norm,
				   activation]
			if activation != 'linear': yield [activation, i]
			w_ = (w + 2 * padding - size) // stride + 1
			h_ = (h + 2 * padding - size) // stride + 1
			w, h, c = w_, h_, n
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[maxpool]':
			stride = d.get('stride', 1)
			size = d.get('size', stride)
			padding = d.get('padding', (size-1) // 2)
			yield ['maxpool', i, size, stride, padding]
			w_ = (w + 2*padding) // d['stride'] 
			h_ = (h + 2*padding) // d['stride']
			w, h = w_, h_
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[avgpool]':
			flat = True; l = c
			yield ['avgpool', i]
		#-----------------------------------------------------
		elif d['type'] == '[softmax]':
			yield ['softmax', i, d['groups']]
		#-----------------------------------------------------
		elif d['type'] == '[connected]':
			if not flat:
				yield ['flatten', i]
				flat = True
			activation = d.get('activation', 'logistic')
			yield ['connected', i, l, d['output'], activation]
			if activation != 'linear': yield [activation, i]
			l = d['output']
		#-----------------------------------------------------
		elif d['type'] == '[dropout]': 
			yield ['dropout', i, d['probability']]
		#-----------------------------------------------------
		elif d['type'] == '[select]':
			if not flat:
				yield ['flatten', i]
				flat = True
			inp = d.get('input', None)
			if type(inp) is str:
				file = inp.split(',')[0]
				layer_num = int(inp.split(',')[1])
				with open(file, 'rb') as f:
					profiles = pickle.load(f, encoding = 'latin1')[0]
				layer = profiles[layer_num]
			else: layer = inp
			activation = d.get('activation', 'logistic')
			d['keep'] = d['keep'].split('/')
			classes = int(d['keep'][-1])
			keep = [int(c) for c in d['keep'][0].split(',')]
			keep_n = len(keep)
			train_from = classes * d['bins']
			for count in range(d['bins']-1):
				for num in keep[-keep_n:]:
					keep += [num + classes]
			k = 1
			while layers[i-k]['type'] not in ['[connected]', '[extract]']:
				k += 1
				if i-k < 0:
					break
			if i-k < 0: l_ = l
			elif layers[i-k]['type'] == 'connected':
				l_ = layers[i-k]['output']
			else:
				l_ = layers[i-k].get('old',[l])[-1]
			yield ['select', i, l_, d['old_output'],
				   activation, layer, d['output'], 
				   keep, train_from]
			if activation != 'linear': yield [activation, i]
			l = d['output']
		#-----------------------------------------------------
		elif d['type'] == '[conv-select]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size // 2
			activation = d.get('activation', 'logistic')
			batch_norm = d.get('batch_normalize', 0) or conv
			d['keep'] = d['keep'].split('/')
			classes = int(d['keep'][-1])
			keep = [int(x) for x in d['keep'][0].split(',')]

			segment = classes + 5
			assert n % segment == 0, \
			'conv-select: segment failed'
			bins = n // segment
			keep_idx = list()
			for j in range(bins):
				offset = j * segment
				for k in range(5):
					keep_idx += [offset + k]
				for k in keep:
					keep_idx += [offset + 5 + k]
			w_ = (w + 2 * padding - size) // stride + 1
			h_ = (h + 2 * padding - size) // stride + 1
			c_ = len(keep_idx)
			yield ['conv-select', i, size, c, n, 
				   stride, padding, batch_norm,
				   activation, keep_idx, c_]
			w, h, c = w_, h_, c_
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[conv-extract]':
			file = d['profile']
			with open(file, 'rb') as f:
				profiles = pickle.load(f, encoding = 'latin1')[0]
			inp_layer = None
			inp = d['input']
			out = d['output']
			inp_layer = None
			if inp >= 0:
				inp_layer = profiles[inp]
			if inp_layer is not None:
				assert len(inp_layer) == c, \
				'Conv-extract does not match input dimension'
			out_layer = profiles[out]

			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size // 2
			activation = d.get('activation', 'logistic')
			batch_norm = d.get('batch_normalize', 0) or conv
			
			k = 1
			find = ['[convolutional]','[conv-extract]']
			while layers[i-k]['type'] not in find:
				k += 1
				if i-k < 0: break
			if i-k >= 0:
				previous_layer = layers[i-k]
				c_ = previous_layer['filters']
			else:
				c_ = c
			
			yield ['conv-extract', i, size, c_, n, 
				   stride, padding, batch_norm,
				   activation, inp_layer, out_layer]
			if activation != 'linear': yield [activation, i]
			w_ = (w + 2 * padding - size) // stride + 1
			h_ = (h + 2 * padding - size) // stride + 1
			w, h, c = w_, h_, len(out_layer)
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[extract]':
			if not flat:
				yield['flatten', i]
				flat = True
			activation = d.get('activation', 'logistic')
			file = d['profile']
			with open(file, 'rb') as f:
				profiles = pickle.load(f, encoding = 'latin1')[0]
			inp_layer = None
			inp = d['input']
			out = d['output']
			if inp >= 0:
				inp_layer = profiles[inp]
			out_layer = profiles[out]
			old = d['old']
			old = [int(x) for x in old.split(',')]
			if inp_layer is not None:
				if len(old) > 2: 
					h_, w_, c_, n_ = old
					new_inp = list()
					for p in range(c_):
						for q in range(h_):
							for r in range(w_):
								if p not in inp_layer:
									continue
								new_inp += [r + w*(q + h*p)]
					inp_layer = new_inp
					old = [h_ * w_ * c_, n_]
				assert len(inp_layer) == l, \
				'Extract does not match input dimension'
			d['old'] = old
			yield ['extract', i] + old + [activation] + [inp_layer, out_layer]
			if activation != 'linear': yield [activation, i]
			l = len(out_layer)
		#-----------------------------------------------------
		elif d['type'] == '[route]': # add new layer here
			routes = d['layers']
			if type(routes) is int:
				routes = [routes]
			else:
				routes = [int(x.strip()) for x in routes.split(',')]
			routes = [i + x if x < 0 else x for x in routes]
			for j, x in enumerate(routes):
				lx = layers[x]; 
				xtype = lx['type']
				_size = lx['_size'][:3]
				if j == 0:
					h, w, c = _size
				else: 
					h_, w_, c_ = _size
					assert w_ == w and h_ == h, \
					'Routing incompatible conv sizes'
					c += c_
			yield ['route', i, routes]
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[reorg]':
			stride = d.get('stride', 1)
			yield ['reorg', i, stride]
			w = w // stride; h = h // stride; 
			c = c * (stride ** 2)
			l = w * h * c
		#-----------------------------------------------------
		else:
			exit('Layer {} not implemented'.format(d['type']))

		d['_size'] = list([h, w, c, l, flat])

	if not flat: meta['out_size'] = [h, w, c]
	else: meta['out_size'] = l

# Layers 
class Layer(object):

    def __init__(self, *args):
        self._signature = list(args)
        self.type = list(args)[0]
        self.number = list(args)[1]

        self.w = dict() # weights
        self.h = dict() # placeholders
        self.wshape = dict() # weight shape
        self.wsize = dict() # weight size
        self.setup(*args[2:]) # set attr up
        self.present()
        for var in self.wshape:
            shp = self.wshape[var]
            size = np.prod(shp)
            self.wsize[var] = size

    def load(self, src_loader):
        var_lay = src_loader.VAR_LAYER
        if self.type not in var_lay: return

        src_type = type(src_loader)
        if src_type is weights_loader:
            wdict = self.load_weights(src_loader)
        else: 
            wdict = self.load_ckpt(src_loader)
        if wdict is not None:
            self.recollect(wdict)

    def load_weights(self, src_loader):
        val = src_loader([self.presenter])
        if val is None: return None
        else: return val.w

    def load_ckpt(self, src_loader):
        result = dict()
        presenter = self.presenter
        for var in presenter.wshape:
            name = presenter.varsig(var)
            shape = presenter.wshape[var]
            key = [name, shape]
            val = src_loader(key)
            result[var] = val
        return result

    @property
    def signature(self):
        return self._signature

    # For comparing two layers
    def __eq__(self, other):
        return self.signature == other.signature
    def __ne__(self, other):
        return not self.__eq__(other)

    def varsig(self, var):
        if var not in self.wshape:
            return None
        sig = str(self.number)
        sig += '-' + self.type
        sig += '/' + var
        return sig

    def recollect(self, w): self.w = w
    def present(self): self.presenter = self
    def setup(self, *args): pass
    def finalize(self): pass 

class local_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, w_, h_, activation):
        self.pad = pad * int(ksize / 2)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.h_out = h_
        self.w_out = w_

        self.dnshape = [h_ * w_, n, c, ksize, ksize]
        self.wshape = dict({
            'biases': [h_ * w_ * n],
            'kernels': [h_ * w_, ksize, ksize, c, n]
        })

    def finalize(self, _):
        weights = self.w['kernels']
        if weights is None: return
        weights = weights.reshape(self.dnshape)
        weights = weights.transpose([0,3,4,2,1])
        self.w['kernels'] = weights

class conv_extract_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, batch_norm, activation,
              inp, out):
        if inp is None: inp = range(c)
        self.activation = activation
        self.batch_norm = batch_norm
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.inp = inp
        self.out = out
        self.wshape = dict({
            'biases': [len(out)], 
            'kernel': [ksize, ksize, len(inp), len(out)]
        })

    @property
    def signature(self):
        sig = ['convolutional']
        sig += self._signature[1:-2]
        return sig

    def present(self):
        args = self.signature
        self.presenter = convolutional_layer(*args)

    def recollect(self, w):
        if w is None:
            self.w = w
            return
        k = w['kernel']
        b = w['biases']
        k = np.take(k, self.inp, 2)
        k = np.take(k, self.out, 3)
        b = np.take(b, self.out)
        assert1 = k.shape == tuple(self.wshape['kernel'])
        assert2 = b.shape == tuple(self.wshape['biases'])
        assert assert1 and assert2, \
        'Dimension not matching in {} recollect'.format(
            self._signature)
        self.w['kernel'] = k
        self.w['biases'] = b


class conv_select_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, batch_norm, activation,
              keep_idx, real_n):
        self.batch_norm = bool(batch_norm)
        self.activation = activation
        self.keep_idx = keep_idx
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.wshape = dict({
            'biases': [real_n], 
            'kernel': [ksize, ksize, c, real_n]
        })
        if self.batch_norm:
            self.wshape.update({
                'moving_variance'  : [real_n], 
                'moving_mean': [real_n], 
                'gamma' : [real_n]
            })
            self.h['is_training'] = {
                'shape': (),
                'feed': True,
                'dfault': False
            }

    @property
    def signature(self):
        sig = ['convolutional']
        sig += self._signature[1:-2]
        return sig
    
    def present(self):
        args = self.signature
        self.presenter = convolutional_layer(*args)

    def recollect(self, w):
        if w is None:
            self.w = w
            return
        idx = self.keep_idx
        k = w['kernel']
        b = w['biases']
        self.w['kernel'] = np.take(k, idx, 3) 
        self.w['biases'] = np.take(b, idx)
        if self.batch_norm:
            m = w['moving_mean']
            v = w['moving_variance']
            g = w['gamma']
            self.w['moving_mean'] = np.take(m, idx)
            self.w['moving_variance'] = np.take(v, idx)
            self.w['gamma'] = np.take(g, idx)

class convolutional_layer(Layer):
    def setup(self, ksize, c, n, stride, 
              pad, batch_norm, activation):
        self.batch_norm = bool(batch_norm)
        self.activation = activation
        self.stride = stride
        self.ksize = ksize
        self.pad = pad
        self.dnshape = [n, c, ksize, ksize] # darknet shape
        self.wshape = dict({
            'biases': [n], 
            'kernel': [ksize, ksize, c, n]
        })
        if self.batch_norm:
            self.wshape.update({
                'moving_variance'  : [n], 
                'moving_mean': [n], 
                'gamma' : [n]
            })
            self.h['is_training'] = {
                'feed': True,
                'dfault': False,
                'shape': ()
            }

    def finalize(self, _):
        """deal with darknet"""
        kernel = self.w['kernel']
        if kernel is None: return
        kernel = kernel.reshape(self.dnshape)
        kernel = kernel.transpose([2,3,1,0])
        self.w['kernel'] = kernel

class extract_layer(Layer):
    def setup(self, old_inp, old_out,
              activation, inp, out):
        if inp is None: inp = range(old_inp)
        self.activation = activation
        self.old_inp = old_inp
        self.old_out = old_out
        self.inp = inp
        self.out = out
        self.wshape = {
            'biases': [len(self.out)],
            'weights': [len(self.inp), len(self.out)]
        }

    @property
    def signature(self):
        sig = ['connected']
        sig += self._signature[1:-2]
        return sig

    def present(self):
        args = self.signature
        self.presenter = connected_layer(*args)

    def recollect(self, val):
        w = val['weights']
        b = val['biases']
        if w is None: self.w = val; return
        w = np.take(w, self.inp, 0)
        w = np.take(w, self.out, 1)
        b = np.take(b, self.out)
        assert1 = w.shape == tuple(self.wshape['weights'])
        assert2 = b.shape == tuple(self.wshape['biases'])
        assert assert1 and assert2, \
        'Dimension does not match in {} recollect'.format(
            self._signature)
        
        self.w['weights'] = w
        self.w['biases'] = b

class select_layer(Layer):
    def setup(self, inp, old, 
              activation, inp_idx,
              out, keep, train):
        self.old = old
        self.keep = keep
        self.train = train
        self.inp_idx = inp_idx
        self.activation = activation
        inp_dim = inp
        if inp_idx is not None:
            inp_dim = len(inp_idx)
        self.inp = inp_dim
        self.out = out
        self.wshape = {
            'biases': [out],
            'weights': [inp_dim, out]
        }

    @property
    def signature(self):
        sig = ['connected']
        sig += self._signature[1:-4]
        return sig

    def present(self):
        args = self.signature
        self.presenter = connected_layer(*args)

    def recollect(self, val):
        w = val['weights']
        b = val['biases']
        if w is None: self.w = val; return
        if self.inp_idx is not None:
            w = np.take(w, self.inp_idx, 0)
            
        keep_b = np.take(b, self.keep)
        keep_w = np.take(w, self.keep, 1)
        train_b = b[self.train:]
        train_w = w[:, self.train:]
        self.w['biases'] = np.concatenate(
            (keep_b, train_b), axis = 0)
        self.w['weights'] = np.concatenate(
            (keep_w, train_w), axis = 1)


class connected_layer(Layer):
    def setup(self, input_size, 
              output_size, activation):
        self.activation = activation
        self.inp = input_size
        self.out = output_size
        self.wshape = {
            'biases': [self.out],
            'weights': [self.inp, self.out]
        }

    def finalize(self, transpose):
        weights = self.w['weights']
        if weights is None: return
        shp = self.wshape['weights']
        if not transpose:
            weights = weights.reshape(shp[::-1])
            weights = weights.transpose([1,0])
        else: weights = weights.reshape(shp)
        self.w['weights'] = weights

class avgpool_layer(Layer):
    pass

class crop_layer(Layer):
    pass

class maxpool_layer(Layer):
    def setup(self, ksize, stride, pad):
        self.stride = stride
        self.ksize = ksize
        self.pad = pad

class softmax_layer(Layer):
    def setup(self, groups):
        self.groups = groups

class dropout_layer(Layer):
    def setup(self, p):
        self.h['pdrop'] = dict({
            'feed': p, # for training
            'dfault': 1.0, # for testing
            'shape': ()
        })

class route_layer(Layer):
    def setup(self, routes):
        self.routes = routes

class reorg_layer(Layer):
    def setup(self, stride):
        self.stride = stride

"""
Darkop Factory
"""

darkops = {
    'dropout': dropout_layer,
    'connected': connected_layer,
    'maxpool': maxpool_layer,
    'convolutional': convolutional_layer,
    'avgpool': avgpool_layer,
    'softmax': softmax_layer,
    'crop': crop_layer,
    'local': local_layer,
    'select': select_layer,
    'route': route_layer,
    'reorg': reorg_layer,
    'conv-select': conv_select_layer,
    'conv-extract': conv_extract_layer,
    'extract': extract_layer
}

def create_darkop(ltype, num, *args):
    op_class = darkops.get(ltype, Layer)
    return op_class(ltype, num, *args)

# ops
FORM = '{:>6} | {:>6} | {:<32} | {}'
FORM_ = '{}+{}+{}+{}'
LINE = FORM_.format('-'*7, '-'*8, '-'*34, '-'*15) 
HEADER = FORM.format(
    'Source', 'Train?','Layer description', 'Output size')

def _shape(tensor): # work for both tf.Tensor & np.ndarray
    if type(tensor) in [tf.Variable, tf.Tensor]: 
        return tensor.get_shape()
    else: return tensor.shape

def _name(tensor):
    return tensor.name.split(':')[0]

class BaseOp(object):
    """
    BaseOp objects initialise with a darknet's `layer` object
    and input tensor of that layer `inp`, it calculates the 
    output of this layer and place the result in self.out
    """

    # let slim take care of the following vars
    _SLIM = ['gamma', 'moving_mean', 'moving_variance']

    def __init__(self, layer, inp, num, roof, feed):
        self.inp = inp # BaseOp
        self.num = num # int
        self.out = None # tf.Tensor
        self.lay = layer

        self.scope = '{}-{}'.format(
            str(self.num), self.lay.type)
        self.gap = roof - self.num
        self.var = not self.gap > 0
        self.act = 'Load '
        self.convert(feed)
        if self.var: self.train_msg = 'Yep! '
        else: self.train_msg = 'Nope '
        self.forward()

    def convert(self, feed):
        """convert self.lay to variables & placeholders"""
        for var in self.lay.wshape:
            self.wrap_variable(var)
        for ph in self.lay.h:
            self.wrap_pholder(ph, feed)

    def wrap_variable(self, var):
        """wrap layer.w into variables"""
        val = self.lay.w.get(var, None)
        if val is None:
            shape = self.lay.wshape[var]
            args = [0., 1e-2, shape]
            if 'moving_mean' in var:
                val = np.zeros(shape)
            elif 'moving_variance' in var:
                val = np.ones(shape)
            else:
                val = np.random.normal(*args)
            self.lay.w[var] = val.astype(np.float32)
            self.act = 'Init '
        if not self.var: return

        val = self.lay.w[var]
        self.lay.w[var] = tf.constant_initializer(val)
        if var in self._SLIM: return
        with tf.variable_scope(self.scope):
            self.lay.w[var] = tf.get_variable(var,
                shape = self.lay.wshape[var],
                dtype = tf.float32,
                initializer = self.lay.w[var])

    def wrap_pholder(self, ph, feed):
        """wrap layer.h into placeholders"""
        phtype = type(self.lay.h[ph])
        if phtype is not dict: return

        sig = '{}/{}'.format(self.scope, ph)
        val = self.lay.h[ph]

        self.lay.h[ph] = tf.placeholder_with_default(
            val['dfault'], val['shape'], name = sig)
        feed[self.lay.h[ph]] = val['feed']

    def verbalise(self): # console speaker
        msg = str()
        inp = _name(self.inp.out)
        if inp == 'input': \
        msg = FORM.format(
            '', '', 'input',
            _shape(self.inp.out)) + '\n'
        if not self.act: return msg
        return msg + FORM.format(
            self.act, self.train_msg, 
            self.speak(), _shape(self.out))
    
    def speak(self): pass

class reorg(BaseOp):
    def _forward(self):
        inp = self.inp.out
        shape = inp.get_shape().as_list()
        _, h, w, c = shape
        s = self.lay.stride
        out = list()
        for i in range(int(h/s)):
            row_i = list()
            for j in range(int(w/s)):
                si, sj = s * i, s * j
                boxij = inp[:, si: si+s, sj: sj+s,:]
                flatij = tf.reshape(boxij, [-1,1,1,c*s*s])
                row_i += [flatij]
            out += [tf.concat(row_i, 2)]

        self.out = tf.concat(out, 1)

    def forward(self):
        inp = self.inp.out
        s = self.lay.stride
        self.out = tf.extract_image_patches(
            inp, [1,s,s,1], [1,s,s,1], [1,1,1,1], 'VALID')

    def speak(self):
        args = [self.lay.stride] * 2
        msg = 'local flatten {}x{}'
        return msg.format(*args)


class local(BaseOp):
    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])

        k = self.lay.w['kernels']
        ksz = self.lay.ksize
        half = int(ksz / 2)
        out = list()
        for i in range(self.lay.h_out):
            row_i = list()
            for j in range(self.lay.w_out):
                kij = k[i * self.lay.w_out + j]
                i_, j_ = i + 1 - half, j + 1 - half
                tij = temp[:, i_ : i_ + ksz, j_ : j_ + ksz,:]
                row_i.append(
                    tf.nn.conv2d(tij, kij, 
                        padding = 'VALID', 
                        strides = [1] * 4))
            out += [tf.concat(row_i, 2)]

        self.out = tf.concat(out, 1)

    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.activation]
        msg = 'loca {}x{}p{}_{}  {}'.format(*args)
        return msg

class convolutional(BaseOp):
    def forward(self):
        pad = [[self.lay.pad, self.lay.pad]] * 2;
        temp = tf.pad(self.inp.out, [[0, 0]] + pad + [[0, 0]])
        temp = tf.nn.conv2d(temp, self.lay.w['kernel'], padding = 'VALID', 
            name = self.scope, strides = [1] + [self.lay.stride] * 2 + [1])
        if self.lay.batch_norm: 
            temp = self.batchnorm(self.lay, temp)
        self.out = tf.nn.bias_add(temp, self.lay.w['biases'])

    def batchnorm(self, layer, inp):
        if not self.var:
            temp = (inp - layer.w['moving_mean'])
            temp /= (np.sqrt(layer.w['moving_variance']) + 1e-5)
            temp *= layer.w['gamma']
            return temp
        else:
            args = dict({
                'center' : False, 'scale' : True,
                'epsilon': 1e-5, 'scope' : self.scope,
                'updates_collections' : None,
                'is_training': layer.h['is_training'],
                'param_initializers': layer.w
                })
            return slim.batch_norm(inp, **args)

    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'conv {}x{}p{}_{}  {}  {}'.format(*args)
        return msg

class conv_select(convolutional):
    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'sele {}x{}p{}_{}  {}  {}'.format(*args)
        return msg

class conv_extract(convolutional):
    def speak(self):
        l = self.lay
        args = [l.ksize] * 2 + [l.pad] + [l.stride]
        args += [l.batch_norm * '+bnorm']
        args += [l.activation]
        msg = 'extr {}x{}p{}_{}  {}  {}'.format(*args)
        return msg

class route(BaseOp):
	def forward(self):
		routes = self.lay.routes
		routes_out = list()
		for r in routes:
			this = self.inp
			while this.lay.number != r:
				this = this.inp
				assert this is not None, \
				'Routing to non-existence {}'.format(r)
			routes_out += [this.out]
		self.out = tf.concat(routes_out, 3)

	def speak(self):
		msg = 'concat {}'
		return msg.format(self.lay.routes)

class connected(BaseOp):
	def forward(self):
		self.out = tf.nn.xw_plus_b(
			self.inp.out,
			self.lay.w['weights'], 
			self.lay.w['biases'], 
			name = self.scope)

	def speak(self):
		layer = self.lay
		args = [layer.inp, layer.out]
		args += [layer.activation]
		msg = 'full {} x {}  {}'
		return msg.format(*args)

class select(connected):
	"""a weird connected layer"""
	def speak(self):
		layer = self.lay
		args = [layer.inp, layer.out]
		args += [layer.activation]
		msg = 'sele {} x {}  {}'
		return msg.format(*args)

class extract(connected):
	"""a weird connected layer"""
	def speak(self):
		layer = self.lay
		args = [len(layer.inp), len(layer.out)]
		args += [layer.activation]
		msg = 'extr {} x {}  {}'
		return msg.format(*args)

class flatten(BaseOp):
	def forward(self):
		temp = tf.transpose(
			self.inp.out, [0,3,1,2])
		self.out = slim.flatten(
			temp, scope = self.scope)

	def speak(self): return 'flat'


class softmax(BaseOp):
	def forward(self):
		self.out = tf.nn.softmax(self.inp.out)

	def speak(self): return 'softmax()'


class avgpool(BaseOp):
	def forward(self):
		self.out = tf.reduce_mean(
			self.inp.out, [1, 2], 
			name = self.scope
		)

	def speak(self): return 'avgpool()'


class dropout(BaseOp):
	def forward(self):
		if self.lay.h['pdrop'] is None:
			self.lay.h['pdrop'] = 1.0
		self.out = tf.nn.dropout(
			self.inp.out, 
			self.lay.h['pdrop'], 
			name = self.scope
		)

	def speak(self): return 'drop'


class crop(BaseOp):
	def forward(self):
		self.out =  self.inp.out * 2. - 1.

	def speak(self):
		return 'scale to (-1, 1)'


class maxpool(BaseOp):
	def forward(self):
		self.out = tf.nn.max_pool(
			self.inp.out, padding = 'SAME',
	        ksize = [1] + [self.lay.ksize]*2 + [1], 
	        strides = [1] + [self.lay.stride]*2 + [1],
	        name = self.scope
	    )
	
	def speak(self):
		l = self.lay
		return 'maxp {}x{}p{}_{}'.format(
			l.ksize, l.ksize, l.pad, l.stride)


class leaky(BaseOp):
	def forward(self):
		self.out = tf.maximum(
			.1 * self.inp.out, 
			self.inp.out, 
			name = self.scope
		)

	def verbalise(self): pass


class identity(BaseOp):
	def __init__(self, inp):
		self.inp = None
		self.out = inp

op_types = {
	'convolutional': convolutional,
	'conv-select': conv_select,
	'connected': connected,
	'maxpool': maxpool,
	'leaky': leaky,
	'dropout': dropout,
	'flatten': flatten,
	'avgpool': avgpool,
	'softmax': softmax,
	'identity': identity,
	'crop': crop,
	'local': local,
	'select': select,
	'route': route,
	'reorg': reorg,
	'conv-extract': conv_extract,
	'extract': extract
}

def op_create(*args):
	layer_type = list(args)[0].type
	return op_types[layer_type](*args)

# loader
class loader(object):
    """
    interface to work with both .weights and .ckpt files
    in loading / recollecting / resolving mode
    """
    VAR_LAYER = ['convolutional', 'connected', 'local', 
                 'select', 'conv-select',
                 'extract', 'conv-extract']

    def __init__(self, *args):
        self.src_key = list()
        self.vals = list()
        self.load(*args)

    def __call__(self, key):
        for idx in range(len(key)):
            val = self.find(key, idx)
            if val is not None: return val
        return None
    
    def find(self, key, idx):
        up_to = min(len(self.src_key), 4)
        for i in range(up_to):
            key_b = self.src_key[i]
            if key_b[idx:] == key[idx:]:
                return self.yields(i)
        return None

    def yields(self, idx):
        del self.src_key[idx]
        temp = self.vals[idx]
        del self.vals[idx]
        return temp

class weights_loader(loader):
    """one who understands .weights files"""
    
    _W_ORDER = dict({ # order of param flattened into .weights file
        'convolutional': [
            'biases','gamma','moving_mean','moving_variance','kernel'
        ],
        'connected': ['biases', 'weights'],
        'local': ['biases', 'kernels']
    })

    def load(self, path, src_layers):
        self.src_layers = src_layers
        walker = weights_walker(path)

        for i, layer in enumerate(src_layers):
            if layer.type not in self.VAR_LAYER: continue
            self.src_key.append([layer])
            
            if walker.eof: new = None
            else: 
                args = layer.signature
                new = create_darkop(*args)
            self.vals.append(new)

            if new is None: continue
            order = self._W_ORDER[new.type]
            for par in order:
                if par not in new.wshape: continue
                val = walker.walk(new.wsize[par])
                new.w[par] = val
            new.finalize(walker.transpose)

        if walker.path is not None:
            assert walker.offset == walker.size, \
            'expect {} bytes, found {}'.format(
                walker.offset, walker.size)
            print('Successfully identified {} bytes'.format(
                walker.offset))

class checkpoint_loader(loader):
    """
    one who understands .ckpt files, very much
    """
    def load(self, ckpt, ignore):
        meta = ckpt + '.meta'
        with tf.Graph().as_default() as graph:
            with tf.Session().as_default() as sess:
                saver = tf.train.import_meta_graph(meta)
                saver.restore(sess, ckpt)
                for var in tf.global_variables():
                    name = var.name.split(':')[0]
                    packet = [name, var.get_shape().as_list()]
                    self.src_key += [packet]
                    self.vals += [var.eval(sess)]

def create_loader(path, cfg = None):
    if path is None:
        load_type = weights_loader
    elif '.weights' in path:
        load_type = weights_loader
    else: 
        load_type = checkpoint_loader
    
    return load_type(path, cfg)

class weights_walker(object):
    """incremental reader of float32 binary files"""
    def __init__(self, path):
        self.eof = False # end of file
        self.path = path  # current pos
        if path is None: 
            self.eof = True
            return
        else: 
            self.size = os.path.getsize(path)# save the path
            major, minor, revision, seen = np.memmap(path,
                shape = (), mode = 'r', offset = 0,
                dtype = '({})i4,'.format(4))
            self.transpose = major > 1000 or minor > 1000
            self.offset = 16

    def walk(self, size):
        if self.eof: return None
        end_point = self.offset + 4 * size
        assert end_point <= self.size, \
        'Over-read {}'.format(self.path)

        float32_1D_array = np.memmap(
            self.path, shape = (), mode = 'r', 
            offset = self.offset,
            dtype='({})float32,'.format(size)
        )

        self.offset = end_point
        if end_point == self.size: 
            self.eof = True
        return float32_1D_array

def model_name(file_path):
    file_name = basename(file_path)
    ext = str()
    if '.' in file_name: # exclude extension
        file_name = file_name.split('.')
        ext = file_name[-1]
        file_name = '.'.join(file_name[:-1])
    if ext == str() or ext == 'meta': # ckpt file
        file_name = file_name.split('-')
        num = int(file_name[-1])
        return '-'.join(file_name[:-1])
    if ext == 'weights':
        return file_name

