from theano import tensor as T
from theano.tensor.nnet import conv
import lasagne


#TODO: debug this layer, make sure initialization makes sense
class HexConvLayer(lasagne.layers.Layer):
    def __init__(
        self, 
        incoming, 
        num_filters, 
        radius, 
        nonlinearity = lasagne.nonlinearity.rectify, 
        W=lasagne.init.HeNormal(gain='relu'), 
        b=lasagne.init.Constant(0),
        padding = 0,
         **kwargs):
            super(HexConvLayer, self).__init__(incoming, **kwargs)
            self.num_filters = num_filters
            self.radius = radius
            self.padding = padding
            self.nonlinearity = nonlinearity
            W_size = 2*sum([i+radius for i in range(radius-1)])+2*radius-1
            self.W_values = self.add_param(W, (num_filters, self.input_shape[1], W_size), name='W')
            self.b = self.add_param(b, (num_filters), name='b')

    def get_output_for(self, input, **kwargs):
        W = T.zeros((self.num_filters, self.input_shape[1], 2*self.radius-1, 2*self.radius-1))
        index=0
        for i in range(self.radius-1):
            W = T.set_subtensor(W[:,:,self.radius-1-i:,i], self.W_values[:,:,index:index+self.radius+i])
            index = index+self.radius+i
        W = T.set_subtensor(W[:,:,:,self.radius], self.W[:,:,index:index+2*self.radius-1])
        index = index+2*self.radius-1
        for i in range(self.radius-1):
            W = T.set_subtensor(W[:,:,:2*(self.radius-1)-i],self.W_values[:,:,index:index+2*self.radius-i])
            index = index+2*self.radius-i

        conv_out = conv.conv2d(
            input = input,
            filters = W,
            filter_shape = (self.num_filters,self.input_shape[1],2*self.radius-1,2*self.radius-1),
            image_shape = self.input_shape
        )

        squished_out = self.nonlinearity(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        if(self.padding == 0):
            padded_out = squished_out
        else:
            padded_out = T.zeros((squished_out.shape[0], self.num_filters, self.input_shape[2], self.input_shape[3]))
            padded_out = T.set_subtensor(padded_out[:,:,self.padding:-self.padding,self.padding:-self.padding], squished_out)

        return padded_out

    def get_output_shape_for(self, input_shape):
        return (self.num_filters, self.input_shape[1], self.input_shape[2], self.input_shape[3])
