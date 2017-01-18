from theano import tensor as T
from theano.tensor.nnet import conv
import lasagne


#TODO: this should be thoroughly validated, I'm sure it has a high bud probability
class HexConvLayer(lasagne.layers.Layer):
    def __init__(
        self, 
        incoming, 
        num_filters, 
        radius, 
        nonlinearity = lasagne.nonlinearities.rectify, 
        W=lasagne.init.HeNormal(gain='relu'), 
        b=lasagne.init.Constant(0),
        pos_dep_bias = False,
        padding = 0,
         **kwargs):
    
        super(HexConvLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.radius = radius
        self.padding = padding
        self.nonlinearity = nonlinearity
        self.pos_dep_bias = pos_dep_bias
        W_size = 2*sum([i+radius for i in range(radius-1)])+2*radius-1
        self.W_values = self.add_param(W, (self.num_filters, self.input_shape[1], W_size), name='W')
        if(self.pos_dep_bias):
            self.b = self.add_param(b, (num_filters, self.input_shape[2]-2*(self.radius-1), self.input_shape[3]-2*(self.radius-1)), name='b')
        else:
            self.b = self.add_param(b, (num_filters,), name='b')

    def get_output_for(self, input, **kwargs):
        W = T.zeros((self.num_filters, self.input_shape[1], 2*self.radius-1, 2*self.radius-1))
        index=0
        for i in range(self.radius-1):
            W = T.set_subtensor(W[:,:,self.radius-1-i:,i], self.W_values[:,:,index:index+self.radius+i])
            index = index+self.radius+i
        W = T.set_subtensor(W[:,:,:,self.radius-1], self.W_values[:,:,index:index+2*self.radius-1])
        index = index+2*self.radius-1
        for i in range(self.radius-1):
            W = T.set_subtensor(W[:,:,:2*self.radius-2-i,self.radius+i],self.W_values[:,:,index:index+2*(self.radius-1)-i])
            index = index+2*(self.radius-1)-i

        conv_out = conv.conv2d(
            input = input,
            filters = W,
            filter_shape = (self.num_filters,self.input_shape[1],2*self.radius-1,2*self.radius-1),
            image_shape = self.input_shape
        )

        if(self.pos_dep_bias):
            squished_out = self.nonlinearity(conv_out + self.b.dimshuffle('x', 0, 1, 2))
        else:
            squished_out = self.nonlinearity(conv_out + self.b.dimshuffle('x',0,'x','x'))

        if(self.padding == 0):
            padded_out = squished_out
        else:
            padded_out = T.zeros((squished_out.shape[0], self.num_filters, self.input_shape[2]+2*(self.padding+1-self.radius), self.input_shape[3]+2*(self.padding+1-self.radius)))
            padded_out = T.set_subtensor(padded_out[:,:,self.padding:-self.padding,self.padding:-self.padding], squished_out)

        return padded_out

    def get_output_shape_for(self, input_shape):
        return (self.input_shape[0], self.num_filters, self.input_shape[2]+2*(self.padding+1-self.radius), self.input_shape[3]+2*(self.padding+1-self.radius))
