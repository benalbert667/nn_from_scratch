import java.util.Random;

public class FeedForwardNetwork {

    private float[][][] w;  // weights
    // Weight w[i][j][k] is the weight from the kth neuron in the i-1th layer to the jth neuron in the ith layer.
    private float[][] b;  // biases
    // Bias b[i][j] is the bias of the jth neuron in the ith layer

    public FeedForwardNetwork(int[] layerSizes) {
        int numLayers = layerSizes.length;
        w = new float[numLayers][][];
        b = new float[numLayers][];
        int prevLayerSize = 1;  // 1 input for each input layer neuron
        for(int i = 0; i < numLayers; i++) {
            int layerSize = layerSizes[i];
            b[i] = new float[layerSize];
            w[i] = new float[layerSize][];
            for(int j = 0; j < layerSize; j++) {
                w[i][j] = new float[prevLayerSize];
            }
            prevLayerSize = layerSize;
        }
        randomize();
    }

    /**
     * Randomizes the weights and biases of the network
     */
    public void randomize() {
        Random r = new Random();
        for(int i = 0; i < w.length; i++) {
            for(int j = 0; j < w[i].length; j++) {
                if(i == 0) {
                    b[i][j] = 0;  // don't include first layer, activations should be equal to input
                } else {
                    b[i][j] = (float) r.nextGaussian();
                }
                for(int k = 0; k < w[i][j].length; k++) {
                    if(i == 0) {
                        w[i][j][k] = 1.0f;  // don't include first layer, activations should be equal to input
                    } else {
                        w[i][j][k] = (float) r.nextGaussian();
                    }
                }
            }
        }
    }

    /**
     * Processes a vector through the network.
     *
     * @param in input vector
     * @return output vector of the network
     */
    public float[] process(float[] in) {
        for(int l = 0; l < w.length; l++) {
            in = layerActivation(in, l)[0];
        }
        return in;
    }

    /**
     * Returns the error and output values of each neuron
     * to be used in gradient descent
     * @param x input vector
     * @param ans expected output of network
     * @return the error and activations for each neuron
     */
    public float[][][] getError(float[] x, float[] ans) {
        float[][] z = new float[w.length][];
        float[][] a = new float[w.length][];
        for(int i = 0; i < w.length; i++) {
            float[][] layer = layerActivation(x, i);
            z[i] = layer[1];
            x = a[i] = layer[0];
        }

        int n_layers = w.length;
        float[][] err = new float[n_layers][];
        err[n_layers - 1] = bp1QC(ans, z[n_layers - 1], a[n_layers - 1]);
        for(int l = n_layers - 2; l >= 0; l--) {
            err[l] = bp2(err[l+1], z[l], l);  // backpropagation
        }

        return new float[][][]{err, a};
    }

    /**
     * Applies gradient descent on the weights and biases
     * @param derivatives partial derivatives of cost function and the activation functions
     * @param lr the learning rate
     */
    public void gradientDescent(float[][][] derivatives, float lr) {
        for(int i = 1; i < w.length; i++) {  // don't include first layer, activations should be equal to input
            for(int j = 0; j < w[i].length; j++) {
                b[i][j] -= lr*derivatives[0][i][j];  // bp4
                for(int k = 0; k < w[i][j].length; k++) {
                    w[i][j][k] -= lr*derivatives[0][i][j]*derivatives[1][i-1][k];  // bp3
                }
            }
        }
    }

    private float[] bp1QC(float[] y, float[] zL, float[] aL) {
        float[] err = new float[y.length];
        for(int i = 0; i < err.length; i++) {
            err[i] = (aL[i] - y[i])*sigPrime(zL[i]);  // equation bp1 with quadratic cost
        }
        return err;
    }

    private float[] bp2(float[] errPrevLayer, float[] zL, int l) {
        float[] err = new float[w[l].length];
        for(int k = 0; k < w[l].length; k++) {
            float weightedError = 0;
            for(int j = 0; j < w[l+1].length; j++) {
                weightedError += w[l+1][j][k]*errPrevLayer[j];
            }
            err[k] = weightedError*sigPrime(zL[k]);
        }
        return err;
    }

    private float[][] layerActivation(float[] in, int layer) {
        float[][] out = new float[2][w[layer].length];
        for(int i = 0; i < w[layer].length; i++) {
            if(layer == 0) {
                out[1][i] = weightedSum(new float[]{in[i]}, layer, i);
            } else {
                out[1][i] = weightedSum(in, layer, i);
            }
            out[0][i] = sigmoid(out[1][i]);
        }
        return out;  // out[0] is a (the layer activation), out[1] is z (the layer weighted sums)
    }

    private float weightedSum(float[] in, int i, int j) {
        float z = 0;
        for(int k = 0; k < in.length; k++) {
            z += in[k] * w[i][j][k];
        }
        z += b[i][j];
        return z;
    }

    private float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    private float sigPrime(float x) {
        float s = sigmoid(x);
        return s*(1 - s);
    }
}
