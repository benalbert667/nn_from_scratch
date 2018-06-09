import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Random;

public class Test {

    public static void main(String[] args) throws IOException {
        float[][][] trainData = getData("./data/train-images-idx3-ubyte",
                "./data/train-labels-idx1-ubyte");
        float[][][] testData = getData("./data/t10k-images-idx3-ubyte",
                "./data/t10k-labels-idx1-ubyte");
        FeedForwardNetwork god = train(trainData[0], trainData[1], testData[0], testData[1]);

    }

    private static float[][][] getData(String imgsFilePath, String lblsFilePath) throws IOException {
        File trainImagesFile = new File(imgsFilePath);
        File trainLabelsFile = new File(lblsFilePath);
        FileInputStream images = new FileInputStream(trainImagesFile);
        FileInputStream labels = new FileInputStream(trainLabelsFile);
        byte[] four = new byte[4];
        images.read(four);
        if(ByteBuffer.wrap(four).getInt() != 2051) {
            System.out.println("Didn't see the magic number...");
            return null;
        } else {
            labels.skip(8);
            images.read(four);
            int numImages = ByteBuffer.wrap(four).getInt();
            images.read(four);
            int h = ByteBuffer.wrap(four).getInt();
            images.read(four);
            int w = ByteBuffer.wrap(four).getInt();
            float[][] imgs = new float[numImages][h*w];
            float[][] lbls = new float[numImages][10];
            System.out.println("Loading images...");
            for(int i = 0; i < numImages; i++) {
                for(int j = 0; j < h*w; j++) {
                    imgs[i][j] = (float) images.read();
                }
                lbls[i] = intToArr(labels.read());
                if(i == numImages*0.25) {
                    System.out.println("25%");
                } else if(i == numImages*0.5) {
                    System.out.println("50%");
                } else if(i == numImages*0.75) {
                    System.out.println("75%");
                } else if(i == numImages - 1) {
                    System.out.println("100%");
                }
            }
            return new float[][][]{imgs, lbls};
        }
    }

    private static FeedForwardNetwork train(float[][] x, float[][] y, float[][] test_x, float[][] test_y) {
        FeedForwardNetwork ffa = new FeedForwardNetwork(new int[]{784, 30, 10});
        ffa.randomize();

        float[][][] derivs = null;
        float learningRate = 3.0f;
        int batchSize = 10;
        int numEpochs = 30;

        int epochs = 0;
        while(epochs < numEpochs) {
            shuffleImgs(x, y);
            // Train
            int numInSum = 0;
            for(int i = 0; i < x.length; i++) {
                // Sum derivatives of batch
                if(derivs == null) {
                    derivs = ffa.getError(x[i], y[i]);
                    numInSum = 1;
                } else {
                    float[][][] toAdd = ffa.getError(x[i], y[i]);
                    for(int j = 0; j < 2; j++) {
                        for(int k = 0; k < derivs[j].length; k++) {
                            for(int l = 0; l < derivs[j][k].length; l++) {
                                derivs[j][k][l] += toAdd[j][k][l];
                            }
                        }
                    }
                    numInSum++;
                }

                // After each batch, perform gradient descent
                if(numInSum == batchSize || i == x.length - 1) {
                    for(int j = 0; j < derivs.length; j++) {
                        for(int k = 0; k < derivs[j].length; k++) {
                            for(int l = 0; l < derivs[j][k].length; l++) {
                                derivs[j][k][l] /= (float) numInSum;
                            }
                        }
                    }
                    ffa.gradientDescent(derivs, learningRate);
                    derivs = null;
                }
            }

            //Test performance
            int score = 0;
            for(int i = 0; i < test_x.length; i++) {
                float[] response = ffa.process(test_x[i]);
                if(arrToInt(test_y[i]) == arrToInt(response)){
                    score++;
                }
            }
            System.out.println("Epoch " + epochs + ": " + score + "/" + test_x.length);
            epochs++;
        }
        return ffa;
    }

    private static int arrToInt(float[] x) {
        float max = Float.MIN_VALUE;
        int maxInd = 0;
        for(int i = 0; i < x.length; i++) {
            if(x[i] > max) {
                max = x[i];
                maxInd = i;
            }
        }
        return maxInd;
    }
    private static float[] intToArr(int x) {
        float[] y = new float[10];
        for(int i = 0; i < 10; i++) {
            if(i == x) {
                y[i] = 1.0f;
            } else {
                y[i] = 0.0f;
            }
        }
        return y;
    }

    private static void shuffleImgs(float[][] x, float[][] y) {
        Random r = new Random();
        for(int i = x.length - 1; i > 0; i--) {
            int indx = r.nextInt(i + 1);
            float[] temp = x[i];
            x[i] = x[indx];
            x[indx] = temp;

            temp = y[i];
            y[i] = y[indx];
            y[indx] = temp;
        }
    }

    private static void display(int[][] data, int width, int height) {
        final BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D)img.getGraphics();
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                int c = data[i][j];
                g.setColor(new Color(c, c, c));
                g.fillRect(j, i, 1, 1);
            }
        }

        JFrame frame = new JFrame(";)");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JPanel panel = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                Graphics2D g2d = (Graphics2D)g;
                g2d.clearRect(0, 0, getWidth(), getHeight());
                g2d.setRenderingHint(
                        RenderingHints.KEY_INTERPOLATION,
                        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                // Or _BICUBIC
                g2d.scale(2, 2);
                g2d.drawImage(img, 0, 0, this);
            }
        };
        panel.setPreferredSize(new Dimension(width*2, height*2));
        frame.getContentPane().add(panel);
        frame.pack();
        frame.setVisible(true);
    }
}
