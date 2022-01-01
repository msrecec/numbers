import org.encog.Encog;
import org.encog.EncogError;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.train.strategy.ResetStrategy;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.platformspecific.j2se.TrainingDialog;
import org.encog.platformspecific.j2se.data.image.ImageMLData;
import org.encog.platformspecific.j2se.data.image.ImageMLDataSet;
import org.encog.util.downsample.Downsample;
import org.encog.util.downsample.RGBDownsample;
import org.encog.util.downsample.SimpleIntensityDownsample;
import org.encog.util.simple.EncogUtility;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.*;
import java.util.List;

public class Main {

    class ImagePair {
        private final File file;
        private final int identity;

        public ImagePair(final File file, final int identity) {
            super();
            this.file = file;
            this.identity = identity;
        }

        public File getFile() {
            return this.file;
        }

        public int getIdentity() {
            return this.identity;
        }
    }

    static public double start;
    static public double end;

    public static void main(String[] args) {
        try {
            final Main program = new Main();
            program.execute("config.txt");
            System.out.println("Vrijeme: " + (end - start) / 1000000000);
        } catch (final Exception e) {
            e.printStackTrace();
        }

        Encog.getInstance().shutdown();
    }

    private final List<ImagePair> imageList = new ArrayList<ImagePair>();
    private final Map<String, String> args = new HashMap<String, String>();
    private final Map<String, Integer> identity2neuron = new HashMap<String, Integer>();
    private final Map<Integer, String> neuron2identity = new HashMap<Integer, String>();
    private ImageMLDataSet training;
    private String line;
    private int outputCount;
    private int downsampleWidth;
    private int downsampleHeight;
    private BasicNetwork network;

    private Downsample downsample;

    private int assignIdentity(final String identity) {

        if (this.identity2neuron.containsKey(identity.toLowerCase())) {
            return this.identity2neuron.get(identity.toLowerCase());
        }

        final int result = this.outputCount;
        this.identity2neuron.put(identity.toLowerCase(), result);
        this.neuron2identity.put(result, identity.toLowerCase());
        this.outputCount++;
        return result;
    }

    public void execute(final String file) throws IOException {
        final FileInputStream fstream = new FileInputStream(file);
        final DataInputStream in = new DataInputStream(fstream);
        final BufferedReader br = new BufferedReader(new InputStreamReader(in));

        while ((this.line = br.readLine()) != null) {
            executeLine();
        }
        in.close();
    }

    private void executeCommand(final String command, final Map<String, String> args) throws IOException {
        if (command.equals("input")) {
            processInput();
        } else if (command.equals("createtraining")) {
            processCreateTraining();
        } else if (command.equals("train")) {
            processTrain();
        } else if (command.equals("network")) {
            processNetwork();
        } else if (command.equals("whatis")) {
            processWhatIs();
        }

    }

    public void executeLine() throws IOException {
        final int index = this.line.indexOf(':');
        if (index == -1) {
            throw new EncogError("Invalid command: " + this.line);
        }

        final String command = this.line.substring(0, index).toLowerCase().trim();
        final String argsStr = this.line.substring(index + 1).trim();
        final StringTokenizer tok = new StringTokenizer(argsStr, ",");
        this.args.clear();
        while (tok.hasMoreTokens()) {
            final String arg = tok.nextToken();
            final int index2 = arg.indexOf(':');
            if (index2 == -1) {
                throw new EncogError("Invalid command: " + this.line);
            }
            final String key = arg.substring(0, index2).toLowerCase().trim();
            final String value = arg.substring(index2 + 1).trim();
            this.args.put(key, value);
        }

        executeCommand(command, this.args);
    }

    private String getArg(final String name) {
        final String result = this.args.get(name);
        if (result == null) {
            throw new EncogError("Missing argument " + name + " on line: " + this.line);
        }
        return result;
    }

    private void processCreateTraining() {
        final String strWidth = getArg("width");
        final String strHeight = getArg("height");
        final String strType = getArg("type");

        this.downsampleHeight = Integer.parseInt(strHeight);
        this.downsampleWidth = Integer.parseInt(strWidth);

        if (strType.equals("RGB")) {
            this.downsample = new RGBDownsample();
        } else {
            this.downsample = new SimpleIntensityDownsample();
        }

        this.training = new ImageMLDataSet(this.downsample, false, 1, -1);
        System.out.println("Training set created");
    }

    private void processInput() throws IOException {
        final String image = getArg("image");
        final String identity = getArg("identity");

        final int idx = assignIdentity(identity);
        final File file = new File(image);

        this.imageList.add(new ImagePair(file, idx));

        System.out.println("Added input image:" + image);
    }

    private void processNetwork() throws IOException {
        System.out.println("Downsampling images...");

        for (final ImagePair pair : this.imageList) {
            final MLData ideal = new BasicMLData(this.outputCount);
            final int idx = pair.getIdentity();
            for (int i = 0; i < this.outputCount; i++) {
                if (i == idx) {
                    ideal.setData(i, 1);
                } else {
                    ideal.setData(i, -1);
                }
            }

            final Image img = ImageIO.read(pair.getFile());
            final ImageMLData data = new ImageMLData(img);
            this.training.add(data, ideal);
        }

        final String strHidden1 = getArg("hidden1");
        final String strHidden2 = getArg("hidden2");

        this.training.downsample(this.downsampleHeight, this.downsampleWidth);

        final int hidden1 = Integer.parseInt(strHidden1);
        final int hidden2 = Integer.parseInt(strHidden2);

        this.network = EncogUtility.simpleFeedForward(this.training.getInputSize(), hidden1, hidden2,
                this.training.getIdealSize(), true);
        System.out.println("Created network: " + this.network.toString());
    }

    private void processTrain() throws IOException {
        final String strMode = getArg("mode");
        final String strStrategyError = getArg("strategyerror");
        final String strStrategyCycles = getArg("strategycycles");

        start = System.nanoTime();

        System.out.println("Training Beginning... Output patterns=" + this.outputCount);

        final double strategyError = Double.parseDouble(strStrategyError);
        final int strategyCycles = Integer.parseInt(strStrategyCycles);

        final ResilientPropagation train = new ResilientPropagation(this.network, this.training);
        train.addStrategy(new ResetStrategy(strategyError, strategyCycles));

        if (strMode.equalsIgnoreCase("gui")) {
            TrainingDialog.trainDialog(train, this.network, this.training);
        } else {
            int i = 0;
            while (network.calculateError(training) > 0.001) {
                train.iteration();
                i++;
                System.out.println("Iteration: " + i + ". with error:" + network.calculateError(training));
            }
            ;
        }
        System.out.println("Training Stopped...");

        end = System.nanoTime();
    }

    public void processWhatIs() throws IOException {
        final String filename = getArg("image");
        final File file = new File(filename);
        final Image img = ImageIO.read(file);
        if (img.getWidth(null) < 50) {
            final ImageMLData input = new ImageMLData(img);
            input.downsample(this.downsample, false, this.downsampleHeight, this.downsampleWidth, 1, -1);

            final int winner = this.network.winner(input);
            System.out.println("What is: " + filename + ", it seems to be: " + this.neuron2identity.get(winner));
        } else {
            String output = "";
            BufferedImage im = ImageIO.read(file);
            BufferedImage sub = im.getSubimage(0, 0, img.getWidth(null), img.getHeight(null));
            File test = new File(".\\numbers\\temp.png");
//            ImageIO.write(sub, "png", test);
            int end = sub.getWidth() / 25;

            System.out.print("What is: " + filename + ", it seems to be: ");
            for (int i = 0; i < end; i++) {
                test = new File(".\\numbers\\temp"+ i +".png");
                ImageIO.write(sub.getSubimage(25 * i, 0, 25, sub.getHeight()), "png", test);
                Image one = ImageIO.read(test);
                final ImageMLData input = new ImageMLData(one);
                input.downsample(this.downsample, false, this.downsampleHeight, this.downsampleWidth, 1, -1);

                final int winner = this.network.winner(input);
                if (this.neuron2identity.get(winner).equals("space"))
                    output += " ";
                else
                    output += this.neuron2identity.get(winner);
                test.delete();
            }
            output = String.join("",output.split("/|-"));
            System.out.println(output);
        }
    }
}
