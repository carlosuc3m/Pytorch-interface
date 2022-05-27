package plugins.carlosuc3m.pytorch.test;

import org.bioimageanalysis.icy.pytorch.v1.tensor.NDarrayBuilder;
import org.bioimageanalysis.icy.pytorch.v1.tensor.Nd4jBuilder;
import org.nd4j.linalg.api.ndarray.INDArray;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * Test Plugin taking the first image of a sequence and converting it to tensor and back to image, using the specified tensor dimension order.
 * 
 * @author Carlos Garcia Lopez de Haro
 */
public class ImageToTensorImageToImageTest// extends EzPlug
{
    //private EzVarSequence varInSequence;
    //private EzVarIntegerArrayNative varInTensorDimOrder;

    //@Override
    protected void initialize()
    {/*
        varInSequence = new EzVarSequence("Sequence (only one image)");
        varInTensorDimOrder = new EzVarIntegerArrayNative("Tensor dimension order", new int[][] {{0, 2, 1}}, true);
        addEzComponent(varInSequence);
        addEzComponent(varInTensorDimOrder);*/
    }

    //@Override
    protected void execute()
    {/*
        Sequence image = varInSequence.getValue(true);
        int[] tensorDimOrder = varInTensorDimOrder.getValue(true);
        tensorDimOrder = new int[] {4,2,1,0};

        long tStart = 0, tTensor = 0, tResult = 0;
        tStart = System.currentTimeMillis();
        try (NDManager man = NDManager.newBaseManager())
        {
            tTensor = System.currentTimeMillis();
        	INDArray aa = SequenceToNd4j.build(image, tensorDimOrder);
        	NDArray nn = NDarrayBuilder.build(aa, man);
        	INDArray aa2 = Nd4jBuilder.build(nn);
            Sequence seq = Nd4jToSequence.build(aa2, tensorDimOrder);
            tResult = System.currentTimeMillis();
            addSequence(seq);
        }

        long tConversion1 = tTensor - tStart;
        long tConversion2 = tResult - tTensor;
        System.out.println("Conversion to tensor = " + tConversion1 + "msec");
        System.out.println("Conversion to image = " + tConversion2 + "msec");
        */
    }

    public static void main(String[] args)
    {/*
        Icy.main(args);
        PluginLauncher.start(PluginLoader.getPlugin(ImageToTensorImageToImageTest.class.getName()));
        */
    }

	//@Override
	public void clean() {
		// TODO Auto-generated method stub
		
	}
}
