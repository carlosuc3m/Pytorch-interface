package org.bioimageanalysis.icy.pytorch.v1.tensor;



/**
 * Class that manages the shape of nd4j arrays
 * @author Carlos Garcia Lopez de Haro
 *
 */
public class Nd4jShapeUtils
{
    /**
     * Creates a tensor shape from an int array
     * 
     * @param shapeArr
     * 	int array with the size of each dimension
     * @return Shape with the image dimensions in the desired order.
     */
    public static long[] fromArray(int[] shapeArr)
    {
        long[] dimensionSizes = new long[shapeArr.length];
        for (int i = 0; i < dimensionSizes.length; i++)
        {
        	dimensionSizes[i] = (long) shapeArr[i];
        }
        return dimensionSizes;
    }

}
