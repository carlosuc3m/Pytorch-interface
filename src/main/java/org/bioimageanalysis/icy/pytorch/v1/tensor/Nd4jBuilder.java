package org.bioimageanalysis.icy.pytorch.v1.tensor;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ai.djl.ndarray.NDArray;

public class Nd4jBuilder {


    /**
     * Creates a {@link INDArray} from a given {@link Tensor} and an array with its dimensions order.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    public static INDArray build(NDArray tensor) throws IllegalArgumentException
    {
        // Create an Icy sequence of the same type of the tensor
        switch (tensor.getDataType())
        {
	        case UINT8:
	        case INT8:
                return buildFromTensorByte(tensor);
            case INT32:
                return buildFromTensorInt(tensor);
            case FLOAT32:
                return buildFromTensorFloat(tensor);
            case FLOAT64:
                return buildFromTensorDouble(tensor);
            default:
                throw new IllegalArgumentException("Unsupported tensor type: " + tensor.getDataType());
        }
    }

    /**
     * Builds a {@link INDArray} from a unsigned byte-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static INDArray buildFromTensorByte(NDArray tensor)
    {
		long[] tensorShape = tensor.getShape().getShape();
		return Nd4j.create(tensor.toByteArray(), tensorShape, DataType.INT8);
	}

    /**
     * Builds a {@link INDArray} from a unsigned integer-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#INT}.
     */
    private static INDArray buildFromTensorInt(NDArray tensor)
    {
		long[] tensorShape = tensor.getShape().getShape();
		return Nd4j.create(tensor.toIntArray(), tensorShape, DataType.INT32);
    }

    /**
     * Builds a {@link INDArray} from a unsigned float-typed {@link Tensor}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static INDArray buildFromTensorFloat(NDArray tensor)
    {
		long[] tensorShape = tensor.getShape().getShape();
		return Nd4j.create(tensor.toFloatArray(), tensorShape, DataType.FLOAT);
    }

    /**
     * Builds a {@link INDArray} from a unsigned double-typed {@link NDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @return The INDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static INDArray buildFromTensorDouble(NDArray tensor)
    {
		long[] tensorShape = tensor.getShape().getShape();
		return Nd4j.create(tensor.toDoubleArray(), tensorShape, DataType.DOUBLE);
    }
}
