package org.bioimageanalysis.icy.pytorch.v1.tensor;

import org.bioimageanalysis.icy.deeplearning.tensor.Tensor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;


public class NDarrayBuilder {


    /**
     * Creates a {@link NDArray} from a given {@link Tensor} and an array with its dimensions order.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    public static NDArray build(Tensor tensor, NDManager manager) throws IllegalArgumentException
    {
        // Create an Icy sequence of the same type of the tensor
    	if (tensor.getDataType() == DataType.INT8) {
            return buildFromTensorByte( tensor.getDataAsNDArray(), manager);
    	} else if (tensor.getDataType() == DataType.INT32) {
            return buildFromTensorInt( tensor.getDataAsNDArray(), manager);
    	} else if (tensor.getDataType() == DataType.FLOAT) {
            return buildFromTensorFloat( tensor.getDataAsNDArray(), manager);
    	} else if (tensor.getDataType() == DataType.DOUBLE) {
            return buildFromTensorDouble( tensor.getDataAsNDArray(), manager);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.getDataType());
    	}
    }
    /**
     * Creates a {@link NDArray} from a given {@link INDArray} and an array with its dimensions order.
     * 
     * @param tensor
     *        The INDArray containing the wanted data.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor.
     * @throws IllegalArgumentException
     *         If the tensor type is not supported.
     */
    public static NDArray build(INDArray tensor, NDManager manager) throws IllegalArgumentException
    {
    	if (tensor.dataType() == DataType.INT8) {
            return buildFromTensorByte( tensor, manager);
    	} else if (tensor.dataType() == DataType.INT32) {
            return buildFromTensorInt( tensor, manager);
    	} else if (tensor.dataType() == DataType.FLOAT) {
            return buildFromTensorFloat( tensor, manager);
    	} else if (tensor.dataType() == DataType.DOUBLE) {
            return buildFromTensorDouble( tensor, manager);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType());
    	}
    }

    /**
     * Builds a {@link NDArray} from a unsigned byte-typed {@link INDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#UBYTE}.
     */
    private static NDArray buildFromTensorByte(INDArray tensor, NDManager manager)
    {
    	long[] tensorShape = tensor.shape();
	 	NDArray ndarray = manager.create(tensor.data().asBytes(), new Shape(tensorShape));
		
		return ndarray;
	}

    /**
     * Builds a {@link NDArray} from a unsigned integer-typed {@link INDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#INT}.
     */
    private static NDArray buildFromTensorInt(INDArray tensor, NDManager manager)
    {
    	long[] tensorShape = tensor.shape();
	 	NDArray ndarray = manager.create(tensor.data().asInt(), new Shape(tensorShape));
	 	return ndarray;
    }

    /**
     * Builds a {@link NDArray} from a unsigned float-typed {@link INDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#FLOAT}.
     */
    private static NDArray buildFromTensorFloat(INDArray tensor, NDManager manager)
    {
    	long[] tensorShape = tensor.shape();
	 	NDArray ndarray = manager.create(tensor.data().asFloat(), new Shape(tensorShape));
	 	return ndarray;
    }

    /**
     * Builds a {@link NDArray} from a unsigned double-typed {@link INDArray}.
     * 
     * @param tensor
     *        The tensor data is read from.
     * @param manager
     *        {@link NDManager} needed to create NDArrays
     * @return The NDArray built from the tensor of type {@link DataType#DOUBLE}.
     */
    private static NDArray buildFromTensorDouble(INDArray tensor, NDManager manager)
    {
    	long[] tensorShape = tensor.shape();
	 	NDArray ndarray = manager.create(tensor.data().asDouble(), new Shape(tensorShape));
	 	return ndarray;
    }
}
