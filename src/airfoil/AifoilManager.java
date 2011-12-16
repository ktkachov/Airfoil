package airfoil;

import com.maxeler.maxcompiler.v1.managers.MAXBoardModel;
import com.maxeler.maxcompiler.v1.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v1.managers.custom.Stream;
import com.maxeler.maxcompiler.v1.managers.custom.blocks.KernelBlock;

public class AifoilManager extends CustomManager {

	public AifoilManager(MAXBoardModel board_model, String name, int numReps) {
		super(board_model, name);

//			KernelBlock adtCalc = addKernel(new ADTKernel(makeKernelParameters("ADTCalcKernel")));
//			Stream q = addStreamFromHost("q");
//			adtCalc.getInput("q") <== q;
//			Stream[] xs = new Stream[4];
//			for (int j = 0; j < xs.length; ++j) {
//				xs[j] = addStreamFromHost("x"+(j+1));
//				adtCalc.getInput("x"+(j+1)) <== xs[j];
//			}
//			Stream adt = addStreamToHost("adt");
//			adt <== adtCalc.getOutput("adt");
		KernelBlock resCalc = addKernel(new ResCalcKernel(makeKernelParameters("ResCalcKernel")));
		Stream in = addStreamFromHost("input");
		resCalc.getInput("input") <== in;
		addStreamToHost("res1") <== resCalc.getOutput("res1");
		addStreamToHost("res2") <== resCalc.getOutput("res2");


		config.setAllowNonMultipleTransitions(true);//FIXME: Must remove later!!!!!!


//		KernelBlock adtCalc = addKernel(new ADTKernel(makeKernelParameters("ADTCalcKernel")));
//		KernelBlock resCalc = addKernel(new ResCalcKernel(makeKernelParameters("ResCalcKernel")));
//		KernelBlock bresCalc = addKernel(new BResCalcKernel(makeKernelParameters("BResCalcKernel")));
//		KernelBlock update = addKernel(new UpdateKernel(makeKernelParameters("UpdateKernel")));


	}


}
