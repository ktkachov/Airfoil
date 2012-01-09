package airfoil;

import com.maxeler.maxcompiler.v1.managers.MAXBoardModel;
import com.maxeler.maxcompiler.v1.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v1.managers.custom.Stream;
import com.maxeler.maxcompiler.v1.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v1.managers.custom.stdlib.MemoryControlGroup.MemoryAccessPattern;

public class AirfoilManager extends CustomManager {

	public AirfoilManager(MAXBoardModel board_model, String name) {
		super(board_model, name, CustomManager.Target.MAXFILE_FOR_HARDWARE);

		KernelBlock resCalc = addKernel(new ResCalcKernel(makeKernelParameters("ResCalcKernel")));
		Stream in_host = addStreamFromHost("input");
		Stream in_dram = addStreamFromOnCardMemory("from_fram", MemoryAccessPattern.LINEAR_1D);


		resCalc.getInput("input_host") <== in_host;
		resCalc.getInput("input_dram") <== in_dram;

		Stream to_host = addStreamToHost("res");
		to_host <== resCalc.getOutput("result_host");
		Stream to_dram = addStreamToOnCardMemory("to_dram", MemoryAccessPattern.LINEAR_1D);
		to_dram <== resCalc.getOutput("result_dram");


		config.setAllowNonMultipleTransitions(true);//FIXME: Must remove later!!!!!!
	}


}
