package airfoil;

import com.maxeler.maxcompiler.v1.managers.MAXBoardModel;
import com.maxeler.maxcompiler.v1.managers.custom.CustomManager;
import com.maxeler.maxcompiler.v1.managers.custom.Stream;
import com.maxeler.maxcompiler.v1.managers.custom.blocks.Fanout;
import com.maxeler.maxcompiler.v1.managers.custom.blocks.KernelBlock;
import com.maxeler.maxcompiler.v1.managers.custom.stdlib.MemoryControlGroup;

public class AifoilManager extends CustomManager {

	public AifoilManager(MAXBoardModel board_model, String name) {
		super(board_model, name, CustomManager.Target.MAXFILE_FOR_HARDWARE);

		MemoryControlGroup control = addMemoryControlGroup("control", MemoryControlGroup.MemoryAccessPattern.LINEAR_1D);

		KernelBlock resCalc = addKernel(new ResCalcKernel(makeKernelParameters("ResCalcKernel")));
		Stream in_host = addStreamFromHost("input");
		Stream in_dram = addStreamFromOnCardMemory("input_dram", control);


		resCalc.getInput("input_host") <== in_host;
		resCalc.getInput("input_dram") <== in_dram;

		Stream result = resCalc.getOutput("result");
		Fanout fanout = fanout("dummy_fanout");
		fanout.getInput() <== result;

		Stream to_host = addStreamToHost("res");
		to_host <== fanout.addOutput("to_host");
		Stream to_dram = addStreamToOnCardMemory("to_dram", control);
		to_dram <== fanout.addOutput("to_dram");


		config.setAllowNonMultipleTransitions(true);//FIXME: Must remove later!!!!!!
	}


}
